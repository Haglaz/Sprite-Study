import { evaluateSprite } from '../critic/evaluator';
import type { CriticCalibration, CriticEvaluation } from '../critic/types';
import {
  applyNeatPixelRefinementPass,
  applyRecipePipeline,
  opDespeckleClusters,
  opEnforceDarkOutline,
  opRemoveOrphans,
  opShiftNwLighting,
  opSnapBinaryAlpha,
  opSnapNearDuplicates,
  opTrimOutlineDoubles,
} from '../evolution/pixelOps';
import type { CoevolutionChampion } from '../evolution/types';
import { retrieveLessonsForQuery } from '../lessons/retrieval';
import type { LessonRecord } from '../lessons/types';
import type { ItemCategory } from '../types/studio';
import {
  buildDeterministicDiffEdits,
  executeBoundedPixelDiff,
  executeDonorComposite,
  executeRampRecolor,
  selectMaterialRampForPrompt,
} from './executors';
import {
  decomposeSpriteToIndexedAndRampRoles,
  findNearestRealSprite,
  retrieveDonorsForPrompt,
} from './indexer';
import type {
  ApprovalLabelRecord,
  GeneratedSpriteRecord,
  GenerationProvenanceSource,
  GenerationTier,
  PixelDiffEdit,
  SpriteIndexState,
} from './types';

export interface BenchmarkPromptSpec {
  id: string;
  prompt: string;
  category: ItemCategory;
  tier: GenerationTier;
  maxDiffPixels: number;
}

/**
 * 10 fixed classic Tibia 7.72 benchmark prompts spanning Equipamentos, Consumíveis, Containers, and Decoração,
 * mixing Tier T1 (recolor + bounded pixel diff) and Tier T2 (donor composite + palette unification + bounded diff).
 */
export const BENCHMARK_10_PROMPTS: readonly BenchmarkPromptSpec[] = [
  {
    id: 'bench-01',
    prompt: 'Armadura azul real com detalhes em ouro estilo paladino 7.72',
    category: 'Equipamentos',
    tier: 'T1',
    maxDiffPixels: 28,
  },
  {
    id: 'bench-02',
    prompt: 'Escudo rubi de fogo forjado com borda dourada e gema central',
    category: 'Equipamentos',
    tier: 'T2',
    maxDiffPixels: 32,
  },
  {
    id: 'bench-03',
    prompt: 'Espada longa de aço prateado com guarda dourada e brilho NW',
    category: 'Equipamentos',
    tier: 'T1',
    maxDiffPixels: 24,
  },
  {
    id: 'bench-04',
    prompt: 'Elmo warlock de obsidiana roxa com adorno carmesim',
    category: 'Equipamentos',
    tier: 'T2',
    maxDiffPixels: 28,
  },
  {
    id: 'bench-05',
    prompt: 'Armadura esmeralda da floresta com fivela de ouro antigo',
    category: 'Equipamentos',
    tier: 'T2',
    maxDiffPixels: 32,
  },
  {
    id: 'bench-06',
    prompt: 'Poção arcana azul safira em frasco de vidro com rolha',
    category: 'Consumíveis',
    tier: 'T1',
    maxDiffPixels: 20,
  },
  {
    id: 'bench-07',
    prompt: 'Elixir de rubi vermelho brilhante de cura vital',
    category: 'Consumíveis',
    tier: 'T2',
    maxDiffPixels: 20,
  },
  {
    id: 'bench-08',
    prompt: 'Baú antigo de madeira bronzeada com fecho dourado',
    category: 'Containers',
    tier: 'T1',
    maxDiffPixels: 28,
  },
  {
    id: 'bench-09',
    prompt: 'Mochila de couro esmeralda reforçada com fivela metálica',
    category: 'Containers',
    tier: 'T2',
    maxDiffPixels: 28,
  },
  {
    id: 'bench-10',
    prompt: 'Cálice sagrado de ouro solar com cristal azul celeste',
    category: 'Decoração',
    tier: 'T1',
    maxDiffPixels: 24,
  },
] as const;

interface GeminiPlanResponse {
  primaryStops: Array<[number, number, number]>;
  accentStops: Array<[number, number, number]>;
  diffEdits: PixelDiffEdit[];
  model: string;
}

function clampByte(v: unknown): number {
  const n = typeof v === 'number' && Number.isFinite(v) ? v : 0;
  return Math.max(0, Math.min(255, Math.round(n)));
}

function calcLuma(r: number, g: number, b: number): number {
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

function parseRgbStops(raw: unknown): Array<[number, number, number]> | null {
  if (!Array.isArray(raw) || raw.length < 3) return null;
  const stops: Array<[number, number, number]> = [];
  for (const item of raw) {
    if (Array.isArray(item) && item.length >= 3) {
      stops.push([clampByte(item[0]), clampByte(item[1]), clampByte(item[2])]);
    }
  }
  if (stops.length < 3) return null;
  // Guarantee monotonic luma order before feeding the executor
  stops.sort((a, b) => calcLuma(a[0], a[1], a[2]) - calcLuma(b[0], b[1], b[2]));
  return stops;
}

function parseDiffEdits(raw: unknown): PixelDiffEdit[] {
  if (!Array.isArray(raw)) return [];
  const out: PixelDiffEdit[] = [];
  for (const item of raw) {
    if (!item || typeof item !== 'object') continue;
    const obj = item as Record<string, unknown>;
    const x = Number(obj.x);
    const y = Number(obj.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (x < 0 || x >= 32 || y < 0 || y >= 32) continue;
    const aRaw = obj.a === undefined ? 255 : Number(obj.a);
    out.push({
      x: Math.round(x),
      y: Math.round(y),
      r: clampByte(obj.r),
      g: clampByte(obj.g),
      b: clampByte(obj.b),
      a: aRaw >= 128 ? 255 : 0,
    });
  }
  return out;
}

/**
 * Calls the server-side Gemini proxy (`/api/gemini/generate`) to obtain a structured
 * palette ramp + bounded pixel diff plan. Throws on any HTTP or parse error (never fails silently).
 */
async function requestGeminiSpritePlan(params: {
  prompt: string;
  category: ItemCategory;
  tier: GenerationTier;
  maxDiffPixels: number;
  donorTags: string[];
  lessonDirectives: string[];
}): Promise<GeminiPlanResponse> {
  const { prompt, category, tier, maxDiffPixels, donorTags, lessonDirectives } = params;

  const systemInstruction = [
    'You are a Classic Tibia 7.72 32x32 pixel-art palette & pixel-diff planner.',
    'Return ONLY strict JSON matching:',
    '{"primaryStops":[[r,g,b],...5 colors darkest-to-brightest], "accentStops":[[r,g,b],...3 colors darkest-to-brightest], "diffEdits":[{"x":0..31,"y":0..31,"r":0..255,"g":0..255,"b":0..255,"a":255}]}',
    'Rules: first primaryStop MUST be a dark hue-shifted outline (luma < 0.18). Zero anti-aliasing. Strictly hue-shifted shadows (cooler/darker) and highlights (warmer/brighter). At most 12 focused interior diffEdits.',
  ].join(' ');

  const userPrompt = [
    `Category: ${category}`,
    `Tier: ${tier}`,
    `User request: ${prompt}`,
    `Donor sprite tags: ${donorTags.slice(0, 8).join(', ')}`,
    `Max diff pixels: ${maxDiffPixels}`,
    lessonDirectives.length > 0
      ? `Active learned lessons:\n- ${lessonDirectives.slice(0, 4).join('\n- ')}`
      : '',
  ]
    .filter(Boolean)
    .join('\n');

  const resp = await fetch('/api/gemini/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: userPrompt,
      systemInstruction,
      responseMimeType: 'application/json',
      temperature: 0.45,
    }),
  });

  const payload = (await resp.json().catch(() => ({}))) as {
    text?: string;
    model?: string;
    error?: string;
  };

  if (!resp.ok || !payload.text) {
    throw new Error(
      payload.error || `Servidor Gemini retornou HTTP ${resp.status} sem resposta válida.`,
    );
  }

  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(payload.text) as Record<string, unknown>;
  } catch {
    throw new Error('Gemini retornou JSON malformado para o plano de paleta/diff.');
  }

  const primaryStops = parseRgbStops(parsed.primaryStops);
  const accentStops = parseRgbStops(parsed.accentStops);
  if (!primaryStops || !accentStops) {
    throw new Error('Plano Gemini sem rampas primaryStops/accentStops válidas (mínimo 3 cores).');
  }

  return {
    primaryStops,
    accentStops,
    diffEdits: parseDiffEdits(parsed.diffEdits).slice(0, maxDiffPixels),
    model: payload.model || 'gemini',
  };
}

function countHardIssues(evaluation: CriticEvaluation): number {
  return evaluation.issues.filter((i) => i.severity === 'hard').length;
}

/**
 * Critic-guided repair pass:
 * 1) Tries promoted Phase 3 co-evolution Champion (if available for the category).
 * 2) Tries targeted discrete pixel-editor operations keyed to the candidate's actual Critic v1 issues.
 * Keeps a repair STRICTLY ONLY IF Critic v1 score improves by > 0.1 without introducing any new hard violation.
 */
export function runCriticGuidedRepair(params: {
  rgba: Uint8ClampedArray;
  baseEval: CriticEvaluation;
  category: ItemCategory;
  calibration: CriticCalibration;
  champion?: CoevolutionChampion | null;
}): {
  rgba: Uint8ClampedArray;
  evaluation: CriticEvaluation;
  repaired: boolean;
  repairDelta: number;
  repairMethod: 'none' | 'champion_coevolution' | 'critic_targeted_ops';
} {
  const { rgba, baseEval, category, calibration, champion } = params;
  const isTile = baseEval.isTile;
  const baseHard = countHardIssues(baseEval);

  let bestRgba: Uint8ClampedArray = rgba;
  let bestEval = baseEval;
  let bestMethod: 'none' | 'champion_coevolution' | 'critic_targeted_ops' = 'none';

  const tryCandidate = (
    nextRgba: Uint8ClampedArray,
    method: 'champion_coevolution' | 'critic_targeted_ops',
  ) => {
    const nextEval = evaluateSprite(nextRgba, { category, calibration, isTileOverride: isTile });
    const nextHard = countHardIssues(nextEval);
    if (nextHard <= baseHard && nextEval.score > bestEval.score + 0.1) {
      bestRgba = nextRgba;
      bestEval = nextEval;
      bestMethod = method;
    }
  };

  // Strategy A: Phase 3 promoted Champion pipeline (Recipe GA + NEAT local brush)
  if (champion) {
    const afterRecipe = applyRecipePipeline(rgba, champion.recipe.steps, isTile);
    const afterNeat = applyNeatPixelRefinementPass(afterRecipe, champion.neat, isTile);
    tryCandidate(afterRecipe, 'champion_coevolution');
    tryCandidate(afterNeat, 'champion_coevolution');
  }

  // Strategy B: Targeted discrete pixel-editor ops based on Critic v1 issues
  const issueCodes = new Set(baseEval.issues.map((i) => i.code));
  let targeted: Uint8ClampedArray = opSnapBinaryAlpha(rgba, 0.5);
  let ranAny = false;

  if (issueCodes.has('ORPHAN_PIXELS') || issueCodes.has('DISCONNECTED_SILHOUETTE')) {
    targeted = opRemoveOrphans(targeted, 0.45);
    ranAny = true;
  }
  if (issueCodes.has('NEAR_DUPLICATE_PALETTE') || issueCodes.has('SMOOTH_GRADIENT')) {
    targeted = opSnapNearDuplicates(targeted, 0.5);
    ranAny = true;
  }
  if (!isTile && (issueCodes.has('MISSING_OUTLINE') || issueCodes.has('BROKEN_OUTLINE'))) {
    targeted = opEnforceDarkOutline(targeted, 0.5, false);
    ranAny = true;
  }
  if (!isTile && issueCodes.has('OUTLINE_DOUBLES')) {
    targeted = opTrimOutlineDoubles(targeted, false);
    ranAny = true;
  }
  if (issueCodes.has('CLUSTER_CONFETTI') || issueCodes.has('SPECKLE_NOISE')) {
    targeted = opDespeckleClusters(targeted, 0.45);
    ranAny = true;
  }
  if (!isTile && (issueCodes.has('WEAK_NW_LIGHTING') || issueCodes.has('PILLOW_SHADING'))) {
    targeted = opShiftNwLighting(targeted, 0.45, false);
    ranAny = true;
  }

  if (ranAny) {
    tryCandidate(targeted, 'critic_targeted_ops');
  } else {
    // Even without flagged hygiene issues, test a conservative 1-op polish (snap near-duplicates + NW light)
    const gentleSnap = opSnapNearDuplicates(rgba, 0.35);
    tryCandidate(gentleSnap, 'critic_targeted_ops');
    if (!isTile) {
      const gentleNw = opShiftNwLighting(rgba, 0.3, false);
      tryCandidate(gentleNw, 'critic_targeted_ops');
    }
  }

  const repairDelta = Math.round((bestEval.score - baseEval.score) * 10) / 10;
  return {
    rgba: bestRgba,
    evaluation: bestEval,
    repaired: bestMethod !== 'none' && repairDelta > 0,
    repairDelta: Math.max(0, repairDelta),
    repairMethod: bestMethod,
  };
}

export interface GenerateCandidatesOptions {
  datasetKey: string;
  prompt: string;
  category: ItemCategory;
  tier: GenerationTier;
  useGemini: boolean;
  maxDiffPixels?: number;
  enableRepair?: boolean;
  candidateCount?: number;
  topKReturn?: number;
  seed?: number;
  index: SpriteIndexState;
  calibration: CriticCalibration;
  lessons?: LessonRecord[];
  approvals?: readonly ApprovalLabelRecord[];
  approvedSprites?: readonly GeneratedSpriteRecord[];
  champion?: CoevolutionChampion | null;
}

export interface GenerateCandidatesResult {
  candidates: GeneratedSpriteRecord[];
  totalEvaluated: number;
  fallbackWarning: string | null;
  appliedLessonIds: string[];
}

/**
 * End-to-end Phase 4/5 generation pipeline:
 * 1) Retrieves top `train`-split + human-approved few-shot donors matching category + prompt.
 * 2) Retrieves active `Lições Aprendidas` for the category (deterministic ordering).
 * 3) Optionally queries `/api/gemini/generate` for a structured palette + diff plan (NEVER silent fallback).
 * 4) Executes `candidateCount` (6..8) pixel-exact Tier T1 or Tier T2 candidates.
 * 5) Ranks all candidates with Critic v1 + nearest-real Originality.
 * 6) Repairs top 2 candidates via Phase 3 Champion / Critic-targeted discrete ops (kept ONLY on strict score gain).
 */
export async function generateSpriteCandidates(
  options: GenerateCandidatesOptions,
): Promise<GenerateCandidatesResult> {
  const {
    datasetKey,
    prompt,
    category,
    tier,
    useGemini,
    maxDiffPixels = 28,
    enableRepair = true,
    candidateCount = 6,
    topKReturn = 4,
    seed = Date.now() & 0x7fffffff,
    index,
    calibration,
    lessons = [],
    approvals = [],
    approvedSprites = [],
    champion = null,
  } = options;

  const donors = retrieveDonorsForPrompt({
    index,
    category,
    prompt,
    topK: Math.max(6, candidateCount),
    excludeTiles: true,
    approvals,
    approvedSprites,
  });

  if (donors.length === 0) {
    throw new Error(
      `Nenhum sprite real de treino indexado para a categoria "${category}". Execute a indexação primeiro.`,
    );
  }

  const applicableLessons = retrieveLessonsForQuery(lessons, {
    category,
    datasetKey,
    limit: 6,
  });
  const appliedLessonIds = applicableLessons.map((r) => r.lesson.id);

  let geminiPlan: GeminiPlanResponse | null = null;
  let fallbackWarning: string | null = null;
  let provenanceSource: GenerationProvenanceSource = tier === 'T1' ? 'local-t1' : 'local-t2';

  if (useGemini) {
    try {
      const donorTags = Array.from(new Set(donors.slice(0, 4).flatMap((d) => d.entry.tags)));
      geminiPlan = await requestGeminiSpritePlan({
        prompt,
        category,
        tier,
        maxDiffPixels,
        donorTags,
        lessonDirectives: applicableLessons.map((r) => `${r.lesson.title}: ${r.lesson.directive}`),
      });
      provenanceSource = tier === 'T1' ? 'gemini-t1' : 'gemini-t2';
    } catch (err) {
      const reason = err instanceof Error ? err.message : 'erro desconhecido na API Gemini';
      // ZERO silent fallback: explicitly switch provenance to local-* AND expose warning to the user
      provenanceSource = tier === 'T1' ? 'local-t1' : 'local-t2';
      fallbackWarning = `Gemini indisponível (${reason}). Executado explicitamente como ${provenanceSource}.`;
    }
  }

  const spliceModes: Array<'center_core' | 'upper_crest' | 'lower_trim'> = [
    'center_core',
    'upper_crest',
    'lower_trim',
  ];

  interface RawCandidate {
    rgba: Uint8ClampedArray;
    donorSpriteIds: number[];
    diffPixelCount: number;
    seed: number;
    evaluation: CriticEvaluation;
    nearestReal: ReturnType<typeof findNearestRealSprite>;
    rankScore: number;
  }

  const rawCandidates: RawCandidate[] = [];

  for (let c = 0; c < candidateCount; c++) {
    const candSeed = (seed + c * 9973) >>> 0;
    const primaryDonor = donors[c % donors.length].entry;
    const ramp = selectMaterialRampForPrompt(prompt, c);

    const primaryStops =
      geminiPlan && c < 2 ? geminiPlan.primaryStops : ramp.primary;
    const accentStops =
      geminiPlan && c < 2 ? geminiPlan.accentStops : ramp.accent;

    let baseRgba: Uint8ClampedArray;
    let donorSpriteIds: number[];

    if (tier === 'T2' && donors.length >= 2) {
      const secondaryDonor = donors[(c + 1 + Math.floor(c / 2)) % donors.length].entry;
      baseRgba = executeDonorComposite({
        primaryRgba: primaryDonor.rgba,
        secondaryRgba: secondaryDonor.rgba,
        primaryStops,
        accentStops,
        spliceMode: spliceModes[c % spliceModes.length],
        isTile: primaryDonor.isTile,
      });
      donorSpriteIds =
        primaryDonor.spriteId === secondaryDonor.spriteId
          ? [primaryDonor.spriteId]
          : [primaryDonor.spriteId, secondaryDonor.spriteId];
    } else {
      baseRgba = executeRampRecolor({
        sourceRgba: primaryDonor.rgba,
        primaryStops,
        accentStops,
        isTile: primaryDonor.isTile,
      }).rgba;
      donorSpriteIds = [primaryDonor.spriteId];
    }

    // Apply bounded pixel-diff edits (strictly <= maxDiffPixels)
    const plannedEdits =
      geminiPlan && c < 2 && geminiPlan.diffEdits.length > 0
        ? geminiPlan.diffEdits
        : buildDeterministicDiffEdits({
            baseRgba,
            accentStops,
            highlightStop: primaryStops[primaryStops.length - 1],
            maxDiffPixels,
            seed: candSeed,
          });

    const diffRes = executeBoundedPixelDiff({
      baseRgba,
      edits: plannedEdits,
      maxDiffPixels,
      preserveOutline: true,
      isTile: primaryDonor.isTile,
    });

    const evalRes = evaluateSprite(diffRes.rgba, {
      category,
      calibration,
      isTileOverride: primaryDonor.isTile,
    });
    const nearestReal = findNearestRealSprite(diffRes.rgba, index);

    // Honest ranking: Critic v1 score dominates, with small penalty only if nearly identical copy (<15% originality)
    const copyPenalty = nearestReal.originalityScore < 15 ? (15 - nearestReal.originalityScore) * 0.8 : 0;
    const rankScore = evalRes.score + Math.min(4, nearestReal.originalityScore * 0.05) - copyPenalty;

    rawCandidates.push({
      rgba: diffRes.rgba,
      donorSpriteIds,
      diffPixelCount: diffRes.appliedCount,
      seed: candSeed,
      evaluation: evalRes,
      nearestReal,
      rankScore,
    });
  }

  rawCandidates.sort((a, b) => b.rankScore - a.rankScore);

  // Repair top 2 candidates with Critic-guided repair (only kept if score strictly improves)
  const createdAt = new Date().toISOString();
  const topSlice = rawCandidates.slice(0, Math.min(topKReturn, rawCandidates.length));
  const records: GeneratedSpriteRecord[] = [];

  for (let i = 0; i < topSlice.length; i++) {
    const cand = topSlice[i];
    let finalRgba = cand.rgba;
    let finalEval = cand.evaluation;
    let repaired = false;
    let repairDelta = 0;
    let repairMethod: 'none' | 'champion_coevolution' | 'critic_targeted_ops' = 'none';
    let finalNearest = cand.nearestReal;

    if (enableRepair && i < 2) {
      const rep = runCriticGuidedRepair({
        rgba: cand.rgba,
        baseEval: cand.evaluation,
        category,
        calibration,
        champion,
      });
      if (rep.repaired) {
        finalRgba = rep.rgba;
        finalEval = rep.evaluation;
        repaired = true;
        repairDelta = rep.repairDelta;
        repairMethod = rep.repairMethod;
        finalNearest = findNearestRealSprite(finalRgba, index);
      }
    }

    const dec = decomposeSpriteToIndexedAndRampRoles(finalRgba);

    records.push({
      id: `gen-${Date.now().toString(36)}-${(cand.seed % 99991).toString(36)}-${i}`,
      datasetKey,
      prompt: prompt.trim(),
      category,
      createdAt,
      rgba: finalRgba,
      paletteHex: dec.paletteHex,
      evaluation: finalEval,
      nearestReal: finalNearest,
      provenance: {
        source: provenanceSource,
        tier,
        model: geminiPlan?.model ?? null,
        fallbackWarning,
        donorSpriteIds: cand.donorSpriteIds,
        diffPixelCount: cand.diffPixelCount,
        diffPixelLimit: maxDiffPixels,
        repaired,
        repairDelta,
        repairMethod,
        appliedLessonIds,
        seed: cand.seed,
      },
      approvalStatus: 'pending',
      readBackVerified: false,
    });
  }

  // Re-sort final returned records by final Critic v1 score descending
  records.sort((a, b) => b.evaluation.score - a.evaluation.score);

  return {
    candidates: records,
    totalEvaluated: rawCandidates.length,
    fallbackWarning,
    appliedLessonIds,
  };
}
