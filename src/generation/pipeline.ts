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
import { encodeRgbaToPngBytes } from '../export/trainingDatasetExport';
import { retrieveLessonsForQuery } from '../lessons/retrieval';
import type { LessonRecord } from '../lessons/types';
import type { ItemCategory } from '../types/studio';
import { SPRITE_SIZE } from '../utils/tibiaParser';
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
import {
  calcRgbLuma,
  hexToRgb,
  upscaleRgbaNearest,
  validateEngineeredPromptJson,
} from './promptEngineer';
import { tibiaifySprite } from './tibiaify';
import type {
  ApprovalLabelRecord,
  EngineeredPromptSpec,
  GeneratedSpriteRecord,
  GenerationProvenanceSource,
  GenerationTier,
  HumannessGateInfo,
  IndexedSpriteEntry,
  PixelDiffEdit,
  SpriteIndexState,
  StyleDossier,
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
  temperature?: number;
}): Promise<GeminiPlanResponse> {
  const {
    prompt,
    category,
    tier,
    maxDiffPixels,
    donorTags,
    lessonDirectives,
    temperature = 0.45,
  } = params;

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
      temperature,
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
  temperature?: number;
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
    temperature = 0.45,
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
        temperature,
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

// ============================================================================
// TWO-STEP HUMAN-GATED FLOW: STEP 1 (ENGINEER PROMPT) & STEP 2 (HYBRID GENERATE)
// ============================================================================

function bytesToBase64(bytes: Uint8Array): string {
  let bin = '';
  const chunk = 0x4000;
  for (let i = 0; i < bytes.length; i += chunk) {
    const slice = bytes.subarray(i, Math.min(i + chunk, bytes.length));
    bin += String.fromCharCode(...slice);
  }
  return btoa(bin);
}

/**
 * Encodes up to `maxCount` donor sprites as 64x64 (2x nearest-neighbor) PNG data URLs
 * for multimodal few-shot conditioning of `/api/gemini/engineer-prompt` and `/api/gemini/generate`.
 */
export function encodeDonorSpritesAsPngDataUrls(
  donors: readonly IndexedSpriteEntry[],
  maxCount = 8,
): string[] {
  const out: string[] = [];
  for (const donor of donors.slice(0, Math.min(10, maxCount))) {
    const up = upscaleRgbaNearest(donor.rgba, SPRITE_SIZE, SPRITE_SIZE, 2);
    const png = encodeRgbaToPngBytes(up.rgba, up.width, up.height);
    out.push(`data:image/png;base64,${bytesToBase64(png)}`);
  }
  return out;
}

/**
 * STEP 1 CLIENT — Calls `POST /api/gemini/engineer-prompt` with user's short idea + train-only
 * style dossier + upscaled donor PNGs (+ optional previousPrompt + adjustmentComment for revisions).
 * Strictly validates JSON output; throws on malformed JSON or 503 (NEVER silently falls back).
 */
export async function requestEngineeredPromptFromServer(params: {
  idea: string;
  categoryHint?: ItemCategory;
  dossier: StyleDossier;
  donorEntries: readonly IndexedSpriteEntry[];
  trainPaletteRgb: ReadonlyArray<[number, number, number]>;
  previousPrompt?: EngineeredPromptSpec | null;
  adjustmentComment?: string | null;
  temperature?: number;
}): Promise<{
  prompt: EngineeredPromptSpec;
  model: string;
}> {
  const {
    idea,
    categoryHint,
    dossier,
    donorEntries,
    trainPaletteRgb,
    previousPrompt = null,
    adjustmentComment = null,
    temperature = 0.4,
  } = params;

  const images = encodeDonorSpritesAsPngDataUrls(donorEntries, 8);
  const resp = await fetch('/api/gemini/engineer-prompt', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      idea: idea.trim(),
      categoryHint,
      styleDossier: dossier,
      images,
      previousPrompt,
      adjustmentComment: adjustmentComment?.trim() || undefined,
      temperature,
    }),
  });

  const payload = (await resp.json().catch(() => ({}))) as {
    prompt?: unknown;
    model?: string;
    error?: string;
    aiReady?: boolean;
  };

  if (!resp.ok) {
    throw new Error(
      payload.error || `Engenheiro de Prompt falhou com HTTP ${resp.status}.`,
    );
  }

  const validDonorIds = new Set(dossier.allowedDonorIds);
  const validated = validateEngineeredPromptJson(payload.prompt, {
    validDonorIds,
    datasetPaletteRgb: trainPaletteRgb,
  });

  return {
    prompt: validated,
    model: payload.model || 'gemini',
  };
}

/**
 * Decodes a structured 32x32 indexed grid JSON returned by Gemini:
 * `{"palette": ["#121826", ...], "rows": ["....111...", ... 32 rows of 32 chars]}`
 * where `.` or `0` = transparent, `1..9` and `A..Z`/`a..z` map to 1-based palette indices.
 */
export function decodeStructuredIndexedGridJson(raw: string): Uint8ClampedArray | null {
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return null;
  }

  if (!Array.isArray(parsed.palette) || !Array.isArray(parsed.rows)) return null;
  if (parsed.rows.length !== SPRITE_SIZE) return null;

  const paletteRgb: Array<[number, number, number]> = [];
  for (const hex of parsed.palette) {
    if (typeof hex !== 'string') return null;
    const rgb = hexToRgb(hex);
    if (!rgb) return null;
    paletteRgb.push(rgb);
  }
  if (paletteRgb.length < 3) return null;

  const symbols = '123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';
  const rgba = new Uint8ClampedArray(SPRITE_SIZE * SPRITE_SIZE * 4);
  const rawRows: unknown[] = parsed.rows;
  let opaque = 0;

  for (let y = 0; y < SPRITE_SIZE; y++) {
    const row: unknown = rawRows[y];
    if (typeof row !== 'string' || row.length !== SPRITE_SIZE) return null;
    for (let x = 0; x < SPRITE_SIZE; x++) {
      const ch = row[x];
      if (ch === '.' || ch === '0' || ch === ' ') continue;
      const idx = symbols.indexOf(ch);
      if (idx < 0 || idx >= paletteRgb.length) continue;
      const col = paletteRgb[idx];
      const off = (y * SPRITE_SIZE + x) * 4;
      rgba[off] = col[0];
      rgba[off + 1] = col[1];
      rgba[off + 2] = col[2];
      rgba[off + 3] = 255;
      opaque++;
    }
  }

  return opaque >= 16 ? rgba : null;
}

async function requestGeminiStructuredGridCandidate(params: {
  approvedPrompt: EngineeredPromptSpec;
  donorEntries: readonly IndexedSpriteEntry[];
  lessonDirectives: string[];
  temperature: number;
  seed: number;
}): Promise<{ rgba: Uint8ClampedArray; model: string }> {
  const { approvedPrompt, donorEntries, lessonDirectives, temperature, seed } = params;
  const images = encodeDonorSpritesAsPngDataUrls(donorEntries, 4);

  const systemInstruction = [
    'You are a master Classic Tibia 7.72 32x32 pixel artist.',
    'Generate a crisp 32x32 indexed pixel-art sprite as strict JSON:',
    '{"palette":["#RRGGBB",...],"rows":[32 strings of exactly 32 characters]}',
    'Use "." for transparent pixels, and "1".."9","A".."Z" for 1-based palette indices.',
    'STRICT RULES: binary alpha only, 1px dark hue-shifted outline around entire silhouette, NW directional lighting, zero anti-aliasing, zero orphan pixels, use ONLY colors from the approved paletteRamps.',
  ].join(' ');

  const userPrompt = [
    `Item: ${approvedPrompt.itemName} (${approvedPrompt.category})`,
    `Visual Brief: ${approvedPrompt.visualBrief}`,
    `Approved Palette Ramps:\n${approvedPrompt.paletteRamps.map((r) => `- ${r.material}: ${r.colors.join(', ')}`).join('\n')}`,
    `Constraints: footprint ~${approvedPrompt.constraints.footprintPx.width}x${approvedPrompt.constraints.footprintPx.height}px, maxColors=${approvedPrompt.constraints.maxColors}, outline="${approvedPrompt.constraints.outlineRule}", noDitheringOrAA=${approvedPrompt.constraints.noDitheringOrAA}, binaryAlpha=${approvedPrompt.constraints.binaryAlpha}`,
    `Avoid: ${approvedPrompt.negativePrompt.join('; ')}`,
    `Variation seed: ${seed}`,
    lessonDirectives.length > 0 ? `Learned lessons:\n- ${lessonDirectives.slice(0, 4).join('\n- ')}` : '',
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
      images,
      temperature,
    }),
  });

  const payload = (await resp.json().catch(() => ({}))) as {
    text?: string;
    model?: string;
    error?: string;
  };
  if (!resp.ok || !payload.text) {
    throw new Error(payload.error || `Gemini retornou HTTP ${resp.status}.`);
  }

  const rgba = decodeStructuredIndexedGridJson(payload.text);
  if (!rgba) {
    throw new Error('Gemini retornou grade 32×32 indexada inválida.');
  }
  return { rgba, model: payload.model || 'gemini' };
}

function medianNumber(values: readonly number[]): number | null {
  if (values.length === 0) return null;
  const s = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function evaluateHumannessGate(params: {
  evaluation: CriticEvaluation;
  nearestReal: ReturnType<typeof findNearestRealSprite>;
}): HumannessGateInfo {
  const { evaluation, nearestReal } = params;
  const hardCount = countHardIssues(evaluation);
  const discProb =
    evaluation.discriminatorScore !== null
      ? Math.max(0, Math.min(1, evaluation.discriminatorScore / 98.5))
      : null;

  // Similarity in 0..1 derived from exact nearest-real match
  const similarity = Math.max(
    0,
    Math.min(1, 1 - nearestReal.originalityScore / 100),
  );
  const nearCopy =
    similarity > 0.9 || (nearestReal.silhouetteIoU > 0.95 && nearestReal.pixelDiffRatio < 0.08);
  const styleOutlier =
    evaluation.calibrated && (evaluation.statisticalScore < 30 || hardCount > 0);

  let rejectReasonPt: string | null = null;
  if (nearCopy) {
    rejectReasonPt = `Cópia muito próxima do real #${nearestReal.spriteId} (similaridade ${Math.round(similarity * 100)}% > 90%)`;
  } else if (hardCount > 0) {
    const firstHard = evaluation.issues.find((i) => i.severity === 'hard');
    rejectReasonPt = `Violação de regra rígida: ${firstHard?.labelPt ?? 'higiene reprovada'}`;
  } else if (styleOutlier) {
    rejectReasonPt = `Outlier estilístico fora da categoria (estatístico ${evaluation.statisticalScore.toFixed(1)})`;
  }

  const passed = rejectReasonPt === null;

  // Originality sweet-spot score: rewards genuine originality 22..82%, penalizes near-copies
  const origTerm =
    nearestReal.originalityScore < 10
      ? 0
      : nearestReal.originalityScore < 25
        ? (nearestReal.originalityScore / 25) * 65
        : Math.min(98, 65 + ((nearestReal.originalityScore - 25) / 60) * 33);

  const discTerm = discProb !== null ? discProb * 100 : evaluation.hygieneScore;
  const humanTerm = evaluation.humanPreferenceScore;

  const baseScore =
    humanTerm !== null
      ? evaluation.score * 0.44 + discTerm * 0.24 + origTerm * 0.18 + humanTerm * 0.14
      : evaluation.score * 0.5 + discTerm * 0.28 + origTerm * 0.22;

  const humannessScore =
    Math.round(Math.max(0, Math.min(99.5, passed ? baseScore : baseScore - 28)) * 10) / 10;

  return {
    passed,
    humannessScore,
    criticTerm: evaluation.score,
    discriminatorProb: discProb !== null ? Math.round(discProb * 1000) / 1000 : null,
    originalityScore: nearestReal.originalityScore,
    similarityToNearest: Math.round(similarity * 1000) / 1000,
    humanPreferenceTerm: humanTerm,
    rejectReasonPt,
  };
}

export interface GenerateFromApprovedPromptOptions {
  datasetKey: string;
  approvedPrompt: EngineeredPromptSpec;
  engineeredPromptId: string;
  engineeredPromptVersion: number;
  promptSource: 'gemini' | 'manual';
  candidateCount?: number; // 4..16, default 8
  useGemini: boolean;
  temperature?: number;
  maxRepairIterations?: number; // K iterations, default 3
  seed?: number;
  index: SpriteIndexState;
  calibration: CriticCalibration;
  lessons?: readonly LessonRecord[];
  approvals?: readonly ApprovalLabelRecord[];
  approvedSprites?: readonly GeneratedSpriteRecord[];
  champion?: CoevolutionChampion | null;
}

export interface GenerateFromApprovedPromptResult {
  topCandidates: GeneratedSpriteRecord[];
  secondaryCandidates: GeneratedSpriteRecord[];
  allCandidates: GeneratedSpriteRecord[];
  totalEvaluated: number;
  passedGateCount: number;
  fallbackWarning: string | null;
  appliedLessonIds: string[];
  categoryValidationRealMedian: number | null;
  categoryValidationHardPassRate: number | null;
  generatedHardPassRate: number;
}

/**
 * STEP 2 HYBRID PIPELINE (executed ONLY after user clicks "OK, gerar"):
 *  a) Generates N candidates (default 8, 4..16) with distinct seeds using approved prompt + donors
 *     (structured 32x32 indexed grid via Gemini when enabled/available, or deterministic local T1/T2 hybrid).
 *  b) Runs EVERY candidate through deterministic `tibiaifySprite` post-processing.
 *  c) Runs Critic-guided iterative repair loop (up to K iterations; accepts ONLY on strict Critic score gain
 *     with zero hard-rule regression).
 *  d) Ranks via the "Humanness Gate" (Critic score, Discriminator P(real), Originality rejecting near-copies
 *     >0.9 similarity & style outliers, and Human Preference when >= 6 mixed votes).
 */
export async function generateFromApprovedPrompt(
  options: GenerateFromApprovedPromptOptions,
): Promise<GenerateFromApprovedPromptResult> {
  const {
    datasetKey,
    approvedPrompt,
    engineeredPromptId,
    engineeredPromptVersion,
    promptSource,
    candidateCount: rawCount = 8,
    useGemini,
    temperature = 0.45,
    maxRepairIterations = 3,
    seed = Date.now() & 0x7fffffff,
    index,
    calibration,
    lessons = [],
    approvals = [],
    approvedSprites = [],
    champion = null,
  } = options;

  const candidateCount = Math.max(4, Math.min(16, Math.round(rawCount)));
  const category = approvedPrompt.category;

  // Build train-only dataset palette RGB for snapping
  const trainEntries = index.entries.filter((e) => e.split === 'train');
  const paletteMap = new Map<string, [number, number, number]>();
  for (const entry of trainEntries) {
    for (let i = 0; i < entry.paletteHex.length; i++) {
      const hex = entry.paletteHex[i];
      if (!paletteMap.has(hex) && entry.paletteRgb[i]) {
        paletteMap.set(hex, entry.paletteRgb[i]);
      }
    }
  }
  const datasetPaletteRgb = Array.from(paletteMap.values());

  // Compute honest held-out validation real median & hard-rule pass rate for this category (NEVER test!)
  const valEntries = index.entries.filter(
    (e) => e.split === 'validation' && e.category === category,
  );
  const categoryValidationRealMedian = medianNumber(valEntries.map((e) => e.criticScore));
  let categoryValidationHardPassRate: number | null = null;
  if (valEntries.length > 0) {
    let valHardClean = 0;
    for (const v of valEntries) {
      const ev = evaluateSprite(v.rgba, { category, calibration, isTileOverride: v.isTile });
      if (countHardIssues(ev) === 0) valHardClean++;
    }
    categoryValidationHardPassRate = Math.round((valHardClean / valEntries.length) * 1000) / 10;
  }

  // Resolve prioritized donors strictly from `train`/`validation` present in `index` (NEVER test)
  const indexById = new Map<number, IndexedSpriteEntry>();
  for (const e of index.entries) {
    if (e.split === 'train' || e.split === 'validation') {
      indexById.set(e.spriteId, e);
    }
  }

  const prioritizedDonors: IndexedSpriteEntry[] = [];
  for (const suggestedId of approvedPrompt.donorSuggestions) {
    const entry = indexById.get(suggestedId);
    if (entry && !entry.isTile) {
      prioritizedDonors.push(entry);
    }
  }
  const retrievedFallback = retrieveDonorsForPrompt({
    index,
    category,
    prompt: `${approvedPrompt.itemName} ${approvedPrompt.visualBrief}`,
    topK: Math.max(8, candidateCount),
    excludeTiles: true,
    approvals,
    approvedSprites,
  });
  for (const r of retrievedFallback) {
    if (!prioritizedDonors.some((d) => d.spriteId === r.entry.spriteId)) {
      prioritizedDonors.push(r.entry);
    }
  }

  if (prioritizedDonors.length === 0) {
    throw new Error(
      `Nenhum sprite doador indexado para "${category}". Execute a indexação do dataset primeiro.`,
    );
  }

  // Extract RGB ramps directly from the approved prompt
  const parseRampToRgb = (colors: string[]): Array<[number, number, number]> => {
    const out: Array<[number, number, number]> = [];
    for (const h of colors) {
      const rgb = hexToRgb(h);
      if (rgb) out.push(rgb);
    }
    out.sort((a, b) => calcRgbLuma(a[0], a[1], a[2]) - calcRgbLuma(b[0], b[1], b[2]));
    return out;
  };

  const fallbackRamp = selectMaterialRampForPrompt(approvedPrompt.itemName, 0);
  const primaryStops =
    approvedPrompt.paletteRamps[0] && parseRampToRgb(approvedPrompt.paletteRamps[0].colors).length >= 3
      ? parseRampToRgb(approvedPrompt.paletteRamps[0].colors)
      : fallbackRamp.primary;
  const accentStops =
    approvedPrompt.paletteRamps[1] && parseRampToRgb(approvedPrompt.paletteRamps[1].colors).length >= 3
      ? parseRampToRgb(approvedPrompt.paletteRamps[1].colors)
      : fallbackRamp.accent;

  const applicableLessons = retrieveLessonsForQuery(lessons, {
    category,
    datasetKey,
    limit: 6,
  });
  const appliedLessonIds = applicableLessons.map((r) => r.lesson.id);
  const lessonDirectives = applicableLessons.map((r) => `${r.lesson.title}: ${r.lesson.directive}`);

  const categoryProfile =
    calibration.profiles[category]?.sampleCount >= 3
      ? calibration.profiles[category]
      : calibration.globalProfile;

  let fallbackWarning: string | null = null;
  let geminiAvailableForBatch = useGemini;

  const spliceModes: Array<'center_core' | 'upper_crest' | 'lower_trim'> = [
    'center_core',
    'upper_crest',
    'lower_trim',
  ];

  const createdAt = new Date().toISOString();
  const builtRecords: GeneratedSpriteRecord[] = [];

  for (let c = 0; c < candidateCount; c++) {
    const candSeed = (seed + c * 9973 + 17) >>> 0;
    const primaryDonor = prioritizedDonors[c % prioritizedDonors.length];
    const secondaryDonor =
      prioritizedDonors[(c + 1 + Math.floor(c / 2)) % prioritizedDonors.length];

    let rawRgba: Uint8ClampedArray | null = null;
    let source: GenerationProvenanceSource = c % 2 === 0 ? 'local-t2' : 'local-t1';
    let tierUsed: GenerationTier = c % 2 === 0 ? 'T2' : 'T1';
    let modelUsed: string | null = null;
    let donorSpriteIds: number[] = [primaryDonor.spriteId];
    let diffPixelCount = 0;
    const diffPixelLimit = 32;

    // Step 2a: Prefer Gemini structured 32x32 indexed grid for first 2 candidates when enabled
    if (geminiAvailableForBatch && c < 2) {
      try {
        const gRes = await requestGeminiStructuredGridCandidate({
          approvedPrompt,
          donorEntries: prioritizedDonors.slice(0, 4),
          lessonDirectives,
          temperature,
          seed: candSeed,
        });
        rawRgba = gRes.rgba;
        source = 'gemini-t2';
        tierUsed = 'T2';
        modelUsed = gRes.model;
        donorSpriteIds = prioritizedDonors.slice(0, 3).map((d) => d.spriteId);
        diffPixelCount = 32;
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'erro desconhecido';
        fallbackWarning = `Gemini indisponível na geração de grade (${msg}). Candidatos sintetizados explicitamente em modo local (local-t1/local-t2) + tibiaify.`;
        geminiAvailableForBatch = false;
      }
    }

    // Local hybrid synthesis using approved ramps + prioritized donors + bounded motifs
    if (!rawRgba) {
      const useComposite = prioritizedDonors.length >= 2 && c % 2 === 0;
      let synthBase: Uint8ClampedArray;
      if (useComposite) {
        synthBase = executeDonorComposite({
          primaryRgba: primaryDonor.rgba,
          secondaryRgba: secondaryDonor.rgba,
          primaryStops,
          accentStops,
          spliceMode: spliceModes[c % spliceModes.length],
          isTile: primaryDonor.isTile,
        });
        source = 'local-t2';
        tierUsed = 'T2';
        donorSpriteIds = Array.from(new Set([primaryDonor.spriteId, secondaryDonor.spriteId]));
      } else {
        synthBase = executeRampRecolor({
          sourceRgba: primaryDonor.rgba,
          primaryStops,
          accentStops,
          isTile: primaryDonor.isTile,
        }).rgba;
        source = 'local-t1';
        tierUsed = 'T1';
        donorSpriteIds = [primaryDonor.spriteId];
      }

      const edits = buildDeterministicDiffEdits({
        baseRgba: synthBase,
        accentStops,
        highlightStop: primaryStops[primaryStops.length - 1],
        maxDiffPixels: diffPixelLimit,
        seed: candSeed,
      });
      const diffRes = executeBoundedPixelDiff({
        baseRgba: synthBase,
        edits,
        maxDiffPixels: diffPixelLimit,
        preserveOutline: true,
        isTile: primaryDonor.isTile,
      });
      rawRgba = diffRes.rgba;
      diffPixelCount = diffRes.appliedCount;
    }

    // Step 2b: Deterministic "Tibia-ification" post-processor on EVERY candidate
    const tib = tibiaifySprite({
      rgba: rawRgba,
      width: SPRITE_SIZE,
      height: SPRITE_SIZE,
      approvedRamps: approvedPrompt.paletteRamps,
      datasetPaletteRgb,
      categoryProfile,
      maxColors: approvedPrompt.constraints.maxColors,
      isTile: primaryDonor.isTile,
      allowDithering: !approvedPrompt.constraints.noDitheringOrAA,
      targetCenterGround: true,
    });

    let currentRgba = tib.rgba;
    let currentEval = evaluateSprite(currentRgba, {
      category,
      calibration,
      isTileOverride: primaryDonor.isTile,
    });
    const preRepairScore = currentEval.score;

    // Step 2c: Critic-guided iterative repair loop (up to K iterations, accept ONLY on strict improvement + 0 hard regressions)
    let repaired = false;
    let repairMethod: 'none' | 'champion_coevolution' | 'critic_targeted_ops' = 'none';
    const maxIters = Math.max(1, Math.min(5, Math.round(maxRepairIterations)));

    for (let iter = 0; iter < maxIters; iter++) {
      const rep = runCriticGuidedRepair({
        rgba: currentRgba,
        baseEval: currentEval,
        category,
        calibration,
        champion,
      });
      if (!rep.repaired) break;
      // Re-verify binary alpha and zero hard-rule regression
      if (
        countHardIssues(rep.evaluation) <= countHardIssues(currentEval) &&
        rep.evaluation.score > currentEval.score + 0.1
      ) {
        currentRgba = rep.rgba;
        currentEval = rep.evaluation;
        repaired = true;
        repairMethod = rep.repairMethod;
      } else {
        break;
      }
    }

    const repairDelta = Math.max(0, Math.round((currentEval.score - preRepairScore) * 10) / 10);
    const nearestReal = findNearestRealSprite(currentRgba, index);
    const humannessGate = evaluateHumannessGate({
      evaluation: currentEval,
      nearestReal,
    });
    const dec = decomposeSpriteToIndexedAndRampRoles(currentRgba);

    builtRecords.push({
      id: `eng-${Date.now().toString(36)}-${(candSeed % 99991).toString(36)}-${c}`,
      datasetKey,
      prompt: `${approvedPrompt.itemName} — ${approvedPrompt.visualBrief.slice(0, 120)}`,
      category,
      createdAt,
      rgba: currentRgba,
      paletteHex: dec.paletteHex,
      evaluation: currentEval,
      nearestReal,
      provenance: {
        source,
        tier: tierUsed,
        model: modelUsed,
        fallbackWarning,
        donorSpriteIds,
        diffPixelCount,
        diffPixelLimit,
        repaired,
        repairDelta,
        repairMethod,
        appliedLessonIds,
        seed: candSeed,
        engineeredPromptId,
        engineeredPromptVersion,
        promptSource,
        approvedPromptSnapshot: approvedPrompt,
        tibiaified: true,
        humannessGate,
        validationCategoryRealMedian: categoryValidationRealMedian,
      },
      approvalStatus: 'pending',
      readBackVerified: false,
    });
  }

  // Step 2d: Rank all candidates by Humanness Gate (passing candidates first, then humannessScore desc)
  builtRecords.sort((a, b) => {
    const ga = a.provenance.humannessGate;
    const gb = b.provenance.humannessGate;
    const passA = ga?.passed ? 1 : 0;
    const passB = gb?.passed ? 1 : 0;
    if (passB !== passA) return passB - passA;
    const hsA = ga?.humannessScore ?? a.evaluation.score;
    const hsB = gb?.humannessScore ?? b.evaluation.score;
    if (hsB !== hsA) return hsB - hsA;
    return b.evaluation.score - a.evaluation.score;
  });

  const passedGateCount = builtRecords.filter((r) => r.provenance.humannessGate?.passed).length;
  const hardCleanCount = builtRecords.filter((r) => countHardIssues(r.evaluation) === 0).length;
  const generatedHardPassRate =
    builtRecords.length > 0 ? Math.round((hardCleanCount / builtRecords.length) * 1000) / 10 : 0;

  const topCount = Math.min(4, builtRecords.length);
  const topCandidates = builtRecords.slice(0, topCount);
  const secondaryCandidates = builtRecords.slice(topCount);

  return {
    topCandidates,
    secondaryCandidates,
    allCandidates: builtRecords,
    totalEvaluated: builtRecords.length,
    passedGateCount,
    fallbackWarning,
    appliedLessonIds,
    categoryValidationRealMedian,
    categoryValidationHardPassRate,
    generatedHardPassRate,
  };
}

