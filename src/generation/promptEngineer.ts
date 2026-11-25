import type { CriticCalibration } from '../critic/types';
import { retrieveLessonsForQuery } from '../lessons/retrieval';
import type { LessonRecord } from '../lessons/types';
import { ALL_ITEM_CATEGORIES, type ItemCategory } from '../types/studio';
import { rgbToHex } from '../utils/pixelImage';
import { SPRITE_SIZE } from '../utils/tibiaParser';
import { selectMaterialRampForPrompt } from './executors';
import { retrieveDonorsForPrompt } from './indexer';
import type {
  ApprovalLabelRecord,
  EngineeredPromptSpec,
  GeneratedSpriteRecord,
  IndexedSpriteEntry,
  PaletteRampSpec,
  PromptDiffChange,
  SpriteIndexState,
  StyleDossier,
  StyleDossierDonor,
} from './types';

export const DEFAULT_NEGATIVE_PROMPTS: readonly string[] = [
  'blur ou suavização',
  'gradientes contínuos',
  'anti-aliasing (alpha semitransparente)',
  'dithering ruidoso ou confete',
  'pixels órfãos ou flutuantes',
  'contorno duplo, grosso ou irregular',
  'formas perfeitamente simétricas artificiais',
  'excesso de cores fora da paleta',
  'direções de luz misturadas ou invertidas (SE mais claro que NW)',
];

const HEX_COLOR_RE = /^#[0-9A-Fa-f]{6}$/;

export function hexToRgb(hex: string): [number, number, number] | null {
  const trimmed = hex.trim();
  if (!HEX_COLOR_RE.test(trimmed)) return null;
  const r = parseInt(trimmed.slice(1, 3), 16);
  const g = parseInt(trimmed.slice(3, 5), 16);
  const b = parseInt(trimmed.slice(5, 7), 16);
  return [r, g, b];
}

export function calcRgbLuma(r: number, g: number, b: number): number {
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

export function rgbDistance(a: readonly [number, number, number], b: readonly [number, number, number]): number {
  const dr = a[0] - b[0];
  const dg = a[1] - b[1];
  const db = a[2] - b[2];
  return Math.sqrt(dr * dr + dg * dg + db * db);
}

export function findNearestPaletteRgb(
  target: readonly [number, number, number],
  paletteRgb: ReadonlyArray<[number, number, number]>,
): { rgb: [number, number, number]; distance: number } {
  if (paletteRgb.length === 0) {
    return { rgb: [target[0], target[1], target[2]], distance: 0 };
  }
  let best = paletteRgb[0];
  let bestDist = Infinity;
  for (let i = 0; i < paletteRgb.length; i++) {
    const d = rgbDistance(target, paletteRgb[i]);
    if (d < bestDist) {
      bestDist = d;
      best = paletteRgb[i];
    }
  }
  return { rgb: [best[0], best[1], best[2]], distance: bestDist };
}

/**
 * Snaps a ramp's hex stops to the dataset's exact RGB palette (or keeps colors already within
 * `maxFreeDistance`), guaranteeing monotonic darkest-to-brightest luminance order.
 */
export function snapRampColorsToDatasetPalette(
  colorsHex: readonly string[],
  datasetPaletteRgb: ReadonlyArray<[number, number, number]>,
  maxFreeDistance = 24,
): string[] {
  const snapped: Array<{ rgb: [number, number, number]; luma: number }> = [];
  for (const hex of colorsHex) {
    const rgb = hexToRgb(hex);
    if (!rgb) continue;
    let finalRgb = rgb;
    if (datasetPaletteRgb.length > 0) {
      const nearest = findNearestPaletteRgb(rgb, datasetPaletteRgb);
      if (nearest.distance > maxFreeDistance) {
        finalRgb = nearest.rgb;
      } else {
        // Prefer exact dataset color when very close so ramps stay ultra-authentic
        finalRgb = nearest.rgb;
      }
    }
    snapped.push({ rgb: finalRgb, luma: calcRgbLuma(finalRgb[0], finalRgb[1], finalRgb[2]) });
  }
  snapped.sort((a, b) => a.luma - b.luma);
  return snapped.map((s) => rgbToHex(s.rgb[0], s.rgb[1], s.rgb[2]));
}

function medianOf(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function round2(v: number): number {
  return Math.round(v * 100) / 100;
}

/**
 * Builds a compact 16x16 ASCII silhouette/tone preview of a 32x32 sprite
 * (useful for token-efficient few-shot conditioning in Step 1 & Step 2).
 */
export function buildCompactAsciiPreview(rgba: Uint8ClampedArray): string {
  const chars = ' .:-=+*#%@';
  const rows: string[] = [];
  for (let by = 0; by < 16; by++) {
    let line = '';
    for (let bx = 0; bx < 16; bx++) {
      let opaque = 0;
      let lumaSum = 0;
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          const i = ((by * 2 + dy) * SPRITE_SIZE + (bx * 2 + dx)) * 4;
          if (rgba[i + 3] >= 128) {
            opaque++;
            lumaSum += calcRgbLuma(rgba[i], rgba[i + 1], rgba[i + 2]);
          }
        }
      }
      if (opaque === 0) {
        line += '.';
      } else {
        const avg = lumaSum / opaque;
        const idx = Math.max(1, Math.min(chars.length - 1, 1 + Math.round(avg * (chars.length - 2))));
        line += chars[idx];
      }
    }
    rows.push(line);
  }
  return rows.join('\n');
}

/**
 * Encodes a 32x32 RGBA sprite into a crisp nearest-neighbor upscaled PNG data URL
 * (browser canvas or pure PNG fallback via caller).
 */
export function upscaleRgbaNearest(
  rgba: Uint8ClampedArray,
  srcW: number,
  srcH: number,
  scale: number,
): { rgba: Uint8ClampedArray; width: number; height: number } {
  const s = Math.max(1, Math.round(scale));
  const outW = srcW * s;
  const outH = srcH * s;
  const out = new Uint8ClampedArray(outW * outH * 4);
  for (let y = 0; y < outH; y++) {
    const sy = Math.floor(y / s);
    for (let x = 0; x < outW; x++) {
      const sx = Math.floor(x / s);
      const srcIdx = (sy * srcW + sx) * 4;
      const dstIdx = (y * outW + x) * 4;
      out[dstIdx] = rgba[srcIdx];
      out[dstIdx + 1] = rgba[srcIdx + 1];
      out[dstIdx + 2] = rgba[srcIdx + 2];
      out[dstIdx + 3] = rgba[srcIdx + 3];
    }
  }
  return { rgba: out, width: outW, height: outH };
}

/**
 * Builds the compact Style Dossier strictly from the `train` split (ZERO `test` leakage).
 * Per-category medians, top dataset palette colors, 6-10 few-shot `train` donors,
 * and ranked active lessons (`mined`, `critic_loop`, `human_feedback`).
 */
export function buildStyleDossier(params: {
  datasetKey: string;
  category: ItemCategory;
  idea: string;
  calibration: CriticCalibration;
  spriteIndex: SpriteIndexState;
  lessons: readonly LessonRecord[];
  approvals?: readonly ApprovalLabelRecord[];
  approvedSprites?: readonly GeneratedSpriteRecord[];
  fewShotCount?: number;
}): {
  dossier: StyleDossier;
  donorEntries: IndexedSpriteEntry[];
  trainPaletteRgb: Array<[number, number, number]>;
} {
  const {
    datasetKey,
    category,
    idea,
    calibration,
    spriteIndex,
    lessons,
    approvals = [],
    approvedSprites = [],
    fewShotCount = 8,
  } = params;

  // Strictly filter to `train` split ONLY for statistics, palette, and few-shot donors
  const trainEntriesAll = spriteIndex.entries.filter((e) => e.split === 'train');
  const trainEntriesCat = trainEntriesAll.filter((e) => e.category === category);
  const trainPool = trainEntriesCat.length >= 3 ? trainEntriesCat : trainEntriesAll;

  // Allowed donor suggestions: train + validation only (NEVER test; spriteIndex itself excludes test)
  const allowedDonorIds = Array.from(
    new Set(
      spriteIndex.entries
        .filter((e) => e.split === 'train' || e.split === 'validation')
        .map((e) => e.spriteId)
        .filter((id) => id > 0),
    ),
  );

  // Extract palette strictly from `train` split (ZERO `test` or `validation` leakage).
  // Include hue-bucket diversity across `trainEntriesAll` so chromatic ideas (e.g. ice, fire, emerald, arcane)
  // get authentic hue-shifted colors that actually exist in the uploaded dataset rather than collapsing to gray.
  const colorFreq = new Map<string, { count: number; rgb: [number, number, number] }>();
  const addPaletteFrom = (entries: readonly IndexedSpriteEntry[], weight: number) => {
    for (const entry of entries) {
      for (let i = 0; i < entry.paletteHex.length; i++) {
        const hex = entry.paletteHex[i].toUpperCase();
        const rgb = entry.paletteRgb[i];
        if (!rgb) continue;
        const cur = colorFreq.get(hex);
        if (cur) cur.count += weight;
        else colorFreq.set(hex, { count: weight, rgb });
      }
    }
  };
  // Category train sprites weighted 2x, plus remaining train sprites weighted 1x for chromatic coverage
  addPaletteFrom(trainPool, 2);
  addPaletteFrom(trainEntriesAll, 1);

  const sortedPalette = Array.from(colorFreq.entries()).sort((a, b) => b[1].count - a[1].count);

  const selectedHex = new Set<string>();
  const topPaletteHex: string[] = [];
  const pushUniqueHex = (hex: string) => {
    if (!selectedHex.has(hex) && topPaletteHex.length < 48) {
      selectedHex.add(hex);
      topPaletteHex.push(hex);
    }
  };

  // 1) Top 16 most common colors (anchors outlines, shadows, steel, wood)
  for (const [hex] of sortedPalette.slice(0, 16)) {
    pushUniqueHex(hex);
  }

  // 2) Up to 4 top colors per chromatic hue bucket (60-degree hue sectors with clear saturation)
  const hueBuckets: string[][] = [[], [], [], [], [], []];
  for (const [hex, { rgb }] of sortedPalette) {
    const r = rgb[0] / 255;
    const g = rgb[1] / 255;
    const b = rgb[2] / 255;
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const d = max - min;
    const sat = max === 0 ? 0 : d / max;
    if (sat < 0.22 || max < 0.16) continue;
    let h = 0;
    if (d > 0) {
      if (max === r) h = ((g - b) / d) % 6;
      else if (max === g) h = (b - r) / d + 2;
      else h = (r - g) / d + 4;
      h = (h * 60 + 360) % 360;
    }
    const bucket = Math.min(5, Math.floor(h / 60));
    if (hueBuckets[bucket].length < 5) {
      hueBuckets[bucket].push(hex);
    }
  }
  for (const bucket of hueBuckets) {
    for (const hex of bucket) {
      pushUniqueHex(hex);
    }
  }
  // 3) Fill remaining slots up to 48 with next most frequent train colors
  for (const [hex] of sortedPalette) {
    if (topPaletteHex.length >= 48) break;
    pushUniqueHex(hex);
  }

  // Keep broad train palette (up to 512 exact train RGB colors) for accurate snapping
  const trainPaletteRgb = sortedPalette.slice(0, 512).map(([, v]) => v.rgb);

  // Per-category medians from Critic calibration (which is mined strictly from `train` split)
  const profile =
    calibration.profiles[category]?.sampleCount >= 3
      ? calibration.profiles[category]
      : calibration.globalProfile;

  const nonTileTrain = trainPool.filter((e) => !e.isTile);
  const bboxPool = nonTileTrain.length > 0 ? nonTileTrain : trainPool;
  const typicalBboxW = Math.max(8, Math.min(30, Math.round(medianOf(bboxPool.map((e) => e.bboxW)) || 22)));
  const typicalBboxH = Math.max(8, Math.min(30, Math.round(medianOf(bboxPool.map((e) => e.bboxH)) || 22)));

  // Retrieve 6-10 few-shot real `train` donors matching category + idea
  const targetFewShot = Math.max(6, Math.min(10, fewShotCount));
  const retrieved = retrieveDonorsForPrompt({
    index: spriteIndex,
    category,
    prompt: idea,
    topK: targetFewShot + 4,
    excludeTiles: true,
    approvals,
    approvedSprites,
  });

  // Strictly real positive spriteIds from `train` split only
  const donorEntries = retrieved
    .map((r) => r.entry)
    .filter((e) => e.split === 'train' && e.spriteId > 0)
    .slice(0, targetFewShot);

  const fewShotDonors: StyleDossierDonor[] = donorEntries.map((e) => ({
    spriteId: e.spriteId,
    split: 'train',
    category: e.category,
    tags: e.tags.slice(0, 8),
    paletteHex: e.paletteHex.slice(0, 16),
    bboxW: e.bboxW,
    bboxH: e.bboxH,
    opaqueCount: e.opaqueCount,
    criticScore: e.criticScore,
    asciiPreview: buildCompactAsciiPreview(e.rgba),
  }));

  // Active lessons ranked by existing deterministic retrieval
  const retrievedLessons = retrieveLessonsForQuery(lessons, {
    category,
    datasetKey,
    limit: 8,
  });

  const lumaLevelsP50 = profile.features.lumaLevels.p50 || 4;
  const rampLengthP50 = Math.max(3, Math.min(5, Math.round(lumaLevelsP50 / 1.5)));

  const dossier: StyleDossier = {
    datasetKey,
    category,
    idea: idea.trim(),
    splitSource: 'train_only',
    trainSampleCount: profile.sampleCount,
    categoryMedians: {
      distinctColorsP50: round2(profile.features.distinctColors.p50),
      distinctColorsP10: round2(profile.features.distinctColors.p10),
      distinctColorsP90: round2(profile.features.distinctColors.p90),
      rampLengthP50,
      outlineCoverageP50: round2(profile.features.outlineCoverage.p50),
      outlineDarknessDeltaP50: round2(profile.features.outlineDarknessDelta.p50),
      outlineDoublesRatioP50: round2(profile.features.outlineDoublesRatio.p50),
      nwLightGradientP50: round2(profile.features.nwLightGradient.p50),
      ditherShareP50: round2(profile.features.ditherShare.p50),
      silhouetteFillRatioP50: round2(profile.features.coverage.p50),
      typicalBboxW,
      typicalBboxH,
    },
    topPaletteHex,
    fewShotDonors,
    allowedDonorIds,
    activeLessons: retrievedLessons.map((r) => ({
      id: r.lesson.id,
      source: r.lesson.source,
      polarity: r.lesson.polarity,
      title: r.lesson.title,
      directive: r.lesson.directive,
    })),
  };

  return { dossier, donorEntries, trainPaletteRgb };
}

export interface ValidateEngineeredPromptOptions {
  validDonorIds?: ReadonlySet<number>;
  datasetPaletteRgb?: ReadonlyArray<[number, number, number]>;
  maxAllowedPaletteDistance?: number;
}

/**
 * Strictly validates JSON output from POST /api/gemini/engineer-prompt (or manual edits).
 * Throws a descriptive PT-BR Error on malformed JSON, missing required fields, invalid category,
 * invalid ramps, or out-of-split donor IDs (preventing `test` leakage). Never silently falls back.
 */
export function validateEngineeredPromptJson(
  raw: string | unknown,
  options: ValidateEngineeredPromptOptions = {},
): EngineeredPromptSpec {
  let obj: unknown = raw;
  if (typeof raw === 'string') {
    const trimmed = raw.trim();
    if (!trimmed) {
      throw new Error('JSON do Engenheiro de Prompt está vazio.');
    }
    try {
      obj = JSON.parse(trimmed);
    } catch {
      throw new Error('Engenheiro de Prompt retornou JSON malformado (falha ao fazer parse).');
    }
  }

  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) {
    throw new Error('Resposta do Engenheiro de Prompt deve ser um objeto JSON válido.');
  }

  const rec = obj as Record<string, unknown>;

  // 1. itemName
  if (typeof rec.itemName !== 'string' || rec.itemName.trim().length < 2) {
    throw new Error('Campo "itemName" ausente ou inválido no JSON do prompt.');
  }
  const itemName = rec.itemName.trim().slice(0, 80);

  // 2. category
  if (
    typeof rec.category !== 'string' ||
    !(ALL_ITEM_CATEGORIES as readonly string[]).includes(rec.category)
  ) {
    throw new Error(
      `Campo "category" inválido ("${String(rec.category)}"). Esperado uma das categorias: ${ALL_ITEM_CATEGORIES.join(', ')}.`,
    );
  }
  const category = rec.category as ItemCategory;

  // 3. visualBrief
  if (typeof rec.visualBrief !== 'string' || rec.visualBrief.trim().length < 15) {
    throw new Error('Campo "visualBrief" ausente ou curto demais (mínimo 15 caracteres).');
  }
  const visualBrief = rec.visualBrief.trim().slice(0, 1200);

  // 4. paletteRamps (1..6 materials, each 3..5 valid hex colors, darkest -> brightest)
  if (!Array.isArray(rec.paletteRamps) || rec.paletteRamps.length === 0) {
    throw new Error('Campo "paletteRamps" deve conter ao menos 1 rampa de material.');
  }
  const paletteRamps: PaletteRampSpec[] = [];
  const maxDist = options.maxAllowedPaletteDistance ?? 95;
  for (let i = 0; i < Math.min(6, rec.paletteRamps.length); i++) {
    const r = rec.paletteRamps[i] as Record<string, unknown>;
    if (!r || typeof r !== 'object') {
      throw new Error(`Rampa #${i + 1} em "paletteRamps" não é um objeto válido.`);
    }
    if (typeof r.material !== 'string' || r.material.trim().length === 0) {
      throw new Error(`Rampa #${i + 1} sem nome de material ("material").`);
    }
    if (!Array.isArray(r.colors) || r.colors.length < 3 || r.colors.length > 5) {
      throw new Error(
        `Rampa "${r.material}" deve ter entre 3 e 5 cores hex (recebido ${Array.isArray(r.colors) ? r.colors.length : 0}).`,
      );
    }
    const parsedColors: Array<{ hex: string; rgb: [number, number, number]; luma: number }> = [];
    for (const c of r.colors) {
      if (typeof c !== 'string') {
        throw new Error(`Cor inválida na rampa "${r.material}".`);
      }
      const rgb = hexToRgb(c);
      if (!rgb) {
        throw new Error(`Cor hex inválida "${c}" na rampa "${r.material}" (esperado #RRGGBB).`);
      }
      let finalRgb = rgb;
      if (options.datasetPaletteRgb && options.datasetPaletteRgb.length > 0) {
        const nearest = findNearestPaletteRgb(rgb, options.datasetPaletteRgb);
        if (nearest.distance > maxDist) {
          finalRgb = nearest.rgb;
        }
      }
      parsedColors.push({
        hex: rgbToHex(finalRgb[0], finalRgb[1], finalRgb[2]),
        rgb: finalRgb,
        luma: calcRgbLuma(finalRgb[0], finalRgb[1], finalRgb[2]),
      });
    }
    // Sort monotonically darkest -> brightest so outline/shadow/base/light/highlight order holds
    parsedColors.sort((a, b) => a.luma - b.luma);
    paletteRamps.push({
      material: r.material.trim().slice(0, 40),
      colors: parsedColors.map((c) => c.hex),
    });
  }

  // 5. constraints
  if (!rec.constraints || typeof rec.constraints !== 'object' || Array.isArray(rec.constraints)) {
    throw new Error('Campo "constraints" ausente ou inválido.');
  }
  const cObj = rec.constraints as Record<string, unknown>;
  const fp = cObj.footprintPx as Record<string, unknown> | undefined;
  const fpW = Number(fp?.width);
  const fpH = Number(fp?.height);
  if (!Number.isFinite(fpW) || !Number.isFinite(fpH) || fpW < 4 || fpW > 32 || fpH < 4 || fpH > 32) {
    throw new Error('Campo "constraints.footprintPx" deve informar width e height entre 4 e 32 px.');
  }
  const maxColors = Number(cObj.maxColors);
  if (!Number.isFinite(maxColors) || maxColors < 4 || maxColors > 64) {
    throw new Error('Campo "constraints.maxColors" deve estar entre 4 e 64.');
  }
  if (typeof cObj.outlineRule !== 'string' || cObj.outlineRule.trim().length < 3) {
    throw new Error('Campo "constraints.outlineRule" é obrigatório.');
  }
  if (typeof cObj.noDitheringOrAA !== 'boolean' || typeof cObj.binaryAlpha !== 'boolean') {
    throw new Error('Campos booleanos "constraints.noDitheringOrAA" e "constraints.binaryAlpha" são obrigatórios.');
  }

  // 6. negativePrompt
  if (!Array.isArray(rec.negativePrompt) || rec.negativePrompt.length < 2) {
    throw new Error('Campo "negativePrompt" deve listar ao menos 2 artefatos a evitar.');
  }
  const negativePrompt = rec.negativePrompt
    .map((s) => (typeof s === 'string' ? s.trim() : ''))
    .filter((s) => s.length > 0)
    .slice(0, 16);
  if (negativePrompt.length < 2) {
    throw new Error('Campo "negativePrompt" não contém itens textuais válidos.');
  }

  // 7. donorSuggestions (3-6 real sprite IDs from train/validation, NEVER test)
  if (!Array.isArray(rec.donorSuggestions) || rec.donorSuggestions.length < 3 || rec.donorSuggestions.length > 6) {
    throw new Error(
      `Campo "donorSuggestions" deve conter entre 3 e 6 IDs de sprites reais (recebido ${Array.isArray(rec.donorSuggestions) ? rec.donorSuggestions.length : 0}).`,
    );
  }
  const donorSuggestions: number[] = [];
  for (const rawId of rec.donorSuggestions) {
    const id = Number(rawId);
    if (!Number.isInteger(id) || id <= 0) {
      throw new Error(`ID de doador inválido "${String(rawId)}" em "donorSuggestions".`);
    }
    if (options.validDonorIds && options.validDonorIds.size > 0 && !options.validDonorIds.has(id)) {
      throw new Error(
        `Sprite doador #${id} não pertence aos splits permitidos (train/validation) — possível vazamento do split test.`,
      );
    }
    if (!donorSuggestions.includes(id)) {
      donorSuggestions.push(id);
    }
  }
  if (donorSuggestions.length < 3) {
    throw new Error('Campo "donorSuggestions" deve conter ao menos 3 IDs únicos válidos.');
  }

  return {
    itemName,
    category,
    visualBrief,
    paletteRamps,
    constraints: {
      footprintPx: { width: Math.round(fpW), height: Math.round(fpH) },
      maxColors: Math.round(maxColors),
      outlineRule: cObj.outlineRule.trim().slice(0, 180),
      noDitheringOrAA: cObj.noDitheringOrAA,
      binaryAlpha: cObj.binaryAlpha,
    },
    negativePrompt,
    donorSuggestions,
  };
}

/**
 * Builds an explicit manual starter template (source: 'manual', model: null) grounded in the
 * train-only Style Dossier so the user can review, edit, and approve a structured prompt even
 * when GEMINI_API_KEY is absent. Never claims AI provenance.
 */
export function buildManualPromptDraft(params: {
  idea: string;
  category: ItemCategory;
  dossier: StyleDossier;
  trainPaletteRgb: ReadonlyArray<[number, number, number]>;
}): EngineeredPromptSpec {
  const { idea, category, dossier, trainPaletteRgb } = params;
  const cleanedIdea = idea.trim() || 'Item clássico Tibia';
  const preset = selectMaterialRampForPrompt(cleanedIdea, 0);

  const primaryHex = snapRampColorsToDatasetPalette(
    preset.primary.map(([r, g, b]) => rgbToHex(r, g, b)),
    trainPaletteRgb,
  );
  const accentHex = snapRampColorsToDatasetPalette(
    preset.accent.map(([r, g, b]) => rgbToHex(r, g, b)),
    trainPaletteRgb,
  );

  // Ensure 3-5 stops per ramp
  const ensureStops = (stops: string[], fallback: string[]): string[] => {
    const uniq = Array.from(new Set(stops));
    if (uniq.length >= 3) return uniq.slice(0, 5);
    return fallback.slice(0, Math.max(3, Math.min(5, fallback.length)));
  };

  const rawPrimaryFallback = preset.primary.map(([r, g, b]) => rgbToHex(r, g, b));
  const rawAccentFallback = preset.accent.map(([r, g, b]) => rgbToHex(r, g, b));

  const donorIds = dossier.fewShotDonors.map((d) => d.spriteId).slice(0, 5);
  while (donorIds.length < 3 && dossier.allowedDonorIds.length > donorIds.length) {
    const next = dossier.allowedDonorIds.find((id) => !donorIds.includes(id));
    if (next === undefined) break;
    donorIds.push(next);
  }

  const maxColors = Math.max(
    8,
    Math.min(28, Math.round(dossier.categoryMedians.distinctColorsP50 + 4)),
  );
  const noDither = dossier.categoryMedians.ditherShareP50 < 0.08;

  const orientationHint =
    category === 'Equipamentos'
      ? 'orientação clássica Tibia 3/4 ou diagonal 45° (ponta superior-direita, cabo inferior-esquerdo)'
      : category === 'Containers'
        ? 'vista 3/4 isométrica compacta com tampa e fecho legíveis'
        : category === 'Consumíveis'
          ? 'silhueta vertical centralizada com brilho especular NW no vidro/líquido'
          : 'objeto decorativo em perspectiva top-down 3/4 com base assentada';

  const capitalizedName = cleanedIdea.charAt(0).toUpperCase() + cleanedIdea.slice(1);

  return {
    itemName: capitalizedName.slice(0, 80),
    category,
    visualBrief: `${capitalizedName} em pixel art clássico de Tibia 32×32 (${orientationHint}). Silhueta forte e legível em 1:1 com clusters limpos de cor, luz direcional vinda do noroeste (NW) e sombra mais fria ao sudeste (SE), contorno externo contínuo de 1px escuro hue-shifted e acabamento fiel aos sprites reais de ${category}.`,
    paletteRamps: [
      {
        material: 'Principal (metal/corpo)',
        colors: ensureStops(primaryHex, rawPrimaryFallback),
      },
      {
        material: 'Detalhe / acabamento',
        colors: ensureStops(accentHex, rawAccentFallback),
      },
    ],
    constraints: {
      footprintPx: {
        width: dossier.categoryMedians.typicalBboxW,
        height: dossier.categoryMedians.typicalBboxH,
      },
      maxColors,
      outlineRule: 'Contorno contínuo 1px escuro hue-shifted em toda a silhueta (sem doubles)',
      noDitheringOrAA: noDither,
      binaryAlpha: true,
    },
    negativePrompt: [...DEFAULT_NEGATIVE_PROMPTS],
    donorSuggestions: donorIds,
  };
}

/**
 * Applies a human adjustment comment locally when Gemini is unavailable, producing a new
 * draft version for user diff review & approval (explicitly labeled manual, never AI).
 */
export function applyLocalAdjustmentToPrompt(params: {
  previous: EngineeredPromptSpec;
  comment: string;
}): EngineeredPromptSpec {
  const { previous, comment } = params;
  const cleaned = comment.trim();
  const norm = cleaned
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '');

  const next: EngineeredPromptSpec = {
    ...previous,
    paletteRamps: previous.paletteRamps.map((r) => ({ ...r, colors: [...r.colors] })),
    constraints: {
      ...previous.constraints,
      footprintPx: { ...previous.constraints.footprintPx },
    },
    negativePrompt: [...previous.negativePrompt],
    donorSuggestions: [...previous.donorSuggestions],
  };

  // Append concrete adjustment instruction to brief
  if (cleaned) {
    next.visualBrief = `${previous.visualBrief.replace(/\s*\[Ajuste:[^\]]*\]$/, '')} [Ajuste: ${cleaned}]`.slice(
      0,
      1200,
    );
  }

  // Darker outline adjustment
  if (norm.includes('contorno') && (norm.includes('escuro') || norm.includes('forte') || norm.includes('definido'))) {
    next.constraints.outlineRule =
      'Contorno 1px bem escuro (luma <= 0.12) contínuo e hue-shifted em toda a silhueta';
    next.paletteRamps = next.paletteRamps.map((ramp) => {
      const first = hexToRgb(ramp.colors[0]);
      if (!first) return ramp;
      const darkened = rgbToHex(
        Math.max(8, Math.round(first[0] * 0.65)),
        Math.max(8, Math.round(first[1] * 0.65)),
        Math.max(12, Math.round(first[2] * 0.72)),
      );
      return { ...ramp, colors: [darkened, ...ramp.colors.slice(1)] };
    });
  }

  // Fewer colors adjustment
  if (norm.includes('menos cor') || norm.includes('reduzir cor') || norm.includes('paleta menor')) {
    next.constraints.maxColors = Math.max(6, previous.constraints.maxColors - 4);
  }

  // No dithering adjustment
  if (norm.includes('sem dither') || norm.includes('menos ruido') || norm.includes('limpo')) {
    next.constraints.noDitheringOrAA = true;
    if (!next.negativePrompt.some((n) => n.toLowerCase().includes('dither'))) {
      next.negativePrompt.push('dithering e pixels isolados');
    }
  }

  // Stronger NW highlight
  if (norm.includes('brilho') || norm.includes('luz') || norm.includes('contraste')) {
    next.paletteRamps = next.paletteRamps.map((ramp) => {
      const last = hexToRgb(ramp.colors[ramp.colors.length - 1]);
      if (!last) return ramp;
      const brightened = rgbToHex(
        Math.min(252, last[0] + 22),
        Math.min(252, last[1] + 22),
        Math.min(252, last[2] + 18),
      );
      return { ...ramp, colors: [...ramp.colors.slice(0, -1), brightened] };
    });
  }

  return next;
}

/**
 * Computes a structured human-readable diff between two EngineeredPromptSpec versions
 * so the user can inspect changes after "Pedir ajuste" before clicking "OK, gerar".
 */
export function computePromptDiff(
  prev: EngineeredPromptSpec,
  next: EngineeredPromptSpec,
): PromptDiffChange[] {
  const changes: PromptDiffChange[] = [];

  if (prev.itemName !== next.itemName) {
    changes.push({
      field: 'itemName',
      labelPt: 'Nome do item',
      before: prev.itemName,
      after: next.itemName,
    });
  }
  if (prev.category !== next.category) {
    changes.push({
      field: 'category',
      labelPt: 'Categoria',
      before: prev.category,
      after: next.category,
    });
  }
  if (prev.visualBrief !== next.visualBrief) {
    changes.push({
      field: 'visualBrief',
      labelPt: 'Brief visual',
      before: prev.visualBrief,
      after: next.visualBrief,
    });
  }

  const prevRampsStr = prev.paletteRamps.map((r) => `${r.material}: ${r.colors.join(', ')}`).join(' | ');
  const nextRampsStr = next.paletteRamps.map((r) => `${r.material}: ${r.colors.join(', ')}`).join(' | ');
  if (prevRampsStr !== nextRampsStr) {
    changes.push({
      field: 'paletteRamps',
      labelPt: 'Rampas de paleta',
      before: prevRampsStr,
      after: nextRampsStr,
    });
  }

  if (prev.constraints.maxColors !== next.constraints.maxColors) {
    changes.push({
      field: 'constraints.maxColors',
      labelPt: 'Máximo de cores',
      before: `${prev.constraints.maxColors} cores`,
      after: `${next.constraints.maxColors} cores`,
    });
  }

  if (
    prev.constraints.footprintPx.width !== next.constraints.footprintPx.width ||
    prev.constraints.footprintPx.height !== next.constraints.footprintPx.height
  ) {
    changes.push({
      field: 'constraints.footprintPx',
      labelPt: 'Área útil (px)',
      before: `${prev.constraints.footprintPx.width}×${prev.constraints.footprintPx.height} px`,
      after: `${next.constraints.footprintPx.width}×${next.constraints.footprintPx.height} px`,
    });
  }

  if (prev.constraints.outlineRule !== next.constraints.outlineRule) {
    changes.push({
      field: 'constraints.outlineRule',
      labelPt: 'Regra de contorno',
      before: prev.constraints.outlineRule,
      after: next.constraints.outlineRule,
    });
  }

  if (prev.constraints.noDitheringOrAA !== next.constraints.noDitheringOrAA) {
    changes.push({
      field: 'constraints.noDitheringOrAA',
      labelPt: 'Sem dithering / AA',
      before: prev.constraints.noDitheringOrAA ? 'Sim' : 'Não',
      after: next.constraints.noDitheringOrAA ? 'Sim' : 'Não',
    });
  }

  const prevNeg = prev.negativePrompt.join('; ');
  const nextNeg = next.negativePrompt.join('; ');
  if (prevNeg !== nextNeg) {
    changes.push({
      field: 'negativePrompt',
      labelPt: 'Prompt negativo',
      before: prevNeg,
      after: nextNeg,
    });
  }

  const prevDonors = prev.donorSuggestions.join(', ');
  const nextDonors = next.donorSuggestions.join(', ');
  if (prevDonors !== nextDonors) {
    changes.push({
      field: 'donorSuggestions',
      labelPt: 'Sprites doadores',
      before: prevDonors,
      after: nextDonors,
    });
  }

  return changes;
}
