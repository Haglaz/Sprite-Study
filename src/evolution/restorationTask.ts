import { evaluateSprite } from '../critic/evaluator';
import type { CriticCalibration, CriticIssueCode } from '../critic/types';
import { createSeededRng } from '../db/split';
import type { ItemCategory } from '../types/studio';
import { SPRITE_SIZE } from '../utils/tibiaParser';
import { applyNeatPixelRefinementPass, applyRecipePipeline } from './pixelOps';
import type {
  EvaluationAggregate,
  NeatGenome,
  RecipeGenome,
  RestorationCorruptionType,
  RestorationSample,
  RestorationTripletPreview,
} from './types';

export const RESTORATION_CORRUPTION_TYPES: readonly RestorationCorruptionType[] = [
  'near_dup_noise',
  'broken_outline_pillow',
  'orphan_confetti_blur',
  'border_scatter',
] as const;

export const CORRUPTION_LABELS_PT: Record<RestorationCorruptionType, string> = {
  near_dup_noise: 'Paleta suja + ruído',
  broken_outline_pillow: 'Contorno quebrado + pillow',
  orphan_confetti_blur: 'Órfãos + halo AA',
  border_scatter: 'Desalinhado + confete',
};

function pIdx(x: number, y: number): number {
  return (y * SPRITE_SIZE + x) * 4;
}

function clampByte(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v)));
}

/**
 * Applies a deterministic, realistic pixel-art corruption to a clean real .spr sprite.
 * Every corruption introduces genuine defects that Critic v1 catches and that pixel-editor ops + NEAT can fix.
 */
export function corruptSpriteForRestoration(
  cleanRgba: Uint8ClampedArray,
  kind: RestorationCorruptionType,
  seed: number,
  isTile = false,
): Uint8ClampedArray {
  const rng = createSeededRng(seed);
  const out = new Uint8ClampedArray(cleanRgba);

  if (kind === 'near_dup_noise') {
    // Perturb interior RGB channels by ±3..8 (dirty near-duplicate palette) + semi-transparent edge + stray pixels
    for (let y = 1; y < SPRITE_SIZE - 1; y++) {
      for (let x = 1; x < SPRITE_SIZE - 1; x++) {
        const i = pIdx(x, y);
        if (out[i + 3] === 0) continue;
        const isEdge =
          cleanRgba[pIdx(x - 1, y) + 3] === 0 ||
          cleanRgba[pIdx(x + 1, y) + 3] === 0 ||
          cleanRgba[pIdx(x, y - 1) + 3] === 0 ||
          cleanRgba[pIdx(x, y + 1) + 3] === 0;
        if (isEdge && !isTile && rng() < 0.25) {
          out[i + 3] = 140 + Math.floor(rng() * 80); // AA halo
        } else if (rng() < 0.65) {
          const dr = Math.floor(rng() * 11) - 5;
          const dg = Math.floor(rng() * 11) - 5;
          const db = Math.floor(rng() * 11) - 5;
          out[i] = clampByte(out[i] + dr);
          out[i + 1] = clampByte(out[i + 1] + dg);
          out[i + 2] = clampByte(out[i + 2] + db);
        }
      }
    }
    return out;
  }

  if (kind === 'broken_outline_pillow') {
    // Strip dark perimeter outline pixels on NW/W/N edges + darken NW interior to induce weak lighting / broken outline
    for (let y = 1; y < SPRITE_SIZE - 1; y++) {
      for (let x = 1; x < SPRITE_SIZE - 1; x++) {
        const i = pIdx(x, y);
        if (cleanRgba[i + 3] === 0) continue;
        const isEdge =
          cleanRgba[pIdx(x - 1, y) + 3] === 0 ||
          cleanRgba[pIdx(x + 1, y) + 3] === 0 ||
          cleanRgba[pIdx(x, y - 1) + 3] === 0 ||
          cleanRgba[pIdx(x, y + 1) + 3] === 0;
        if (isEdge && !isTile && rng() < 0.72) {
          // Brighten perimeter pixel so dark outline is broken
          out[i] = clampByte(out[i] + 105);
          out[i + 1] = clampByte(out[i + 1] + 95);
          out[i + 2] = clampByte(out[i + 2] + 85);
        } else if (!isEdge && x < 16 && y < 16 && rng() < 0.45) {
          // Flatten NW highlight
          out[i] = clampByte(out[i] * 0.72);
          out[i + 1] = clampByte(out[i + 1] * 0.72);
          out[i + 2] = clampByte(out[i + 2] * 0.72);
        }
      }
    }
    return out;
  }

  if (kind === 'orphan_confetti_blur') {
    // Add 4 detached orphan pixels + AA halo around several border pixels + interior confetti
    if (!isTile) {
      const corners: [number, number][] = [
        [2, 2],
        [29, 3],
        [3, 29],
        [28, 28],
      ];
      for (const [ox, oy] of corners) {
        const oi = pIdx(ox, oy);
        if (out[oi + 3] === 0) {
          out[oi] = 175 + Math.floor(rng() * 50);
          out[oi + 1] = 140 + Math.floor(rng() * 50);
          out[oi + 2] = 95 + Math.floor(rng() * 50);
          out[oi + 3] = 255;
        }
      }
    }
    // Add mild semi-transparent halo outside silhouette + occasional interior confetti
    for (let y = 1; y < SPRITE_SIZE - 1; y++) {
      for (let x = 1; x < SPRITE_SIZE - 1; x++) {
        const i = pIdx(x, y);
        if (cleanRgba[i + 3] === 0 && !isTile) {
          const adjOpaque =
            cleanRgba[pIdx(x - 1, y) + 3] > 0 || cleanRgba[pIdx(x, y - 1) + 3] > 0;
          if (adjOpaque && rng() < 0.25) {
            out[i] = 48;
            out[i + 1] = 48;
            out[i + 2] = 56;
            out[i + 3] = 92; // AA fringe
          }
        } else if (cleanRgba[i + 3] > 0 && rng() < 0.14) {
          out[i] = clampByte(out[i] + (rng() < 0.5 ? 45 : -45));
          out[i + 1] = clampByte(out[i + 1] + (rng() < 0.5 ? 40 : -40));
          out[i + 2] = clampByte(out[i + 2] + (rng() < 0.5 ? 35 : -35));
        }
      }
    }
    return out;
  }

  // border_scatter: nudge non-tile sprites toward top-left border + sprinkle near-duplicate drift
  let minX = SPRITE_SIZE;
  let minY = SPRITE_SIZE;
  for (let y = 0; y < SPRITE_SIZE; y++) {
    for (let x = 0; x < SPRITE_SIZE; x++) {
      if (cleanRgba[pIdx(x, y) + 3] > 0) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
      }
    }
  }
  const shifted = new Uint8ClampedArray(SPRITE_SIZE * SPRITE_SIZE * 4);
  const shiftX = !isTile && minX >= 2 ? minX : 0;
  const shiftY = !isTile && minY >= 2 ? minY : 0;

  for (let y = 0; y < SPRITE_SIZE; y++) {
    for (let x = 0; x < SPRITE_SIZE; x++) {
      const src = pIdx(x, y);
      if (cleanRgba[src + 3] === 0) continue;
      const nx = x - shiftX;
      const ny = y - shiftY;
      if (nx < 0 || ny < 0 || nx >= SPRITE_SIZE || ny >= SPRITE_SIZE) continue;
      const dst = pIdx(nx, ny);
      const jitter = rng() < 0.5 ? Math.floor(rng() * 9) - 4 : 0;
      shifted[dst] = clampByte(cleanRgba[src] + jitter);
      shifted[dst + 1] = clampByte(cleanRgba[src + 1] + jitter);
      shifted[dst + 2] = clampByte(cleanRgba[src + 2] - jitter);
      shifted[dst + 3] = 255;
    }
  }
  return shifted;
}

/**
 * Builds a deterministic restoration benchmark batch from real sprites of a single split (`train` or `validation`).
 * Never mixes splits.
 */
export function buildRestorationSamples(params: {
  sprites: Array<{ spriteId: number; rgba: Uint8ClampedArray }>;
  category: ItemCategory;
  calibration: CriticCalibration;
  seed: number;
  maxCount: number;
}): RestorationSample[] {
  const { sprites, category, calibration, seed, maxCount } = params;
  const selected = sprites.slice(0, maxCount);
  const samples: RestorationSample[] = [];

  for (let i = 0; i < selected.length; i++) {
    const item = selected[i];
    const cleanEval = evaluateSprite(item.rgba, { category, calibration });
    const corruptionType = RESTORATION_CORRUPTION_TYPES[i % RESTORATION_CORRUPTION_TYPES.length];
    const corruptedRgba = corruptSpriteForRestoration(
      item.rgba,
      corruptionType,
      seed + item.spriteId * 31 + i * 17,
      cleanEval.isTile,
    );
    const corruptedEval = evaluateSprite(corruptedRgba, { category, calibration });
    const corruptedHardViolations = corruptedEval.issues.filter((iss) => iss.severity === 'hard').length;

    samples.push({
      spriteId: item.spriteId,
      category,
      corruptionType,
      cleanRgba: item.rgba,
      corruptedRgba,
      cleanScore: cleanEval.score,
      corruptedScore: corruptedEval.score,
      corruptedHardViolations,
      isTile: cleanEval.isTile,
    });
  }

  return samples;
}

/**
 * Computes translation-invariant binary silhouette IoU and interior exact-color similarity
 * so centering a shifted sprite isn't unfairly penalized while erasing/deforming is caught.
 */
export function computeStructuralFidelity(
  cleanRgba: Uint8ClampedArray,
  restoredRgba: Uint8ClampedArray,
): { silhouetteIoU: number; pixelSimilarity: number } {
  const getBbox = (buf: Uint8ClampedArray) => {
    let minX = SPRITE_SIZE;
    let maxX = -1;
    let minY = SPRITE_SIZE;
    let maxY = -1;
    for (let y = 0; y < SPRITE_SIZE; y++) {
      for (let x = 0; x < SPRITE_SIZE; x++) {
        if (buf[pIdx(x, y) + 3] >= 128) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    return { minX, maxX, minY, maxY };
  };

  const b1 = getBbox(cleanRgba);
  const b2 = getBbox(restoredRgba);
  if (b1.maxX < b1.minX || b2.maxX < b2.minX) {
    return { silhouetteIoU: 0, pixelSimilarity: 0 };
  }

  // Align centroids (integer pixel offset) to handle center_and_pad restoration cleanly
  const dx = Math.round((b2.minX + b2.maxX) / 2 - (b1.minX + b1.maxX) / 2);
  const dy = Math.round((b2.minY + b2.maxY) / 2 - (b1.minY + b1.maxY) / 2);

  let inter = 0;
  let union = 0;
  let colorSimSum = 0;

  for (let y = 0; y < SPRITE_SIZE; y++) {
    for (let x = 0; x < SPRITE_SIZE; x++) {
      const i1 = pIdx(x, y);
      const a1 = cleanRgba[i1 + 3] >= 128;

      const rx = x + dx;
      const ry = y + dy;
      const inBounds = rx >= 0 && rx < SPRITE_SIZE && ry >= 0 && ry < SPRITE_SIZE;
      const i2 = inBounds ? pIdx(rx, ry) : -1;
      const a2 = inBounds && restoredRgba[i2 + 3] >= 128;

      if (a1 || a2) union++;
      if (a1 && a2) {
        inter++;
        const dist =
          (Math.abs(cleanRgba[i1] - restoredRgba[i2]) +
            Math.abs(cleanRgba[i1 + 1] - restoredRgba[i2 + 1]) +
            Math.abs(cleanRgba[i1 + 2] - restoredRgba[i2 + 2])) /
          765;
        colorSimSum += Math.max(0, 1 - dist * 1.4);
      }
    }
  }

  const silhouetteIoU = union > 0 ? inter / union : 0;
  const pixelSimilarity = inter > 0 ? colorSimSum / inter : 0;
  return { silhouetteIoU, pixelSimilarity };
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function round1(v: number): number {
  return Math.round(v * 10) / 10;
}

function round3(v: number): number {
  return Math.round(v * 1000) / 1000;
}

/**
 * Applies a cooperative (RecipeGenome -> NeatGenome) pair to a single sprite.
 */
export function applyEditorPairToSprite(
  corruptedRgba: Uint8ClampedArray,
  recipe: RecipeGenome,
  neat: NeatGenome | null,
  isTile = false,
): Uint8ClampedArray {
  const afterRecipe = applyRecipePipeline(corruptedRgba, recipe.steps, isTile);
  if (!neat) return afterRecipe;
  return applyNeatPixelRefinementPass(afterRecipe, neat, isTile);
}

/**
 * Evaluates a (RecipeGenome, NeatGenome) editor pair over a restoration batch using Critic v1 + structural fidelity.
 */
export function evaluateEditorPairOnBatch(params: {
  samples: RestorationSample[];
  recipe: RecipeGenome;
  neat: NeatGenome | null;
  calibration: CriticCalibration;
  parsimonyPenaltyPerStep?: number;
  collectPreviews?: boolean;
}): { aggregate: EvaluationAggregate; previews: RestorationTripletPreview[] } {
  const {
    samples,
    recipe,
    neat,
    calibration,
    parsimonyPenaltyPerStep = 0.25,
    collectPreviews = false,
  } = params;

  if (samples.length === 0) {
    return {
      aggregate: {
        meanRestoredScore: 0,
        medianRestoredScore: 0,
        meanCorruptedScore: 0,
        medianCorruptedScore: 0,
        meanDelta: 0,
        medianDelta: 0,
        meanSilhouetteIoU: 0,
        meanPixelSimilarity: 0,
        hardViolationDelta: 0,
        combinedFitness: 0,
      },
      previews: [],
    };
  }

  const restoredScores: number[] = [];
  const corruptedScores: number[] = [];
  const deltas: number[] = [];
  const ious: number[] = [];
  const sims: number[] = [];
  let hardViolationDeltaTotal = 0;
  const previews: RestorationTripletPreview[] = [];

  for (const sample of samples) {
    const restoredRgba = applyEditorPairToSprite(
      sample.corruptedRgba,
      recipe,
      neat,
      sample.isTile,
    );
    const restoredEval = evaluateSprite(restoredRgba, {
      category: sample.category,
      calibration,
      isTileOverride: sample.isTile,
    });
    const { silhouetteIoU, pixelSimilarity } = computeStructuralFidelity(
      sample.cleanRgba,
      restoredRgba,
    );

    const delta = restoredEval.score - sample.corruptedScore;
    const restoredHard = restoredEval.issues.filter((i) => i.severity === 'hard').length;
    hardViolationDeltaTotal += restoredHard - sample.corruptedHardViolations;

    restoredScores.push(restoredEval.score);
    corruptedScores.push(sample.corruptedScore);
    deltas.push(delta);
    ious.push(silhouetteIoU);
    sims.push(pixelSimilarity);

    if (collectPreviews && previews.length < 4) {
      const corruptedEval = evaluateSprite(sample.corruptedRgba, {
        category: sample.category,
        calibration,
        isTileOverride: sample.isTile,
      });
      const beforeSet = new Set<CriticIssueCode>(corruptedEval.issues.map((i) => i.code));
      const afterSet = new Set<CriticIssueCode>(restoredEval.issues.map((i) => i.code));
      const issuesFixed = Array.from(beforeSet).filter((c) => !afterSet.has(c));
      const issuesRemaining = Array.from(afterSet);

      previews.push({
        spriteId: sample.spriteId,
        corruptionType: sample.corruptionType,
        cleanScore: sample.cleanScore,
        corruptedScore: sample.corruptedScore,
        restoredScore: restoredEval.score,
        deltaScore: round1(delta),
        silhouetteIoU: round3(silhouetteIoU),
        issuesFixed,
        issuesRemaining,
        cleanRgba: sample.cleanRgba,
        corruptedRgba: sample.corruptedRgba,
        restoredRgba,
      });
    }
  }

  const n = samples.length;
  const meanRestoredScore = restoredScores.reduce((a, b) => a + b, 0) / n;
  const meanCorruptedScore = corruptedScores.reduce((a, b) => a + b, 0) / n;
  const meanDelta = deltas.reduce((a, b) => a + b, 0) / n;
  const meanSilhouetteIoU = ious.reduce((a, b) => a + b, 0) / n;
  const meanPixelSimilarity = sims.reduce((a, b) => a + b, 0) / n;
  const activeStepCount = recipe.steps.filter((s) => s.enabled).length;

  // Honest multi-term fitness: Critic v1 score + restoration gain + shape/color preservation - step parsimony
  const shapePenalty = meanSilhouetteIoU < 0.72 ? (0.72 - meanSilhouetteIoU) * 60 : 0;
  const hardPenalty = hardViolationDeltaTotal > 0 ? hardViolationDeltaTotal * 5 : 0;
  const combinedFitness =
    meanRestoredScore * 0.55 +
    meanDelta * 0.65 +
    meanSilhouetteIoU * 18 +
    meanPixelSimilarity * 12 -
    activeStepCount * parsimonyPenaltyPerStep -
    shapePenalty -
    hardPenalty;

  return {
    aggregate: {
      meanRestoredScore: round1(meanRestoredScore),
      medianRestoredScore: round1(median(restoredScores)),
      meanCorruptedScore: round1(meanCorruptedScore),
      medianCorruptedScore: round1(median(corruptedScores)),
      meanDelta: round1(meanDelta),
      medianDelta: round1(median(deltas)),
      meanSilhouetteIoU: round3(meanSilhouetteIoU),
      meanPixelSimilarity: round3(meanPixelSimilarity),
      hardViolationDelta: hardViolationDeltaTotal,
      combinedFitness: round1(combinedFitness),
    },
    previews,
  };
}
