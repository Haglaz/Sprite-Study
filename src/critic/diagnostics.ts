import { createSeededRng, DEFAULT_SPLIT_SEED } from '../db/split';
import type { StoredDatasetSplit } from '../db/schema';
import {
  ALL_ITEM_CATEGORIES,
  type ItemCategory,
  type ParsedDatArchive,
  type ParsedItemMetadata,
  type ParsedSprArchive,
} from '../types/studio';
import { decodeSpritePixels } from '../utils/tibiaParser';
import { trainLogisticDiscriminator } from './discriminator';
import { evaluateSprite } from './evaluator';
import { extractSpriteFeatures } from './features';
import { buildStandardJunkFloorFixtures } from './fixtures';
import { rebuildHumanPreferenceModel } from '../generation/store';
import type { ApprovalLabelRecord } from '../generation/types';
import { mineCategoryProfiles, type LabeledSpriteSample } from './profile';
import type {
  BaselineStats,
  CriticCalibration,
  DiagnosticReport,
  DiagnosticSpriteSample,
  HumanPreferenceModel,
  SpriteFeatureVector,
} from './types';

function computeMedian(sorted: readonly number[]): number {
  const n = sorted.length;
  if (n === 0) return 0;
  const mid = Math.floor(n / 2);
  return n % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Computes summary baseline statistics (mean, median, min, max) plus a deterministic
 * 95% bootstrap confidence interval for the median.
 */
export function computeBaselineStats(
  scores: readonly number[],
  seed = DEFAULT_SPLIT_SEED,
  bootstrapIterations = 300,
): BaselineStats {
  const n = scores.length;
  if (n === 0) {
    return { n: 0, mean: 0, median: 0, min: 0, max: 0, ciLow: 0, ciHigh: 0 };
  }
  const sorted = scores.slice().sort((a, b) => a - b);
  let sum = 0;
  for (let i = 0; i < n; i++) sum += sorted[i];
  const mean = Math.round((sum / n) * 10) / 10;
  const median = Math.round(computeMedian(sorted) * 10) / 10;

  if (n === 1) {
    return {
      n: 1,
      mean,
      median,
      min: sorted[0],
      max: sorted[0],
      ciLow: sorted[0],
      ciHigh: sorted[0],
    };
  }

  const rng = createSeededRng(seed);
  const bootMedians: number[] = new Array(bootstrapIterations);
  const sampleBuf = new Array<number>(n);

  for (let b = 0; b < bootstrapIterations; b++) {
    for (let i = 0; i < n; i++) {
      sampleBuf[i] = scores[Math.floor(rng() * n)];
    }
    sampleBuf.sort((a, bVal) => a - bVal);
    bootMedians[b] = computeMedian(sampleBuf);
  }
  bootMedians.sort((a, b) => a - b);
  const ciLow = Math.round(bootMedians[Math.floor(bootstrapIterations * 0.025)] * 10) / 10;
  const ciHigh = Math.round(bootMedians[Math.min(bootstrapIterations - 1, Math.floor(bootstrapIterations * 0.975))] * 10) / 10;

  return {
    n,
    mean,
    median,
    min: sorted[0],
    max: sorted[n - 1],
    ciLow,
    ciHigh,
  };
}

function subsampleDeterministic<T>(items: readonly T[], maxCount: number, seed: number): T[] {
  if (items.length <= maxCount) return items.slice();
  const rng = createSeededRng(seed);
  const indices = items.map((_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }
  return indices
    .slice(0, maxCount)
    .sort((a, b) => a - b)
    .map((idx) => items[idx]);
}

function buildItemBySpriteMap(dat: ParsedDatArchive): Map<number, ParsedItemMetadata> {
  const map = new Map<number, ParsedItemMetadata>();
  for (const item of dat.items) {
    if (item.primarySpriteId > 0 && !map.has(item.primarySpriteId)) {
      map.set(item.primarySpriteId, item);
    }
  }
  return map;
}

export interface CalibrateFromSamplesParams {
  datasetKey: string;
  trainSamples: readonly LabeledSpriteSample[];
  validationSamples: readonly LabeledSpriteSample[];
  splitTotals: StoredDatasetSplit['totals'];
  extraApprovedFeatures?: readonly SpriteFeatureVector[];
  extraRejectedFeatures?: readonly SpriteFeatureVector[];
  humanPreferences?: HumanPreferenceModel | null;
  seed?: number;
  calibratedAt?: string;
}

/**
 * Pure calibration + diagnostic pipeline operating on already-decoded train & validation samples.
 * Strictly NEVER touches the `test` split.
 */
export function calibrateCriticAndBuildDiagnostics(params: CalibrateFromSamplesParams): {
  calibration: CriticCalibration;
  diagnostics: DiagnosticReport;
} {
  const seed = params.seed ?? DEFAULT_SPLIT_SEED;
  const nowIso = params.calibratedAt ?? new Date().toISOString();

  // 1. Mine per-category distributions strictly from train split
  const { profiles, globalProfile, featureVectors: trainFeatureRows } = mineCategoryProfiles(
    params.trainSamples,
  );

  // 2. Extract validation feature vectors (held-out)
  const valEntries: Array<{ sample: LabeledSpriteSample; features: SpriteFeatureVector }> = [];
  for (const s of params.validationSamples) {
    const feat = extractSpriteFeatures(s.rgba);
    if (feat.opaquePixels > 0) {
      valEntries.push({ sample: s, features: feat });
    }
  }

  // 3. Train logistic discriminator on train positives (+ human approvals) vs synthetic corruptions (+ human rejections); score held-out validation AUC
  const discriminator = trainLogisticDiscriminator({
    trainRealRgbas: trainFeatureRows.map((r) => r.sample.rgba),
    trainRealFeatures: trainFeatureRows.map((r) => r.features),
    validationRealRgbas: valEntries.map((r) => r.sample.rgba),
    validationRealFeatures: valEntries.map((r) => r.features),
    extraApprovedFeatures: params.extraApprovedFeatures,
    extraRejectedFeatures: params.extraRejectedFeatures,
    seed,
    trainedAt: nowIso,
  });

  const calibration: CriticCalibration = {
    datasetKey: params.datasetKey,
    seed,
    calibratedAt: nowIso,
    profiles,
    globalProfile,
    discriminator,
    humanPreferences: params.humanPreferences ?? null,
  };

  // 4. Evaluate held-out validation REAL sprites ("real ceiling")
  const realScores: number[] = [];
  const realEvaluated: DiagnosticSpriteSample[] = [];

  for (const entry of valEntries) {
    const ev = evaluateSprite(entry.sample.rgba, {
      category: entry.sample.category,
      isTileOverride: entry.sample.isTile,
      calibration,
    });
    realScores.push(ev.score);
    realEvaluated.push({
      id: `real-${entry.sample.spriteId}`,
      label: `Real #${entry.sample.spriteId} (${entry.sample.category})`,
      kind: 'real',
      category: entry.sample.category,
      spriteId: entry.sample.spriteId,
      rgba: entry.sample.rgba,
      evaluation: ev,
    });
  }

  // 5. Evaluate canonical JUNK fixtures ("junk floor") with the exact same critic
  const junkFixtures = buildStandardJunkFloorFixtures(seed);
  const junkScores: number[] = [];
  const junkEvaluated: DiagnosticSpriteSample[] = [];

  for (const jf of junkFixtures) {
    const ev = evaluateSprite(jf.rgba, {
      category: jf.category,
      calibration,
    });
    junkScores.push(ev.score);
    junkEvaluated.push({
      id: jf.id,
      label: jf.label,
      kind: 'junk',
      category: jf.category,
      rgba: jf.rgba,
      evaluation: ev,
    });
  }

  // Pick a balanced deterministic showcase of held-out real sprites (up to 12 across categories)
  const realPreview: DiagnosticSpriteSample[] = [];
  for (const cat of ALL_ITEM_CATEGORIES) {
    const catReal = realEvaluated.filter((s) => s.category === cat);
    realPreview.push(...subsampleDeterministic(catReal, 3, seed + cat.length * 19));
  }

  const diagnostics: DiagnosticReport = {
    generatedAt: nowIso,
    seed,
    realCeiling: computeBaselineStats(realScores, seed + 41),
    junkFloor: computeBaselineStats(junkScores, seed + 53),
    discriminatorValidationAuc: discriminator.validationAuc,
    discriminatorValPosCount: discriminator.validationPosCount,
    discriminatorValNegCount: discriminator.validationNegCount,
    splitTotals: { ...params.splitTotals },
    categoryProfiles: profiles,
    realSamples: realPreview,
    junkSamples: junkEvaluated,
  };

  return { calibration, diagnostics };
}

/**
 * Decodes stratified `train` and `validation` sprite samples from a parsed Tibia `.dat`/`.spr`
 * dataset (bounded per category so browser calibration completes cleanly in under ~1.5s) and
 * produces the calibrated Critic v1 + full DiagnosticReport.
 * Never reads the `test` split.
 */
export function calibrateCriticFromArchive(params: {
  datasetKey: string;
  dat: ParsedDatArchive;
  spr: ParsedSprArchive;
  split: StoredDatasetSplit;
  extraApprovedFeatures?: readonly SpriteFeatureVector[];
  extraRejectedFeatures?: readonly SpriteFeatureVector[];
  humanPreferences?: HumanPreferenceModel | null;
  maxTrainPerCategory?: number;
  maxValPerCategory?: number;
  seed?: number;
}): {
  calibration: CriticCalibration;
  diagnostics: DiagnosticReport;
} {
  const seed = params.seed ?? params.split.seed ?? DEFAULT_SPLIT_SEED;
  const maxTrainPerCategory = params.maxTrainPerCategory ?? 140;
  const maxValPerCategory = params.maxValPerCategory ?? 45;
  const itemBySprite = buildItemBySpriteMap(params.dat);

  const trainSamples: LabeledSpriteSample[] = [];
  const validationSamples: LabeledSpriteSample[] = [];

  for (let cIdx = 0; cIdx < ALL_ITEM_CATEGORIES.length; cIdx++) {
    const cat: ItemCategory = ALL_ITEM_CATEGORIES[cIdx];
    const part = params.split.byCategory[cat];

    const trainIds = subsampleDeterministic(part.train, maxTrainPerCategory, seed + cIdx * 101);
    const valIds = subsampleDeterministic(part.validation, maxValPerCategory, seed + cIdx * 211);

    for (const sid of trainIds) {
      const rgba = decodeSpritePixels(params.spr, sid);
      const item = itemBySprite.get(sid);
      const isTileItem = Boolean(
        item &&
          (item.flags.ground ||
            item.flags.groundBorder ||
            item.width > 1 ||
            item.height > 1),
      );
      trainSamples.push({
        spriteId: sid,
        category: cat,
        rgba,
        isTile: isTileItem,
      });
    }

    for (const sid of valIds) {
      const rgba = decodeSpritePixels(params.spr, sid);
      const item = itemBySprite.get(sid);
      const isTileItem = Boolean(
        item &&
          (item.flags.ground ||
            item.flags.groundBorder ||
            item.width > 1 ||
            item.height > 1),
      );
      validationSamples.push({
        spriteId: sid,
        category: cat,
        rgba,
        isTile: isTileItem,
      });
    }
  }

  return calibrateCriticAndBuildDiagnostics({
    datasetKey: params.datasetKey,
    trainSamples,
    validationSamples,
    splitTotals: params.split.totals,
    extraApprovedFeatures: params.extraApprovedFeatures,
    extraRejectedFeatures: params.extraRejectedFeatures,
    humanPreferences: params.humanPreferences,
    seed,
  });
}

/**
 * Automatic Critic Retraining on new human Approve/Reject labels:
 * Feeds approved generated sprite features as additional positives and rejected features
 * as additional hard negatives into the L2 logistic discriminator, rebuilds `humanPreferences`,
 * and re-evaluates honest held-out validation ROC-AUC & baselines without touching `test`.
 */
export function retrainCriticWithHumanApprovals(params: {
  datasetKey: string;
  dat: ParsedDatArchive;
  spr: ParsedSprArchive;
  split: StoredDatasetSplit;
  approvals: readonly ApprovalLabelRecord[];
  previousCalibration?: CriticCalibration | null;
  seed?: number;
}): {
  calibration: CriticCalibration;
  diagnostics: DiagnosticReport;
  previousValidationAuc: number | null;
  currentValidationAuc: number;
  approvedUsed: number;
  rejectedUsed: number;
} {
  const approvedFeats = params.approvals
    .filter((a) => a.decision === 'approved')
    .map((a) => a.features);
  const rejectedFeats = params.approvals
    .filter((a) => a.decision === 'rejected')
    .map((a) => a.features);
  const humanPreferences = rebuildHumanPreferenceModel([...params.approvals]);

  const { calibration, diagnostics } = calibrateCriticFromArchive({
    datasetKey: params.datasetKey,
    dat: params.dat,
    spr: params.spr,
    split: params.split,
    extraApprovedFeatures: approvedFeats,
    extraRejectedFeatures: rejectedFeats,
    humanPreferences,
    seed: params.seed ?? params.split.seed,
  });

  return {
    calibration,
    diagnostics,
    previousValidationAuc: params.previousCalibration?.discriminator.validationAuc ?? null,
    currentValidationAuc: calibration.discriminator.validationAuc,
    approvedUsed: approvedFeats.length,
    rejectedUsed: rejectedFeats.length,
  };
}
