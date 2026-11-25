import { createSeededRng } from '../db/split';
import { extractProfileNumericArray, extractSpriteFeatures } from './features';
import {
  corruptSpriteForNegativeSample,
  makeFlatRectangleSprite,
  makeRandomNoiseSprite,
  makeSmoothGradientSprite,
  type CorruptionKind,
} from './fixtures';
import {
  PROFILE_FEATURE_KEYS,
  type DiscriminatorModel,
  type ProfileFeatureKey,
  type SpriteFeatureVector,
} from './types';

const CORRUPTION_KINDS: readonly CorruptionKind[] = [
  'noise',
  'blur_requantize',
  'gradient_overlay',
  'shuffled_pixels',
  'mixel_2x',
  'pillow_blob',
];

function sigmoid(z: number): number {
  if (z >= 24) return 1 / (1 + Math.exp(-24));
  if (z <= -24) return 1 / (1 + Math.exp(24));
  return 1 / (1 + Math.exp(-z));
}

/**
 * Computes exact ROC-AUC (area under the ROC curve) via pairwise ranking with tie handling.
 */
export function computeRocAuc(scores: readonly number[], labels: readonly (0 | 1)[]): number {
  let posCount = 0;
  let negCount = 0;
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] === 1) posCount++;
    else negCount++;
  }
  if (posCount === 0 || negCount === 0) return 0.5;

  let wins = 0;
  for (let i = 0; i < scores.length; i++) {
    if (labels[i] !== 1) continue;
    for (let j = 0; j < scores.length; j++) {
      if (labels[j] !== 0) continue;
      if (scores[i] > scores[j]) wins += 1;
      else if (scores[i] === scores[j]) wins += 0.5;
    }
  }
  return wins / (posCount * negCount);
}

export function predictDiscriminatorProbability(
  model: DiscriminatorModel,
  features: SpriteFeatureVector,
): number {
  const raw = extractProfileNumericArray(features, model.featureKeys);
  let logit = model.bias;
  for (let d = 0; d < raw.length; d++) {
    const std = model.stds[d] > 1e-6 ? model.stds[d] : 1;
    const norm = Math.max(-6, Math.min(6, (raw[d] - model.means[d]) / std));
    logit += model.weights[d] * norm;
  }
  return sigmoid(logit);
}

function buildSyntheticNegatives(
  realRgbas: readonly Uint8ClampedArray[],
  seed: number,
  targetCount: number,
): SpriteFeatureVector[] {
  const rng = createSeededRng(seed);
  const negatives: SpriteFeatureVector[] = [];
  if (realRgbas.length === 0) return negatives;

  // Include canonical synthetic junk shapes first
  negatives.push(extractSpriteFeatures(makeFlatRectangleSprite()));
  negatives.push(extractSpriteFeatures(makeSmoothGradientSprite()));
  negatives.push(extractSpriteFeatures(makeRandomNoiseSprite(seed + 17)));

  let idx = 0;
  while (negatives.length < targetCount) {
    const src = realRgbas[Math.floor(rng() * realRgbas.length)];
    const kind = CORRUPTION_KINDS[idx % CORRUPTION_KINDS.length];
    const corrupted = corruptSpriteForNegativeSample(src, kind, seed + 1009 + idx * 31);
    const feat = extractSpriteFeatures(corrupted);
    if (feat.opaquePixels > 0) {
      negatives.push(feat);
    }
    idx++;
    if (idx > targetCount * 3) break;
  }

  return negatives;
}

export interface TrainDiscriminatorParams {
  trainRealRgbas: readonly Uint8ClampedArray[];
  trainRealFeatures: readonly SpriteFeatureVector[];
  validationRealRgbas: readonly Uint8ClampedArray[];
  validationRealFeatures: readonly SpriteFeatureVector[];
  extraRejectedFeatures?: readonly SpriteFeatureVector[];
  extraApprovedFeatures?: readonly SpriteFeatureVector[];
  seed: number;
  trainedAt?: string;
}

/**
 * Trains an L2-regularized Logistic Regression real-vs-fake discriminator in-browser
 * using strictly `train` split positives vs synthetic corruptions + user rejections,
 * then evaluates exact ROC-AUC on held-out `validation` positives vs held-out negatives.
 */
export function trainLogisticDiscriminator(params: TrainDiscriminatorParams): DiscriminatorModel {
  const featureKeys: readonly ProfileFeatureKey[] = PROFILE_FEATURE_KEYS;
  const dim = featureKeys.length;

  const trainPos = [
    ...params.trainRealFeatures,
    ...(params.extraApprovedFeatures ?? []),
  ];
  const targetTrainNegCount = Math.max(16, Math.min(trainPos.length, 320));
  const syntheticTrainNeg = buildSyntheticNegatives(
    params.trainRealRgbas,
    params.seed + 303,
    targetTrainNegCount,
  );
  const trainNeg = [
    ...syntheticTrainNeg,
    ...(params.extraRejectedFeatures ?? []),
  ];

  // Combine training dataset
  const xTrain: number[][] = [];
  const yTrain: Array<0 | 1> = [];

  for (const feat of trainPos) {
    xTrain.push(extractProfileNumericArray(feat, featureKeys));
    yTrain.push(1);
  }
  for (const feat of trainNeg) {
    xTrain.push(extractProfileNumericArray(feat, featureKeys));
    yTrain.push(0);
  }

  const n = xTrain.length;
  const means = new Array<number>(dim).fill(0);
  const stds = new Array<number>(dim).fill(1);

  if (n > 0) {
    for (let d = 0; d < dim; d++) {
      let sum = 0;
      for (let i = 0; i < n; i++) sum += xTrain[i][d];
      means[d] = sum / n;
      let sq = 0;
      for (let i = 0; i < n; i++) {
        const diff = xTrain[i][d] - means[d];
        sq += diff * diff;
      }
      stds[d] = Math.max(1e-4, Math.sqrt(sq / n));
    }
  }

  // Standardize & clip training matrix
  const xNorm: number[][] = xTrain.map((row) =>
    row.map((val, d) => Math.max(-6, Math.min(6, (val - means[d]) / stds[d]))),
  );

  // Class-balanced batch gradient descent with L2 regularization
  const weights = new Array<number>(dim).fill(0);
  let bias = 0;

  const posWeight = trainPos.length > 0 ? n / (2 * trainPos.length) : 1;
  const negWeight = trainNeg.length > 0 ? n / (2 * trainNeg.length) : 1;
  const lr = 0.18;
  const l2Lambda = 0.015;
  const epochs = 180;

  if (n > 0) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      const gradW = new Array<number>(dim).fill(0);
      let gradB = 0;

      for (let i = 0; i < n; i++) {
        let logit = bias;
        const row = xNorm[i];
        for (let d = 0; d < dim; d++) logit += weights[d] * row[d];
        const pred = sigmoid(logit);
        const wClass = yTrain[i] === 1 ? posWeight : negWeight;
        const err = (pred - yTrain[i]) * wClass;
        gradB += err;
        for (let d = 0; d < dim; d++) {
          gradW[d] += err * row[d];
        }
      }

      const step = lr / (1 + epoch * 0.008);
      bias -= (step * gradB) / n;
      for (let d = 0; d < dim; d++) {
        weights[d] -= step * (gradW[d] / n + l2Lambda * weights[d]);
      }
    }
  }

  const draftModel: DiscriminatorModel = {
    featureKeys,
    means,
    stds,
    weights,
    bias,
    trainPosCount: trainPos.length,
    trainNegCount: trainNeg.length,
    validationAuc: 0.5,
    validationPosCount: 0,
    validationNegCount: 0,
    trainedAt: params.trainedAt ?? new Date().toISOString(),
    seed: params.seed,
  };

  // Compute honest held-out validation ROC-AUC using ONLY validation sprites & corruptions of validation sprites
  const valPos = params.validationRealFeatures;
  const valNeg = buildSyntheticNegatives(
    params.validationRealRgbas,
    params.seed + 909,
    Math.max(12, Math.min(valPos.length, 160)),
  );

  const valScores: number[] = [];
  const valLabels: Array<0 | 1> = [];

  for (const feat of valPos) {
    valScores.push(predictDiscriminatorProbability(draftModel, feat));
    valLabels.push(1);
  }
  for (const feat of valNeg) {
    valScores.push(predictDiscriminatorProbability(draftModel, feat));
    valLabels.push(0);
  }

  draftModel.validationAuc = computeRocAuc(valScores, valLabels);
  draftModel.validationPosCount = valPos.length;
  draftModel.validationNegCount = valNeg.length;

  return draftModel;
}
