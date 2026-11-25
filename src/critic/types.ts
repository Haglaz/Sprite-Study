import type { ItemCategory, ParsedItemFlags } from '../types/studio';

export type CriticIssueSeverity = 'hard' | 'hygiene' | 'statistical';

export type CriticIssueCode =
  | 'EMPTY_SPRITE'
  | 'BINARY_ALPHA'
  | 'COLOR_COUNT_OUT_OF_RANGE'
  | 'FLAT_FILL'
  | 'SMOOTH_GRADIENT'
  | 'SPECKLE_NOISE'
  | 'ORPHAN_PIXELS'
  | 'MISSING_OUTLINE'
  | 'DISCONNECTED_SILHOUETTE'
  | 'TOUCHING_BORDER'
  | 'OFF_CENTER'
  | 'NEAR_DUPLICATE_PALETTE'
  | 'MIXEL_RESAMPLE'
  | 'OUTLINE_DOUBLES'
  | 'BROKEN_OUTLINE'
  | 'STAIRCASE_JAGGIES'
  | 'CLUSTER_CONFETTI'
  | 'PILLOW_SHADING'
  | 'BANDING_DETECTED'
  | 'DITHER_MISMATCH'
  | 'WEAK_NW_LIGHTING'
  | 'LOW_HUE_SHIFT'
  | 'STATISTICAL_OUTLIER'
  | 'UNCALIBRATED';

export interface CriticIssue {
  code: CriticIssueCode;
  severity: CriticIssueSeverity;
  labelPt: string;
  detailPt: string;
  measured?: number;
  expected?: string;
}

export interface SpriteFeatureVector {
  // Structural & alpha
  opaquePixels: number;
  coverage: number; // 0..1
  semiTransparentPixels: number;
  distinctColors: number;
  lumaLevels: number;
  borderTouchCount: number;
  centerOffsetNorm: number; // 0..1 (distance from 15.5,15.5 divided by 16)
  bboxWidth: number;
  bboxHeight: number;
  componentCount: number;
  orphanPixels: number;
  orphanRatio: number;

  // Pixel-exact hygiene
  nearDuplicateColorPairs: number;
  nearDuplicateRatio: number; // 0..1
  mixelScore: number; // 0..1 (higher = resampled/upscaled 2x2 mixel look)
  smoothGradientRatio: number; // 0..1
  speckleNoiseScore: number; // 0..1
  strayColorPixelRatio: number; // 0..1
  meanClusterSize: number;

  // Outline & line quality
  outlineCoverage: number; // 0..1
  outlineDarknessDelta: number; // interiorLuma - outlineLuma (-1..1)
  outlineDoublesRatio: number; // 0..1
  outlineDoubleThickRatio: number; // 0..1
  outlineHueShiftedRatio: number; // 0..1
  brokenOutlineRatio: number; // 0..1
  staircaseIrregularity: number; // 0..1

  // Shading & color ramps
  nwLightGradient: number; // -1..1 (positive = NW brighter than SE)
  pillowShadingScore: number; // 0..1
  bandingScore: number; // 0..1
  ditherShare: number; // 0..1
  hueShiftDegrees: number; // 0..180
}

export interface FeatureDistribution {
  mean: number;
  std: number;
  min: number;
  max: number;
  p10: number;
  p25: number;
  p50: number;
  p75: number;
  p90: number;
  n: number;
}

export type ProfileFeatureKey =
  | 'coverage'
  | 'distinctColors'
  | 'lumaLevels'
  | 'nearDuplicateRatio'
  | 'mixelScore'
  | 'smoothGradientRatio'
  | 'speckleNoiseScore'
  | 'strayColorPixelRatio'
  | 'meanClusterSize'
  | 'outlineCoverage'
  | 'outlineDarknessDelta'
  | 'outlineDoublesRatio'
  | 'outlineHueShiftedRatio'
  | 'nwLightGradient'
  | 'pillowShadingScore'
  | 'bandingScore'
  | 'ditherShare'
  | 'hueShiftDegrees';

export const PROFILE_FEATURE_KEYS: readonly ProfileFeatureKey[] = [
  'coverage',
  'distinctColors',
  'lumaLevels',
  'nearDuplicateRatio',
  'mixelScore',
  'smoothGradientRatio',
  'speckleNoiseScore',
  'strayColorPixelRatio',
  'meanClusterSize',
  'outlineCoverage',
  'outlineDarknessDelta',
  'outlineDoublesRatio',
  'outlineHueShiftedRatio',
  'nwLightGradient',
  'pillowShadingScore',
  'bandingScore',
  'ditherShare',
  'hueShiftDegrees',
] as const;

export interface CategoryProfile {
  category: ItemCategory;
  sampleCount: number;
  tileSampleCount: number;
  features: Record<ProfileFeatureKey, FeatureDistribution>;
}

export interface DiscriminatorModel {
  featureKeys: readonly ProfileFeatureKey[];
  means: number[];
  stds: number[];
  weights: number[];
  bias: number;
  trainPosCount: number;
  trainNegCount: number;
  validationAuc: number; // 0..1 computed on held-out validation positives vs held-out negatives
  validationPosCount: number;
  validationNegCount: number;
  trainedAt: string;
  seed: number;
}

export interface HumanPreferenceModel {
  approvedCount: number;
  rejectedCount: number;
  approvedMean: number[];
  rejectedMean: number[];
}

export interface CriticCalibration {
  datasetKey: string;
  seed: number;
  calibratedAt: string;
  profiles: Record<ItemCategory, CategoryProfile>;
  globalProfile: CategoryProfile;
  discriminator: DiscriminatorModel;
  humanPreferences?: HumanPreferenceModel | null;
}

export interface CriticEvaluationContext {
  category: ItemCategory;
  flags?: Partial<ParsedItemFlags>;
  isTileOverride?: boolean;
  calibration?: CriticCalibration | null;
}

export interface CriticEvaluation {
  score: number; // 0..99.9 (honest runtime value, never forced to flat 100)
  calibrated: boolean;
  calibrationLabel: string;
  isTile: boolean;
  hardRuleCap: number;
  hygieneScore: number;
  statisticalScore: number;
  discriminatorScore: number | null;
  humanPreferenceScore: number | null;
  features: SpriteFeatureVector;
  issues: CriticIssue[];
}

export interface BaselineStats {
  n: number;
  mean: number;
  median: number;
  min: number;
  max: number;
  ciLow: number; // 95% bootstrap CI lower bound for the median
  ciHigh: number; // 95% bootstrap CI upper bound for the median
}

export interface DiagnosticSpriteSample {
  id: string;
  label: string;
  kind: 'real' | 'junk';
  category: ItemCategory;
  spriteId?: number;
  rgba: Uint8ClampedArray;
  evaluation: CriticEvaluation;
}

export interface DiagnosticReport {
  generatedAt: string;
  seed: number;
  realCeiling: BaselineStats;
  junkFloor: BaselineStats;
  discriminatorValidationAuc: number;
  discriminatorValPosCount: number;
  discriminatorValNegCount: number;
  splitTotals: {
    train: number;
    validation: number;
    test: number;
    total: number;
  };
  categoryProfiles: Record<ItemCategory, CategoryProfile>;
  realSamples: DiagnosticSpriteSample[];
  junkSamples: DiagnosticSpriteSample[];
}
