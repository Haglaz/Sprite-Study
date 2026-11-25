import type { CriticEvaluation } from '../critic/types';
import type { NearestRealMatch } from '../generation/types';
import type { ItemCategory } from '../types/studio';

export type T3MaterialRole = 'outline' | 'shadow' | 'base' | 'light' | 'highlight';
export type T3TokenRole = 'transparent' | T3MaterialRole;

export const T3_MATERIAL_ROLES: readonly T3MaterialRole[] = [
  'outline',
  'shadow',
  'base',
  'light',
  'highlight',
] as const;

export const T3_ROLE_LABELS_PT: Record<T3MaterialRole, string> = {
  outline: 'Contorno (outline)',
  shadow: 'Sombra (shadow)',
  base: 'Base (base)',
  light: 'Luz (light)',
  highlight: 'Brilho (highlight)',
};

export interface RampRoleVocabularyEntry {
  tokenId: number; // 1..K (0 is reserved for transparent)
  rgb: [number, number, number];
  hex: string;
  luma: number;
  hueDeg: number;
  saturation: number;
  materialFamily: string;
  role: T3MaterialRole;
  autoRole: T3MaterialRole;
  pixelCount: number;
  perimeterRatio: number; // 0..1 fraction of occurrences on outer silhouette boundary
  userOverridden: boolean;
}

export interface MaterialClusterSummary {
  materialFamily: string;
  tokenCount: number;
  totalPixels: number;
  roleCounts: Record<T3MaterialRole, number>;
}

export interface T3RampVocabulary {
  id: string; // `t3-vocab:${datasetKey}`
  datasetKey: string;
  splitHash: string;
  builtAt: string;
  updatedAt: string;
  trainSpriteCount: number;
  vocabSize: number; // 1 (transparent) + entries.length
  entries: RampRoleVocabularyEntry[];
  materialSummaries: MaterialClusterSummary[];
  overrideCount: number;
  readBackVerified: boolean;
}

export interface T3SamplerSettings {
  mode: 'masked-token' | 'autoregressive';
  gridSize: number; // 32
  tokenCount: number; // 1024
  vocabSize: number; // transparent(0) + palette tokens + optional mask
  maskTokenId: number;
  steps: number;
  temperature: number;
  topK: number;
  cfgScale: number;
  maxResampleAttempts: number;
}

export interface T3ModelCard {
  schema?: string;
  version: string;
  steps: number;
  splitHash: string;
  vocabulary: Array<{
    tokenId: number;
    rgb: [number, number, number];
    hex: string;
    materialFamily: string;
    role: T3MaterialRole;
  }>;
  samplerSettings: T3SamplerSettings;
  categories: ItemCategory[];
  tagVocab: string[];
  generatorMetrics?: {
    paramCount?: number;
    trainLoss?: number;
    valLoss?: number;
    hygienePassRates?: {
      binaryAlpha: number;
      outlineClosure: number;
      connectedSilhouette: number;
      borderClean: number;
    };
  };
  rerankerMetrics?: {
    heldOutValAuc: number;
    positiveCount?: number;
    negativeCount?: number;
  };
  exportedAt?: string;
}

export interface T3ConstraintCheckResult {
  passed: boolean;
  binaryAlphaOk: boolean;
  outlineClosureOk: boolean;
  connectedSilhouetteOk: boolean;
  failures: string[];
  mainComponentRatio: number;
  boundaryDarkRatio: number;
}

export interface T3SamplingCandidate {
  candidateIndex: number;
  tokenGrid: Uint8Array; // 1024 entries (0..K)
  rgba: Uint8ClampedArray; // 32*32*4
  constraints: T3ConstraintCheckResult;
  resampleAttemptsUsed: number;
  rerankerProbability: number | null; // 0..1 from t3_reranker.onnx if loaded
  evaluation: CriticEvaluation;
  nearestReal: NearestRealMatch | null;
  memorizedCopyWarning: boolean;
  combinedRankScore: number;
}

export interface T3SamplingBatchResult {
  executionProvider: 'webgpu' | 'wasm' | 'custom-runner';
  requestedCount: number;
  totalSamplingRounds: number;
  constraintFailuresResampled: number;
  candidates: T3SamplingCandidate[];
  samplingMs: number;
}

export interface T3MethodBenchmarkSummary {
  label: string;
  sampleCount: number;
  medianScore: number;
  meanScore: number;
  minScore: number;
  maxScore: number;
  meanOriginality: number | null;
  binaryAlphaRate: number;
  outlineClosureRate: number;
  connectedSilhouetteRate: number;
  memorizationRate: number | null;
}

export interface T3TestSplitComparisonReport {
  evaluatedAt: string;
  splitHash: string;
  testSpriteCount: number;
  benchmarkPromptCount: number;
  realTest: T3MethodBenchmarkSummary;
  junkFloor: T3MethodBenchmarkSummary;
  t1: T3MethodBenchmarkSummary;
  t2: T3MethodBenchmarkSummary;
  t3: T3MethodBenchmarkSummary;
  t3IsWorse: boolean;
  recommendedEnabled: boolean;
  verdictSummaryPt: string;
}

export interface T3ModelPackageRecord {
  id: string; // `t3-model:${datasetKey}`
  datasetKey: string;
  modelCard: T3ModelCard;
  generatorOnnx: ArrayBuffer;
  rerankerOnnx: ArrayBuffer | null;
  generatorFileName: string;
  rerankerFileName: string | null;
  enabled: boolean;
  disabledReasonPt: string | null;
  lastTestEvaluation: T3TestSplitComparisonReport | null;
  importedAt: string;
  updatedAt: string;
  readBackVerified: boolean;
}
