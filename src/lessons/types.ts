import type { CriticIssueCode, ProfileFeatureKey } from '../critic/types';
import type { ItemCategory } from '../types/studio';

export type LessonCategory = ItemCategory | 'Global';

export type LessonSource = 'human_feedback' | 'critic_loop' | 'mined';

export type LessonStatus = 'active' | 'conflicted' | 'superseded' | 'archived';

export type LessonPolarity = 'enforce' | 'avoid';

export type LessonScopeKey =
  | 'palette_size'
  | 'near_duplicate_palette'
  | 'outline_coverage'
  | 'outline_quality'
  | 'nw_lighting'
  | 'pillow_shading'
  | 'cluster_hygiene'
  | 'dither_share'
  | 'hue_shift'
  | 'mixel_resample'
  | 'border_margin'
  | `custom:${string}`;

export interface LessonTargetRange {
  min?: number;
  max?: number;
  p10?: number;
  p50?: number;
  p90?: number;
  unit?: string;
}

export interface LessonEvidence {
  sampleSize: number;
  positiveCount: number;
  negativeCount: number;
  criticDeltaMean: number | null; // Observed average Critic v1 score delta; null if unmeasured
  referenceSpriteIds: number[];
  confidence: number; // Strictly empirical in [0.05, 0.98], never flat 1.0
}

export interface LessonRecord {
  id: string;
  datasetKey: string | null;
  category: LessonCategory;
  scopeKey: LessonScopeKey;
  title: string;
  source: LessonSource;
  status: LessonStatus;
  polarity: LessonPolarity;
  directive: string;
  validationCheck: string;
  metricKey?: ProfileFeatureKey;
  relatedIssueCodes: CriticIssueCode[];
  targetRange?: LessonTargetRange;
  evidence: LessonEvidence;
  conflictGroupId: string | null;
  supersededById: string | null;
  createdAt: string;
  updatedAt: string;
  readBackVerified: boolean;
}

export type LessonConflictReason =
  | 'opposite_polarity'
  | 'disjoint_range'
  | 'competing_directive';

export interface LessonConflict {
  id: string;
  category: LessonCategory;
  scopeKey: LessonScopeKey;
  reason: LessonConflictReason;
  reasonLabelPt: string;
  lessonAId: string;
  lessonBId: string;
  lessonA: LessonRecord;
  lessonB: LessonRecord;
}

export type ConflictResolutionStrategy =
  | 'auto_precedence'
  | 'keep_lesson'
  | 'merge_range';

export interface ConflictResolutionResult {
  winner: LessonRecord;
  loser: LessonRecord;
  strategy: ConflictResolutionStrategy;
  rationalePt: string;
  resolvedAt: string;
}

export interface LessonRetrievalQuery {
  category: ItemCategory;
  issueCodes?: readonly CriticIssueCode[];
  searchText?: string;
  datasetKey?: string | null;
  includeConflicted?: boolean;
  includeSuperseded?: boolean;
  includeArchived?: boolean;
  limit?: number;
  nowMs?: number;
}

export interface RetrievalScoreBreakdown {
  categoryMatch: number;
  sourceAuthority: number;
  issueRelevance: number;
  evidenceStrength: number;
  recency: number;
  textMatch: number;
  total: number;
}

export interface RetrievedLesson {
  lesson: LessonRecord;
  score: RetrievalScoreBreakdown;
}

export const LESSON_SCOPE_OPTIONS: ReadonlyArray<{
  key: LessonScopeKey;
  labelPt: string;
  relatedIssueCodes: CriticIssueCode[];
}> = [
  { key: 'palette_size', labelPt: 'Tamanho da paleta', relatedIssueCodes: ['COLOR_COUNT_OUT_OF_RANGE', 'FLAT_FILL'] },
  { key: 'near_duplicate_palette', labelPt: 'Integridade da paleta (sem quase-duplicadas)', relatedIssueCodes: ['NEAR_DUPLICATE_PALETTE', 'SMOOTH_GRADIENT'] },
  { key: 'outline_coverage', labelPt: 'Contorno escuro externo', relatedIssueCodes: ['MISSING_OUTLINE', 'BROKEN_OUTLINE'] },
  { key: 'outline_quality', labelPt: 'Qualidade de traço (doubles/jaggies)', relatedIssueCodes: ['OUTLINE_DOUBLES', 'STAIRCASE_JAGGIES'] },
  { key: 'nw_lighting', labelPt: 'Direção de luz NW → SE', relatedIssueCodes: ['WEAK_NW_LIGHTING'] },
  { key: 'pillow_shading', labelPt: 'Anti-pillow shading / banding', relatedIssueCodes: ['PILLOW_SHADING', 'BANDING_DETECTED'] },
  { key: 'cluster_hygiene', labelPt: 'Higiene de clusters e ruído', relatedIssueCodes: ['CLUSTER_CONFETTI', 'SPECKLE_NOISE', 'ORPHAN_PIXELS'] },
  { key: 'dither_share', labelPt: 'Proporção de dithering', relatedIssueCodes: ['DITHER_MISMATCH'] },
  { key: 'hue_shift', labelPt: 'Hue shift sombra → luz', relatedIssueCodes: ['LOW_HUE_SHIFT'] },
  { key: 'mixel_resample', labelPt: 'Anti-mixel / escala nativa 1:1', relatedIssueCodes: ['MIXEL_RESAMPLE', 'BINARY_ALPHA'] },
  { key: 'border_margin', labelPt: 'Margem e centralização 32×32', relatedIssueCodes: ['TOUCHING_BORDER', 'OFF_CENTER', 'DISCONNECTED_SILHOUETTE'] },
];

export const SOURCE_AUTHORITY_RANK: Record<LessonSource, number> = {
  human_feedback: 3,
  critic_loop: 2,
  mined: 1,
};

/**
 * Computes an honest, data-driven confidence in [0.05, 0.98] using the Wilson score
 * lower bound (95% z=1.96) over positive vs total observations. Never returns flat 1.0.
 */
export function computeWilsonConfidence(positiveCount: number, negativeCount: number): number {
  const pos = Math.max(0, positiveCount);
  const neg = Math.max(0, negativeCount);
  const total = pos + neg;
  if (total === 0) return 0.25;

  const z = 1.96;
  const p = pos / total;
  const z2 = z * z;
  const denom = 1 + z2 / total;
  const center = p + z2 / (2 * total);
  const margin = z * Math.sqrt((p * (1 - p) + z2 / (4 * total)) / total);
  const lower = (center - margin) / denom;

  return Math.round(Math.max(0.05, Math.min(0.98, lower)) * 1000) / 1000;
}
