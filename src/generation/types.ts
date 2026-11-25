import type { CriticEvaluation, CriticIssueCode, SpriteFeatureVector } from '../critic/types';
import type { ItemCategory, ParsedItemFlags } from '../types/studio';

export type GenerationTier = 'T1' | 'T2' | 'T3';

export type GenerationProvenanceSource =
  | 'gemini-t1'
  | 'gemini-t2'
  | 'local-t1'
  | 'local-t2'
  | 'onnx-t3';

export type RampRoleCode = 0 | 1 | 2 | 3 | 4 | 5;
// 0 = transparent, 1 = outline, 2 = shadow, 3 = midtone, 4 = highlight, 5 = accent

export const RAMP_ROLE_NAMES: Record<RampRoleCode, string> = {
  0: 'transparent',
  1: 'outline',
  2: 'shadow',
  3: 'midtone',
  4: 'highlight',
  5: 'accent',
};

export interface IndexedSpriteEntry {
  spriteId: number;
  itemId: number;
  category: ItemCategory;
  split: 'train' | 'validation'; // Test split is strictly excluded from retrieval/training index
  flags: ParsedItemFlags;
  tags: string[];
  paletteHex: string[];
  paletteRgb: Array<[number, number, number]>;
  indexedGrid: Uint8Array; // 1024 entries: 0=transparent, 1..N=palette index + 1
  rampRoles: Uint8Array; // 1024 entries: RampRoleCode
  silhouetteMask: Uint8Array; // 1024 entries: 0 | 1
  lumaSignature: number[]; // 64 values (8x8 pooled luminance 0..1)
  rgba: Uint8ClampedArray;
  criticScore: number;
  isTile: boolean;
  opaqueCount: number;
  bboxW: number;
  bboxH: number;
  dominantHue: string;
}

export interface SpriteIndexState {
  datasetKey: string;
  indexedAt: string;
  entries: IndexedSpriteEntry[];
  trainCount: number;
  validationCount: number;
  excludedTestCount: number;
}

export interface RetrievedDonor {
  entry: IndexedSpriteEntry;
  relevanceScore: number;
  matchedTags: string[];
}

export interface PixelDiffEdit {
  x: number;
  y: number;
  r: number;
  g: number;
  b: number;
  a: 0 | 255;
}

export interface NearestRealMatch {
  spriteId: number;
  category: ItemCategory;
  split: 'train' | 'validation';
  rgba: Uint8ClampedArray;
  criticScore: number;
  silhouetteIoU: number;
  pixelDiffRatio: number; // 0..1 fraction of pixels meaningfully changed
  colorDiffMean: number; // 0..1 mean RGB delta across union
  originalityScore: number; // 0..99.5 honest originality metric
}

export interface GenerationProvenance {
  source: GenerationProvenanceSource;
  tier: GenerationTier;
  model: string | null;
  fallbackWarning: string | null;
  donorSpriteIds: number[];
  diffPixelCount: number;
  diffPixelLimit: number;
  repaired: boolean;
  repairDelta: number;
  repairMethod: 'none' | 'champion_coevolution' | 'critic_targeted_ops';
  appliedLessonIds: string[];
  seed: number;
}

export interface GeneratedSpriteRecord {
  id: string;
  datasetKey: string;
  prompt: string;
  category: ItemCategory;
  createdAt: string;
  rgba: Uint8ClampedArray;
  paletteHex: string[];
  evaluation: CriticEvaluation;
  nearestReal: NearestRealMatch;
  provenance: GenerationProvenance;
  approvalStatus: 'pending' | 'approved' | 'rejected';
  approvalUpdatedAt?: string;
  readBackVerified: boolean;
}

export interface ApprovalLabelRecord {
  id: string; // matches sprite record id
  datasetKey: string;
  spriteRecordId: string;
  category: ItemCategory;
  decision: 'approved' | 'rejected';
  criticScore: number;
  originalityScore: number;
  issues: CriticIssueCode[];
  features: SpriteFeatureVector;
  source: GenerationProvenanceSource;
  note?: string;
  createdAt: string;
  readBackVerified: boolean;
}
