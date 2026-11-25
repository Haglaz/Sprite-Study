import type { CriticIssueCode } from '../critic/types';
import type { ItemCategory } from '../types/studio';

// ============================================================================
// 1. REAL NEAT GENOME & POPULATION TYPES
// ============================================================================

export type NeatNodeType = 'input' | 'bias' | 'hidden' | 'output';
export type NeatActivation = 'tanh' | 'sigmoid' | 'relu' | 'linear';

export interface NeatNodeGene {
  id: number;
  type: NeatNodeType;
  activation: NeatActivation;
  layer: number; // 0 for input/bias, 1 for output, (0,1) for hidden nodes (strictly acyclic DAG)
}

export interface NeatConnectionGene {
  innovation: number;
  inNode: number;
  outNode: number;
  weight: number;
  enabled: boolean;
}

export interface NeatGenome {
  id: string;
  nodes: NeatNodeGene[];
  connections: NeatConnectionGene[];
  fitness: number;
  adjustedFitness: number;
  speciesId: number | null;
}

export interface NeatSpecies {
  id: number;
  representative: NeatGenome;
  memberIds: string[];
  bestFitness: number;
  stagnantGenerations: number;
  averageAdjustedFitness: number;
}

export interface InnovationTrackerState {
  nextInnovation: number;
  nextNodeId: number;
  // Encoded as `${inNode}->${outNode}` -> innovation
  connectionInnovations: Record<string, number>;
  // Encoded as `${originalInnovation}` -> { nodeId, inInnov, outInnov }
  splitInnovations: Record<string, { nodeId: number; inInnov: number; outInnov: number }>;
}

export interface NeatConfig {
  populationSize: number;
  inputCount: number;
  outputCount: number;
  c1Excess: number;
  c2Disjoint: number;
  c3Weight: number;
  compatibilityThreshold: number;
  targetSpeciesCount: number;
  weightMutateRate: number;
  weightPerturbScale: number;
  weightReplaceRate: number;
  addConnectionRate: number;
  addNodeRate: number;
  toggleEnableRate: number;
  crossoverRate: number;
  elitismPerSpecies: number;
  stagnationLimit: number;
}

export interface NeatPopulationState {
  generation: number;
  config: NeatConfig;
  tracker: InnovationTrackerState;
  genomes: NeatGenome[];
  species: NeatSpecies[];
  nextSpeciesId: number;
  nextGenomeSeq: number;
  bestGenome: NeatGenome | null;
  rngState: number;
}

// ============================================================================
// 2. PIXEL-EDITOR OPERATIONS & RECIPE GA TYPES
// ============================================================================

export type PixelOpId =
  | 'snap_binary_alpha'
  | 'remove_orphans'
  | 'snap_near_duplicates'
  | 'quantize_palette'
  | 'enforce_dark_outline'
  | 'trim_outline_doubles'
  | 'despeckle_clusters'
  | 'shift_nw_lighting'
  | 'cleanup_dither'
  | 'center_and_pad';

export interface RecipeStepGene {
  op: PixelOpId;
  enabled: boolean;
  /** 0..1 normalized operator strength/aggressiveness */
  intensity: number;
  /** 0..1 normalized threshold parameter */
  threshold: number;
}

export interface RecipeGenome {
  id: string;
  steps: RecipeStepGene[];
  fitness: number;
  valScore: number | null;
}

export interface RecipeGaConfig {
  populationSize: number;
  minSteps: number;
  maxSteps: number;
  tournamentSize: number;
  crossoverRate: number;
  mutationRate: number;
  elitismCount: number;
  parsimonyPenaltyPerStep: number;
}

export interface RecipeGaPopulationState {
  generation: number;
  config: RecipeGaConfig;
  genomes: RecipeGenome[];
  bestGenome: RecipeGenome | null;
  nextGenomeSeq: number;
  rngState: number;
}

// ============================================================================
// 3. RESTORATION BENCHMARK, CO-EVOLUTION, PROMOTION & CHECKPOINT TYPES
// ============================================================================

export type RestorationCorruptionType =
  | 'near_dup_noise'
  | 'broken_outline_pillow'
  | 'orphan_confetti_blur'
  | 'border_scatter';

export interface RestorationSample {
  spriteId: number;
  category: ItemCategory;
  corruptionType: RestorationCorruptionType;
  cleanRgba: Uint8ClampedArray;
  corruptedRgba: Uint8ClampedArray;
  cleanScore: number;
  corruptedScore: number;
  corruptedHardViolations: number;
  isTile: boolean;
}

export interface RestorationTripletPreview {
  spriteId: number;
  corruptionType: RestorationCorruptionType;
  cleanScore: number;
  corruptedScore: number;
  restoredScore: number;
  deltaScore: number;
  silhouetteIoU: number;
  issuesFixed: CriticIssueCode[];
  issuesRemaining: CriticIssueCode[];
  cleanRgba: Uint8ClampedArray;
  corruptedRgba: Uint8ClampedArray;
  restoredRgba: Uint8ClampedArray;
}

export interface EvaluationAggregate {
  meanRestoredScore: number;
  medianRestoredScore: number;
  meanCorruptedScore: number;
  medianCorruptedScore: number;
  meanDelta: number;
  medianDelta: number;
  meanSilhouetteIoU: number;
  meanPixelSimilarity: number;
  hardViolationDelta: number; // restoredHardViolations - corruptedHardViolations (<= 0 desired)
  combinedFitness: number;
}

export interface PromotionRulesConfig {
  minValidationImprovement: number;
  minInitialGainOverCorrupted: number;
  maxOverfitGap: number;
  minSilhouetteIoU: number;
  maxNewHardViolations: number;
  minMarginAboveJunkFloor: number;
}

export interface PromotionLogEntry {
  id: string;
  generation: number;
  timestamp: string;
  decision: 'promoted' | 'rejected';
  reasonPt: string;
  trainScore: number;
  valScore: number;
  prevChampionValScore: number | null;
  valCorruptedScore: number;
  valDeltaFromCorrupted: number;
  silhouetteIoU: number;
  hardViolationDelta: number;
  recipeSummary: string;
  neatSummary: string;
}

export interface EvolutionGenerationPoint {
  generation: number;
  trainBestScore: number;
  trainMeanScore: number;
  valBestScore: number;
  valMedianScore: number;
  valCorruptedMedian: number;
  valDelta: number;
  valSilhouetteIoU: number;
  neatSpeciesCount: number;
  neatNodes: number;
  neatEnabledConnections: number;
  recipeActiveSteps: number;
  promoted: boolean;
  elapsedMs: number;
}

export interface CoevolutionChampion {
  generation: number;
  promotedAt: string;
  recipe: RecipeGenome;
  neat: NeatGenome;
  trainStats: EvaluationAggregate;
  valStats: EvaluationAggregate;
}

export interface CoevolutionRunConfig {
  category: ItemCategory;
  targetGenerations: number;
  populationSize: number;
  seed: number;
  trainBatchSize: number;
  valBatchSize: number;
  promotionRules: PromotionRulesConfig;
}

export interface EvolutionCheckpoint {
  id: string; // `${datasetKey}:${category}`
  datasetKey: string;
  category: ItemCategory;
  config: CoevolutionRunConfig;
  generation: number;
  status: 'idle' | 'running' | 'paused' | 'completed';
  neatState: NeatPopulationState;
  gaState: RecipeGaPopulationState;
  champion: CoevolutionChampion | null;
  history: EvolutionGenerationPoint[];
  promotionLog: PromotionLogEntry[];
  valBaselineCorruptedMedian: number;
  realCeilingMedian: number;
  junkFloorMedian: number;
  trainSampleCount: number;
  valSampleCount: number;
  updatedAt: string;
  readBackVerified: boolean;
}

export interface EliteArchiveRecord {
  id: string; // `${datasetKey}:${category}:gen-${generation}`
  datasetKey: string;
  category: ItemCategory;
  generation: number;
  promotedAt: string;
  recipe: RecipeGenome;
  neat: NeatGenome;
  trainScore: number;
  valScore: number;
  valDelta: number;
  silhouetteIoU: number;
  readBackVerified: boolean;
}
