import type { CriticIssueCode } from '../critic/types';
import type { GeneratedSpriteRecord, GenerationTier } from '../generation/types';
import type { LessonPolarity, LessonScopeKey } from '../lessons/types';
import type { ItemCategory } from '../types/studio';

export type AgentRepairPolicy = 'critic_ops' | 'champion_coevolution' | 'none';

export interface EloMatchSummary {
  matchId: string;
  opponentId: string;
  opponentName: string;
  outcome: 'win' | 'loss' | 'draw';
  eloBefore: number;
  eloAfter: number;
  eloDelta: number;
  prompt: string;
  playedAt: string;
}

export interface AgentProfileRecord {
  id: string;
  datasetKey: string;
  name: string;
  descriptionPt: string;
  tier: GenerationTier;
  repairPolicy: AgentRepairPolicy;
  maxDiffPixels: number;
  championCategory?: ItemCategory;
  elo: number;
  wins: number;
  losses: number;
  draws: number;
  matchesPlayed: number;
  isActive: boolean;
  matchHistory: EloMatchSummary[];
  createdAt: string;
  updatedAt: string;
  readBackVerified: boolean;
}

export interface CommentToLessonDraft {
  category: ItemCategory;
  scopeKey: LessonScopeKey;
  polarity: LessonPolarity;
  title: string;
  directive: string;
  validationCheck: string;
  relatedIssueCodes: CriticIssueCode[];
  sourceSpriteId: string;
}

export interface FieldTestPromptOutcome {
  promptId: string;
  prompt: string;
  category: ItemCategory;
  baselineRecord: GeneratedSpriteRecord;
  candidateRecord: GeneratedSpriteRecord;
  scoreDelta: number;
  originalityDelta: number;
  winner: 'candidate' | 'baseline' | 'tie';
  candidateBinaryAlphaOk: boolean;
  candidateNoNewHardViolations: boolean;
}

export interface FieldTestReport {
  runAt: string;
  datasetKey: string;
  baselineAgentId: string;
  baselineAgentName: string;
  candidateAgentId: string;
  candidateAgentName: string;
  promptCount: number;
  candidateMeanScore: number;
  baselineMeanScore: number;
  meanScoreDelta: number;
  medianScoreDelta: number;
  candidateMeanOriginality: number;
  baselineMeanOriginality: number;
  candidateWins: number;
  baselineWins: number;
  ties: number;
  binaryAlphaPassRate: number; // 0..1
  hardViolationFreeRate: number; // 0..1
  verdict: 'APROVADO EM CAMPO' | 'EMPATE TÉCNICO' | 'REGRESSÃO DETECTADA';
  verdictRationalePt: string;
  outcomes: FieldTestPromptOutcome[];
}

export interface CriticRetrainTelemetry {
  retrainCount: number;
  lastRetrainedAt: string | null;
  approvedUsed: number;
  rejectedUsed: number;
  previousValidationAuc: number | null;
  currentValidationAuc: number;
  humanPreferenceActive: boolean;
}
