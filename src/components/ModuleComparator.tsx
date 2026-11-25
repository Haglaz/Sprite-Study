import { useMemo, useState } from 'react';
import {
  AlertTriangle,
  Award,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Database,
  Eye,
  EyeOff,
  FlaskConical,
  Loader2,
  MessageSquarePlus,
  RefreshCw,
  Scale,
  Swords,
  ThumbsDown,
  ThumbsUp,
  Trophy,
} from 'lucide-react';
import { parseCommentToLessonDraft } from '../comparator/elo';
import { runFieldTestC6 } from '../comparator/fieldTest';
import type {
  AgentProfileRecord,
  CommentToLessonDraft,
  CriticRetrainTelemetry,
  FieldTestReport,
} from '../comparator/types';
import { evaluateSprite } from '../critic/evaluator';
import type { CriticCalibration, DiagnosticReport } from '../critic/types';
import type { StoredDatasetSplit } from '../db/schema';
import type { CoevolutionChampion } from '../evolution/types';
import { BENCHMARK_10_PROMPTS, generateSpriteCandidates } from '../generation/pipeline';
import type {
  ApprovalLabelRecord,
  GeneratedSpriteRecord,
  SpriteIndexState,
} from '../generation/types';
import type { LessonRecord } from '../lessons/types';
import type { ItemCategory, LocalDatasetSummary } from '../types/studio';
import { SpriteThumb, type ThumbBackdrop } from './SpriteThumb';

interface ModuleComparatorProps {
  dataset: LocalDatasetSummary | null;
  activeSplit: StoredDatasetSplit | null;
  calibration: CriticCalibration | null;
  diagnosticReport: DiagnosticReport | null;
  spriteIndex: SpriteIndexState | null;
  isIndexing: boolean;
  agents: AgentProfileRecord[];
  lessons: LessonRecord[];
  championsByCategory: Partial<Record<ItemCategory, CoevolutionChampion>>;
  generatedSprites: GeneratedSpriteRecord[];
  approvals: ApprovalLabelRecord[];
  retrainTelemetry: CriticRetrainTelemetry | null;
  isRetrainingCritic: boolean;
  onGoToUpload: () => void;
  onGoToStudio: () => void;
  onRunIndexing: () => Promise<SpriteIndexState | null>;
  onRecordApproval: (
    sprite: GeneratedSpriteRecord,
    decision: 'approved' | 'rejected',
  ) => Promise<void>;
  onSaveGeneratedBatch: (records: GeneratedSpriteRecord[]) => Promise<void>;
  onRecordEloMatch: (params: {
    agentAId: string;
    agentBId: string;
    winner: 'A' | 'B' | 'draw';
    prompt: string;
  }) => Promise<{ deltaA: number; deltaB: number }>;
  onSetActiveAgent: (agentId: string) => Promise<void>;
  onCreateLessonFromComment: (draft: CommentToLessonDraft) => Promise<void>;
  onManualRetrainCritic: () => Promise<void>;
}

type ComparatorSubTab = 'arena' | 'ranking' | 'field_test';
type ArenaMode = 'elo_duel' | 'vs_real';

interface ActiveDuelState {
  prompt: string;
  category: ItemCategory;
  leftAgent: AgentProfileRecord;
  rightAgent: AgentProfileRecord;
  leftSprite: GeneratedSpriteRecord;
  rightSprite: GeneratedSpriteRecord;
  revealed: boolean;
  votedWinner?: 'A' | 'B' | 'draw';
  eloDeltaLeft?: number;
  eloDeltaRight?: number;
}

function deterministicFlip(key: string): boolean {
  let h = 2166136261;
  for (let i = 0; i < key.length; i++) {
    h ^= key.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h & 1) === 1;
}

export function ModuleComparator({
  dataset,
  activeSplit,
  calibration,
  diagnosticReport,
  spriteIndex,
  isIndexing,
  agents,
  lessons,
  championsByCategory,
  generatedSprites,
  approvals,
  retrainTelemetry,
  isRetrainingCritic,
  onGoToUpload,
  onGoToStudio,
  onRunIndexing,
  onRecordApproval,
  onSaveGeneratedBatch,
  onRecordEloMatch,
  onSetActiveAgent,
  onCreateLessonFromComment,
  onManualRetrainCritic,
}: ModuleComparatorProps) {
  const [subTab, setSubTab] = useState<ComparatorSubTab>('arena');
  const [arenaMode, setArenaMode] = useState<ArenaMode>('elo_duel');
  const [backdrop, setBackdrop] = useState<ThumbBackdrop>('transparent');

  // Elo Duel state
  const [duelPromptIdx, setDuelPromptIdx] = useState(0);
  const [duelAgentAId, setDuelAgentAId] = useState<string>('');
  const [duelAgentBId, setDuelAgentBId] = useState<string>('');
  const [duelState, setDuelState] = useState<ActiveDuelState | null>(null);
  const [isRunningDuel, setIsRunningDuel] = useState(false);

  // Vs Real .spr state
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [blindVsReal, setBlindVsReal] = useState(true);
  const [revealedVsRealIds, setRevealedVsRealIds] = useState<Record<string, boolean>>({});

  // Comment-to-lesson state
  const [commentText, setCommentText] = useState('');
  const [isSavingLesson, setIsSavingLesson] = useState(false);

  // C.6 Field Test state
  const [fieldBaselineId, setFieldBaselineId] = useState<string>('');
  const [fieldCandidateId, setFieldCandidateId] = useState<string>('');
  const [fieldPromptCount, setFieldPromptCount] = useState<number>(6);
  const [isRunningFieldTest, setIsRunningFieldTest] = useState(false);
  const [fieldTestReport, setFieldTestReport] = useState<FieldTestReport | null>(null);

  // Feedback notices
  const [statusNotice, setStatusNotice] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isVoting, setIsVoting] = useState(false);
  const [activatingId, setActivatingId] = useState<string | null>(null);

  const activeAgent = useMemo(
    () => agents.find((a) => a.isActive) ?? agents[0] ?? null,
    [agents],
  );

  const resolvedDuelAgentA = useMemo(() => {
    if (duelAgentAId) {
      const found = agents.find((a) => a.id === duelAgentAId);
      if (found) return found;
    }
    return activeAgent ?? agents[0] ?? null;
  }, [agents, duelAgentAId, activeAgent]);

  const resolvedDuelAgentB = useMemo(() => {
    if (duelAgentBId && duelAgentBId !== resolvedDuelAgentA?.id) {
      const found = agents.find((a) => a.id === duelAgentBId);
      if (found) return found;
    }
    return agents.find((a) => a.id !== resolvedDuelAgentA?.id) ?? null;
  }, [agents, duelAgentBId, resolvedDuelAgentA]);

  const resolvedFieldBaseline = useMemo(() => {
    if (fieldBaselineId) {
      const found = agents.find((a) => a.id === fieldBaselineId);
      if (found) return found;
    }
    return (
      agents.find((a) => a.repairPolicy === 'none') ??
      agents.find((a) => !a.isActive) ??
      agents[0] ??
      null
    );
  }, [agents, fieldBaselineId]);

  const resolvedFieldCandidate = useMemo(() => {
    if (fieldCandidateId && fieldCandidateId !== resolvedFieldBaseline?.id) {
      const found = agents.find((a) => a.id === fieldCandidateId);
      if (found) return found;
    }
    return (
      agents.find((a) => a.isActive && a.id !== resolvedFieldBaseline?.id) ??
      agents.find((a) => a.id !== resolvedFieldBaseline?.id) ??
      null
    );
  }, [agents, fieldCandidateId, resolvedFieldBaseline]);

  const safeIndex =
    generatedSprites.length > 0 ? Math.min(selectedIndex, generatedSprites.length - 1) : 0;
  const currentGenerated = generatedSprites[safeIndex] ?? null;

  const targetCommentSprite: GeneratedSpriteRecord | null =
    arenaMode === 'elo_duel' && duelState
      ? duelState.leftSprite
      : currentGenerated;

  const parsedCommentPreview = useMemo(() => {
    if (!targetCommentSprite || commentText.trim().length < 5) return null;
    return parseCommentToLessonDraft(commentText, targetCommentSprite);
  }, [commentText, targetCommentSprite]);

  const realEvalForCurrent = useMemo(() => {
    if (!currentGenerated || !calibration) return null;
    return evaluateSprite(currentGenerated.nearestReal.rgba, {
      category: currentGenerated.nearestReal.category,
      calibration,
    });
  }, [currentGenerated, calibration]);

  const approvedCount = approvals.filter((a) => a.decision === 'approved').length;
  const rejectedCount = approvals.filter((a) => a.decision === 'rejected').length;
  const totalVotes = approvedCount + rejectedCount;
  const humanPrefActive = totalVotes >= 6 && approvedCount > 0 && rejectedCount > 0;
  const approvedSprites = useMemo(
    () => generatedSprites.filter((s) => s.approvalStatus === 'approved'),
    [generatedSprites],
  );

  if (!dataset || !activeSplit || !calibration || !diagnosticReport) {
    return (
      <section className="max-w-3xl mx-auto py-10">
        <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-6 space-y-4">
          <div className="flex items-start gap-3">
            <Database className="w-5 h-5 text-[#E5A93C] mt-0.5 shrink-0" aria-hidden="true" />
            <div className="space-y-1">
              <h1 className="text-lg font-semibold text-[#F3F5F8]">Comparador A/B & Elo</h1>
              <p className="text-sm text-[#98A2B3]">
                Carregue e calibre um pacote <code className="font-mono text-[#E6EBF3]">.dat</code>{' '}
                e <code className="font-mono text-[#E6EBF3]">.spr</code> para habilitar duelos
                cegos A/B com Elo, curadoria com retreino automático do Crítico v1, conversão de
                comentários em lições e Teste de Campo (C.6).
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onGoToUpload}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-[#E5A93C] text-[#0D0F13] hover:bg-[#EEB955] transition-colors"
          >
            Ir para Pacote .dat/.spr
          </button>
        </div>
      </section>
    );
  }

  const ensureIndex = async (): Promise<SpriteIndexState | null> => {
    if (spriteIndex && spriteIndex.entries.length > 0) return spriteIndex;
    return onRunIndexing();
  };

  const handleStartBlindDuel = async () => {
    if (isRunningDuel || isIndexing || !resolvedDuelAgentA || !resolvedDuelAgentB) return;
    setIsRunningDuel(true);
    setErrorMsg(null);
    setStatusNotice(null);

    try {
      const idx = await ensureIndex();
      if (!idx) throw new Error('Não foi possível indexar o dataset atual.');

      const spec = BENCHMARK_10_PROMPTS[duelPromptIdx % BENCHMARK_10_PROMPTS.length];
      const seed = (Date.now() & 0x7ffff) + duelPromptIdx * 7919;

      const [resA, resB] = await Promise.all([
        generateSpriteCandidates({
          datasetKey: activeSplit.datasetKey,
          prompt: spec.prompt,
          category: spec.category,
          tier: resolvedDuelAgentA.tier,
          useGemini: false,
          maxDiffPixels: resolvedDuelAgentA.maxDiffPixels,
          enableRepair: resolvedDuelAgentA.repairPolicy !== 'none',
          candidateCount: 6,
          topKReturn: 1,
          seed,
          index: idx,
          calibration,
          lessons,
          approvals,
          approvedSprites,
          champion:
            resolvedDuelAgentA.repairPolicy === 'champion_coevolution'
              ? championsByCategory[spec.category] ?? null
              : null,
        }),
        generateSpriteCandidates({
          datasetKey: activeSplit.datasetKey,
          prompt: spec.prompt,
          category: spec.category,
          tier: resolvedDuelAgentB.tier,
          useGemini: false,
          maxDiffPixels: resolvedDuelAgentB.maxDiffPixels,
          enableRepair: resolvedDuelAgentB.repairPolicy !== 'none',
          candidateCount: 6,
          topKReturn: 1,
          seed,
          index: idx,
          calibration,
          lessons,
          approvals,
          approvedSprites,
          champion:
            resolvedDuelAgentB.repairPolicy === 'champion_coevolution'
              ? championsByCategory[spec.category] ?? null
              : null,
        }),
      ]);

      const spriteA = resA.candidates[0];
      const spriteB = resB.candidates[0];
      if (!spriteA || !spriteB) {
        throw new Error('Não foi possível gerar os dois candidatos do duelo.');
      }

      // Persist both generated sprites with read-back verification so Approve/Reject works seamlessly
      await onSaveGeneratedBatch([spriteA, spriteB]);

      const swap = deterministicFlip(`${spriteA.id}:${spriteB.id}`);
      setDuelState({
        prompt: spec.prompt,
        category: spec.category,
        leftAgent: swap ? resolvedDuelAgentB : resolvedDuelAgentA,
        rightAgent: swap ? resolvedDuelAgentA : resolvedDuelAgentB,
        leftSprite: swap ? spriteB : spriteA,
        rightSprite: swap ? spriteA : spriteB,
        revealed: false,
      });
      setDuelPromptIdx((i) => (i + 1) % BENCHMARK_10_PROMPTS.length);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Falha ao iniciar duelo cego.');
    } finally {
      setIsRunningDuel(false);
    }
  };

  const handleVoteDuel = async (winner: 'A' | 'B' | 'draw') => {
    if (!duelState || duelState.revealed || isVoting) return;
    setIsVoting(true);
    setErrorMsg(null);
    try {
      const res = await onRecordEloMatch({
        agentAId: duelState.leftAgent.id,
        agentBId: duelState.rightAgent.id,
        winner,
        prompt: duelState.prompt,
      });
      setDuelState((prev) =>
        prev
          ? {
              ...prev,
              revealed: true,
              votedWinner: winner,
              eloDeltaLeft: res.deltaA,
              eloDeltaRight: res.deltaB,
            }
          : null,
      );
      setStatusNotice(
        `Duelo Elo gravado e verificado (read-back): ${duelState.leftAgent.name} (${res.deltaA >= 0 ? '+' : ''}${res.deltaA.toFixed(1)}) vs ${duelState.rightAgent.name} (${res.deltaB >= 0 ? '+' : ''}${res.deltaB.toFixed(1)}).`,
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Falha ao registrar duelo Elo.');
    } finally {
      setIsVoting(false);
    }
  };

  const handleVoteApproval = async (
    sprite: GeneratedSpriteRecord,
    decision: 'approved' | 'rejected',
  ) => {
    if (isVoting) return;
    setIsVoting(true);
    setErrorMsg(null);
    try {
      await onRecordApproval(sprite, decision);
      setStatusNotice(
        decision === 'approved'
          ? 'Sprite aprovado, salvo no pool few-shot e Crítico v1 re-treinado automaticamente.'
          : 'Sprite rejeitado como negativo difícil e Crítico v1 re-treinado automaticamente.',
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Falha ao registrar aprovação.');
    } finally {
      setIsVoting(false);
    }
  };

  const handleActivateAgent = async (agent: AgentProfileRecord) => {
    if (activatingId) return;
    setActivatingId(agent.id);
    setErrorMsg(null);
    try {
      await onSetActiveAgent(agent.id);
      setStatusNotice(
        `Agente "${agent.name}" promovido como ativo e verificado (read-back) no IndexedDB.`,
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Falha ao ativar agente.');
    } finally {
      setActivatingId(null);
    }
  };

  const handleSubmitCommentLesson = async () => {
    if (!parsedCommentPreview || isSavingLesson) return;
    setIsSavingLesson(true);
    setErrorMsg(null);
    try {
      await onCreateLessonFromComment(parsedCommentPreview);
      setCommentText('');
      setStatusNotice(
        `Lição "${parsedCommentPreview.title}" criada a partir do comentário e verificada (read-back) no banco de Lições Aprendidas.`,
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Falha ao converter comentário em lição.');
    } finally {
      setIsSavingLesson(false);
    }
  };

  const handleRunFieldTest = async () => {
    if (
      isRunningFieldTest ||
      isIndexing ||
      !resolvedFieldBaseline ||
      !resolvedFieldCandidate
    ) {
      return;
    }
    setIsRunningFieldTest(true);
    setErrorMsg(null);
    setStatusNotice(null);
    try {
      const idx = await ensureIndex();
      if (!idx) throw new Error('Indexe o dataset antes de executar o Teste de Campo.');

      await new Promise((r) => window.setTimeout(r, 15));
      const report = await runFieldTestC6({
        datasetKey: activeSplit.datasetKey,
        baselineAgent: resolvedFieldBaseline,
        candidateAgent: resolvedFieldCandidate,
        index: idx,
        calibration,
        lessons,
        approvals,
        approvedSprites,
        championsByCategory,
        promptCount: fieldPromptCount,
      });
      setFieldTestReport(report);
      setStatusNotice(
        `Teste de Campo (C.6) concluído em ${report.promptCount} prompts: ${report.verdict} (Δ médio ${report.meanScoreDelta >= 0 ? '+' : ''}${report.meanScoreDelta.toFixed(1)} pts).`,
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Falha ao executar Teste de Campo C.6.');
    } finally {
      setIsRunningFieldTest(false);
    }
  };

  const recentMatches = useMemo(() => {
    const seen = new Set<string>();
    const rows: Array<{
      matchId: string;
      agentName: string;
      opponentName: string;
      outcome: 'win' | 'loss' | 'draw';
      eloBefore: number;
      eloAfter: number;
      eloDelta: number;
      prompt: string;
      playedAt: string;
    }> = [];
    for (const a of agents) {
      for (const m of a.matchHistory) {
        if (seen.has(m.matchId)) continue;
        seen.add(m.matchId);
        rows.push({
          matchId: m.matchId,
          agentName: a.name,
          opponentName: m.opponentName,
          outcome: m.outcome,
          eloBefore: m.eloBefore,
          eloAfter: m.eloAfter,
          eloDelta: m.eloDelta,
          prompt: m.prompt,
          playedAt: m.playedAt,
        });
      }
    }
    return rows.sort((a, b) => Date.parse(b.playedAt) - Date.parse(a.playedAt)).slice(0, 12);
  }, [agents]);

  return (
    <section className="space-y-6">
      {/* Header & Sub-Navigation */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 border-b border-[#1E232D] pb-4">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold tracking-tight text-[#F3F5F8]">
            Comparador A/B, Elo & Teste de Campo (C.6)
          </h1>
          <p className="text-sm text-[#98A2B3]">
            Duelos cegos com ranking Elo persistido, promoção de agente ativo, comentário → lição,
            aprovações no pool few-shot e retreino automático do Crítico v1.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          {activeAgent && (
            <div
              data-testid="active-agent-summary"
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#111E17] border border-[#1F4A31] text-xs font-mono text-[#86EFAC]"
            >
              <Award className="w-3.5 h-3.5 shrink-0" />
              <span>
                Ativo: {activeAgent.name} (Elo {activeAgent.elo.toFixed(0)})
              </span>
            </div>
          )}

          <div className="flex items-center gap-1 text-xs">
            {(['transparent', 'dark', 'magenta'] as ThumbBackdrop[]).map((bg) => (
              <button
                key={bg}
                type="button"
                onClick={() => setBackdrop(bg)}
                className={`px-2 py-1.5 rounded border transition-colors ${
                  backdrop === bg
                    ? 'bg-[#1D2432] border-[#E5A93C] text-[#F3F5F8]'
                    : 'bg-[#0D0F13] border-[#262C38] text-[#8C97AB]'
                }`}
              >
                {bg === 'transparent' ? 'Grade' : bg === 'dark' ? 'Escuro' : 'Magenta'}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Sub-Tabs */}
      <div className="flex items-center gap-6 border-b border-[#1E232D] text-sm font-medium">
        {(
          [
            ['arena', 'Arena A/B Cega (Elo + Lições)', Swords],
            ['ranking', `Ranking Elo & Tornar ativo (${agents.length})`, Trophy],
            ['field_test', 'Teste de Campo (C.6)', FlaskConical],
          ] as const
        ).map(([id, label, Icon]) => {
          const active = subTab === id;
          return (
            <button
              key={id}
              type="button"
              onClick={() => setSubTab(id)}
              data-testid={`comparator-tab-${id}`}
              className={`inline-flex items-center gap-2 pb-2.5 border-b-2 transition-colors whitespace-nowrap ${
                active
                  ? 'border-[#E5A93C] text-[#F3F5F8]'
                  : 'border-transparent text-[#8C97AB] hover:text-[#E6EBF3]'
              }`}
            >
              <Icon className="w-4 h-4" />
              <span>{label}</span>
            </button>
          );
        })}
      </div>

      {/* Automatic Critic Retraining & Human Labels Telemetry Strip */}
      <div
        data-testid="critic-retrain-strip"
        className="flex flex-wrap lg:flex-nowrap items-center justify-between gap-3 px-4 py-2.5 rounded-xl bg-[#12151C] border border-[#1E232D] text-xs font-mono"
      >
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[#CBD3E1] tabular-nums min-w-0">
          <span>
            Rótulos:{' '}
            <strong className="text-[#86EFAC]">{approvedCount}A</strong> /{' '}
            <strong className="text-[#FCA5A5]">{rejectedCount}R</strong>
          </span>
          <span>
            Pool few-shot:{' '}
            <strong className="text-[#60A5FA]">{approvedSprites.length}</strong>{' '}
            {approvedSprites.length === 1 ? 'sprite' : 'sprites'}
          </span>
          <span>
            AUC (val):{' '}
            <strong className="text-[#F3F5F8]">
              {calibration.discriminator.validationAuc.toFixed(3)}
            </strong>
            {retrainTelemetry && retrainTelemetry.previousValidationAuc !== null && (
              <span className="text-[#8C97AB]">
                {' '}
                (antes {retrainTelemetry.previousValidationAuc.toFixed(3)} ·{' '}
                {retrainTelemetry.retrainCount}×)
              </span>
            )}
          </span>
          <span className={humanPrefActive ? 'text-[#86EFAC]' : 'text-[#FBBF24]'}>
            {humanPrefActive ? 'Termo humano ATIVO (12%)' : `Termo humano ${totalVotes}/6 votos`}
          </span>
        </div>

        <button
          type="button"
          onClick={() => void onManualRetrainCritic()}
          disabled={isRetrainingCritic}
          data-testid="manual-retrain-critic-btn"
          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded border border-[#29303E] bg-[#0D0F13] text-[#CBD3E1] hover:border-[#E5A93C] disabled:opacity-50 shrink-0 whitespace-nowrap"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRetrainingCritic ? 'animate-spin text-[#E5A93C]' : ''}`} />
          <span>{isRetrainingCritic ? 'Re-treinando...' : 'Re-treinar Crítico'}</span>
        </button>
      </div>

      {/* Notices */}
      {errorMsg && (
        <div className="flex items-start gap-2.5 px-4 py-3 rounded-lg bg-[#231417] border border-[#5E2328] text-sm text-[#FCA5A5]">
          <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}
      {statusNotice && (
        <div
          data-testid="comparator-status-notice"
          className="flex items-start gap-2.5 px-4 py-3 rounded-lg bg-[#111E17] border border-[#1F4A31] text-xs text-[#86EFAC]"
        >
          <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0" />
          <span>{statusNotice}</span>
        </div>
      )}

      {/* ==================================================================== */}
      {/* TAB 1: ARENA A/B CEGA (ELO DUEL + VS REAL + COMMENT-TO-LESSON)       */}
      {/* ==================================================================== */}
      {subTab === 'arena' && (
        <div className="space-y-6">
          {/* Arena Mode Toggle */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-xs">
              <button
                type="button"
                onClick={() => setArenaMode('elo_duel')}
                data-testid="arena-mode-elo"
                className={`px-3 py-1.5 rounded-lg border font-medium transition-colors ${
                  arenaMode === 'elo_duel'
                    ? 'bg-[#1D2432] border-[#E5A93C] text-[#F3F5F8]'
                    : 'bg-[#12151C] border-[#1E232D] text-[#8C97AB]'
                }`}
              >
                Duelo Cego Agente vs Agente (Elo)
              </button>
              <button
                type="button"
                onClick={() => setArenaMode('vs_real')}
                data-testid="arena-mode-real"
                className={`px-3 py-1.5 rounded-lg border font-medium transition-colors ${
                  arenaMode === 'vs_real'
                    ? 'bg-[#1D2432] border-[#E5A93C] text-[#F3F5F8]'
                    : 'bg-[#12151C] border-[#1E232D] text-[#8C97AB]'
                }`}
              >
                Gerado vs Real .spr ({generatedSprites.length})
              </button>
            </div>
          </div>

          {arenaMode === 'elo_duel' ? (
            <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-5 space-y-5">
              {/* Matchmaker Controls */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-3 items-end">
                <label className="space-y-1">
                  <span className="block text-xs text-[#98A2B3]">Agente 1</span>
                  <select
                    value={resolvedDuelAgentA?.id ?? ''}
                    onChange={(e) => setDuelAgentAId(e.target.value)}
                    className="w-full px-3 py-2 text-xs rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8]"
                  >
                    {agents.map((a) => (
                      <option key={a.id} value={a.id}>
                        {a.name} (Elo {a.elo.toFixed(0)})
                      </option>
                    ))}
                  </select>
                </label>

                <label className="space-y-1">
                  <span className="block text-xs text-[#98A2B3]">Agente 2</span>
                  <select
                    value={resolvedDuelAgentB?.id ?? ''}
                    onChange={(e) => setDuelAgentBId(e.target.value)}
                    className="w-full px-3 py-2 text-xs rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8]"
                  >
                    {agents
                      .filter((a) => a.id !== resolvedDuelAgentA?.id)
                      .map((a) => (
                        <option key={a.id} value={a.id}>
                          {a.name} (Elo {a.elo.toFixed(0)})
                        </option>
                      ))}
                  </select>
                </label>

                <label className="space-y-1">
                  <span className="block text-xs text-[#98A2B3]">Prompt do duelo</span>
                  <select
                    value={duelPromptIdx}
                    onChange={(e) => setDuelPromptIdx(Number(e.target.value))}
                    className="w-full px-3 py-2 text-xs rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8]"
                  >
                    {BENCHMARK_10_PROMPTS.map((p, idx) => (
                      <option key={p.id} value={idx}>
                        [{p.category}] {p.prompt.slice(0, 42)}...
                      </option>
                    ))}
                  </select>
                </label>

                <button
                  type="button"
                  onClick={() => void handleStartBlindDuel()}
                  disabled={isRunningDuel || isIndexing || !resolvedDuelAgentA || !resolvedDuelAgentB}
                  data-testid="start-blind-duel-btn"
                  className="inline-flex items-center justify-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg bg-[#E5A93C] text-[#0D0F13] hover:bg-[#EEB955] disabled:opacity-50 transition-colors"
                >
                  {isRunningDuel || isIndexing ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Swords className="w-4 h-4" />
                  )}
                  <span>{isRunningDuel ? 'Gerando duelo...' : 'Iniciar duelo cego A/B'}</span>
                </button>
              </div>

              {/* Active Duel Display */}
              {!duelState ? (
                <div className="bg-[#0D0F13] border border-[#1B202A] rounded-lg p-8 text-center space-y-1.5">
                  <p className="text-sm text-[#CBD3E1]">
                    Clique em <strong>Iniciar duelo cego A/B</strong> para gerar dois candidatos em
                    paralelo com identidades ocultas e atualizar o ranking Elo.
                  </p>
                  <p className="text-xs text-[#8C97AB]">
                    Ambos os agentes recebem o mesmo prompt e seed sobre o split{' '}
                    <code className="font-mono">train</code> + pool few-shot aprovado.
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <div className="text-xs font-mono text-[#8C97AB]">
                        {duelState.category} ·{' '}
                        {duelState.revealed ? 'Identidades reveladas' : 'Identidades ocultas (cego)'}
                      </div>
                      <h2 className="text-base font-medium text-[#F3F5F8]">{duelState.prompt}</h2>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {(
                      [
                        ['A', duelState.leftAgent, duelState.leftSprite, duelState.eloDeltaLeft],
                        ['B', duelState.rightAgent, duelState.rightSprite, duelState.eloDeltaRight],
                      ] as const
                    ).map(([label, agent, spr, eloDelta]) => (
                      <div
                        key={label}
                        className="bg-[#0D0F13] border border-[#1E232D] rounded-xl p-4 flex flex-col items-center gap-3.5"
                      >
                        <div className="w-full flex items-center justify-between text-xs font-mono">
                          <span className="font-semibold text-[#E5A93C]">Candidato {label}</span>
                          <span className="text-[#98A2B3] truncate max-w-[70%]">
                            {duelState.revealed
                              ? `${agent.name} (${eloDelta !== undefined ? `${eloDelta >= 0 ? '+' : ''}${eloDelta.toFixed(1)} Elo` : ''})`
                              : 'Agente oculto'}
                          </span>
                        </div>

                        <SpriteThumb
                          rgba={spr.rgba}
                          scale={4}
                          backdrop={backdrop}
                          alt={`Duelo candidato ${label}`}
                        />

                        {duelState.revealed ? (
                          <div className="w-full space-y-2.5">
                            <div className="grid grid-cols-3 gap-2 text-center font-mono text-xs bg-[#12151C] border border-[#1E232D] rounded-lg p-2 tabular-nums">
                              <div>
                                <div className="text-[10px] text-[#8C97AB]">Crítico v1</div>
                                <div className="text-sm font-semibold text-[#F3F5F8]">
                                  {spr.evaluation.score.toFixed(1)}
                                </div>
                              </div>
                              <div>
                                <div className="text-[10px] text-[#8C97AB]">Originalidade</div>
                                <div className="text-sm text-[#60A5FA]">
                                  {spr.nearestReal.originalityScore.toFixed(1)}%
                                </div>
                              </div>
                              <div>
                                <div className="text-[10px] text-[#8C97AB]">Cores / Diff</div>
                                <div className="text-sm text-[#CBD3E1]">
                                  {spr.paletteHex.length}c · {spr.provenance.diffPixelCount}px
                                </div>
                              </div>
                            </div>

                            <div className="flex items-center justify-center gap-2">
                              <button
                                type="button"
                                disabled={isVoting}
                                onClick={() => void handleVoteApproval(spr, 'approved')}
                                data-testid={`duel-approve-${label}`}
                                className="inline-flex items-center gap-1 px-3 py-1.5 text-xs rounded-lg border border-[#1F4A31] bg-[#111E17] text-[#86EFAC] hover:bg-[#16291F]"
                              >
                                <ThumbsUp className="w-3.5 h-3.5" />
                                Aprovar {label}
                              </button>
                              <button
                                type="button"
                                disabled={isVoting}
                                onClick={() => void handleVoteApproval(spr, 'rejected')}
                                data-testid={`duel-reject-${label}`}
                                className="inline-flex items-center gap-1 px-3 py-1.5 text-xs rounded-lg border border-[#5E2328] bg-[#231417] text-[#FCA5A5] hover:bg-[#2E191D]"
                              >
                                <ThumbsDown className="w-3.5 h-3.5" />
                                Rejeitar {label}
                              </button>
                            </div>
                          </div>
                        ) : (
                          <button
                            type="button"
                            disabled={isVoting}
                            onClick={() => void handleVoteDuel(label)}
                            data-testid={`duel-vote-${label}`}
                            className="w-full py-2 text-xs font-semibold rounded-lg bg-[#1A202B] border border-[#323B4D] text-[#F3F5F8] hover:border-[#E5A93C] transition-colors"
                          >
                            Escolher {label} como melhor sprite 7.72
                          </button>
                        )}
                      </div>
                    ))}
                  </div>

                  {!duelState.revealed && (
                    <div className="flex justify-center">
                      <button
                        type="button"
                        disabled={isVoting}
                        onClick={() => void handleVoteDuel('draw')}
                        data-testid="duel-vote-draw"
                        className="px-4 py-1.5 text-xs rounded-lg border border-[#29303E] bg-[#0D0F13] text-[#98A2B3] hover:text-[#F3F5F8] hover:border-[#3D475B]"
                      >
                        Registrar empate técnico (0.5 / 0.5 Elo)
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            /* VS REAL .SPR INSPECTION */
            <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-5 space-y-5">
              {!currentGenerated || !realEvalForCurrent ? (
                <div className="text-center py-6 space-y-3">
                  <p className="text-sm text-[#98A2B3]">
                    Gere sprites no Estúdio Prompt ou inicie um duelo acima para comparar gerado vs
                    real do <code className="font-mono">.spr</code>.
                  </p>
                  <button
                    type="button"
                    onClick={onGoToStudio}
                    className="px-4 py-2 text-xs font-semibold rounded-lg bg-[#E5A93C] text-[#0D0F13]"
                  >
                    Abrir Estúdio Prompt
                  </button>
                </div>
              ) : (
                <>
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <div className="text-xs font-mono text-[#8C97AB]">
                        {currentGenerated.category} · Originalidade{' '}
                        {currentGenerated.nearestReal.originalityScore.toFixed(1)}%
                      </div>
                      <h2 className="text-base font-medium text-[#F3F5F8]">
                        {currentGenerated.prompt}
                      </h2>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => setBlindVsReal((v) => !v)}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#CBD3E1]"
                      >
                        {blindVsReal ? <EyeOff className="w-3.5 h-3.5 text-[#E5A93C]" /> : <Eye className="w-3.5 h-3.5 text-[#60A5FA]" />}
                        <span>{blindVsReal ? 'Modo Cego' : 'Modo Aberto'}</span>
                      </button>
                      <button
                        type="button"
                        disabled={safeIndex <= 0}
                        onClick={() => setSelectedIndex((i) => Math.max(0, i - 1))}
                        className="p-1.5 rounded border border-[#262C38] bg-[#0D0F13] text-[#CBD3E1] disabled:opacity-40"
                      >
                        <ChevronLeft className="w-4 h-4" />
                      </button>
                      <span className="text-xs font-mono tabular-nums text-[#98A2B3]">
                        {safeIndex + 1}/{generatedSprites.length}
                      </span>
                      <button
                        type="button"
                        disabled={safeIndex >= generatedSprites.length - 1}
                        onClick={() =>
                          setSelectedIndex((i) => Math.min(generatedSprites.length - 1, i + 1))
                        }
                        className="p-1.5 rounded border border-[#262C38] bg-[#0D0F13] text-[#CBD3E1] disabled:opacity-40"
                      >
                        <ChevronRight className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  {(() => {
                    const isRev = !blindVsReal || Boolean(revealedVsRealIds[currentGenerated.id]);
                    const flip = blindVsReal && deterministicFlip(currentGenerated.id);
                    const sA = flip
                      ? {
                          kind: 'real' as const,
                          rgba: currentGenerated.nearestReal.rgba,
                          title: `Real #${currentGenerated.nearestReal.spriteId}`,
                          ev: realEvalForCurrent,
                        }
                      : {
                          kind: 'generated' as const,
                          rgba: currentGenerated.rgba,
                          title: `Gerado (${currentGenerated.provenance.source})`,
                          ev: currentGenerated.evaluation,
                        };
                    const sB = flip
                      ? {
                          kind: 'generated' as const,
                          rgba: currentGenerated.rgba,
                          title: `Gerado (${currentGenerated.provenance.source})`,
                          ev: currentGenerated.evaluation,
                        }
                      : {
                          kind: 'real' as const,
                          rgba: currentGenerated.nearestReal.rgba,
                          title: `Real #${currentGenerated.nearestReal.spriteId}`,
                          ev: realEvalForCurrent,
                        };

                    return (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {(
                          [
                            ['A', sA],
                            ['B', sB],
                          ] as const
                        ).map(([lbl, slot]) => (
                          <div
                            key={lbl}
                            className="bg-[#0D0F13] border border-[#1E232D] rounded-xl p-4 flex flex-col items-center gap-3"
                          >
                            <div className="w-full flex items-center justify-between text-xs font-mono">
                              <span className="font-semibold text-[#E5A93C]">Slot {lbl}</span>
                              <span className="text-[#98A2B3]">
                                {isRev ? slot.title : 'Oculto (cego)'}
                              </span>
                            </div>
                            <SpriteThumb
                              rgba={slot.rgba}
                              scale={4}
                              backdrop={backdrop}
                              alt={`Slot ${lbl}`}
                            />
                            {!isRev ? (
                              <button
                                type="button"
                                onClick={() => {
                                  setRevealedVsRealIds((p) => ({
                                    ...p,
                                    [currentGenerated.id]: true,
                                  }));
                                  void handleVoteApproval(
                                    currentGenerated,
                                    slot.kind === 'generated' ? 'approved' : 'rejected',
                                  );
                                }}
                                className="w-full py-2 text-xs font-semibold rounded-lg bg-[#1A202B] border border-[#323B4D] text-[#F3F5F8] hover:border-[#E5A93C]"
                              >
                                {lbl} parece mais real (revelar)
                              </button>
                            ) : (
                              <div className="text-xs font-mono text-[#CBD3E1]">
                                Crítico v1: <strong>{slot.ev.score.toFixed(1)}</strong> · Higiene{' '}
                                {slot.ev.hygieneScore.toFixed(1)}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    );
                  })()}
                </>
              )}
            </div>
          )}

          {/* COMMENT-TO-LESSON PANEL */}
          <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-5 space-y-3.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <MessageSquarePlus className="w-4 h-4 text-[#E5A93C]" />
                <h2 className="text-sm font-semibold text-[#F3F5F8]">
                  Comentário → Lição Aprendida
                </h2>
              </div>
              <span className="text-xs font-mono text-[#8C97AB]">
                {targetCommentSprite
                  ? `Alvo: ${targetCommentSprite.category} (${targetCommentSprite.id})`
                  : 'Gere ou inicie um duelo para associar uma categoria'}
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              {[
                'Evitar contorno quebrado nas bordas externas da armadura',
                'Reforçar iluminação NW forte no topo com brilho especular',
                'Evitar cores quase-duplicadas e manter paleta discreta limpa',
              ].map((sample) => (
                <button
                  key={sample}
                  type="button"
                  onClick={() => setCommentText(sample)}
                  className="px-2.5 py-1 rounded border border-[#262C38] bg-[#0D0F13] text-[#98A2B3] hover:text-[#F3F5F8] hover:border-[#3D475B]"
                >
                  {sample}
                </button>
              ))}
            </div>

            <div className="flex flex-col sm:flex-row gap-2.5">
              <input
                type="text"
                value={commentText}
                onChange={(e) => setCommentText(e.target.value)}
                data-testid="comment-to-lesson-input"
                placeholder="Descreva o que melhorar ou evitar (ex: Evitar contorno quebrado nas bordas da armadura)..."
                className="flex-1 px-3.5 py-2 text-sm rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8] placeholder:text-[#5B6476] focus:outline-none focus:border-[#E5A93C]"
              />
              <button
                type="button"
                disabled={!parsedCommentPreview || isSavingLesson}
                onClick={() => void handleSubmitCommentLesson()}
                data-testid="convert-comment-to-lesson-btn"
                className="inline-flex items-center justify-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg bg-[#1A202B] border border-[#E5A93C] text-[#F3F5F8] hover:bg-[#232B3A] disabled:opacity-40 whitespace-nowrap"
              >
                {isSavingLesson ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <BookOpen className="w-3.5 h-3.5 text-[#E5A93C]" />
                )}
                <span>Converter em Lição</span>
              </button>
            </div>

            {parsedCommentPreview && (
              <div
                data-testid="comment-lesson-preview"
                className="px-3.5 py-2.5 rounded-lg bg-[#0D0F13] border border-[#1E232D] text-xs font-mono text-[#98A2B3] space-y-1"
              >
                <div className="flex flex-wrap items-center gap-3 text-[#E6EBF3]">
                  <span>
                    Escopo: <strong className="text-[#60A5FA]">{parsedCommentPreview.scopeKey}</strong>
                  </span>
                  <span>
                    Polaridade:{' '}
                    <strong
                      className={
                        parsedCommentPreview.polarity === 'avoid'
                          ? 'text-[#FCA5A5]'
                          : 'text-[#86EFAC]'
                      }
                    >
                      {parsedCommentPreview.polarity === 'avoid' ? 'EVITAR' : 'EXIGIR'}
                    </strong>
                  </span>
                  <span>Categoria: {parsedCommentPreview.category}</span>
                </div>
                <div>Verificação: {parsedCommentPreview.validationCheck}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ==================================================================== */}
      {/* TAB 2: RANKING ELO & "TORNAR ATIVO"                                  */}
      {/* ==================================================================== */}
      {subTab === 'ranking' && (
        <div className="space-y-6">
          <div className="bg-[#12151C] border border-[#1E232D] rounded-xl overflow-hidden">
            <div className="px-5 py-3.5 border-b border-[#1E232D] flex items-center justify-between">
              <h2 className="text-sm font-semibold text-[#F3F5F8]">
                Classificação Elo dos Agentes (IndexedDB · read-back verificado)
              </h2>
              <span className="text-xs font-mono text-[#8C97AB]">K = 32 · Base 1200</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="bg-[#0D0F13] text-[#8C97AB] border-b border-[#1E232D] font-mono">
                  <tr>
                    <th className="text-left px-4 py-2.5 font-medium">#</th>
                    <th className="text-left px-4 py-2.5 font-medium">Agente</th>
                    <th className="text-left px-4 py-2.5 font-medium">Configuração</th>
                    <th className="text-right px-4 py-2.5 font-medium">Elo</th>
                    <th className="text-right px-4 py-2.5 font-medium">V / E / D</th>
                    <th className="text-right px-4 py-2.5 font-medium">Ação</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1A1F28]">
                  {agents.map((agent, idx) => (
                    <tr key={agent.id} data-testid={`agent-row-${agent.id}`}>
                      <td className="px-4 py-3 font-mono text-[#8C97AB] tabular-nums">{idx + 1}</td>
                      <td className="px-4 py-3">
                        <div className="font-medium text-[#F3F5F8] flex items-center gap-2">
                          <span>{agent.name}</span>
                          {agent.isActive && (
                            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#111E17] border border-[#1F4A31] text-[#86EFAC]">
                              ATIVO
                            </span>
                          )}
                        </div>
                        <div className="text-[11px] text-[#8C97AB] mt-0.5">
                          {agent.descriptionPt}
                        </div>
                      </td>
                      <td className="px-4 py-3 font-mono text-[#98A2B3] whitespace-nowrap">
                        {agent.tier} · diff {agent.maxDiffPixels}px · {agent.repairPolicy}
                      </td>
                      <td className="px-4 py-3 text-right font-mono font-semibold text-sm text-[#E5A93C] tabular-nums">
                        {agent.elo.toFixed(1)}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-[#CBD3E1] tabular-nums whitespace-nowrap">
                        {agent.wins} / {agent.draws} / {agent.losses}
                      </td>
                      <td className="px-4 py-3 text-right whitespace-nowrap">
                        {agent.isActive ? (
                          <span className="inline-flex items-center gap-1 text-xs font-mono text-[#86EFAC]">
                            <Check className="w-3.5 h-3.5" />
                            Ativo no Estúdio
                          </span>
                        ) : (
                          <button
                            type="button"
                            disabled={activatingId === agent.id}
                            onClick={() => void handleActivateAgent(agent)}
                            data-testid={`activate-agent-btn-${agent.id}`}
                            className="px-3 py-1.5 text-xs font-medium rounded-lg bg-[#1A202B] border border-[#323B4D] text-[#F3F5F8] hover:border-[#E5A93C] disabled:opacity-50"
                          >
                            {activatingId === agent.id ? 'Ativando...' : 'Tornar ativo'}
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Recent Elo Match Log */}
          <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-5 space-y-3">
            <h3 className="text-sm font-semibold text-[#F3F5F8]">
              Histórico de Duelos Elo ({recentMatches.length})
            </h3>
            {recentMatches.length === 0 ? (
              <p className="text-xs text-[#8C97AB]">
                Nenhum duelo cego registrado ainda. Jogue na aba Arena A/B Cega para calibrar o Elo.
              </p>
            ) : (
              <div className="space-y-2">
                {recentMatches.map((m) => (
                  <div
                    key={m.matchId}
                    className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 rounded-lg bg-[#0D0F13] border border-[#1A1F28] text-xs font-mono tabular-nums"
                  >
                    <div className="text-[#E6EBF3]">
                      <strong>{m.agentName}</strong> vs {m.opponentName} —{' '}
                      <span
                        className={
                          m.outcome === 'win'
                            ? 'text-[#86EFAC]'
                            : m.outcome === 'loss'
                              ? 'text-[#FCA5A5]'
                              : 'text-[#FBBF24]'
                        }
                      >
                        {m.outcome === 'win' ? 'Vitória' : m.outcome === 'loss' ? 'Derrota' : 'Empate'}
                      </span>
                    </div>
                    <div className="text-[#98A2B3]">
                      {m.eloBefore.toFixed(1)} → {m.eloAfter.toFixed(1)} (
                      {m.eloDelta >= 0 ? '+' : ''}
                      {m.eloDelta.toFixed(1)})
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ==================================================================== */}
      {/* TAB 3: TESTE DE CAMPO (C.6)                                          */}
      {/* ==================================================================== */}
      {subTab === 'field_test' && (
        <div className="space-y-6">
          <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-5 space-y-4">
            <div className="space-y-1">
              <h2 className="text-sm font-semibold text-[#F3F5F8]">
                Teste de Campo Pareado (Seção C.6)
              </h2>
              <p className="text-xs text-[#98A2B3]">
                Avalia um agente candidato contra um baseline nos mesmos prompts fixos e seeds
                (usando apenas o split <code className="font-mono">train</code> + pool aprovado;
                split <code className="font-mono">test</code> intocado), exigindo 100% alpha binário
                e zero regressões de regras duras.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 items-end">
              <label className="space-y-1">
                <span className="block text-xs text-[#98A2B3]">Agente Baseline</span>
                <select
                  value={resolvedFieldBaseline?.id ?? ''}
                  onChange={(e) => setFieldBaselineId(e.target.value)}
                  className="w-full px-3 py-2 text-xs rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8]"
                >
                  {agents.map((a) => (
                    <option key={a.id} value={a.id}>
                      {a.name}
                    </option>
                  ))}
                </select>
              </label>

              <label className="space-y-1">
                <span className="block text-xs text-[#98A2B3]">Agente Candidato</span>
                <select
                  value={resolvedFieldCandidate?.id ?? ''}
                  onChange={(e) => setFieldCandidateId(e.target.value)}
                  className="w-full px-3 py-2 text-xs rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8]"
                >
                  {agents
                    .filter((a) => a.id !== resolvedFieldBaseline?.id)
                    .map((a) => (
                      <option key={a.id} value={a.id}>
                        {a.name}
                      </option>
                    ))}
                </select>
              </label>

              <label className="space-y-1">
                <span className="block text-xs text-[#98A2B3]">Prompts do teste</span>
                <select
                  value={fieldPromptCount}
                  onChange={(e) => setFieldPromptCount(Number(e.target.value))}
                  className="w-full px-3 py-2 text-xs rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8]"
                >
                  <option value={4}>4 prompts (rápido)</option>
                  <option value={6}>6 prompts (padrão C.6)</option>
                  <option value={10}>10 prompts (completo)</option>
                </select>
              </label>

              <button
                type="button"
                disabled={
                  isRunningFieldTest ||
                  isIndexing ||
                  !resolvedFieldBaseline ||
                  !resolvedFieldCandidate
                }
                onClick={() => void handleRunFieldTest()}
                data-testid="run-field-test-btn"
                className="inline-flex items-center justify-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg bg-[#E5A93C] text-[#0D0F13] hover:bg-[#EEB955] disabled:opacity-50"
              >
                {isRunningFieldTest || isIndexing ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <FlaskConical className="w-4 h-4" />
                )}
                <span>{isRunningFieldTest ? 'Executando C.6...' : 'Executar Teste de Campo'}</span>
              </button>
            </div>
          </div>

          {fieldTestReport && (
            <div className="space-y-4" data-testid="field-test-results">
              {/* Verdict Banner */}
              <div
                className={`px-4 py-3.5 rounded-xl border flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 ${
                  fieldTestReport.verdict === 'APROVADO EM CAMPO'
                    ? 'bg-[#111E17] border-[#1F4A31] text-[#86EFAC]'
                    : fieldTestReport.verdict === 'REGRESSÃO DETECTADA'
                      ? 'bg-[#231417] border-[#5E2328] text-[#FCA5A5]'
                      : 'bg-[#1D1A12] border-[#5B451B] text-[#FCD34D]'
                }`}
              >
                <div>
                  <div className="text-xs font-mono font-semibold tracking-wider">
                    VEREDITO C.6: {fieldTestReport.verdict}
                  </div>
                  <div className="text-xs opacity-90 mt-0.5">
                    {fieldTestReport.verdictRationalePt}
                  </div>
                </div>
                {fieldTestReport.verdict !== 'REGRESSÃO DETECTADA' &&
                  resolvedFieldCandidate &&
                  !resolvedFieldCandidate.isActive && (
                    <button
                      type="button"
                      onClick={() => void handleActivateAgent(resolvedFieldCandidate)}
                      className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-[#0D0F13] border border-current whitespace-nowrap"
                    >
                      Tornar candidato ativo
                    </button>
                  )}
              </div>

              {/* 4 Headline Field Metrics */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 font-mono tabular-nums">
                <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-4">
                  <div className="text-[11px] text-[#8C97AB]">Candidato vs Baseline</div>
                  <div className="text-xl font-semibold text-[#F3F5F8] mt-1">
                    {fieldTestReport.candidateMeanScore.toFixed(1)} vs{' '}
                    {fieldTestReport.baselineMeanScore.toFixed(1)}
                  </div>
                  <div className="text-[11px] text-[#8C97AB]">
                    Δ médio {fieldTestReport.meanScoreDelta >= 0 ? '+' : ''}
                    {fieldTestReport.meanScoreDelta.toFixed(1)} · mediana{' '}
                    {fieldTestReport.medianScoreDelta >= 0 ? '+' : ''}
                    {fieldTestReport.medianScoreDelta.toFixed(1)}
                  </div>
                </div>

                <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-4">
                  <div className="text-[11px] text-[#8C97AB]">Placar Pareado (V/E/D)</div>
                  <div className="text-xl font-semibold text-[#E5A93C] mt-1">
                    {fieldTestReport.candidateWins}V / {fieldTestReport.ties}E /{' '}
                    {fieldTestReport.baselineWins}D
                  </div>
                  <div className="text-[11px] text-[#8C97AB]">
                    {fieldTestReport.promptCount} prompts fixos
                  </div>
                </div>

                <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-4">
                  <div className="text-[11px] text-[#8C97AB]">Originalidade Média</div>
                  <div className="text-xl font-semibold text-[#60A5FA] mt-1">
                    {fieldTestReport.candidateMeanOriginality.toFixed(1)}%
                  </div>
                  <div className="text-[11px] text-[#8C97AB]">
                    Baseline: {fieldTestReport.baselineMeanOriginality.toFixed(1)}%
                  </div>
                </div>

                <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-4">
                  <div className="text-[11px] text-[#8C97AB]">Higiene & Regras Duras</div>
                  <div className="text-xl font-semibold text-[#22C55E] mt-1">
                    {Math.round(fieldTestReport.binaryAlphaPassRate * fieldTestReport.promptCount)}/
                    {fieldTestReport.promptCount} ·{' '}
                    {Math.round(fieldTestReport.hardViolationFreeRate * fieldTestReport.promptCount)}/
                    {fieldTestReport.promptCount}
                  </div>
                  <div className="text-[11px] text-[#8C97AB]">
                    Alpha binário · Sem violações duras
                  </div>
                </div>
              </div>

              {/* Paired Visual Outcomes Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {fieldTestReport.outcomes.map((o) => (
                  <div
                    key={o.promptId}
                    className="bg-[#12151C] border border-[#1E232D] rounded-xl p-4 space-y-3"
                  >
                    <div className="flex items-center justify-between gap-2 text-xs">
                      <span className="font-medium text-[#F3F5F8] truncate">{o.prompt}</span>
                      <span
                        className={`font-mono font-semibold shrink-0 ${
                          o.winner === 'candidate'
                            ? 'text-[#86EFAC]'
                            : o.winner === 'baseline'
                              ? 'text-[#FCA5A5]'
                              : 'text-[#98A2B3]'
                        }`}
                      >
                        Δ {o.scoreDelta >= 0 ? '+' : ''}
                        {o.scoreDelta.toFixed(1)}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-3 bg-[#0D0F13] border border-[#1B202A] rounded-lg p-3">
                      <div className="flex items-center gap-3">
                        <SpriteThumb
                          rgba={o.baselineRecord.rgba}
                          scale={3}
                          backdrop={backdrop}
                          alt={`Baseline: ${o.prompt}`}
                        />
                        <div className="text-xs font-mono tabular-nums space-y-0.5">
                          <div className="text-[10px] text-[#8C97AB]">BASELINE</div>
                          <div className="text-base font-semibold text-[#CBD3E1]">
                            {o.baselineRecord.evaluation.score.toFixed(1)}
                          </div>
                          <div className="text-[10px] text-[#8C97AB]">
                            Orig {o.baselineRecord.nearestReal.originalityScore.toFixed(1)}%
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-3 border-l border-[#1E232D] pl-3">
                        <SpriteThumb
                          rgba={o.candidateRecord.rgba}
                          scale={3}
                          backdrop={backdrop}
                          alt={`Candidato: ${o.prompt}`}
                        />
                        <div className="text-xs font-mono tabular-nums space-y-0.5">
                          <div className="text-[10px] text-[#E5A93C]">CANDIDATO</div>
                          <div className="text-base font-semibold text-[#F3F5F8]">
                            {o.candidateRecord.evaluation.score.toFixed(1)}
                          </div>
                          <div className="text-[10px] text-[#60A5FA]">
                            Orig {o.candidateRecord.nearestReal.originalityScore.toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
