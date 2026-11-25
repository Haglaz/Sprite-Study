import { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  Archive,
  Award,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Database,
  Download,
  Layers,
  Loader2,
  MessageSquarePlus,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  Wand2,
  Wrench,
} from 'lucide-react';
import { parseCommentToLessonDraft } from '../comparator/elo';
import type { AgentProfileRecord, CommentToLessonDraft } from '../comparator/types';
import type { CriticCalibration, DiagnosticReport } from '../critic/types';
import type { StoredDatasetSplit } from '../db/schema';
import type { CoevolutionChampion } from '../evolution/types';
import {
  buildTrainingDatasetExportZip,
  encodeRgbaToPngBytes,
  type TrainingDatasetExportResult,
} from '../export/trainingDatasetExport';
import {
  generateFromApprovedPrompt,
  requestEngineeredPromptFromServer,
  type GenerateFromApprovedPromptResult,
} from '../generation/pipeline';
import {
  applyLocalAdjustmentToPrompt,
  buildManualPromptDraft,
  buildStyleDossier,
  computePromptDiff,
} from '../generation/promptEngineer';
import type {
  ApprovalLabelRecord,
  EngineeredPromptSpec,
  GeneratedSpriteRecord,
  IndexedSpriteEntry,
  PromptDiffChange,
  PromptHistoryRecord,
  SpriteIndexState,
} from '../generation/types';
import type { LessonRecord } from '../lessons/types';
import type {
  T3MaterialRole,
  T3ModelPackageRecord,
  T3RampVocabulary,
  T3TestSplitComparisonReport,
} from '../t3/types';
import { ALL_ITEM_CATEGORIES, type ItemCategory, type LocalDatasetSummary } from '../types/studio';
import { PromptEngineerCard } from './PromptEngineerCard';
import { SpriteThumb, type ThumbBackdrop } from './SpriteThumb';
import { T3WorkbenchPanel } from './T3WorkbenchPanel';

interface ModulePromptStudioProps {
  dataset: LocalDatasetSummary | null;
  activeSplit: StoredDatasetSplit | null;
  calibration: CriticCalibration | null;
  diagnosticReport: DiagnosticReport | null;
  spriteIndex: SpriteIndexState | null;
  isIndexing: boolean;
  indexProgress: { done: number; total: number } | null;
  lessons: LessonRecord[];
  activeAgent: AgentProfileRecord | null;
  championsByCategory: Partial<Record<ItemCategory, CoevolutionChampion>>;
  generatedSprites: GeneratedSpriteRecord[];
  approvals: ApprovalLabelRecord[];
  promptHistory: PromptHistoryRecord[];
  t3Vocabulary: T3RampVocabulary | null;
  t3ModelPackage: T3ModelPackageRecord | null;
  onGoToUpload: () => void;
  onRunIndexing: () => Promise<SpriteIndexState | null>;
  onSavePromptVersion: (record: PromptHistoryRecord) => Promise<PromptHistoryRecord>;
  onSaveGeneratedBatch: (records: GeneratedSpriteRecord[]) => Promise<void>;
  onRecordApproval: (
    sprite: GeneratedSpriteRecord,
    decision: 'approved' | 'rejected',
  ) => Promise<void>;
  onDeleteGenerated: (id: string) => Promise<void>;
  onCreateLessonFromComment: (draft: CommentToLessonDraft) => Promise<void>;
  onUpdateT3VocabRole: (tokenId: number, role: T3MaterialRole) => Promise<void>;
  onRebuildT3Vocab: () => Promise<void>;
  onImportT3ModelPackage: (params: {
    modelCardJson: string;
    generatorOnnx: ArrayBuffer;
    rerankerOnnx: ArrayBuffer | null;
    generatorFileName: string;
    rerankerFileName: string | null;
  }) => Promise<T3ModelPackageRecord>;
  onSaveT3TestEvaluation: (report: T3TestSplitComparisonReport) => Promise<void>;
  onToggleT3Enabled: (enabled: boolean) => Promise<void>;
}

const SOURCE_LABELS: Record<string, string> = {
  'gemini-t1': 'gemini-t1',
  'gemini-t2': 'gemini-t2',
  'local-t1': 'local-t1',
  'local-t2': 'local-t2',
  'onnx-t3': 'onnx-t3',
};

function triggerBlobDownload(bytes: Uint8Array, mimeType: string, filename: string): void {
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  const blob = new Blob([copy.buffer], { type: mimeType });
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}

export function ModulePromptStudio({
  dataset,
  activeSplit,
  calibration,
  diagnosticReport,
  spriteIndex,
  isIndexing,
  indexProgress,
  lessons,
  activeAgent,
  championsByCategory,
  generatedSprites,
  approvals,
  promptHistory,
  t3Vocabulary,
  t3ModelPackage,
  onGoToUpload,
  onRunIndexing,
  onSavePromptVersion,
  onSaveGeneratedBatch,
  onRecordApproval,
  onDeleteGenerated,
  onCreateLessonFromComment,
  onUpdateT3VocabRole,
  onRebuildT3Vocab,
  onImportT3ModelPackage,
  onSaveT3TestEvaluation,
  onToggleT3Enabled,
}: ModulePromptStudioProps) {
  const [studioTab, setStudioTab] = useState<'gated' | 't3'>('gated');
  const [idea, setIdea] = useState<string>('espada de gelo');
  const [category, setCategory] = useState<ItemCategory>('Equipamentos');
  const [candidateCount, setCandidateCount] = useState<number>(8);
  const [temperature, setTemperature] = useState<number>(0.45);
  const [backdrop, setBackdrop] = useState<ThumbBackdrop>('transparent');
  const [statusFilter, setStatusFilter] = useState<'all' | 'pending' | 'approved' | 'rejected'>(
    'all',
  );

  // Server Gemini health state (checked honestly via GET /api/health)
  const [aiReady, setAiReady] = useState<boolean | null>(null);
  const [serverModels, setServerModels] = useState<{ primary: string; fallback: string } | null>(
    null,
  );

  // Step 1: Active engineered prompt state + diff against previous version
  const [activePromptRecord, setActivePromptRecord] = useState<PromptHistoryRecord | null>(null);
  const [promptDiff, setPromptDiff] = useState<PromptDiffChange[]>([]);
  const [isEngineering, setIsEngineering] = useState(false);

  // Step 2: Generation state + latest batch stats + expandable lower-ranked candidates
  const [isGenerating, setIsGenerating] = useState(false);
  const [lastBatchSummary, setLastBatchSummary] = useState<{
    topCount: number;
    secondaryCount: number;
    passedGateCount: number;
    totalEvaluated: number;
    topScore: number | null;
    categoryValRealMedian: number | null;
    categoryValHardPassRate: number | null;
    generatedHardPassRate: number;
    secondaryIds: Set<string>;
  } | null>(null);
  const [showSecondaryCandidates, setShowSecondaryCandidates] = useState(false);

  // Review loop: "Pedir ajuste" per candidate
  const [adjustingSpriteId, setAdjustingSpriteId] = useState<string | null>(null);
  const [adjustmentComment, setAdjustmentComment] = useState<string>('');
  const [isApplyingAdjustment, setIsApplyingAdjustment] = useState(false);

  // Export & notifications
  const [isExportingDataset, setIsExportingDataset] = useState(false);
  const [lastExportSummary, setLastExportSummary] = useState<Omit<
    TrainingDatasetExportResult,
    'zipBytes' | 'labelsCsv' | 'filePaths'
  > | null>(null);
  const [statusNotice, setStatusNotice] = useState<string | null>(null);
  const [fallbackWarning, setFallbackWarning] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [votingId, setVotingId] = useState<string | null>(null);

  // Check server Gemini availability honestly on mount
  useEffect(() => {
    let cancelled = false;
    fetch('/api/health')
      .then((r) => r.json())
      .then((data: { aiReady?: boolean; models?: { primary: string; fallback: string } }) => {
        if (cancelled) return;
        setAiReady(Boolean(data.aiReady));
        if (data.models) setServerModels(data.models);
      })
      .catch(() => {
        if (!cancelled) setAiReady(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Restore latest persisted prompt draft if none is active yet
  useEffect(() => {
    if (!activePromptRecord && promptHistory.length > 0) {
      setActivePromptRecord(promptHistory[0]);
      setIdea(promptHistory[0].idea);
      setCategory(promptHistory[0].engineeredPrompt.category);
    }
  }, [promptHistory, activePromptRecord]);

  const approvedSprites = useMemo(
    () => generatedSprites.filter((s) => s.approvalStatus === 'approved'),
    [generatedSprites],
  );

  const donorLookup = useMemo(() => {
    const map = new Map<number, IndexedSpriteEntry>();
    if (spriteIndex) {
      for (const e of spriteIndex.entries) {
        map.set(e.spriteId, e);
      }
    }
    return map;
  }, [spriteIndex]);

  const availableCategoryTrainDonors = useMemo(() => {
    if (!spriteIndex) return [];
    return spriteIndex.entries
      .filter((e) => e.split === 'train' && e.category === category && !e.isTile)
      .sort((a, b) => b.criticScore - a.criticScore)
      .slice(0, 12);
  }, [spriteIndex, category]);

  const activeRequestVersions = useMemo(() => {
    if (!activePromptRecord) return [];
    return promptHistory
      .filter((p) => p.requestId === activePromptRecord.requestId)
      .sort((a, b) => b.version - a.version);
  }, [promptHistory, activePromptRecord]);

  const activeCategoryLessons = useMemo(
    () =>
      lessons.filter(
        (l) => l.status === 'active' && (l.category === category || l.category === 'Global'),
      ).length,
    [lessons, category],
  );

  const activeChampion = championsByCategory[category] ?? null;

  const filteredSprites = useMemo(() => {
    if (statusFilter === 'all') return generatedSprites;
    return generatedSprites.filter((s) => s.approvalStatus === statusFilter);
  }, [generatedSprites, statusFilter]);

  const { primaryDisplaySprites, secondaryDisplaySprites } = useMemo(() => {
    const rankScore = (s: GeneratedSpriteRecord): number =>
      s.provenance.humannessGate?.humannessScore ?? s.evaluation.score;
    const rankCompare = (a: GeneratedSpriteRecord, b: GeneratedSpriteRecord): number => {
      const aPass = a.provenance.humannessGate ? (a.provenance.humannessGate.passed ? 1 : 0) : 1;
      const bPass = b.provenance.humannessGate ? (b.provenance.humannessGate.passed ? 1 : 0) : 1;
      if (bPass !== aPass) return bPass - aPass;
      const diff = rankScore(b) - rankScore(a);
      if (Math.abs(diff) > 1e-6) return diff;
      return b.evaluation.score - a.evaluation.score;
    };

    if (!lastBatchSummary || lastBatchSummary.secondaryIds.size === 0) {
      return {
        primaryDisplaySprites: filteredSprites.slice().sort(rankCompare),
        secondaryDisplaySprites: [] as GeneratedSpriteRecord[],
      };
    }
    const primary: GeneratedSpriteRecord[] = [];
    const secondary: GeneratedSpriteRecord[] = [];
    for (const s of filteredSprites) {
      if (lastBatchSummary.secondaryIds.has(s.id)) {
        secondary.push(s);
      } else {
        primary.push(s);
      }
    }
    primary.sort(rankCompare);
    secondary.sort(rankCompare);
    return { primaryDisplaySprites: primary, secondaryDisplaySprites: secondary };
  }, [filteredSprites, lastBatchSummary]);

  const headlineMetrics = useMemo(() => {
    const n = generatedSprites.length;
    if (n === 0) {
      return {
        count: 0,
        meanScore: null as number | null,
        meanOriginality: null as number | null,
        minOriginality: null as number | null,
        maxOriginality: null as number | null,
        repairedCount: 0,
        meanRepairDelta: 0,
        approvedCount: approvals.filter((a) => a.decision === 'approved').length,
        rejectedCount: approvals.filter((a) => a.decision === 'rejected').length,
      };
    }

    const scores = generatedSprites.map((s) => s.evaluation.score);
    const origs = generatedSprites.map((s) => s.nearestReal.originalityScore);
    const repaired = generatedSprites.filter((s) => s.provenance.repaired);

    const meanScore = scores.reduce((a, b) => a + b, 0) / n;
    const meanOriginality = origs.reduce((a, b) => a + b, 0) / n;
    const meanRepairDelta =
      repaired.length > 0
        ? repaired.reduce((a, b) => a + b.provenance.repairDelta, 0) / repaired.length
        : 0;

    return {
      count: n,
      meanScore,
      meanOriginality,
      minOriginality: Math.min(...origs),
      maxOriginality: Math.max(...origs),
      repairedCount: repaired.length,
      meanRepairDelta,
      approvedCount: approvals.filter((a) => a.decision === 'approved').length,
      rejectedCount: approvals.filter((a) => a.decision === 'rejected').length,
    };
  }, [generatedSprites, approvals]);

  if (!dataset || !activeSplit || !calibration || !diagnosticReport) {
    return (
      <section className="max-w-3xl mx-auto py-10">
        <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-6 space-y-4">
          <div className="flex items-start gap-3">
            <Database className="w-5 h-5 text-[#E5A93C] mt-0.5 shrink-0" aria-hidden="true" />
            <div className="space-y-1">
              <h1 className="text-lg font-semibold text-[#F3F5F8]">Estúdio Prompt</h1>
              <p className="text-sm text-[#98A2B3]">
                Carregue e calibre um pacote <code className="font-mono text-[#E6EBF3]">.dat</code>{' '}
                e <code className="font-mono text-[#E6EBF3]">.spr</code> para montar o dossiê de
                estilo (split train), engenheirar prompts com aprovação humana e gerar sprites 32×32
                fiéis ao Tibia.
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

  const ensureIndexReady = async (): Promise<SpriteIndexState | null> => {
    if (spriteIndex && spriteIndex.entries.length > 0) return spriteIndex;
    return onRunIndexing();
  };

  /**
   * STEP 1: Engineer Prompt (NEVER generates sprites — only creates/revises the prompt for user review).
   */
  const handleRunPromptEngineer = async (opts?: {
    forceManual?: boolean;
    regenerate?: boolean;
    revisionComment?: string;
    baseRecord?: PromptHistoryRecord;
  }) => {
    const trimmedIdea = idea.trim();
    if (!trimmedIdea || isEngineering || isIndexing) return;

    setIsEngineering(true);
    setErrorMsg(null);
    setFallbackWarning(null);
    setStatusNotice(null);

    try {
      const idx = await ensureIndexReady();
      if (!idx) throw new Error('Não foi possível indexar o dataset atual.');

      const targetCategory = opts?.baseRecord?.engineeredPrompt.category ?? category;
      const { dossier, donorEntries, trainPaletteRgb } = buildStyleDossier({
        datasetKey: activeSplit.datasetKey,
        category: targetCategory,
        idea: trimmedIdea,
        calibration,
        spriteIndex: idx,
        lessons,
        approvals,
        approvedSprites,
        fewShotCount: 8,
      });

      const baseRec = opts?.baseRecord ?? (opts?.regenerate ? activePromptRecord : null);
      const requestId = baseRec ? baseRec.requestId : `req-${Date.now().toString(36)}`;
      const existingForReq = promptHistory.filter((p) => p.requestId === requestId);
      const nextVersion =
        existingForReq.reduce((max, p) => Math.max(max, p.version), baseRec?.version ?? 0) + 1;

      let nextSpec: EngineeredPromptSpec;
      let source: 'gemini' | 'manual';
      let model: string | null = null;

      if (opts?.forceManual || aiReady === false) {
        if (opts?.revisionComment && baseRec) {
          nextSpec = applyLocalAdjustmentToPrompt({
            previous: baseRec.engineeredPrompt,
            comment: opts.revisionComment,
          });
        } else {
          nextSpec = buildManualPromptDraft({
            idea: trimmedIdea,
            category: targetCategory,
            dossier,
            trainPaletteRgb,
          });
        }
        source = 'manual';
        model = null;
        if (!opts?.forceManual) {
          setFallbackWarning(
            'Gemini indisponível (GEMINI_API_KEY não configurada no servidor). Rascunho estruturado criado em modo manual para você editar e aprovar antes de gerar.',
          );
        }
      } else {
        // Real server-side Gemini Prompt Engineer (strict JSON validation — NEVER silently falls back)
        const serverRes = await requestEngineeredPromptFromServer({
          idea: trimmedIdea,
          categoryHint: targetCategory,
          dossier,
          donorEntries,
          trainPaletteRgb,
          previousPrompt: baseRec?.engineeredPrompt ?? null,
          adjustmentComment: opts?.revisionComment ?? null,
          temperature,
        });
        nextSpec = serverRes.prompt;
        source = 'gemini';
        model = serverRes.model;
      }

      const newRecord: PromptHistoryRecord = {
        id: `${requestId}-v${nextVersion}-${Date.now().toString(36)}`,
        datasetKey: activeSplit.datasetKey,
        requestId,
        version: nextVersion,
        idea: trimmedIdea,
        parentVersionId: baseRec?.id ?? null,
        adjustmentComment: opts?.revisionComment?.trim() || null,
        source,
        model,
        engineeredPrompt: nextSpec,
        approved: false,
        approvedAt: null,
        createdAt: new Date().toISOString(),
        readBackVerified: false,
      };

      const saved = await onSavePromptVersion(newRecord);
      const diff = baseRec
        ? computePromptDiff(baseRec.engineeredPrompt, saved.engineeredPrompt)
        : [];
      setPromptDiff(diff);
      setActivePromptRecord(saved);
      setCategory(saved.engineeredPrompt.category);
      setStatusNotice(
        `Etapa 1 concluída (v${saved.version} · ${source === 'gemini' ? `Gemini ${model}` : 'manual'}). Revise o prompt estruturado e clique em "OK, gerar" para sintetizar os sprites.`,
      );
    } catch (err) {
      setErrorMsg(
        err instanceof Error
          ? err.message
          : 'Falha ao executar o Engenheiro de Prompt.',
      );
    } finally {
      setIsEngineering(false);
    }
  };

  /**
   * Saves manual user edits to the engineered prompt as a new version in IndexedDB.
   */
  const handleSaveEditedPrompt = async (edited: EngineeredPromptSpec) => {
    if (!activePromptRecord) return;
    const existingForReq = promptHistory.filter(
      (p) => p.requestId === activePromptRecord.requestId,
    );
    const nextVersion =
      existingForReq.reduce((max, p) => Math.max(max, p.version), activePromptRecord.version) + 1;

    const nextRecord: PromptHistoryRecord = {
      id: `${activePromptRecord.requestId}-v${nextVersion}-${Date.now().toString(36)}`,
      datasetKey: activeSplit.datasetKey,
      requestId: activePromptRecord.requestId,
      version: nextVersion,
      idea: activePromptRecord.idea,
      parentVersionId: activePromptRecord.id,
      adjustmentComment: 'Edição manual do usuário',
      source: 'manual',
      model: null,
      engineeredPrompt: edited,
      approved: false,
      approvedAt: null,
      createdAt: new Date().toISOString(),
      readBackVerified: false,
    };

    const saved = await onSavePromptVersion(nextRecord);
    setPromptDiff(computePromptDiff(activePromptRecord.engineeredPrompt, saved.engineeredPrompt));
    setActivePromptRecord(saved);
    setStatusNotice(`Edição salva como v${saved.version} no IndexedDB. Clique em "OK, gerar" quando estiver pronto.`);
  };

  /**
   * STEP 2: Executed STRICTLY ONLY after user clicks "OK, gerar" on the engineered prompt.
   */
  const handleApproveAndGenerate = async (specToApprove: EngineeredPromptSpec) => {
    if (!activePromptRecord || isGenerating || isIndexing) return;
    setIsGenerating(true);
    setErrorMsg(null);
    setFallbackWarning(null);
    setStatusNotice(null);

    try {
      const idx = await ensureIndexReady();
      if (!idx) throw new Error('Não foi possível indexar o dataset atual.');

      // 1. Mark prompt approved & persist read-back verified in IndexedDB 'promptHistory'
      const approvedRecord = await onSavePromptVersion({
        ...activePromptRecord,
        engineeredPrompt: specToApprove,
        approved: true,
        approvedAt: new Date().toISOString(),
      });
      setActivePromptRecord(approvedRecord);
      setPromptDiff([]);

      // 2. Run hybrid Step 2 pipeline (N candidates -> tibiaify -> Critic repair -> Humanness gate)
      const res: GenerateFromApprovedPromptResult = await generateFromApprovedPrompt({
        datasetKey: activeSplit.datasetKey,
        approvedPrompt: approvedRecord.engineeredPrompt,
        engineeredPromptId: approvedRecord.id,
        engineeredPromptVersion: approvedRecord.version,
        promptSource: approvedRecord.source,
        candidateCount,
        useGemini: Boolean(aiReady),
        temperature,
        maxRepairIterations: 3,
        seed: (activeSplit.seed + Date.now()) & 0x7fffffff,
        index: idx,
        calibration,
        lessons,
        approvals,
        approvedSprites,
        champion: championsByCategory[approvedRecord.engineeredPrompt.category] ?? activeChampion,
      });

      if (res.fallbackWarning) {
        setFallbackWarning(res.fallbackWarning);
      }

      await onSaveGeneratedBatch(res.allCandidates);

      const topScore = res.topCandidates[0]?.evaluation.score ?? null;
      setLastBatchSummary({
        topCount: res.topCandidates.length,
        secondaryCount: res.secondaryCandidates.length,
        passedGateCount: res.passedGateCount,
        totalEvaluated: res.totalEvaluated,
        topScore,
        categoryValRealMedian: res.categoryValidationRealMedian,
        categoryValHardPassRate: res.categoryValidationHardPassRate,
        generatedHardPassRate: res.generatedHardPassRate,
        secondaryIds: new Set(res.secondaryCandidates.map((c) => c.id)),
      });
      setShowSecondaryCandidates(false);

      const valCompareStr =
        res.categoryValidationRealMedian !== null && topScore !== null
          ? ` Top Crítico ${topScore.toFixed(1)} vs mediana real validation ${res.categoryValidationRealMedian.toFixed(1)} (regras rígidas: ${res.generatedHardPassRate.toFixed(0)}% limpos).`
          : '';
      setStatusNotice(
        `Etapa 2 concluída: ${res.totalEvaluated} candidatos 32×32 tibiaificados e ranqueados pelo Humanness Gate (${res.passedGateCount} aprovados no gate).${valCompareStr}`,
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Falha na geração da Etapa 2.');
    } finally {
      setIsGenerating(false);
    }
  };

  /**
   * REVIEW LOOP: "Pedir ajuste" — converts user comment into a `human_feedback` lesson AND
   * triggers a prompt-engineer revision showing the diff before user clicks "OK, gerar" again.
   */
  const handleSubmitAdjustmentRequest = async (sprite: GeneratedSpriteRecord) => {
    const comment = adjustmentComment.trim();
    if (!comment || isApplyingAdjustment) return;
    setIsApplyingAdjustment(true);
    setErrorMsg(null);

    try {
      // 1. Convert free-text comment to a real `human_feedback` lesson in IndexedDB
      const lessonDraft = parseCommentToLessonDraft(comment, sprite);
      await onCreateLessonFromComment(lessonDraft);

      // 2. Locate parent prompt record (or use current activePromptRecord)
      const parentRec =
        (sprite.provenance.engineeredPromptId
          ? promptHistory.find((p) => p.id === sprite.provenance.engineeredPromptId)
          : null) ?? activePromptRecord;

      if (parentRec) {
        await handleRunPromptEngineer({
          baseRecord: parentRec,
          revisionComment: comment,
        });
      } else {
        setStatusNotice(
          `Lição de feedback humano registrada ("${lessonDraft.title}"). Gere um prompt na Etapa 1 para aplicar.`,
        );
      }

      setAdjustingSpriteId(null);
      setAdjustmentComment('');
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Falha ao solicitar ajuste.');
    } finally {
      setIsApplyingAdjustment(false);
    }
  };

  const handleExportTrainingDataset = async () => {
    if (isExportingDataset || isIndexing) return;
    setIsExportingDataset(true);
    setErrorMsg(null);
    setStatusNotice(null);

    try {
      const idx = await ensureIndexReady();
      if (!idx) throw new Error('Indexe o dataset antes de exportar.');

      await new Promise((r) => window.setTimeout(r, 15));
      const result = buildTrainingDatasetExportZip({
        index: idx,
        split: activeSplit,
      });

      triggerBlobDownload(
        result.zipBytes,
        'application/zip',
        `atelier-dataset-treino-${new Date().toISOString().slice(0, 10)}.zip`,
      );

      setLastExportSummary({
        fileCount: result.fileCount,
        exportedSpriteCount: result.exportedSpriteCount,
        trainCount: result.trainCount,
        validationCount: result.validationCount,
        excludedTestCount: result.excludedTestCount,
        humanApprovedCount: result.humanApprovedCount,
        humanRejectedCount: result.humanRejectedCount,
        hasVocabularyJson: result.hasVocabularyJson,
        humanLabelsCsv: result.humanLabelsCsv,
      });
      setStatusNotice(
        `Dataset de treino exportado (.zip): ${result.exportedSpriteCount} PNGs 32×32 com alpha real (${result.trainCount} train + ${result.validationCount} val), grades indexadas, paletas, mapas de rampa e labels.csv. Split test excluído (${result.excludedTestCount} sprites).`,
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Falha ao exportar dataset de treino.');
    } finally {
      setIsExportingDataset(false);
    }
  };

  const handleVote = async (
    sprite: GeneratedSpriteRecord,
    decision: 'approved' | 'rejected',
  ) => {
    if (votingId) return;
    setVotingId(sprite.id);
    setErrorMsg(null);
    try {
      await onRecordApproval(sprite, decision);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Falha ao gravar aprovação.');
    } finally {
      setVotingId(null);
    }
  };

  const totalVotes = headlineMetrics.approvedCount + headlineMetrics.rejectedCount;
  const humanPrefActive =
    totalVotes >= 6 && headlineMetrics.approvedCount > 0 && headlineMetrics.rejectedCount > 0;

  const renderCandidateCard = (rec: GeneratedSpriteRecord, rankIndex: number) => {
    const gate = rec.provenance.humannessGate;
    // Collect 2-3 real donor sprites to show directly next to the candidate
    const donorThumbs: IndexedSpriteEntry[] = [];
    for (const did of rec.provenance.donorSpriteIds) {
      const entry = donorLookup.get(did);
      if (entry && !donorThumbs.some((d) => d.spriteId === entry.spriteId)) {
        donorThumbs.push(entry);
      }
    }
    if (rec.provenance.approvedPromptSnapshot) {
      for (const did of rec.provenance.approvedPromptSnapshot.donorSuggestions) {
        if (donorThumbs.length >= 3) break;
        const entry = donorLookup.get(did);
        if (entry && !donorThumbs.some((d) => d.spriteId === entry.spriteId)) {
          donorThumbs.push(entry);
        }
      }
    }
    const isAdjustOpen = adjustingSpriteId === rec.id;
    const hardIssues = rec.evaluation.issues.filter((i) => i.severity === 'hard');

    return (
      <article
        key={rec.id}
        data-testid="generated-sprite-card"
        className={`bg-[#12151C] border rounded-xl p-4 space-y-3.5 ${
          gate && !gate.passed ? 'border-[#5E2328]/80' : 'border-[#1E232D]'
        }`}
      >
        {/* Header & Actions */}
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 space-y-0.5">
            <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-mono text-[#8C97AB]">
              <span className="text-[#CBD3E1]">#{rankIndex + 1}</span>
              <span>·</span>
              <span className="text-[#E5A93C] font-semibold">
                {SOURCE_LABELS[rec.provenance.source] ?? rec.provenance.source}
              </span>
              {rec.provenance.model && (
                <>
                  <span>·</span>
                  <span className="text-[#93C5FD]">{rec.provenance.model}</span>
                </>
              )}
              {rec.provenance.engineeredPromptVersion && (
                <>
                  <span>·</span>
                  <span>Prompt v{rec.provenance.engineeredPromptVersion}</span>
                </>
              )}
              {rec.provenance.tibiaified && (
                <>
                  <span>·</span>
                  <span className="text-[#86EFAC]">tibiaify</span>
                </>
              )}
              {rec.readBackVerified && (
                <>
                  <span>·</span>
                  <span className="inline-flex items-center gap-0.5 text-[#86EFAC]">
                    <Check className="w-3 h-3" />
                    IDB ok
                  </span>
                </>
              )}
            </div>
            <h3 className="text-sm font-medium text-[#F3F5F8] truncate" title={rec.prompt}>
              {rec.prompt}
            </h3>
          </div>

          <div className="flex items-center gap-1 shrink-0">
            <button
              type="button"
              onClick={() => {
                const png = encodeRgbaToPngBytes(rec.rgba, 32, 32);
                triggerBlobDownload(png, 'image/png', `${rec.id}.png`);
              }}
              title="Baixar PNG 32×32 nativo com alpha real"
              className="p-1.5 rounded border border-[#262C38] text-[#98A2B3] hover:text-[#F3F5F8] hover:border-[#3D475B]"
            >
              <Download className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={() => void onDeleteGenerated(rec.id)}
              title="Remover sprite"
              className="p-1.5 rounded border border-[#262C38] text-[#98A2B3] hover:text-[#FCA5A5] hover:border-[#5E2328]"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Native 32x32 + Crisp Zoomed 3x + 2-3 Real Donors Comparison */}
        <div className="grid grid-cols-1 sm:grid-cols-[auto_1fr] gap-3 bg-[#0D0F13] border border-[#1B202A] rounded-lg p-3">
          {/* Candidate Native 1x + Zoomed 3x */}
          <div className="flex items-end gap-2.5">
            <div className="flex flex-col items-center gap-1">
              <SpriteThumb
                rgba={rec.rgba}
                scale={3}
                backdrop={backdrop}
                alt={`Candidato ampliado 3x: ${rec.prompt}`}
              />
              <span className="text-[10px] font-mono text-[#8C97AB]">3× (96px)</span>
            </div>
            <div className="flex flex-col items-center gap-1">
              <SpriteThumb
                rgba={rec.rgba}
                scale={1}
                backdrop={backdrop}
                alt={`Candidato nativo 32x32: ${rec.prompt}`}
              />
              <span className="text-[10px] font-mono text-[#8C97AB]">1:1 (32px)</span>
            </div>
          </div>

          {/* 2-3 Real Donors + Nearest Real for Side-by-Side Comparison */}
          <div className="sm:border-l sm:border-[#1E232D] sm:pl-3 space-y-2">
            <div className="text-[10px] font-mono uppercase tracking-wider text-[#8C97AB]">
              Comparação visual · Doadores reais ({donorThumbs.length > 0 ? donorThumbs.length : 1})
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {donorThumbs.length > 0 ? (
                donorThumbs.slice(0, 3).map((d) => (
                  <div key={d.spriteId} className="flex flex-col items-center gap-0.5">
                    <SpriteThumb
                      rgba={d.rgba}
                      scale={2}
                      backdrop={backdrop}
                      alt={`Doador real #${d.spriteId}`}
                    />
                    <span className="text-[10px] font-mono text-[#98A2B3]">
                      #{d.spriteId} · {d.criticScore.toFixed(0)}
                    </span>
                  </div>
                ))
              ) : (
                <div className="flex flex-col items-center gap-0.5">
                  <SpriteThumb
                    rgba={rec.nearestReal.rgba}
                    scale={2}
                    backdrop={backdrop}
                    alt={`Real mais próximo #${rec.nearestReal.spriteId}`}
                  />
                  <span className="text-[10px] font-mono text-[#98A2B3]">
                    Real #{rec.nearestReal.spriteId}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Critic v1 Honest Breakdown */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-[11px] font-mono bg-[#0F1218] border border-[#1A1F28] rounded-lg px-3 py-2 tabular-nums">
          <div>
            <span className="block text-[10px] text-[#8C97AB]">Crítico v1</span>
            <span className="text-sm font-semibold text-[#F3F5F8]">
              {rec.evaluation.calibrated ? rec.evaluation.score.toFixed(1) : 'não calibrado'}
            </span>
          </div>
          <div>
            <span className="block text-[10px] text-[#8C97AB]">Higiene / Estat.</span>
            <span className="text-[#CBD3E1]">
              {rec.evaluation.hygieneScore.toFixed(0)} / {rec.evaluation.statisticalScore.toFixed(0)}
            </span>
          </div>
          <div>
            <span className="block text-[10px] text-[#8C97AB]">P(real) Discrim.</span>
            <span className="text-[#93C5FD]">
              {rec.evaluation.discriminatorScore !== null
                ? `${Math.round((rec.evaluation.discriminatorScore / 98.5) * 100)}%`
                : 'não calibrado'}
            </span>
          </div>
          <div>
            <span className="block text-[10px] text-[#8C97AB]">Originalidade</span>
            <span className="text-[#60A5FA]">
              {rec.nearestReal.originalityScore.toFixed(1)}%
            </span>
          </div>
          <div>
            <span className="block text-[10px] text-[#8C97AB]">Paleta / Alpha</span>
            <span className={hardIssues.length === 0 ? 'text-[#86EFAC]' : 'text-[#FCA5A5]'}>
              {rec.paletteHex.length}c ·{' '}
              {rec.evaluation.features.semiTransparentPixels === 0 ? 'bin' : 'AA!'}
            </span>
          </div>
        </div>

        {/* Humanness Gate / Repair status */}
        <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-[#8C97AB]">
          <div className="flex flex-wrap items-center gap-2">
            {gate && (
              <span
                className={`px-2 py-0.5 rounded border ${
                  gate.passed
                    ? 'bg-[#111E17] border-[#1F4A31] text-[#86EFAC]'
                    : 'bg-[#231417] border-[#5E2328] text-[#FCA5A5]'
                }`}
              >
                {gate.passed
                  ? `Humanness Gate OK (${gate.humannessScore.toFixed(1)})`
                  : `Filtrado: ${gate.rejectReasonPt}`}
              </span>
            )}
            {rec.provenance.repaired && (
              <span className="inline-flex items-center gap-1 text-[#86EFAC]">
                <Wrench className="w-3 h-3" />+{rec.provenance.repairDelta.toFixed(1)} pts reparo
              </span>
            )}
          </div>
          <span>Real mais próximo #{rec.nearestReal.spriteId}</span>
        </div>

        {/* Review Loop Controls: Aprovar, Rejeitar, Pedir ajuste */}
        <div className="flex flex-wrap items-center justify-between gap-2 pt-1 border-t border-[#1A1F28]">
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              type="button"
              disabled={votingId === rec.id}
              onClick={() => void handleVote(rec, 'approved')}
              data-testid={`approve-sprite-${rec.id}`}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors ${
                rec.approvalStatus === 'approved'
                  ? 'bg-[#12261B] border-[#22C55E] text-[#86EFAC]'
                  : 'bg-[#151921] border-[#262C38] text-[#CBD3E1] hover:border-[#22C55E] hover:text-[#86EFAC]'
              }`}
            >
              <ThumbsUp className="w-3.5 h-3.5" />
              <span>{rec.approvalStatus === 'approved' ? 'Aprovado' : 'Aprovar'}</span>
            </button>

            <button
              type="button"
              disabled={votingId === rec.id}
              onClick={() => void handleVote(rec, 'rejected')}
              data-testid={`reject-sprite-${rec.id}`}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors ${
                rec.approvalStatus === 'rejected'
                  ? 'bg-[#271417] border-[#EF4444] text-[#FCA5A5]'
                  : 'bg-[#151921] border-[#262C38] text-[#CBD3E1] hover:border-[#EF4444] hover:text-[#FCA5A5]'
              }`}
            >
              <ThumbsDown className="w-3.5 h-3.5" />
              <span>{rec.approvalStatus === 'rejected' ? 'Rejeitado' : 'Rejeitar'}</span>
            </button>

            <button
              type="button"
              onClick={() => {
                setAdjustingSpriteId(isAdjustOpen ? null : rec.id);
                setAdjustmentComment('');
              }}
              data-testid={`request-adjustment-btn-${rankIndex}`}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors ${
                isAdjustOpen
                  ? 'bg-[#1B263B] border-[#60A5FA] text-[#93C5FD]'
                  : 'bg-[#151921] border-[#262C38] text-[#CBD3E1] hover:border-[#60A5FA] hover:text-[#93C5FD]'
              }`}
            >
              <MessageSquarePlus className="w-3.5 h-3.5" />
              <span>Pedir ajuste</span>
            </button>
          </div>
        </div>

        {/* Inline "Pedir ajuste" Form -> Prompt Revision Diff + Human Feedback Lesson */}
        {isAdjustOpen && (
          <div
            data-testid="adjustment-comment-panel"
            className="bg-[#0D131D] border border-[#284366] rounded-lg p-3 space-y-2"
          >
            <label className="block text-xs text-[#93C5FD] font-medium">
              Qual ajuste você quer no prompt? (ex.: &ldquo;contorno mais escuro&rdquo;, &ldquo;menos cores&rdquo;, &ldquo;mais brilho NW&rdquo;)
            </label>
            <div className="flex flex-col sm:flex-row gap-2">
              <input
                type="text"
                value={adjustmentComment}
                onChange={(e) => setAdjustmentComment(e.target.value)}
                placeholder="Ex: contorno mais escuro e menos cores"
                data-testid="adjustment-comment-input"
                className="flex-1 px-3 py-1.5 text-xs rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8]"
              />
              <button
                type="button"
                disabled={!adjustmentComment.trim() || isApplyingAdjustment}
                onClick={() => void handleSubmitAdjustmentRequest(rec)}
                data-testid="submit-adjustment-btn"
                className="inline-flex items-center justify-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold rounded-lg bg-[#3B82F6] text-white hover:bg-[#2563EB] disabled:opacity-50"
              >
                {isApplyingAdjustment ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Wand2 className="w-3.5 h-3.5" />
                )}
                <span>Revisar prompt + salvar lição</span>
              </button>
            </div>
            <p className="text-[11px] text-[#8C97AB]">
              Salva seu comentário como lição <code className="font-mono">human_feedback</code> e
              abre o diff do prompt revisado na Etapa 1 para você confirmar com &ldquo;OK, gerar&rdquo;.
            </p>
          </div>
        )}
      </article>
    );
  };

  return (
    <section className="space-y-6">
      {/* Header & Index/Export Bar */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 border-b border-[#1E232D] pb-4">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold tracking-tight text-[#F3F5F8]">
            Estúdio Prompt — Fluxo em 2 Etapas com Aprovação Humana
          </h1>
          <p className="text-sm text-[#98A2B3]">
            Ideia → <strong>Etapa 1 (Engenheiro de Prompt)</strong> → Revisão/Edição humana →{' '}
            <strong>Etapa 2 (Gerar + Tibiaify + Reparo + Humanness Gate)</strong> → Aprovar /
            Rejeitar / Pedir ajuste. Nenhum sprite é gerado antes de clicar em{' '}
            <code className="font-mono text-[#E5A93C]">OK, gerar</code>.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={() => void onRunIndexing()}
            disabled={isIndexing || isGenerating || isEngineering}
            data-testid="index-dataset-btn"
            className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium rounded-lg bg-[#151921] border border-[#29303E] text-[#E6EBF3] hover:border-[#3D475B] disabled:opacity-50 transition-colors"
          >
            {isIndexing ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin text-[#E5A93C]" />
            ) : (
              <Layers className="w-3.5 h-3.5 text-[#60A5FA]" />
            )}
            <span>
              {isIndexing && indexProgress
                ? `Indexando ${indexProgress.done}/${indexProgress.total}...`
                : spriteIndex
                  ? `Reindexar (${spriteIndex.trainCount} train · ${spriteIndex.validationCount} val)`
                  : 'Indexar dataset (train/val)'}
            </span>
          </button>

          <button
            type="button"
            onClick={() => void handleExportTrainingDataset()}
            disabled={isExportingDataset || isIndexing}
            data-testid="export-training-dataset-btn"
            className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium rounded-lg bg-[#151921] border border-[#2B3B52] text-[#93C5FD] hover:bg-[#192231] disabled:opacity-50 transition-colors"
          >
            {isExportingDataset ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Archive className="w-3.5 h-3.5" />
            )}
            <span>Exportar dataset de treino (.zip)</span>
          </button>
        </div>
      </div>

      {/* Sub-view Switcher: 2-Step Prompt Studio vs T3 Vocab/Colab/ONNX */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="inline-flex rounded-lg bg-[#12151C] border border-[#1E232D] p-1 gap-1">
          <button
            type="button"
            onClick={() => setStudioTab('gated')}
            data-testid="studio-subtab-t1t2"
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors ${
              studioTab === 'gated'
                ? 'bg-[#E5A93C] text-[#0D0F13] font-semibold'
                : 'text-[#98A2B3] hover:text-[#F3F5F8]'
            }`}
          >
            Engenheiro &amp; Gerador ({generatedSprites.length})
          </button>
          <button
            type="button"
            onClick={() => setStudioTab('t3')}
            data-testid="studio-subtab-t3"
            className={`inline-flex items-center gap-2 px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors ${
              studioTab === 't3'
                ? 'bg-[#E5A93C] text-[#0D0F13] font-semibold'
                : 'text-[#98A2B3] hover:text-[#F3F5F8]'
            }`}
          >
            <span>Modelo T3 (Vocab · Colab · ONNX)</span>
            <span
              className={`px-1.5 py-0.5 text-[10px] font-mono rounded ${
                studioTab === 't3'
                  ? 'bg-[#0D0F13]/20 text-[#0D0F13]'
                  : t3ModelPackage?.enabled
                    ? 'bg-[#12261B] text-[#86EFAC]'
                    : 'bg-[#1F2430] text-[#98A2B3]'
              }`}
            >
              {t3ModelPackage ? (t3ModelPackage.enabled ? 'ATIVO' : 'OFF') : 'SEM ONNX'}
            </span>
          </button>
        </div>

        {/* Backdrop toggle */}
        <div className="flex items-center gap-1.5 text-xs text-[#98A2B3]">
          <span>Fundo:</span>
          {(['transparent', 'dark', 'magenta'] as ThumbBackdrop[]).map((bg) => (
            <button
              key={bg}
              type="button"
              onClick={() => setBackdrop(bg)}
              className={`px-2 py-1 rounded border text-xs transition-colors ${
                backdrop === bg
                  ? 'bg-[#1D2432] border-[#E5A93C] text-[#F3F5F8]'
                  : 'bg-[#0D0F13] border-[#262C38] text-[#8C97AB] hover:text-[#E6EBF3]'
              }`}
            >
              {bg === 'transparent' ? 'Grade' : bg === 'dark' ? 'Escuro' : 'Magenta'}
            </button>
          ))}
        </div>
      </div>

      {studioTab === 't3' ? (
        <T3WorkbenchPanel
          dataset={dataset}
          activeSplit={activeSplit}
          calibration={calibration}
          spriteIndex={spriteIndex}
          lessons={lessons}
          generatedSprites={generatedSprites}
          approvals={approvals}
          vocabulary={t3Vocabulary}
          modelPackage={t3ModelPackage}
          backdrop={backdrop}
          onEnsureIndexReady={ensureIndexReady}
          onUpdateVocabRole={onUpdateT3VocabRole}
          onRebuildVocab={onRebuildT3Vocab}
          onImportModelPackage={onImportT3ModelPackage}
          onSaveTestEvaluation={onSaveT3TestEvaluation}
          onToggleT3Enabled={onToggleT3Enabled}
          onSaveGeneratedBatch={onSaveGeneratedBatch}
        />
      ) : (
        <>
          {/* Honest Server / Index / Active Agent Status Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 rounded-lg bg-[#12151C] border border-[#1E232D] text-xs font-mono text-[#98A2B3]">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
              {aiReady === false ? (
                <span data-testid="gemini-unavailable-badge" className="text-[#FCD34D]">
                  Gemini indisponível (sem GEMINI_API_KEY) · modo manual + local pixel-exato
                </span>
              ) : aiReady === true ? (
                <span className="text-[#86EFAC]">
                  Gemini pronto ({serverModels?.primary ?? 'servidor'})
                </span>
              ) : (
                <span>Verificando servidor Gemini...</span>
              )}
              {activeAgent && (
                <span
                  data-testid="studio-active-agent-badge"
                  className="inline-flex items-center gap-1 text-[#CBD3E1]"
                >
                  <Award className="w-3.5 h-3.5 text-[#E5A93C]" />
                  <span>{activeAgent.name}</span>
                </span>
              )}
              {spriteIndex && (
                <span>
                  Índice: <strong className="text-[#86EFAC]">{spriteIndex.trainCount}</strong> train ·{' '}
                  <strong className="text-[#93C5FD]">{spriteIndex.validationCount}</strong> val ·{' '}
                  <strong className="text-[#FBBF24]">{spriteIndex.excludedTestCount}</strong> test
                  excluídos
                </span>
              )}
            </div>
            {lastExportSummary && (
              <span className="text-[#86EFAC]" data-testid="export-summary-banner">
                ZIP: {lastExportSummary.exportedSpriteCount} sprites (0 test)
              </span>
            )}
          </div>

          {/* Alerts */}
          {errorMsg && (
            <div
              data-testid="studio-error-banner"
              className="flex items-start gap-2.5 px-4 py-3 rounded-lg bg-[#231417] border border-[#5E2328] text-sm text-[#FCA5A5]"
            >
              <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}
          {fallbackWarning && (
            <div
              data-testid="studio-fallback-warning"
              className="flex items-start gap-2.5 px-4 py-3 rounded-lg bg-[#221C11] border border-[#5B451B] text-xs text-[#FCD34D]"
            >
              <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
              <span>{fallbackWarning}</span>
            </div>
          )}
          {statusNotice && (
            <div
              data-testid="studio-status-notice"
              className="flex items-start gap-2.5 px-4 py-3 rounded-lg bg-[#111E17] border border-[#1F4A31] text-xs text-[#86EFAC]"
            >
              <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0" />
              <span>{statusNotice}</span>
            </div>
          )}

          {/* STEP 1 INPUT: Idea -> "Gerar prompt" (NEVER triggers generation directly) */}
          <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-5 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="text-xs font-mono uppercase tracking-wider text-[#E5A93C]">
                Etapa 1 · Ideia curta para o Engenheiro de Prompt
              </div>
              <div className="flex items-center gap-3 text-[11px] font-mono text-[#8C97AB]">
                <span>Lições ativas ({category}): {activeCategoryLessons}</span>
                <span>Histórico salvo: {promptHistory.length} versão(ões)</span>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-3">
              <label className="lg:col-span-5 space-y-1">
                <span className="block text-xs font-medium text-[#98A2B3]">
                  Ideia do sprite (ex.: &ldquo;espada de gelo&rdquo;, &ldquo;orc com machado&rdquo;)
                </span>
                <input
                  type="text"
                  value={idea}
                  onChange={(e) => setIdea(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && idea.trim()) {
                      e.preventDefault();
                      void handleRunPromptEngineer();
                    }
                  }}
                  placeholder="Ex: espada de gelo"
                  data-testid="studio-idea-input"
                  className="w-full px-3.5 py-2 text-sm rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8] placeholder:text-[#5B6476] focus:outline-none focus:border-[#E5A93C]"
                />
              </label>

              <label className="lg:col-span-3 space-y-1">
                <span className="block text-xs font-medium text-[#98A2B3]">Categoria</span>
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value as ItemCategory)}
                  data-testid="studio-category-select"
                  className="w-full px-3 py-2 text-sm rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8] focus:outline-none focus:border-[#E5A93C]"
                >
                  {ALL_ITEM_CATEGORIES.map((cat) => (
                    <option key={cat} value={cat}>
                      {cat} ({activeSplit.byCategory[cat]?.train.length ?? 0} train)
                    </option>
                  ))}
                </select>
              </label>

              <label className="lg:col-span-2 space-y-1">
                <span className="flex items-center justify-between text-xs font-medium text-[#98A2B3]">
                  <span>Candidatos</span>
                  <span className="font-mono text-[#E5A93C] tabular-nums">{candidateCount}</span>
                </span>
                <input
                  type="range"
                  min={4}
                  max={16}
                  step={1}
                  value={candidateCount}
                  onChange={(e) => setCandidateCount(Number(e.target.value))}
                  data-testid="studio-candidate-count-range"
                  className="w-full accent-[#E5A93C] mt-2"
                />
              </label>

              <label className="lg:col-span-2 space-y-1">
                <span className="flex items-center justify-between text-xs font-medium text-[#98A2B3]">
                  <span>Temp.</span>
                  <span className="font-mono text-[#93C5FD] tabular-nums">
                    {temperature.toFixed(2)}
                  </span>
                </span>
                <input
                  type="range"
                  min={0}
                  max={1.5}
                  step={0.05}
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  data-testid="studio-temperature-range"
                  className="w-full accent-[#60A5FA] mt-2"
                />
              </label>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => void handleRunPromptEngineer()}
                  disabled={isEngineering || isGenerating || isIndexing || !idea.trim()}
                  data-testid="engineer-prompt-btn"
                  className="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-lg bg-[#E5A93C] text-[#0D0F13] hover:bg-[#EEB955] disabled:opacity-50 transition-colors"
                >
                  {isEngineering ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Wand2 className="w-4 h-4" />
                  )}
                  <span>
                    {aiReady === false
                      ? 'Gerar prompt (rascunho manual)'
                      : 'Gerar prompt'}
                  </span>
                </button>

                {aiReady === true && (
                  <button
                    type="button"
                    onClick={() => void handleRunPromptEngineer({ forceManual: true })}
                    disabled={isEngineering || isGenerating || isIndexing || !idea.trim()}
                    data-testid="manual-prompt-draft-btn"
                    className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg bg-[#151921] border border-[#29303E] text-[#CBD3E1] hover:border-[#3D475B] disabled:opacity-50"
                  >
                    <span>Criar rascunho manual</span>
                  </button>
                )}
              </div>

              {promptHistory.length > 0 && (
                <label className="inline-flex items-center gap-2 text-xs text-[#98A2B3]">
                  <span>Histórico:</span>
                  <select
                    value={activePromptRecord?.id ?? ''}
                    onChange={(e) => {
                      const found = promptHistory.find((p) => p.id === e.target.value);
                      if (found) {
                        setActivePromptRecord(found);
                        setPromptDiff([]);
                        setIdea(found.idea);
                        setCategory(found.engineeredPrompt.category);
                      }
                    }}
                    className="px-2.5 py-1 text-xs font-mono rounded bg-[#0D0F13] border border-[#262C38] text-[#E6EBF3] max-w-[240px]"
                  >
                    {promptHistory.slice(0, 20).map((rec) => (
                      <option key={rec.id} value={rec.id}>
                        {rec.engineeredPrompt.itemName} · v{rec.version} ({rec.source})
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </div>
          </div>

          {/* STEP 1 REVIEW CARD (Editable Structured Prompt + Human Gate "OK, gerar") */}
          {activePromptRecord && (
            <PromptEngineerCard
              activeRecord={activePromptRecord}
              requestVersions={activeRequestVersions}
              diffChanges={promptDiff}
              donorLookup={donorLookup}
              availableDonors={availableCategoryTrainDonors}
              backdrop={backdrop}
              isEngineering={isEngineering}
              isGenerating={isGenerating}
              candidateCount={candidateCount}
              onSelectVersion={(rec) => {
                setActivePromptRecord(rec);
                setPromptDiff([]);
              }}
              onRegeneratePrompt={() => void handleRunPromptEngineer({ regenerate: true })}
              onSaveEditedPrompt={handleSaveEditedPrompt}
              onApproveAndGenerate={handleApproveAndGenerate}
            />
          )}

          {/* Latest Step 2 Validation Comparison Strip */}
          {lastBatchSummary && (
            <div
              data-testid="step2-validation-comparison"
              className="scroll-mt-20 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3"
            >
              <div className="bg-[#12151C] border border-[#1F4A31] rounded-xl p-4 space-y-1">
                <div className="text-xs text-[#8C97AB]">Top Candidato vs Validation Real</div>
                <div className="text-xl font-mono font-semibold tabular-nums text-[#86EFAC]">
                  {lastBatchSummary.topScore !== null ? lastBatchSummary.topScore.toFixed(1) : '—'}
                  <span className="text-xs text-[#98A2B3] font-normal">
                    {' '}
                    vs{' '}
                    {lastBatchSummary.categoryValRealMedian !== null
                      ? lastBatchSummary.categoryValRealMedian.toFixed(1)
                      : '—'}
                  </span>
                </div>
                <div className="text-[11px] font-mono text-[#8C97AB]">
                  Mediana real held-out (validation)
                </div>
              </div>

              <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-4 space-y-1">
                <div className="text-xs text-[#8C97AB]">Regras Rígidas (0 violações)</div>
                <div className="text-xl font-mono font-semibold tabular-nums text-[#F3F5F8]">
                  {lastBatchSummary.generatedHardPassRate.toFixed(0)}%
                  <span className="text-xs text-[#98A2B3] font-normal">
                    {' '}
                    vs{' '}
                    {lastBatchSummary.categoryValHardPassRate !== null
                      ? `${lastBatchSummary.categoryValHardPassRate.toFixed(0)}%`
                      : '—'}{' '}
                    real
                  </span>
                </div>
                <div className="text-[11px] font-mono text-[#8C97AB]">
                  Alpha binário · contorno 1px · paleta
                </div>
              </div>

              <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-4 space-y-1">
                <div className="text-xs text-[#8C97AB]">Humanness Gate</div>
                <div className="text-xl font-mono font-semibold tabular-nums text-[#60A5FA]">
                  {lastBatchSummary.passedGateCount}/{lastBatchSummary.totalEvaluated}
                </div>
                <div className="text-[11px] font-mono text-[#8C97AB]">
                  Crítico + Discriminador + Originalidade
                </div>
              </div>

              <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-4 space-y-1">
                <div className="text-xs text-[#8C97AB]">Curadoria Humana</div>
                <div className="text-xl font-mono font-semibold tabular-nums text-[#E5A93C]">
                  {headlineMetrics.approvedCount}A / {headlineMetrics.rejectedCount}R
                </div>
                <div className="text-[11px] font-mono text-[#8C97AB]">
                  {humanPrefActive ? 'Termo humano ativo' : `Mínimo 6 votos mistos (${totalVotes}/6)`}
                </div>
              </div>
            </div>
          )}

          {/* Gallery Header & Filters */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-semibold text-[#F3F5F8]">
                Etapa 2 · Candidatos 32×32 (1:1 + Zoom + Doadores Reais)
              </h2>
              <span className="text-xs font-mono text-[#8C97AB] tabular-nums">
                ({primaryDisplaySprites.length} principais
                {secondaryDisplaySprites.length > 0
                  ? ` · +${secondaryDisplaySprites.length} filtrados`
                  : ''}
                )
              </span>
            </div>

            <div className="flex items-center gap-1.5 text-xs">
              {(
                [
                  ['all', 'Todos'],
                  ['pending', 'Pendentes'],
                  ['approved', 'Aprovados'],
                  ['rejected', 'Rejeitados'],
                ] as const
              ).map(([val, label]) => (
                <button
                  key={val}
                  type="button"
                  onClick={() => setStatusFilter(val)}
                  className={`px-2.5 py-1 rounded border transition-colors ${
                    statusFilter === val
                      ? 'bg-[#1D2432] border-[#E5A93C] text-[#F3F5F8]'
                      : 'bg-[#12151C] border-[#1E232D] text-[#8C97AB] hover:text-[#E6EBF3]'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Candidate Grid */}
          {primaryDisplaySprites.length === 0 ? (
            <div
              data-testid="no-candidates-empty-state"
              className="bg-[#12151C] border border-[#1E232D] rounded-xl p-10 text-center space-y-2"
            >
              <Sparkles className="w-5 h-5 text-[#E5A93C] mx-auto" />
              <p className="text-sm text-[#CBD3E1]">
                Nenhum candidato gerado ainda nesta visualização.
              </p>
              <p className="text-xs text-[#8C97AB]">
                Digite uma ideia na Etapa 1, clique em <strong>Gerar prompt</strong>, revise/edite o
                brief e clique em <strong>OK, gerar</strong> para iniciar a síntese.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <div
                className="grid grid-cols-1 lg:grid-cols-2 gap-4"
                data-testid="generated-sprites-grid"
              >
                {primaryDisplaySprites.map((rec, idx) => renderCandidateCard(rec, idx))}
              </div>

              {secondaryDisplaySprites.length > 0 && (
                <div className="space-y-3">
                  <button
                    type="button"
                    onClick={() => setShowSecondaryCandidates((v) => !v)}
                    data-testid="toggle-secondary-candidates-btn"
                    className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium rounded-lg bg-[#151921] border border-[#29303E] text-[#CBD3E1] hover:border-[#E5A93C]"
                  >
                    {showSecondaryCandidates ? (
                      <ChevronUp className="w-3.5 h-3.5" />
                    ) : (
                      <ChevronDown className="w-3.5 h-3.5" />
                    )}
                    <span>
                      {showSecondaryCandidates
                        ? `Ocultar demais candidatos (${secondaryDisplaySprites.length})`
                        : `Ver outros candidatos ranqueados / filtrados (${secondaryDisplaySprites.length})`}
                    </span>
                  </button>

                  {showSecondaryCandidates && (
                    <div
                      className="grid grid-cols-1 lg:grid-cols-2 gap-4"
                      data-testid="secondary-sprites-grid"
                    >
                      {secondaryDisplaySprites.map((rec, idx) =>
                        renderCandidateCard(rec, primaryDisplaySprites.length + idx),
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </section>
  );
}
