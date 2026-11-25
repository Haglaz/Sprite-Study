import { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  Archive,
  Award,
  Check,
  CheckCircle2,
  Database,
  Download,
  Layers,
  Loader2,
  Play,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  Wrench,
} from 'lucide-react';
import type { AgentProfileRecord } from '../comparator/types';
import type { CriticCalibration, DiagnosticReport } from '../critic/types';
import type { StoredDatasetSplit } from '../db/schema';
import type { CoevolutionChampion } from '../evolution/types';
import {
  buildTrainingDatasetExportZip,
  encodeRgbaToPngBytes,
  type TrainingDatasetExportResult,
} from '../export/trainingDatasetExport';
import {
  BENCHMARK_10_PROMPTS,
  generateSpriteCandidates,
} from '../generation/pipeline';
import type {
  ApprovalLabelRecord,
  GeneratedSpriteRecord,
  GenerationTier,
  SpriteIndexState,
} from '../generation/types';
import type { LessonRecord } from '../lessons/types';
import { runT3SamplingAndRanking } from '../t3/sampler';
import type {
  T3MaterialRole,
  T3ModelPackageRecord,
  T3RampVocabulary,
  T3TestSplitComparisonReport,
} from '../t3/types';
import { createWorkerT3Client } from '../t3/workerClient';
import { ALL_ITEM_CATEGORIES, type ItemCategory, type LocalDatasetSummary } from '../types/studio';
import { SPRITE_SIZE } from '../utils/tibiaParser';
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
  t3Vocabulary: T3RampVocabulary | null;
  t3ModelPackage: T3ModelPackageRecord | null;
  onGoToUpload: () => void;
  onRunIndexing: () => Promise<SpriteIndexState | null>;
  onSaveGeneratedBatch: (records: GeneratedSpriteRecord[]) => Promise<void>;
  onRecordApproval: (
    sprite: GeneratedSpriteRecord,
    decision: 'approved' | 'rejected',
  ) => Promise<void>;
  onDeleteGenerated: (id: string) => Promise<void>;
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
  t3Vocabulary,
  t3ModelPackage,
  onGoToUpload,
  onRunIndexing,
  onSaveGeneratedBatch,
  onRecordApproval,
  onDeleteGenerated,
  onUpdateT3VocabRole,
  onRebuildT3Vocab,
  onImportT3ModelPackage,
  onSaveT3TestEvaluation,
  onToggleT3Enabled,
}: ModulePromptStudioProps) {
  const [studioTab, setStudioTab] = useState<'t1t2' | 't3'>('t1t2');
  const [category, setCategory] = useState<ItemCategory>('Equipamentos');
  const [tier, setTier] = useState<GenerationTier>('T1');
  const [engineMode, setEngineMode] = useState<'local' | 'gemini'>('local');
  const [maxDiffPixels, setMaxDiffPixels] = useState<number>(28);
  const [prompt, setPrompt] = useState<string>(
    'Armadura azul real com detalhes em ouro estilo paladino 7.72',
  );
  const [backdrop, setBackdrop] = useState<ThumbBackdrop>('transparent');
  const [statusFilter, setStatusFilter] = useState<'all' | 'pending' | 'approved' | 'rejected'>(
    'all',
  );

  const [isGenerating, setIsGenerating] = useState(false);
  const [batchProgress, setBatchProgress] = useState<{ done: number; total: number } | null>(null);
  const [isExportingDataset, setIsExportingDataset] = useState(false);
  const [lastExportSummary, setLastExportSummary] = useState<Omit<
    TrainingDatasetExportResult,
    'zipBytes' | 'labelsCsv' | 'filePaths'
  > | null>(null);
  const [statusNotice, setStatusNotice] = useState<string | null>(null);
  const [fallbackWarning, setFallbackWarning] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [votingId, setVotingId] = useState<string | null>(null);

  // Sync composer defaults whenever the user promotes a new active agent via "Tornar ativo"
  useEffect(() => {
    if (!activeAgent) return;
    setTier(activeAgent.tier);
    setMaxDiffPixels(activeAgent.maxDiffPixels);
  }, [activeAgent?.id]);

  // If T3 gets disabled after evaluation, fall back cleanly to T1
  useEffect(() => {
    if (tier === 'T3' && !t3ModelPackage?.enabled) {
      setTier('T1');
    }
  }, [tier, t3ModelPackage?.enabled]);

  const approvedSprites = useMemo(
    () => generatedSprites.filter((s) => s.approvalStatus === 'approved'),
    [generatedSprites],
  );

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

  const headlineMetrics = useMemo(() => {
    const n = generatedSprites.length;
    if (n === 0) {
      return {
        count: 0,
        meanScore: null as number | null,
        medianScore: null as number | null,
        meanOriginality: null as number | null,
        minOriginality: null as number | null,
        maxOriginality: null as number | null,
        repairedCount: 0,
        meanRepairDelta: 0,
        approvedCount: approvals.filter((a) => a.decision === 'approved').length,
        rejectedCount: approvals.filter((a) => a.decision === 'rejected').length,
      };
    }

    const scores = generatedSprites.map((s) => s.evaluation.score).sort((a, b) => a - b);
    const origs = generatedSprites.map((s) => s.nearestReal.originalityScore);
    const repaired = generatedSprites.filter((s) => s.provenance.repaired);

    const meanScore = scores.reduce((a, b) => a + b, 0) / n;
    const medianScore =
      n % 2 === 1 ? scores[(n - 1) / 2] : (scores[n / 2 - 1] + scores[n / 2]) / 2;
    const meanOriginality = origs.reduce((a, b) => a + b, 0) / n;
    const meanRepairDelta =
      repaired.length > 0
        ? repaired.reduce((a, b) => a + b.provenance.repairDelta, 0) / repaired.length
        : 0;

    return {
      count: n,
      meanScore,
      medianScore,
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
              <h1 className="text-lg font-semibold text-[#F3F5F8]">Estúdio Prompt (T1 / T2)</h1>
              <p className="text-sm text-[#98A2B3]">
                Carregue e calibre um pacote <code className="font-mono text-[#E6EBF3]">.dat</code>{' '}
                e <code className="font-mono text-[#E6EBF3]">.spr</code> para indexar sprites reais
                de treino, gerar variações pixel-exatas (Tier T1 e T2) e exportar o dataset de
                treino.
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

  const handleGenerateSingle = async () => {
    if (isGenerating || isIndexing || !prompt.trim()) return;
    setIsGenerating(true);
    setErrorMsg(null);
    setFallbackWarning(null);
    setStatusNotice(null);

    try {
      const idx = await ensureIndexReady();
      if (!idx) throw new Error('Não foi possível indexar o dataset atual.');

      if (tier === 'T3' && t3ModelPackage?.enabled) {
        const client = await createWorkerT3Client(t3ModelPackage);
        try {
          const t3Batch = await runT3SamplingAndRanking({
            runner: client.runner,
            modelCard: t3ModelPackage.modelCard,
            category,
            prompt,
            calibration,
            spriteIndex: idx,
            candidateCount: 4,
            seed: activeSplit.seed + (Date.now() % 100000),
          });
          const records: GeneratedSpriteRecord[] = t3Batch.candidates
            .filter((c) => c.nearestReal !== null)
            .map((c) => {
              const seen = new Set<string>();
              const paletteHex: string[] = [];
              for (let p = 0; p < SPRITE_SIZE * SPRITE_SIZE; p++) {
                const off = p * 4;
                if (c.rgba[off + 3] < 128) continue;
                const hex =
                  '#' +
                  [c.rgba[off], c.rgba[off + 1], c.rgba[off + 2]]
                    .map((v) => v.toString(16).padStart(2, '0'))
                    .join('')
                    .toUpperCase();
                if (!seen.has(hex)) {
                  seen.add(hex);
                  paletteHex.push(hex);
                }
              }
              return {
                id: `t3-${Date.now()}-${c.candidateIndex}`,
                datasetKey: activeSplit.datasetKey,
                prompt,
                category,
                createdAt: new Date().toISOString(),
                rgba: c.rgba,
                paletteHex,
                evaluation: c.evaluation,
                nearestReal: c.nearestReal!,
                provenance: {
                  source: 'onnx-t3',
                  tier: 'T3',
                  model: t3ModelPackage.modelCard.version,
                  fallbackWarning: c.memorizedCopyWarning
                    ? 'Aviso de memorização: cópia muito próxima do split train.'
                    : null,
                  donorSpriteIds: [c.nearestReal!.spriteId],
                  diffPixelCount: Math.round(c.nearestReal!.pixelDiffRatio * 1024),
                  diffPixelLimit: 1024,
                  repaired: false,
                  repairDelta: 0,
                  repairMethod: 'none',
                  appliedLessonIds: [],
                  seed: activeSplit.seed + c.candidateIndex,
                },
                approvalStatus: 'pending',
                readBackVerified: true,
              };
            });
          await onSaveGeneratedBatch(records);
          setStatusNotice(
            `${records.length} candidatos T3 gerados em Web Worker (${t3Batch.executionProvider.toUpperCase()}) e salvos no IndexedDB.`,
          );
          return;
        } finally {
          await client.dispose();
        }
      }

      const effectiveTier: 'T1' | 'T2' = tier === 'T2' ? 'T2' : 'T1';
      const res = await generateSpriteCandidates({
        datasetKey: activeSplit.datasetKey,
        prompt,
        category,
        tier: effectiveTier,
        useGemini: engineMode === 'gemini',
        maxDiffPixels,
        enableRepair: activeAgent ? activeAgent.repairPolicy !== 'none' : true,
        candidateCount: 8,
        topKReturn: 4,
        index: idx,
        calibration,
        lessons,
        approvals,
        approvedSprites,
        champion: activeChampion,
      });

      if (res.fallbackWarning) {
        setFallbackWarning(res.fallbackWarning);
      }

      await onSaveGeneratedBatch(res.candidates);
      setStatusNotice(
        `${res.candidates.length} melhores candidatos (de ${res.totalEvaluated} avaliados pelo Crítico v1) salvos e verificados (read-back) no IndexedDB.`,
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Falha ao gerar candidatos.');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleGenerateBenchmark10 = async () => {
    if (isGenerating || isIndexing) return;
    setIsGenerating(true);
    setErrorMsg(null);
    setFallbackWarning(null);
    setStatusNotice(null);
    setBatchProgress({ done: 0, total: BENCHMARK_10_PROMPTS.length });

    try {
      const idx = await ensureIndexReady();
      if (!idx) throw new Error('Não foi possível indexar o dataset atual.');

      const winners: GeneratedSpriteRecord[] = [];
      let lastWarn: string | null = null;

      for (let i = 0; i < BENCHMARK_10_PROMPTS.length; i++) {
        const spec = BENCHMARK_10_PROMPTS[i];
        // Yield a frame so progress bar paints smoothly
        await new Promise((r) => window.setTimeout(r, 8));

        const res = await generateSpriteCandidates({
          datasetKey: activeSplit.datasetKey,
          prompt: spec.prompt,
          category: spec.category,
          tier: spec.tier,
          useGemini: engineMode === 'gemini',
          maxDiffPixels: spec.maxDiffPixels,
          candidateCount: 6,
          topKReturn: 1,
          seed: 77200 + i * 131,
          index: idx,
          calibration,
          lessons,
          approvals,
          approvedSprites,
          champion: championsByCategory[spec.category] ?? null,
        });

        if (res.fallbackWarning) lastWarn = res.fallbackWarning;
        if (res.candidates[0]) winners.push(res.candidates[0]);
        setBatchProgress({ done: i + 1, total: BENCHMARK_10_PROMPTS.length });
      }

      if (lastWarn) setFallbackWarning(lastWarn);
      await onSaveGeneratedBatch(winners);
      setStatusNotice(
        `Lote benchmark concluído: ${winners.length} sprites gerados (mix T1/T2), ranqueados pelo Crítico v1 e verificados (read-back) no IndexedDB.`,
      );
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Falha ao executar lote benchmark.');
    } finally {
      setIsGenerating(false);
      setBatchProgress(null);
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

  return (
    <section className="space-y-6">
      {/* Header & Index/Export Bar */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 border-b border-[#1E232D] pb-4">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold tracking-tight text-[#F3F5F8]">
            Estúdio Prompt — Geração Pixel-Exata (T1 / T2 / Modelo T3)
          </h1>
          <p className="text-sm text-[#98A2B3]">
            Recuperação exclusiva do split <code className="font-mono text-[#CBD3E1]">train</code>,
            recolor com ordem de luminância preservada, diff limitado em pixels, modelo indexado
            T3 (Colab + ONNX Web Worker) e comparação direta com o sprite real mais próximo.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={() => void onRunIndexing()}
            disabled={isIndexing || isGenerating}
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

      {/* Sub-view Switcher: T1/T2 Studio vs T3 Vocab/Colab/ONNX */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="inline-flex rounded-lg bg-[#12151C] border border-[#1E232D] p-1 gap-1">
          <button
            type="button"
            onClick={() => setStudioTab('t1t2')}
            data-testid="studio-subtab-t1t2"
            className={`px-3.5 py-1.5 text-xs font-medium rounded-md transition-colors ${
              studioTab === 't1t2'
                ? 'bg-[#E5A93C] text-[#0D0F13] font-semibold'
                : 'text-[#98A2B3] hover:text-[#F3F5F8]'
            }`}
          >
            Gerador T1 / T2 ({generatedSprites.length})
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

      {/* Index / Active Agent / Export status strip */}
      {(spriteIndex || lastExportSummary || activeAgent) && (
        <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 rounded-lg bg-[#12151C] border border-[#1E232D] text-xs font-mono text-[#98A2B3]">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
            {activeAgent && (
              <span
                data-testid="studio-active-agent-badge"
                className="inline-flex items-center gap-1.5 text-[#86EFAC]"
              >
                <Award className="w-3.5 h-3.5" />
                <span>
                  Agente ativo: <strong>{activeAgent.name}</strong> (Elo {activeAgent.elo.toFixed(0)})
                </span>
              </span>
            )}
            {spriteIndex ? (
              <>
                <span className="text-[#E6EBF3]">
                  Índice: <strong className="text-[#86EFAC]">{spriteIndex.trainCount}</strong> train
                </span>
                <span>
                  <strong className="text-[#93C5FD]">{spriteIndex.validationCount}</strong> val
                </span>
                <span>
                  <strong className="text-[#FBBF24]">{spriteIndex.excludedTestCount}</strong> test
                  excluídos
                </span>
              </>
            ) : (
              <span>Índice será construído automaticamente ao gerar ou exportar.</span>
            )}
            {approvedSprites.length > 0 && (
              <span className="text-[#60A5FA]">
                Pool few-shot: +{approvedSprites.length} aprovado(s)
              </span>
            )}
          </div>
          {lastExportSummary && (
            <span className="text-[#86EFAC]" data-testid="export-summary-banner">
              ZIP gerado: {lastExportSummary.exportedSpriteCount} sprites ({lastExportSummary.fileCount}{' '}
              arquivos · 0 test)
            </span>
          )}
        </div>
      )}

      {/* Alerts */}
      {errorMsg && (
        <div className="flex items-start gap-2.5 px-4 py-3 rounded-lg bg-[#231417] border border-[#5E2328] text-sm text-[#FCA5A5]">
          <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}
      {fallbackWarning && (
        <div className="flex items-start gap-2.5 px-4 py-3 rounded-lg bg-[#221C11] border border-[#5B451B] text-xs text-[#FCD34D]">
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

      {/* Composer Panel */}
      <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-5 space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <label className="space-y-1.5">
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

          <label className="space-y-1.5">
            <span className="block text-xs font-medium text-[#98A2B3]">Tier de geração</span>
            <select
              value={tier}
              onChange={(e) => setTier(e.target.value as GenerationTier)}
              data-testid="studio-tier-select"
              className="w-full px-3 py-2 text-sm rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8] focus:outline-none focus:border-[#E5A93C]"
            >
              <option value="T1">Tier T1 · Recolor semântico + Diff</option>
              <option value="T2">Tier T2 · Compósito de doadores + Diff</option>
              <option value="T3" disabled={!t3ModelPackage?.enabled}>
                {t3ModelPackage?.enabled
                  ? 'Tier T3 · Modelo ONNX (Web Worker)'
                  : 'Tier T3 · Desativado até passar no split test'}
              </option>
            </select>
          </label>

          <label className="space-y-1.5">
            <span className="block text-xs font-medium text-[#98A2B3]">Motor (sem fallback oculto)</span>
            <select
              value={engineMode}
              onChange={(e) => setEngineMode(e.target.value as 'local' | 'gemini')}
              data-testid="studio-engine-select"
              className="w-full px-3 py-2 text-sm rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8] focus:outline-none focus:border-[#E5A93C]"
            >
              <option value="local">Local Pixel-Exato (determinístico)</option>
              <option value="gemini">Gemini API + Executor Pixel-Exato</option>
            </select>
          </label>

          <label className="space-y-1.5">
            <span className="flex items-center justify-between text-xs font-medium text-[#98A2B3]">
              <span>Limite de diff</span>
              <span className="font-mono text-[#E5A93C] tabular-nums">{maxDiffPixels} px</span>
            </span>
            <input
              type="range"
              min={8}
              max={64}
              step={4}
              value={maxDiffPixels}
              onChange={(e) => setMaxDiffPixels(Number(e.target.value))}
              className="w-full accent-[#E5A93C] mt-2"
            />
          </label>
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-xs text-[#98A2B3]">
            <label htmlFor="studio-prompt-input" className="font-medium">
              Descrição do item (materiais, paleta e detalhes 32×32)
            </label>
            <div className="flex items-center gap-3 font-mono text-[11px]">
              <span>Lições ativas: {activeCategoryLessons}</span>
              <span>
                Campeão Fase 3:{' '}
                {activeChampion ? `Gen ${activeChampion.generation}` : 'nenhum (usa ops do Crítico)'}
              </span>
            </div>
          </div>
          <textarea
            id="studio-prompt-input"
            rows={2}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Ex: Armadura azul real com detalhes em ouro estilo paladino 7.72"
            className="w-full px-3.5 py-2.5 text-sm rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8] placeholder:text-[#5B6476] focus:outline-none focus:border-[#E5A93C]"
          />
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => void handleGenerateSingle()}
              disabled={isGenerating || isIndexing || !prompt.trim()}
              data-testid="generate-candidates-btn"
              className="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-lg bg-[#E5A93C] text-[#0D0F13] hover:bg-[#EEB955] disabled:opacity-50 transition-colors"
            >
              {isGenerating && !batchProgress ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4" />
              )}
              <span>Gerar candidatos ({tier})</span>
            </button>

            <button
              type="button"
              onClick={() => void handleGenerateBenchmark10()}
              disabled={isGenerating || isIndexing}
              data-testid="generate-benchmark-10-btn"
              className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg bg-[#1A202B] border border-[#323B4D] text-[#F3F5F8] hover:border-[#E5A93C] disabled:opacity-50 transition-colors"
            >
              {batchProgress ? (
                <Loader2 className="w-4 h-4 animate-spin text-[#E5A93C]" />
              ) : (
                <Play className="w-4 h-4 text-[#E5A93C]" />
              )}
              <span>
                {batchProgress
                  ? `Gerando lote ${batchProgress.done}/${batchProgress.total}...`
                  : 'Gerar lote (10 sprites)'}
              </span>
            </button>
          </div>

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
      </div>

      {/* 4-Column Headline Metrics Strip (strictly runtime-computed, with Real Ceiling & Junk Floor) */}
      <div
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3"
        data-testid="headline-metrics-strip"
      >
        <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-4 space-y-1">
          <div className="text-xs text-[#8C97AB]">Nota Média Crítico v1</div>
          <div className="text-2xl font-mono font-semibold tabular-nums text-[#F3F5F8]">
            {headlineMetrics.meanScore !== null ? headlineMetrics.meanScore.toFixed(1) : '—'}
          </div>
          <div className="text-[11px] font-mono text-[#8C97AB] tabular-nums">
            Teto real {diagnosticReport.realCeiling.median.toFixed(1)} · Piso lixo{' '}
            {diagnosticReport.junkFloor.median.toFixed(1)}
          </div>
        </div>

        <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-4 space-y-1">
          <div className="text-xs text-[#8C97AB]">Originalidade Média</div>
          <div className="text-2xl font-mono font-semibold tabular-nums text-[#60A5FA]">
            {headlineMetrics.meanOriginality !== null
              ? `${headlineMetrics.meanOriginality.toFixed(1)}%`
              : '—'}
          </div>
          <div className="text-[11px] font-mono text-[#8C97AB] tabular-nums">
            {headlineMetrics.minOriginality !== null && headlineMetrics.maxOriginality !== null
              ? `Faixa ${headlineMetrics.minOriginality.toFixed(1)}%–${headlineMetrics.maxOriginality.toFixed(1)}% vs .spr`
              : 'Distância ao sprite real mais próximo'}
          </div>
        </div>

        <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-4 space-y-1">
          <div className="text-xs text-[#8C97AB]">Reparo Guiado pelo Crítico</div>
          <div className="text-2xl font-mono font-semibold tabular-nums text-[#22C55E]">
            {headlineMetrics.count > 0
              ? `${headlineMetrics.repairedCount}/${headlineMetrics.count}`
              : '—'}
          </div>
          <div className="text-[11px] font-mono text-[#8C97AB] tabular-nums">
            {headlineMetrics.repairedCount > 0
              ? `Ganho médio +${headlineMetrics.meanRepairDelta.toFixed(1)} pts`
              : 'Aplicado apenas se melhorar sem violação'}
          </div>
        </div>

        <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-4 space-y-1">
          <div className="text-xs text-[#8C97AB]">Curadoria Humana (Crítico)</div>
          <div className="text-2xl font-mono font-semibold tabular-nums text-[#E5A93C]">
            {headlineMetrics.approvedCount}A / {headlineMetrics.rejectedCount}R
          </div>
          <div className="text-[11px] font-mono text-[#8C97AB] tabular-nums">
            {humanPrefActive
              ? 'Termo humano ativo no Crítico v1'
              : `Mínimo 6 votos mistos (${totalVotes}/6)`}
          </div>
        </div>
      </div>

      {/* Gallery Filter Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-[#F3F5F8]">
            Sprites Gerados vs Sprite Real Mais Próximo
          </h2>
          <span className="text-xs font-mono text-[#8C97AB] tabular-nums">
            ({filteredSprites.length} de {generatedSprites.length})
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

      {/* Side-by-side Generated vs Nearest Real Grid */}
      {filteredSprites.length === 0 ? (
        <div className="bg-[#12151C] border border-[#1E232D] rounded-xl p-10 text-center space-y-2">
          <p className="text-sm text-[#CBD3E1]">
            Nenhum sprite gerado ainda nesta visualização.
          </p>
          <p className="text-xs text-[#8C97AB]">
            Clique em <strong>Gerar candidatos</strong> ou <strong>Gerar lote (10 sprites)</strong>{' '}
            para ver cada sprite gerado lado a lado com o sprite real mais próximo do seu .spr.
          </p>
        </div>
      ) : (
        <div
          className="grid grid-cols-1 lg:grid-cols-2 gap-4"
          data-testid="generated-sprites-grid"
        >
          {filteredSprites.map((rec) => {
            const deltaVsNearest = rec.evaluation.score - rec.nearestReal.criticScore;
            return (
              <article
                key={rec.id}
                data-testid="generated-sprite-card"
                className="bg-[#12151C] border border-[#1E232D] rounded-xl p-4 space-y-3.5"
              >
                {/* Prompt & Category Header */}
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 space-y-0.5">
                    <div className="flex items-center gap-2 text-[11px] font-mono text-[#8C97AB]">
                      <span className="text-[#E5A93C] font-semibold">
                        {SOURCE_LABELS[rec.provenance.source] ?? rec.provenance.source}
                      </span>
                      <span>·</span>
                      <span>{rec.category}</span>
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

                {/* Side-by-side Sprites: Generated vs Nearest Real .spr */}
                <div className="grid grid-cols-2 gap-3 bg-[#0D0F13] border border-[#1B202A] rounded-lg p-3">
                  <div className="flex items-center gap-3">
                    <SpriteThumb
                      rgba={rec.rgba}
                      scale={3}
                      backdrop={backdrop}
                      alt={`Gerado: ${rec.prompt}`}
                    />
                    <div className="min-w-0 space-y-1">
                      <div className="text-[11px] uppercase tracking-wider text-[#8C97AB]">
                        Gerado ({rec.provenance.tier})
                      </div>
                      <div className="text-lg font-mono font-semibold tabular-nums text-[#F3F5F8]">
                        {rec.evaluation.score.toFixed(1)}
                      </div>
                      <div className="text-[11px] font-mono text-[#8C97AB]">
                        {rec.paletteHex.length} cores ·{' '}
                        {rec.evaluation.features.semiTransparentPixels === 0 ? 'alpha bin' : 'AA!'}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 border-l border-[#1E232D] pl-3">
                    <SpriteThumb
                      rgba={rec.nearestReal.rgba}
                      scale={3}
                      backdrop={backdrop}
                      alt={`Real mais próximo #${rec.nearestReal.spriteId}`}
                    />
                    <div className="min-w-0 space-y-1">
                      <div className="text-[11px] uppercase tracking-wider text-[#8C97AB]">
                        Real #{rec.nearestReal.spriteId} ({rec.nearestReal.split})
                      </div>
                      <div className="text-lg font-mono font-semibold tabular-nums text-[#CBD3E1]">
                        {rec.nearestReal.criticScore.toFixed(1)}
                      </div>
                      <div className="text-[11px] font-mono text-[#8C97AB] tabular-nums">
                        Δ {deltaVsNearest >= 0 ? '+' : ''}
                        {deltaVsNearest.toFixed(1)} · IoU{' '}
                        {Math.round(rec.nearestReal.silhouetteIoU * 100)}%
                      </div>
                    </div>
                  </div>
                </div>

                {/* Originality + Provenance Metadata */}
                <div className="grid grid-cols-3 gap-2 text-[11px] font-mono bg-[#0F1218] border border-[#1A1F28] rounded-lg px-3 py-2 tabular-nums">
                  <div>
                    <span className="block text-[#6F7A8F]">Originalidade</span>
                    <span className="text-[#60A5FA] font-semibold">
                      {rec.nearestReal.originalityScore.toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span className="block text-[#6F7A8F]">Doadores / Diff</span>
                    <span className="text-[#E6EBF3]">
                      #{rec.provenance.donorSpriteIds.join('+')} · {rec.provenance.diffPixelCount}/
                      {rec.provenance.diffPixelLimit}px
                    </span>
                  </div>
                  <div>
                    <span className="block text-[#6F7A8F]">Reparo Crítico</span>
                    <span className={rec.provenance.repaired ? 'text-[#22C55E]' : 'text-[#8C97AB]'}>
                      {rec.provenance.repaired ? (
                        <span className="inline-flex items-center gap-1">
                          <Wrench className="w-3 h-3" />+{rec.provenance.repairDelta.toFixed(1)}
                        </span>
                      ) : (
                        'inalterado'
                      )}
                    </span>
                  </div>
                </div>

                {/* Palette Swatches + Top Issues */}
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex flex-wrap items-center gap-1" title="Paleta exata do sprite">
                    {rec.paletteHex.slice(0, 16).map((hex) => (
                      <span
                        key={hex}
                        title={hex}
                        className="w-3 h-3 rounded-[2px] border border-[#2C3340]"
                        style={{ backgroundColor: hex }}
                      />
                    ))}
                  </div>
                  {rec.evaluation.issues.length > 0 ? (
                    <span className="text-[11px] text-[#FBBF24] truncate max-w-[60%]">
                      {rec.evaluation.issues[0].labelPt}
                      {rec.evaluation.issues.length > 1
                        ? ` (+${rec.evaluation.issues.length - 1})`
                        : ''}
                    </span>
                  ) : (
                    <span className="text-[11px] text-[#86EFAC]">0 defeitos críticos</span>
                  )}
                </div>

                {/* Approve / Reject Human Curation Controls */}
                <div className="flex items-center justify-between gap-2 pt-2 border-t border-[#1E232D]">
                  <span className="text-[11px] text-[#8C97AB]">
                    {rec.approvalStatus === 'approved'
                      ? 'Aprovado para treino do Crítico'
                      : rec.approvalStatus === 'rejected'
                        ? 'Rejeitado (exemplo negativo no Crítico)'
                        : 'Avalie para alimentar o Crítico v1'}
                  </span>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      disabled={votingId === rec.id}
                      onClick={() => void handleVote(rec, 'approved')}
                      data-testid={`approve-btn-${rec.id}`}
                      className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors ${
                        rec.approvalStatus === 'approved'
                          ? 'bg-[#12261B] border-[#22C55E] text-[#86EFAC]'
                          : 'bg-[#151921] border-[#262C38] text-[#CBD3E1] hover:border-[#22C55E] hover:text-[#86EFAC]'
                      }`}
                    >
                      <ThumbsUp className="w-3.5 h-3.5" />
                      <span>Aprovar</span>
                    </button>
                    <button
                      type="button"
                      disabled={votingId === rec.id}
                      onClick={() => void handleVote(rec, 'rejected')}
                      data-testid={`reject-btn-${rec.id}`}
                      className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors ${
                        rec.approvalStatus === 'rejected'
                          ? 'bg-[#281418] border-[#EF4444] text-[#FCA5A5]'
                          : 'bg-[#151921] border-[#262C38] text-[#CBD3E1] hover:border-[#EF4444] hover:text-[#FCA5A5]'
                      }`}
                    >
                      <ThumbsDown className="w-3.5 h-3.5" />
                      <span>Rejeitar</span>
                    </button>
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      )}
        </>
      )}
    </section>
  );
}
