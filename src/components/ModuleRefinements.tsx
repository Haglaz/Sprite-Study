import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowRight,
  BookPlus,
  CheckCircle2,
  Cpu,
  Layers,
  Pause,
  Play,
  RotateCcw,
  ShieldAlert,
  Sparkles,
} from 'lucide-react';
import type { CriticCalibration, DiagnosticReport } from '../critic/types';
import { buildDatasetKey, defaultDbBackend } from '../db/persistence';
import type { StoredDatasetSplit } from '../db/schema';
import { createDefaultPromotionRules } from '../evolution/coevolution';
import { getNeatComplexitySummary } from '../evolution/neat';
import { PIXEL_OP_LABELS_PT } from '../evolution/pixelOps';
import { formatRecipeSummaryPt } from '../evolution/recipeGa';
import {
  buildRestorationSamples,
  CORRUPTION_LABELS_PT,
  evaluateEditorPairOnBatch,
} from '../evolution/restorationTask';
import {
  deleteEvolutionCheckpoint,
  loadEvolutionCheckpoint,
  saveEvolutionCheckpointVerified,
} from '../evolution/store';
import type { TrainingWorkerEvent } from '../evolution/trainingWorker';
import type {
  CoevolutionRunConfig,
  EvolutionCheckpoint,
  RestorationTripletPreview,
} from '../evolution/types';
import {
  createEvolutionWorkerClient,
  extractSplitSpritesForCategory,
  type EvolutionWorkerHandle,
} from '../evolution/workerClient';
import { ALL_ITEM_CATEGORIES, type ItemCategory, type LocalDatasetSummary } from '../types/studio';
import { SpriteThumb } from './SpriteThumb';
import { TrainingCurvesChart } from './TrainingCurvesChart';

export interface CriticLoopLessonPayload {
  category: ItemCategory;
  title: string;
  directive: string;
  validationCheck: string;
  criticDeltaMean: number;
  sampleSize: number;
}

interface ModuleRefinementsProps {
  dataset: LocalDatasetSummary | null;
  activeSplit: StoredDatasetSplit | null;
  calibration: CriticCalibration | null;
  diagnosticReport: DiagnosticReport | null;
  onGoToUpload: () => void;
  onRegisterChampionLesson?: (payload: CriticLoopLessonPayload) => Promise<void>;
}

const DEFAULT_TRAIN_BATCH = 12;
const DEFAULT_VAL_BATCH = 12;

export function ModuleRefinements({
  dataset,
  activeSplit,
  calibration,
  diagnosticReport,
  onGoToUpload,
  onRegisterChampionLesson,
}: ModuleRefinementsProps) {
  const [selectedCategory, setSelectedCategory] = useState<ItemCategory>('Equipamentos');
  const [targetGenerations, setTargetGenerations] = useState<number>(6);
  const [populationSize, setPopulationSize] = useState<number>(10);
  const [seed, setSeed] = useState<number>(772);

  const [checkpoint, setCheckpoint] = useState<EvolutionCheckpoint | null>(null);
  const [previews, setPreviews] = useState<RestorationTripletPreview[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isPausing, setIsPausing] = useState(false);
  const [isLoadingCheckpoint, setIsLoadingCheckpoint] = useState(false);
  const [confirmResetOpen, setConfirmResetOpen] = useState(false);
  const [statusNotice, setStatusNotice] = useState<string | null>(null);
  const [errorNotice, setErrorNotice] = useState<string | null>(null);
  const [isRecordingLesson, setIsRecordingLesson] = useState(false);

  const workerRef = useRef<EvolutionWorkerHandle | null>(null);
  const persistChainRef = useRef<Promise<void>>(Promise.resolve());

  const datasetKey = useMemo(
    () => (dataset ? buildDatasetKey(dataset.datSha256, dataset.sprSha256) : null),
    [dataset],
  );

  // Terminate worker cleanly on unmount
  useEffect(() => {
    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, []);

  // Recompute held-out validation triplet previews whenever a saved checkpoint is restored
  const rebuildPreviewsForCheckpoint = useCallback(
    (ckpt: EvolutionCheckpoint | null) => {
      if (!ckpt || !dataset || !activeSplit || !calibration) {
        setPreviews([]);
        return;
      }
      const { valSprites } = extractSplitSpritesForCategory({
        dataset,
        split: activeSplit,
        category: ckpt.category,
        maxTrain: ckpt.config.trainBatchSize,
        maxVal: ckpt.config.valBatchSize,
      });
      if (valSprites.length === 0) {
        setPreviews([]);
        return;
      }
      const valSamples = buildRestorationSamples({
        sprites: valSprites,
        category: ckpt.category,
        calibration,
        seed: ckpt.config.seed + 503,
        maxCount: ckpt.config.valBatchSize,
      });
      const recipe = ckpt.champion?.recipe ?? ckpt.gaState.bestGenome ?? ckpt.gaState.genomes[0];
      const neat = ckpt.champion?.neat ?? ckpt.neatState.bestGenome ?? ckpt.neatState.genomes[0] ?? null;
      if (!recipe) {
        setPreviews([]);
        return;
      }
      const res = evaluateEditorPairOnBatch({
        samples: valSamples,
        recipe,
        neat,
        calibration,
        collectPreviews: true,
      });
      setPreviews(res.previews);
    },
    [dataset, activeSplit, calibration],
  );

  // Auto-load persisted checkpoint when dataset or selected category changes
  useEffect(() => {
    let cancelled = false;
    if (!datasetKey) {
      setCheckpoint(null);
      setPreviews([]);
      return;
    }

    async function loadSaved() {
      setIsLoadingCheckpoint(true);
      setErrorNotice(null);
      setConfirmResetOpen(false);
      try {
        const saved = await loadEvolutionCheckpoint(defaultDbBackend, datasetKey!, selectedCategory);
        if (cancelled) return;
        setCheckpoint(saved);
        if (saved) {
          setTargetGenerations(saved.config.targetGenerations);
          setPopulationSize(saved.config.populationSize);
          setSeed(saved.config.seed);
          setStatusNotice(
            `Checkpoint restaurado do IndexedDB (Gen ${saved.generation}/${saved.config.targetGenerations}, verificado por read-back).`,
          );
          rebuildPreviewsForCheckpoint(saved);
        } else {
          setStatusNotice(null);
          setPreviews([]);
        }
      } catch (err) {
        if (!cancelled) {
          setErrorNotice(
            err instanceof Error ? err.message : 'Falha ao carregar checkpoint salvo.',
          );
        }
      } finally {
        if (!cancelled) setIsLoadingCheckpoint(false);
      }
    }

    void loadSaved();
    return () => {
      cancelled = true;
    };
  }, [datasetKey, selectedCategory, rebuildPreviewsForCheckpoint]);

  const queueVerifiedCheckpointSave = useCallback(
    (nextCkpt: EvolutionCheckpoint, label: string) => {
      persistChainRef.current = persistChainRef.current
        .then(async () => {
          const verified = await saveEvolutionCheckpointVerified(defaultDbBackend, nextCkpt);
          setCheckpoint(verified);
          setStatusNotice(label);
        })
        .catch((err) => {
          setErrorNotice(
            err instanceof Error
              ? `Falha ao verificar checkpoint (read-back): ${err.message}`
              : 'Falha ao persistir checkpoint no IndexedDB.',
          );
        });
    },
    [],
  );

  const handleStartOrResume = useCallback(
    (additionalGenerations = 0) => {
      if (!dataset || !activeSplit || !calibration || !diagnosticReport || !datasetKey || isRunning) {
        return;
      }

      const { trainSprites, valSprites } = extractSplitSpritesForCategory({
        dataset,
        split: activeSplit,
        category: selectedCategory,
        maxTrain: DEFAULT_TRAIN_BATCH,
        maxVal: DEFAULT_VAL_BATCH,
      });

      if (trainSprites.length === 0 || valSprites.length === 0) {
        setErrorNotice(
          `Categoria "${selectedCategory}" não possui sprites suficientes no split treino/validação (${trainSprites.length} treino, ${valSprites.length} validação).`,
        );
        return;
      }

      setErrorNotice(null);
      setConfirmResetOpen(false);
      setIsRunning(true);
      setIsPausing(false);

      const resumeFrom = checkpoint && checkpoint.category === selectedCategory ? checkpoint : null;
      const effectiveTarget = resumeFrom
        ? Math.max(targetGenerations, resumeFrom.generation + Math.max(1, additionalGenerations || 4))
        : targetGenerations;

      if (effectiveTarget !== targetGenerations) {
        setTargetGenerations(effectiveTarget);
      }

      const runConfig: CoevolutionRunConfig = {
        category: selectedCategory,
        targetGenerations: effectiveTarget,
        populationSize,
        seed,
        trainBatchSize: Math.min(DEFAULT_TRAIN_BATCH, trainSprites.length),
        valBatchSize: Math.min(DEFAULT_VAL_BATCH, valSprites.length),
        promotionRules: createDefaultPromotionRules(),
      };

      workerRef.current?.terminate();

      const handleEvent = (ev: TrainingWorkerEvent) => {
        if (ev.type === 'started') {
          setStatusNotice(
            ev.resumed
              ? `Web Worker retomado da Gen ${ev.generation} → ${ev.targetGenerations} (${ev.trainSampleCount} treino / ${ev.valSampleCount} validação held-out).`
              : `Web Worker iniciado: Gen 1 → ${ev.targetGenerations} (${ev.trainSampleCount} treino / ${ev.valSampleCount} validação held-out).`,
          );
          return;
        }

        if (ev.type === 'generation') {
          setCheckpoint(ev.checkpoint);
          setPreviews(ev.previews);
          queueVerifiedCheckpointSave(
            ev.checkpoint,
            `Gen ${ev.point.generation}/${ev.checkpoint.config.targetGenerations} salva e verificada (read-back) · Treino ${ev.point.trainBestScore.toFixed(1)} / Val ${ev.point.valBestScore.toFixed(1)} (${ev.point.promoted ? 'Promovido' : 'Rejeitado'}).`,
          );
          return;
        }

        if (ev.type === 'paused') {
          setIsRunning(false);
          setIsPausing(false);
          workerRef.current?.terminate();
          workerRef.current = null;
          queueVerifiedCheckpointSave(
            ev.checkpoint,
            `Treino pausado na Gen ${ev.checkpoint.generation}. Checkpoint salvo e verificado (read-back) no IndexedDB.`,
          );
          return;
        }

        if (ev.type === 'completed') {
          setIsRunning(false);
          setIsPausing(false);
          if (ev.previews.length > 0) setPreviews(ev.previews);
          workerRef.current?.terminate();
          workerRef.current = null;
          queueVerifiedCheckpointSave(
            ev.checkpoint,
            `Treino concluído (Gen ${ev.checkpoint.generation}/${ev.checkpoint.config.targetGenerations}). Checkpoint e elite salvos e verificados (read-back) no IndexedDB.`,
          );
          return;
        }

        if (ev.type === 'error') {
          setIsRunning(false);
          setIsPausing(false);
          setErrorNotice(ev.message);
          workerRef.current?.terminate();
          workerRef.current = null;
        }
      };

      const worker = createEvolutionWorkerClient(handleEvent);
      workerRef.current = worker;

      worker.start({
        type: 'start',
        datasetKey,
        config: runConfig,
        calibration,
        realCeilingMedian: diagnosticReport.realCeiling.median,
        junkFloorMedian: diagnosticReport.junkFloor.median,
        trainSprites,
        valSprites,
        resumeCheckpoint: resumeFrom,
      });
    },
    [
      dataset,
      activeSplit,
      calibration,
      diagnosticReport,
      datasetKey,
      isRunning,
      selectedCategory,
      checkpoint,
      targetGenerations,
      populationSize,
      seed,
      queueVerifiedCheckpointSave,
    ],
  );

  const handlePause = useCallback(() => {
    if (!isRunning || !workerRef.current) return;
    setIsPausing(true);
    setStatusNotice('Pausando ao concluir a geração atual e salvando checkpoint...');
    workerRef.current.pause();
  }, [isRunning]);

  const handleConfirmReset = useCallback(async () => {
    if (!datasetKey || isRunning) return;
    try {
      await deleteEvolutionCheckpoint(defaultDbBackend, datasetKey, selectedCategory);
      setCheckpoint(null);
      setPreviews([]);
      setConfirmResetOpen(false);
      setStatusNotice(`Checkpoint de ${selectedCategory} removido do IndexedDB.`);
      setErrorNotice(null);
    } catch (err) {
      setErrorNotice(
        err instanceof Error ? err.message : 'Falha ao remover checkpoint.',
      );
    }
  }, [datasetKey, isRunning, selectedCategory]);

  const handlePromoteChampionToLesson = useCallback(async () => {
    if (!checkpoint?.champion || !onRegisterChampionLesson || isRecordingLesson) return;
    setIsRecordingLesson(true);
    setErrorNotice(null);
    try {
      const champ = checkpoint.champion;
      const recipeText = formatRecipeSummaryPt(champ.recipe);
      const neatComp = getNeatComplexitySummary(champ.neat);
      await onRegisterChampionLesson({
        category: checkpoint.category,
        title: `Pipeline co-evolutivo Gen ${champ.generation} para ${checkpoint.category}`,
        directive: `Aplicar sequência discreta (${recipeText}) seguida por refinamento NEAT (${neatComp.totalNodes} nós, ${neatComp.enabledConnections} conexões) preservando silhueta (IoU ${(champ.valStats.meanSilhouetteIoU * 100).toFixed(1)}%).`,
        validationCheck: `Validação held-out ${champ.valStats.medianRestoredScore.toFixed(1)} (+${champ.valStats.medianDelta.toFixed(1)} sobre corrompido) sem novas violações rígidas`,
        criticDeltaMean: champ.valStats.meanDelta,
        sampleSize: checkpoint.valSampleCount,
      });
      setStatusNotice(
        `Lição do campeão Gen ${champ.generation} registrada em "Lições Aprendidas" e verificada (read-back).`,
      );
    } catch (err) {
      setErrorNotice(
        err instanceof Error ? err.message : 'Falha ao registrar lição do campeão.',
      );
    } finally {
      setIsRecordingLesson(false);
    }
  }, [checkpoint, onRegisterChampionLesson, isRecordingLesson]);

  // Empty gate when no .dat/.spr dataset is calibrated yet
  if (!dataset || !activeSplit || !calibration || !diagnosticReport) {
    return (
      <section
        aria-labelledby="training-empty-heading"
        className="rounded-xl border border-[#1E232D] bg-[#12151C] p-6 space-y-4"
      >
        <div className="flex items-start gap-3">
          <Cpu className="w-5 h-5 text-[#E5A93C] mt-0.5 shrink-0" aria-hidden="true" />
          <div className="space-y-1">
            <h1 id="training-empty-heading" className="text-lg font-semibold text-[#F3F5F8]">
              Treino e Refinos
            </h1>
            <p className="text-xs text-[#8C97AB] max-w-2xl">
              Co-evolução real em Web Workers (NEAT com especiação + Algoritmo Genético de receitas de edição pixel-a-pixel), avaliada exclusivamente pelo Crítico v1 sobre a tarefa de restauração em splits separados de treino e validação.
            </p>
          </div>
        </div>

        <div className="rounded-lg border border-[#2B3344] bg-[#161B26] p-4 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-xs text-[#CBD5E1]">
            <Layers className="w-4 h-4 text-[#E5A93C]" aria-hidden="true" />
            <span>Crítico v1 não calibrado. Carregue primeiro um pacote .dat/.spr real.</span>
          </div>
          <button
            type="button"
            onClick={onGoToUpload}
            className="px-4 py-2 text-xs font-semibold rounded-lg bg-[#E5A93C] text-[#111318] hover:bg-[#EEB952] transition-colors whitespace-nowrap shrink-0"
          >
            Ir para Pacote .dat/.spr
          </button>
        </div>
      </section>
    );
  }

  const catSplit = activeSplit.byCategory[selectedCategory];
  const latestPoint =
    checkpoint && checkpoint.history.length > 0
      ? checkpoint.history[checkpoint.history.length - 1]
      : null;
  const champion = checkpoint?.champion ?? null;
  const promotedCount = checkpoint
    ? checkpoint.promotionLog.filter((p) => p.decision === 'promoted').length
    : 0;
  const neatComp = champion ? getNeatComplexitySummary(champion.neat) : null;
  const overfitGap = latestPoint
    ? Math.round((latestPoint.trainBestScore - latestPoint.valBestScore) * 10) / 10
    : null;

  const isCompleted = checkpoint?.status === 'completed';
  const canResume = Boolean(checkpoint && checkpoint.generation > 0 && !isCompleted);

  return (
    <div className="space-y-6">
      {/* Section 1: Header, Category Selector & Worker Controls */}
      <section
        aria-labelledby="training-heading"
        className="rounded-xl border border-[#1E232D] bg-[#12151C] p-5 space-y-5"
      >
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="space-y-1 max-w-2xl">
            <h1 id="training-heading" className="text-lg font-semibold text-[#F3F5F8]">
              Treino e Refinos — Co-Evolução NEAT + AG
            </h1>
            <p className="text-xs text-[#8C97AB]">
              Evolução real fora da thread principal (Web Worker) na tarefa de restauração pixel-a-pixel. Otimiza exclusivamente no split <span className="font-mono text-[#CBD5E1]">treino</span> e promove campeões apenas quando vencem na <span className="font-mono text-[#93C5FD]">validação held-out</span> sem overfitting nem destruição de silhueta.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 shrink-0">
            {isRunning ? (
              <button
                type="button"
                onClick={handlePause}
                disabled={isPausing}
                className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg bg-[#2A2215] border border-[#5A431D] text-[#FDE68A] hover:bg-[#3A2E1B] disabled:opacity-60 transition-colors whitespace-nowrap shrink-0"
              >
                <Pause className="w-3.5 h-3.5" aria-hidden="true" />
                <span>{isPausing ? 'Pausando...' : 'Pausar e salvar'}</span>
              </button>
            ) : (
              <>
                <button
                  type="button"
                  onClick={() => handleStartOrResume(isCompleted ? 4 : 0)}
                  disabled={isLoadingCheckpoint}
                  data-testid="start-training-btn"
                  className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg bg-[#E5A93C] text-[#111318] hover:bg-[#EEB952] disabled:opacity-50 transition-colors whitespace-nowrap shrink-0"
                >
                  <Play className="w-3.5 h-3.5" aria-hidden="true" />
                  <span>
                    {canResume
                      ? `Retomar (Gen ${checkpoint!.generation} → ${Math.max(targetGenerations, checkpoint!.generation + 1)})`
                      : isCompleted
                        ? `Continuar +4 gerações (Gen ${checkpoint!.generation})`
                        : 'Iniciar treino'}
                  </span>
                </button>

                {checkpoint && !confirmResetOpen && (
                  <button
                    type="button"
                    onClick={() => setConfirmResetOpen(true)}
                    className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg border border-[#2A3140] text-[#A6B0C3] hover:text-[#F3F5F8] hover:border-[#3D475B] transition-colors whitespace-nowrap shrink-0"
                  >
                    <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
                    <span>Reiniciar</span>
                  </button>
                )}

                {confirmResetOpen && (
                  <div className="inline-flex items-center gap-2 rounded-lg border border-[#5E2328] bg-[#241316] px-3 py-1.5 text-xs">
                    <span className="text-[#FCA5A5]">Apagar checkpoint?</span>
                    <button
                      type="button"
                      onClick={handleConfirmReset}
                      className="font-semibold text-[#FECACA] hover:underline"
                    >
                      Confirmar
                    </button>
                    <button
                      type="button"
                      onClick={() => setConfirmResetOpen(false)}
                      className="text-[#94A3B8] hover:text-[#F1F5F9]"
                    >
                      Cancelar
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        </div>

        {/* Category segmented selector + Compact numeric parameters */}
        <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-[#1D2330]">
          <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Categoria de treino">
            {ALL_ITEM_CATEGORIES.map((cat) => {
              const splitInfo = activeSplit.byCategory[cat];
              const active = selectedCategory === cat;
              return (
                <button
                  key={cat}
                  type="button"
                  disabled={isRunning}
                  onClick={() => setSelectedCategory(cat)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap shrink-0 disabled:opacity-50 ${
                    active
                      ? 'bg-[#202736] text-[#F3F5F8] border border-[#3B465E]'
                      : 'text-[#8C97AB] hover:text-[#E2E8F0] border border-transparent'
                  }`}
                >
                  <span>{cat}</span>
                  <span className="ml-1.5 font-mono tabular-nums text-[11px] text-[#7D889D]">
                    {splitInfo?.train.length ?? 0}/{splitInfo?.validation.length ?? 0}
                  </span>
                </button>
              );
            })}
          </div>

          <div className="flex flex-wrap items-center gap-4 text-xs text-[#94A3B8]">
            <label className="inline-flex items-center gap-2">
              <span>Gerações</span>
              <select
                value={targetGenerations}
                disabled={isRunning}
                onChange={(e) => setTargetGenerations(Number(e.target.value))}
                className="rounded-md bg-[#0E1117] border border-[#262D3B] px-2 py-1 font-mono tabular-nums text-xs text-[#F1F5F9]"
              >
                {[4, 6, 8, 12, 16, 24].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>

            <label className="inline-flex items-center gap-2">
              <span>População</span>
              <select
                value={populationSize}
                disabled={isRunning}
                onChange={(e) => setPopulationSize(Number(e.target.value))}
                className="rounded-md bg-[#0E1117] border border-[#262D3B] px-2 py-1 font-mono tabular-nums text-xs text-[#F1F5F9]"
              >
                {[8, 10, 12, 16, 20].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>

            <label className="inline-flex items-center gap-2">
              <span>Seed</span>
              <input
                type="number"
                value={seed}
                disabled={isRunning}
                onChange={(e) => setSeed(Math.max(1, Number(e.target.value) || 772))}
                className="w-20 rounded-md bg-[#0E1117] border border-[#262D3B] px-2 py-1 font-mono tabular-nums text-xs text-[#F1F5F9]"
              />
            </label>
          </div>
        </div>

        {/* Read-back verified status / error feedback */}
        {(statusNotice || errorNotice) && (
          <div className="space-y-2">
            {statusNotice && (
              <div
                data-testid="training-status-notice"
                className="flex items-center gap-2 text-xs text-[#A7F3D0] bg-[#101D17] border border-[#1E402E] rounded-lg px-3.5 py-2"
              >
                <CheckCircle2 className="w-3.5 h-3.5 text-[#22C55E] shrink-0" aria-hidden="true" />
                <span>{statusNotice}</span>
              </div>
            )}
            {errorNotice && (
              <div className="flex items-center gap-2 text-xs text-[#FCA5A5] bg-[#241316] border border-[#5E2328] rounded-lg px-3.5 py-2">
                <ShieldAlert className="w-3.5 h-3.5 text-[#F87171] shrink-0" aria-hidden="true" />
                <span>{errorNotice}</span>
              </div>
            )}
          </div>
        )}
      </section>

      {/* Section 2: 4-Column Honest Metrics Strip (Single Elevation, Hairline Dividers) */}
      <section
        aria-label="Métricas honestas da co-evolução"
        className="rounded-xl border border-[#1E232D] bg-[#12151C] grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-[#1E232D]"
      >
        <div className="p-4 space-y-1">
          <div className="text-xs text-[#7D889D]">Campeão na Validação (Held-Out)</div>
          <div className="font-mono tabular-nums text-xl font-semibold text-[#93C5FD]">
            {champion ? champion.valStats.meanRestoredScore.toFixed(1) : '—'}
          </div>
          <div className="text-[11px] font-mono tabular-nums text-[#8C97AB]">
            {champion
              ? `Corrompido ${champion.valStats.meanCorruptedScore.toFixed(1)} · Δ +${champion.valStats.meanDelta.toFixed(1)} · IoU ${(champion.valStats.meanSilhouetteIoU * 100).toFixed(1)}%`
              : `Split ${selectedCategory}: ${catSplit?.train.length ?? 0} treino / ${catSplit?.validation.length ?? 0} val`}
          </div>
        </div>

        <div className="p-4 space-y-1">
          <div className="text-xs text-[#7D889D]">Treino vs Validação (Gap)</div>
          <div className="font-mono tabular-nums text-xl font-semibold text-[#F3F5F8]">
            {latestPoint
              ? `${latestPoint.trainBestScore.toFixed(1)} / ${latestPoint.valBestScore.toFixed(1)}`
              : '—'}
          </div>
          <div className="text-[11px] font-mono tabular-nums text-[#8C97AB]">
            {overfitGap !== null
              ? `Gap ${overfitGap >= 0 ? `+${overfitGap.toFixed(1)}` : overfitGap.toFixed(1)} pts (limite ≤ 18.0)`
              : 'Sem gerações avaliadas ainda'}
          </div>
        </div>

        <div className="p-4 space-y-1">
          <div className="text-xs text-[#7D889D]">Baselines Crítico v1 (Mesma Métrica)</div>
          <div className="font-mono tabular-nums text-xl font-semibold text-[#86EFAC]">
            {diagnosticReport.realCeiling.median.toFixed(1)}{' '}
            <span className="text-xs font-normal text-[#7D889D]">vs</span>{' '}
            <span className="text-[#FCA5A5]">{diagnosticReport.junkFloor.median.toFixed(1)}</span>
          </div>
          <div className="text-[11px] font-mono tabular-nums text-[#8C97AB]">
            Teto real (.spr) vs Piso lixo · AUC {diagnosticReport.discriminatorValidationAuc.toFixed(3)}
          </div>
        </div>

        <div className="p-4 space-y-1">
          <div className="text-xs text-[#7D889D]">Co-Evolução NEAT + AG</div>
          <div className="font-mono tabular-nums text-xl font-semibold text-[#E5A93C]">
            {checkpoint ? `Gen ${checkpoint.generation}/${checkpoint.config.targetGenerations}` : 'Pronto'}
          </div>
          <div className="text-[11px] font-mono tabular-nums text-[#8C97AB]">
            {checkpoint
              ? `${promotedCount} promoção(ões) · ${checkpoint.neatState.species.length} espécie(s) NEAT`
              : `População ${populationSize} AG / ${Math.max(6, Math.round(populationSize * 0.75))} NEAT`}
          </div>
        </div>
      </section>

      {/* Section 3: Primary Focal Anchor — Train vs Validation Curves */}
      <section
        aria-labelledby="curves-heading"
        className="rounded-xl border border-[#1E232D] bg-[#12151C] p-5 space-y-4"
      >
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <h2 id="curves-heading" className="text-sm font-semibold text-[#F3F5F8]">
              Curvas de Treino vs Validação Held-Out ({selectedCategory})
            </h2>
            <p className="text-xs text-[#7D889D]">
              Losangos verdes indicam gerações promovidas pelas regras de validação; círculos azuis indicam candidatas rejeitadas.
            </p>
          </div>
          {checkpoint && (
            <div className="text-xs font-mono tabular-nums text-[#8C97AB]">
              Amostras: {checkpoint.trainSampleCount} treino · {checkpoint.valSampleCount} validação · test intocado ({catSplit?.test.length ?? 0})
            </div>
          )}
        </div>

        <TrainingCurvesChart
          history={checkpoint?.history ?? []}
          valCorruptedMedian={
            latestPoint?.valCorruptedMedian ??
            checkpoint?.valBaselineCorruptedMedian ??
            diagnosticReport.junkFloor.median
          }
          realCeilingMedian={checkpoint?.realCeilingMedian ?? diagnosticReport.realCeiling.median}
          junkFloorMedian={checkpoint?.junkFloorMedian ?? diagnosticReport.junkFloor.median}
        />
      </section>

      {/* Section 4: Held-Out Validation Triplets + Promoted Champion Inspection */}
      {previews.length > 0 && (
        <section
          aria-labelledby="restoration-preview-heading"
          className="rounded-xl border border-[#1E232D] bg-[#12151C] p-5 space-y-5"
        >
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 id="restoration-preview-heading" className="text-sm font-semibold text-[#F3F5F8]">
                Restauração Real na Validação Held-Out (Corrompido → Restaurado → Original .spr)
              </h2>
              <p className="text-xs text-[#7D889D]">
                Sprites nunca vistos durante a evolução. Todos os pixels restaurados pertencem à paleta exata sem anti-aliasing.
              </p>
            </div>

            {champion && onRegisterChampionLesson && (
              <button
                type="button"
                disabled={isRecordingLesson}
                onClick={handlePromoteChampionToLesson}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-[#2D384B] bg-[#161C27] text-[#CBD5E1] hover:border-[#4B5E7E] hover:text-[#F8FAFC] disabled:opacity-50 transition-colors whitespace-nowrap shrink-0"
              >
                <BookPlus className="w-3.5 h-3.5 text-[#E5A93C]" aria-hidden="true" />
                <span>
                  {isRecordingLesson ? 'Gravando...' : 'Registrar lição do campeão'}
                </span>
              </button>
            )}
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            {previews.map((item) => (
              <div
                key={item.spriteId}
                className="rounded-lg border border-[#202633] bg-[#0F1218] p-4 space-y-3"
              >
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                  <div className="text-[#CBD5E1] font-medium">
                    Sprite <span className="font-mono tabular-nums text-[#F3F5F8]">#{item.spriteId}</span>
                    <span className="mx-1.5 text-[#475569]">·</span>
                    <span className="text-[#8C97AB]">{CORRUPTION_LABELS_PT[item.corruptionType]}</span>
                  </div>
                  <div className="font-mono tabular-nums text-[11px]">
                    <span className={item.deltaScore >= 0 ? 'text-[#86EFAC]' : 'text-[#FCA5A5]'}>
                      {item.deltaScore >= 0 ? `+${item.deltaScore.toFixed(1)}` : item.deltaScore.toFixed(1)} pts
                    </span>
                    <span className="mx-1.5 text-[#475569]">·</span>
                    <span className="text-[#94A3B8]">IoU {(item.silhouetteIoU * 100).toFixed(1)}%</span>
                  </div>
                </div>

                <div className="flex items-center justify-between gap-2">
                  <div className="flex flex-col items-center gap-1.5">
                    <SpriteThumb
                      rgba={item.corruptedRgba}
                      scale={3}
                      backdrop="dark"
                      alt={`Sprite #${item.spriteId} corrompido`}
                    />
                    <span className="text-[11px] font-mono tabular-nums text-[#FCA5A5]">
                      Corrompido {item.corruptedScore.toFixed(1)}
                    </span>
                  </div>

                  <ArrowRight className="w-4 h-4 text-[#475569] shrink-0" aria-hidden="true" />

                  <div className="flex flex-col items-center gap-1.5">
                    <SpriteThumb
                      rgba={item.restoredRgba}
                      scale={3}
                      backdrop="dark"
                      alt={`Sprite #${item.spriteId} restaurado`}
                    />
                    <span className="text-[11px] font-mono tabular-nums text-[#93C5FD]">
                      Restaurado {item.restoredScore.toFixed(1)}
                    </span>
                  </div>

                  <ArrowRight className="w-4 h-4 text-[#475569] shrink-0" aria-hidden="true" />

                  <div className="flex flex-col items-center gap-1.5">
                    <SpriteThumb
                      rgba={item.cleanRgba}
                      scale={3}
                      backdrop="dark"
                      alt={`Sprite #${item.spriteId} original do .spr`}
                    />
                    <span className="text-[11px] font-mono tabular-nums text-[#86EFAC]">
                      Real .spr {item.cleanScore.toFixed(1)}
                    </span>
                  </div>
                </div>

                <div className="text-[11px] text-[#7D889D] pt-2 border-t border-[#1B202C] flex flex-wrap gap-x-3 gap-y-1">
                  <span>
                    Corrigidos:{' '}
                    <span className="font-mono text-[#A7F3D0]">
                      {item.issuesFixed.length > 0 ? item.issuesFixed.join(', ') : 'nenhum novo'}
                    </span>
                  </span>
                  {item.issuesRemaining.length > 0 && (
                    <span>
                      Restantes:{' '}
                      <span className="font-mono text-[#CBD5E1]">
                        {item.issuesRemaining.slice(0, 3).join(', ')}
                      </span>
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Decoded Champion Architecture (Recipe GA + NEAT Topology) */}
          {champion && neatComp && (
            <div className="pt-4 border-t border-[#1D2330] grid grid-cols-1 lg:grid-cols-2 gap-6 text-xs">
              <div className="space-y-2">
                <div className="flex items-center gap-1.5 font-semibold text-[#F3F5F8]">
                  <Sparkles className="w-3.5 h-3.5 text-[#E5A93C]" aria-hidden="true" />
                  <span>Receita AG Promovida (Gen {champion.generation})</span>
                </div>
                <ol className="space-y-1 text-[#CBD5E1]">
                  {champion.recipe.steps
                    .filter((s) => s.enabled)
                    .map((step, idx) => (
                      <li
                        key={`${step.op}-${idx}`}
                        className="flex items-center justify-between py-1 border-b border-[#1A202B] last:border-b-0"
                      >
                        <span>
                          <span className="font-mono text-[#7D889D] mr-2">0{idx + 1}.</span>
                          {PIXEL_OP_LABELS_PT[step.op]}
                        </span>
                        <span className="font-mono tabular-nums text-[11px] text-[#8C97AB]">
                          int {step.intensity.toFixed(2)} · lim {step.threshold.toFixed(2)}
                        </span>
                      </li>
                    ))}
                </ol>
              </div>

              <div className="space-y-2">
                <div className="font-semibold text-[#F3F5F8]">
                  Política Pixel-a-Pixel NEAT Promovida
                </div>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-[#CBD5E1]">
                  <div className="flex justify-between py-1 border-b border-[#1A202B]">
                    <dt className="text-[#7D889D]">Topologia</dt>
                    <dd className="font-mono tabular-nums">
                      {neatComp.totalNodes} nós ({neatComp.hiddenNodes} ocultos)
                    </dd>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#1A202B]">
                    <dt className="text-[#7D889D]">Conexões ativas</dt>
                    <dd className="font-mono tabular-nums">
                      {neatComp.enabledConnections}/{neatComp.totalConnections}
                    </dd>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#1A202B]">
                    <dt className="text-[#7D889D]">Espécies NEAT</dt>
                    <dd className="font-mono tabular-nums">
                      {checkpoint?.neatState.species.length ?? 0} (δ={checkpoint?.neatState.config.compatibilityThreshold.toFixed(2)})
                    </dd>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#1A202B]">
                    <dt className="text-[#7D889D]">Similaridade pixel</dt>
                    <dd className="font-mono tabular-nums">
                      {(champion.valStats.meanPixelSimilarity * 100).toFixed(1)}%
                    </dd>
                  </div>
                </dl>
                <p className="text-[11px] text-[#7D889D]">
                  Cada decisão da rede NEAT seleciona estritamente um índice exato da paleta do sprite (manter, contorno escuro, vizinho modal ou ±1 degrau de luminância), garantindo alpha binário e zero anti-aliasing.
                </p>
              </div>
            </div>
          )}
        </section>
      )}

      {/* Section 5: Promotion Log Table */}
      <section
        aria-labelledby="promotion-log-heading"
        data-testid="promotion-log-section"
        className="rounded-xl border border-[#1E232D] bg-[#12151C] p-5 space-y-3"
      >
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <h2 id="promotion-log-heading" className="text-sm font-semibold text-[#F3F5F8]">
              Registro de Promoções e Rejeições
            </h2>
            <p className="text-xs text-[#7D889D]">
              Regras ativas: melhora mínima na validação (+0.2 pts), IoU ≥ 72%, 0 novas violações rígidas, gap treino–validação ≤ 18 pts e margem acima do piso lixo.
            </p>
          </div>
          <span className="text-xs font-mono tabular-nums text-[#8C97AB]">
            {checkpoint?.promotionLog.length ?? 0} registro(s) · {promotedCount} promovido(s)
          </span>
        </div>

        {!checkpoint || checkpoint.promotionLog.length === 0 ? (
          <div className="py-6 text-center text-xs text-[#7D889D] border-t border-[#1D2330]">
            Nenhuma candidatura avaliada ainda para {selectedCategory}.
          </div>
        ) : (
          <div className="overflow-x-auto border-t border-[#1D2330]">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-[#1E2532] text-[11px] text-[#7D889D]">
                  <th className="py-2.5 pr-3 font-medium">Gen</th>
                  <th className="py-2.5 px-3 font-medium">Decisão</th>
                  <th className="py-2.5 px-3 font-medium text-right">Treino</th>
                  <th className="py-2.5 px-3 font-medium text-right">Validação</th>
                  <th className="py-2.5 px-3 font-medium text-right">Δ Corromp.</th>
                  <th className="py-2.5 px-3 font-medium text-right">IoU</th>
                  <th className="py-2.5 pl-3 font-medium">Justificativa e Critério</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#181D27]">
                {checkpoint.promotionLog.map((entry) => {
                  const promoted = entry.decision === 'promoted';
                  return (
                    <tr
                      key={entry.id}
                      className="hover:bg-[#161B25] transition-colors align-top"
                    >
                      <td className="py-2.5 pr-3 font-mono tabular-nums text-[#CBD5E1] whitespace-nowrap">
                        G{entry.generation}
                      </td>
                      <td className="py-2.5 px-3 whitespace-nowrap">
                        <span
                          className={`font-medium ${
                            promoted ? 'text-[#86EFAC]' : 'text-[#94A3B8]'
                          }`}
                        >
                          {promoted ? 'Promovido' : 'Rejeitado'}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 font-mono tabular-nums text-right text-[#E5A93C] whitespace-nowrap">
                        {entry.trainScore.toFixed(1)}
                      </td>
                      <td className="py-2.5 px-3 font-mono tabular-nums text-right text-[#93C5FD] whitespace-nowrap">
                        {entry.valScore.toFixed(1)}
                      </td>
                      <td
                        className={`py-2.5 px-3 font-mono tabular-nums text-right whitespace-nowrap ${
                          entry.valDeltaFromCorrupted >= 0 ? 'text-[#86EFAC]' : 'text-[#FCA5A5]'
                        }`}
                      >
                        {entry.valDeltaFromCorrupted >= 0
                          ? `+${entry.valDeltaFromCorrupted.toFixed(1)}`
                          : entry.valDeltaFromCorrupted.toFixed(1)}
                      </td>
                      <td className="py-2.5 px-3 font-mono tabular-nums text-right text-[#CBD5E1] whitespace-nowrap">
                        {(entry.silhouetteIoU * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 pl-3 text-[#CBD5E1]">
                        <div>{entry.reasonPt}</div>
                        <div className="text-[11px] text-[#7D889D] mt-0.5 truncate max-w-xl">
                          {entry.recipeSummary} · NEAT: {entry.neatSummary}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
