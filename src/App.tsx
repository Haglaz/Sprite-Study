import { useCallback, useEffect, useState } from 'react';
import { CheckCircle2, Database } from 'lucide-react';
import { ModuleComparator } from './components/ModuleComparator';
import { ModuleLessons, type NewLessonFormPayload } from './components/ModuleLessons';
import { ModulePromptStudio } from './components/ModulePromptStudio';
import {
  ModuleRefinements,
  type CriticLoopLessonPayload,
} from './components/ModuleRefinements';
import { ModuleUpload } from './components/ModuleUpload';
import {
  ensureDefaultAgentsVerified,
  recordBlindEloMatchVerified,
  setActiveAgentVerified,
} from './comparator/store';
import type {
  AgentProfileRecord,
  CommentToLessonDraft,
  CriticRetrainTelemetry,
} from './comparator/types';
import {
  calibrateCriticFromArchive,
  retrainCriticWithHumanApprovals,
} from './critic/diagnostics';
import type { CriticCalibration, DiagnosticReport } from './critic/types';
import {
  buildDatasetKey,
  defaultDbBackend,
  exportFullBackupJson,
  importFullBackupJson,
  loadActiveDataset,
  loadDatasetSplit,
  requestPersistentStorage,
  saveActiveDataset,
  saveCriticState,
  saveDatasetSplit,
  type BackupImportResult,
  type PersistentStorageStatus,
} from './db/persistence';
import type { StoredDatasetRecord, StoredDatasetSplit } from './db/schema';
import {
  collectUniqueNonEmptySprites,
  createStratifiedDatasetSplit,
  DEFAULT_SPLIT_SEED,
} from './db/split';
import { listEvolutionCheckpoints } from './evolution/store';
import type { CoevolutionChampion } from './evolution/types';
import { runSpriteIndexingJob } from './generation/indexer';
import { loadPromptHistory, savePromptVersionVerified } from './generation/promptStore';
import {
  deleteGeneratedSprite,
  loadApprovals,
  loadGeneratedSprites,
  rebuildHumanPreferenceModel,
  recordApprovalLabelVerified,
  saveGeneratedSpriteVerified,
} from './generation/store';
import type {
  ApprovalLabelRecord,
  GeneratedSpriteRecord,
  PromptHistoryRecord,
  SpriteIndexState,
} from './generation/types';
import { mineLessonsFromCalibration } from './lessons/miner';
import {
  createLessonVerified,
  loadAllLessons,
  recordLessonEvidenceVerified,
  resolveAllConflictsAutoVerified,
  resolveConflictVerified,
  setLessonArchiveStateVerified,
  syncMinedLessonsVerified,
} from './lessons/store';
import type {
  ConflictResolutionStrategy,
  LessonConflict,
  LessonRecord,
} from './lessons/types';
import {
  importT3ModelPackageVerified,
  loadT3ModelPackage,
  loadT3Vocabulary,
  saveT3ModelPackageVerified,
  saveT3VocabularyVerified,
} from './t3/store';
import type {
  T3MaterialRole,
  T3ModelPackageRecord,
  T3RampVocabulary,
  T3TestSplitComparisonReport,
} from './t3/types';
import {
  applyVocabularyRoleCorrection,
  buildRampRoleVocabularyFromTrainSplit,
  computeSplitHash,
} from './t3/vocabulary';
import type { ItemCategory, LocalDatasetSummary, StudioModuleId } from './types/studio';
import {
  downloadPeonsoTibiaFiles,
  type DiscoveredPeonsoPackage,
  type PeonsoFetchProgress,
} from './utils/peonsoGithubLoader';
import { sha256Hex } from './utils/pixelImage';
import {
  extractDominantPalette,
  parseTibiaDat,
  parseTibiaSpr,
  TibiaParseError,
} from './utils/tibiaParser';

const NAV_MODULES: { id: StudioModuleId; label: string; mobileLabel: string }[] = [
  { id: 'upload', label: 'Pacote .dat/.spr', mobileLabel: 'Pacote' },
  { id: 'refinements', label: 'Treino e Refinos', mobileLabel: 'Treinar' },
  { id: 'lessons', label: 'Lições Aprendidas', mobileLabel: 'Lições' },
  { id: 'comparator', label: 'Comparador', mobileLabel: 'Comparar' },
  { id: 'studio', label: 'Estúdio Prompt', mobileLabel: 'Estúdio' },
];

export default function App() {
  const [activeModule, setActiveModule] = useState<StudioModuleId>('upload');

  // Pending selected files before a complete .dat + .spr pair is ready
  const [datFile, setDatFile] = useState<File | null>(null);
  const [sprFile, setSprFile] = useState<File | null>(null);

  // Active verified dataset + Split + Critic v1 calibration + Diagnostic report
  const [dataset, setDataset] = useState<LocalDatasetSummary | null>(null);
  const [activeSplit, setActiveSplit] = useState<StoredDatasetSplit | null>(null);
  const [calibration, setCalibration] = useState<CriticCalibration | null>(null);
  const [diagnosticReport, setDiagnosticReport] = useState<DiagnosticReport | null>(null);

  // Real Lessons Database state (IndexedDB 'lessons' store)
  const [lessons, setLessons] = useState<LessonRecord[]>([]);
  const [lessonConflicts, setLessonConflicts] = useState<LessonConflict[]>([]);
  const [isLessonsBusy, setIsLessonsBusy] = useState(false);
  const [lessonsStatusMsg, setLessonsStatusMsg] = useState<string | null>(null);
  const [lessonsErrorMsg, setLessonsErrorMsg] = useState<string | null>(null);

  // Phase 4 & 5: Indexing, Generated Sprites, Approvals, Phase 3 Champions, Elo Agents, & Retrain Telemetry
  const [spriteIndex, setSpriteIndex] = useState<SpriteIndexState | null>(null);
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexProgress, setIndexProgress] = useState<{ done: number; total: number } | null>(null);
  const [generatedSprites, setGeneratedSprites] = useState<GeneratedSpriteRecord[]>([]);
  const [approvals, setApprovals] = useState<ApprovalLabelRecord[]>([]);
  const [promptHistory, setPromptHistory] = useState<PromptHistoryRecord[]>([]);
  const [championsByCategory, setChampionsByCategory] = useState<
    Partial<Record<ItemCategory, CoevolutionChampion>>
  >({});
  const [agents, setAgents] = useState<AgentProfileRecord[]>([]);
  const [retrainTelemetry, setRetrainTelemetry] = useState<CriticRetrainTelemetry | null>(null);
  const [isRetrainingCritic, setIsRetrainingCritic] = useState(false);

  // Phase 7: Trained Indexed Model (T3) Vocabulary & ONNX Package
  const [t3Vocabulary, setT3Vocabulary] = useState<T3RampVocabulary | null>(null);
  const [t3ModelPackage, setT3ModelPackage] = useState<T3ModelPackageRecord | null>(null);

  // Lifecycle & persistence status
  const [isRestoring, setIsRestoring] = useState(true);
  const [isParsing, setIsParsing] = useState(false);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [isFetchingGithub, setIsFetchingGithub] = useState(false);
  const [githubFetchProgress, setGithubFetchProgress] = useState<PeonsoFetchProgress | null>(null);
  const [githubSourcePkg, setGithubSourcePkg] = useState<DiscoveredPeonsoPackage | null>(null);
  const [storageStatus, setStorageStatus] = useState<PersistentStorageStatus | null>(null);
  const [persistenceNotice, setPersistenceNotice] = useState<string | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);

  const refreshLessonsFromDb = useCallback(async () => {
    const loaded = await loadAllLessons(defaultDbBackend);
    setLessons(loaded.lessons);
    setLessonConflicts(loaded.conflicts);
  }, []);

  /**
   * Parses raw .dat/.spr buffers, ensures a persisted deterministic 70/15/15 split,
   * runs Critic v1 calibration + diagnostics, and verifies IndexedDB read-back.
   */
  const hydrateAndCalibrateRecord = useCallback(
    async (record: StoredDatasetRecord, restoredFromDb: boolean) => {
      setIsCalibrating(true);
      setParseError(null);

      try {
        const parsedSpr = parseTibiaSpr(record.sprBuffer);
        const parsedDat = parseTibiaDat(
          record.datBuffer,
          parsedSpr.spriteCount,
          parsedSpr.isExtended,
        );
        const palette = extractDominantPalette(parsedSpr, parsedDat.items, 16);
        const datasetKey = buildDatasetKey(record.datSha256, record.sprSha256);

        const summary: LocalDatasetSummary = {
          datFileName: record.datFileName,
          sprFileName: record.sprFileName,
          datSizeBytes: record.datSizeBytes,
          sprSizeBytes: record.sprSizeBytes,
          datSha256: record.datSha256,
          sprSha256: record.sprSha256,
          dat: parsedDat,
          spr: parsedSpr,
          dominantPaletteHex: palette,
          savedAt: record.savedAt,
          restoredFromDb,
        };
        setDataset(summary);

        // Load or create deterministic 70/15/15 stratified split and verify persistence
        let split = await loadDatasetSplit(defaultDbBackend, datasetKey);
        if (!split) {
          const candidates = collectUniqueNonEmptySprites(parsedDat.items, parsedSpr);
          const created = createStratifiedDatasetSplit({
            datasetKey,
            candidates,
            seed: DEFAULT_SPLIT_SEED,
          });
          const verifiedSplit = await saveDatasetSplit(defaultDbBackend, created);
          split = verifiedSplit.readBack;
        }
        setActiveSplit(split);

        // Load persisted generated sprites, human approvals, prompt history, promoted champions, and Phase 7 T3 state
        const [
          savedSprites,
          savedApprovals,
          savedPromptHistory,
          savedCheckpoints,
          savedVocab,
          savedT3Model,
        ] = await Promise.all([
          loadGeneratedSprites(defaultDbBackend, datasetKey),
          loadApprovals(defaultDbBackend, datasetKey),
          loadPromptHistory(defaultDbBackend, datasetKey),
          listEvolutionCheckpoints(defaultDbBackend, datasetKey),
          loadT3Vocabulary(defaultDbBackend, datasetKey),
          loadT3ModelPackage(defaultDbBackend, datasetKey),
        ]);

        const champMap: Partial<Record<ItemCategory, CoevolutionChampion>> = {};
        for (const cp of savedCheckpoints) {
          if (cp.champion) champMap[cp.category] = cp.champion;
        }

        // Seed/load Elo competing agents (preserving existing Elo & active agent across reloads)
        const hydratedAgents = await ensureDefaultAgentsVerified(
          defaultDbBackend,
          datasetKey,
          champMap,
        );

        // Build or restore Material Ramp-Role Vocabulary strictly from `train` split
        const currentSplitHash = computeSplitHash(split);
        let activeVocab = savedVocab;
        if (!activeVocab || activeVocab.splitHash !== currentSplitHash) {
          const built = buildRampRoleVocabularyFromTrainSplit({
            datasetKey,
            dat: parsedDat,
            spr: parsedSpr,
            split,
          });
          activeVocab = await saveT3VocabularyVerified(defaultDbBackend, built);
        }

        // Yield one frame so the calibrating spinner renders cleanly before heavy pixel decoding
        await new Promise((r) => window.setTimeout(r, 20));

        const humanPrefs = rebuildHumanPreferenceModel(savedApprovals);
        const approvedFeats = savedApprovals
          .filter((a) => a.decision === 'approved')
          .map((a) => a.features);
        const rejectedFeats = savedApprovals
          .filter((a) => a.decision === 'rejected')
          .map((a) => a.features);

        const { calibration: enrichedCalibration, diagnostics: nextDiagnostics } =
          calibrateCriticFromArchive({
            datasetKey,
            dat: parsedDat,
            spr: parsedSpr,
            split,
            extraApprovedFeatures: approvedFeats,
            extraRejectedFeatures: rejectedFeats,
            humanPreferences: humanPrefs,
            seed: split.seed,
          });

        // Persist critic calibration state and verify read-back before showing saved notice
        await saveCriticState(defaultDbBackend, {
          id: datasetKey,
          datasetKey,
          calibratedAt: enrichedCalibration.calibratedAt,
          seed: enrichedCalibration.seed,
          payload: enrichedCalibration,
        });

        setCalibration(enrichedCalibration);
        setDiagnosticReport(nextDiagnostics);
        setGeneratedSprites(savedSprites);
        setApprovals(savedApprovals);
        setPromptHistory(savedPromptHistory);
        setChampionsByCategory(champMap);
        setAgents(hydratedAgents);
        setT3Vocabulary(activeVocab);
        setT3ModelPackage(savedT3Model);
        setRetrainTelemetry({
          retrainCount: 0,
          lastRetrainedAt: enrichedCalibration.calibratedAt,
          approvedUsed: approvedFeats.length,
          rejectedUsed: rejectedFeats.length,
          previousValidationAuc: null,
          currentValidationAuc: enrichedCalibration.discriminator.validationAuc,
          humanPreferenceActive: Boolean(humanPrefs),
        });
        setSpriteIndex(null);
        setPersistenceNotice(
          restoredFromDb
            ? 'Dataset, split 70/15/15, agentes Elo e Crítico v1 restaurados e verificados no IndexedDB'
            : 'Dataset, split 70/15/15, agentes Elo e Crítico v1 salvos e verificados (read-back) no IndexedDB',
        );
      } catch (err) {
        setCalibration(null);
        setDiagnosticReport(null);
        if (err instanceof TibiaParseError) {
          setParseError(err.message);
        } else {
          setParseError(
            err instanceof Error
              ? err.message
              : 'Falha ao analisar o pacote .dat/.spr ou calibrar o Crítico v1.',
          );
        }
      } finally {
        setIsCalibrating(false);
      }
    },
    [],
  );

  // Startup auto-restore from IndexedDB + navigator.storage.persist()
  useEffect(() => {
    let cancelled = false;

    async function boot() {
      try {
        const [persistStatus, stored, loadedLessons] = await Promise.all([
          requestPersistentStorage(),
          loadActiveDataset(defaultDbBackend),
          loadAllLessons(defaultDbBackend),
        ]);
        if (cancelled) return;

        setStorageStatus(persistStatus);
        setLessons(loadedLessons.lessons);
        setLessonConflicts(loadedLessons.conflicts);

        if (stored && stored.datBuffer instanceof ArrayBuffer && stored.sprBuffer instanceof ArrayBuffer) {
          await hydrateAndCalibrateRecord(stored, true);
        }
      } catch (err) {
        if (!cancelled) {
          setParseError(
            err instanceof Error
              ? `Erro ao restaurar banco local: ${err.message}`
              : 'Erro ao restaurar banco local.',
          );
        }
      } finally {
        if (!cancelled) setIsRestoring(false);
      }
    }

    void boot();
    return () => {
      cancelled = true;
    };
  }, [hydrateAndCalibrateRecord]);

  const ingestDatSprPair = useCallback(
    async (nextDat: File, nextSpr: File) => {
      setIsParsing(true);
      setParseError(null);
      setPersistenceNotice(null);
      setCalibration(null);
      setDiagnosticReport(null);

      try {
        const [datBuf, sprBuf] = await Promise.all([
          nextDat.arrayBuffer(),
          nextSpr.arrayBuffer(),
        ]);

        const parsedSpr = parseTibiaSpr(sprBuf);
        const parsedDat = parseTibiaDat(datBuf, parsedSpr.spriteCount, parsedSpr.isExtended);

        const [datSha256, sprSha256] = await Promise.all([
          sha256Hex(datBuf),
          sha256Hex(sprBuf),
        ]);

        const record: StoredDatasetRecord = {
          id: 'active',
          datFileName: nextDat.name,
          sprFileName: nextSpr.name,
          datSizeBytes: nextDat.size,
          sprSizeBytes: nextSpr.size,
          datSha256,
          sprSha256,
          datBuffer: datBuf,
          sprBuffer: sprBuf,
          itemCount: parsedDat.items.length,
          spriteCount: parsedSpr.spriteCount,
          nonEmptySprites: parsedSpr.nonEmptySprites,
          versionHint: parsedDat.versionHint,
          savedAt: new Date().toISOString(),
        };

        const verified = await saveActiveDataset(defaultDbBackend, record);
        setIsParsing(false);
        await hydrateAndCalibrateRecord(verified.readBack, false);
      } catch (err) {
        setIsParsing(false);
        setDataset(null);
        setActiveSplit(null);
        setCalibration(null);
        setDiagnosticReport(null);
        if (err instanceof TibiaParseError) {
          setParseError(err.message);
        } else {
          setParseError(
            err instanceof Error
              ? err.message
              : 'Não foi possível ler e persistir os arquivos .dat e .spr selecionados.',
          );
        }
      }
    },
    [hydrateAndCalibrateRecord],
  );

  const handleFilesSelected = (files: File[]) => {
    let nextDat = datFile;
    let nextSpr = sprFile;
    let matchedAny = false;

    for (const file of files) {
      const lower = file.name.toLowerCase();
      if (lower.endsWith('.dat')) {
        nextDat = file;
        matchedAny = true;
      } else if (lower.endsWith('.spr')) {
        nextSpr = file;
        matchedAny = true;
      }
    }

    if (!matchedAny) {
      setParseError('Selecione arquivos com extensão .dat e .spr.');
      return;
    }

    setParseError(null);
    setGithubSourcePkg(null);
    setDatFile(nextDat);
    setSprFile(nextSpr);

    if (nextDat && nextSpr) {
      void ingestDatSprPair(nextDat, nextSpr);
    }
  };

  const handleFetchPeonsoFromGithub = useCallback(async () => {
    if (isFetchingGithub || isParsing || isCalibrating) return;
    setIsFetchingGithub(true);
    setParseError(null);
    setGithubFetchProgress({
      stage: 'searching',
      message: 'Localizando .dat e .spr nos repositórios do peonso no GitHub...',
    });

    try {
      const { pkg, datFile: downloadedDat, sprFile: downloadedSpr } =
        await downloadPeonsoTibiaFiles({
          onProgress: (p) => setGithubFetchProgress(p),
        });

      setDatFile(downloadedDat);
      setSprFile(downloadedSpr);
      setGithubSourcePkg(pkg);
      await ingestDatSprPair(downloadedDat, downloadedSpr);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'Não foi possível baixar o .dat e .spr do peonso no GitHub.';
      setParseError(
        `${message} Verifique sua conexão com o GitHub ou envie os arquivos manualmente.`,
      );
    } finally {
      setIsFetchingGithub(false);
      setGithubFetchProgress(null);
    }
  }, [isFetchingGithub, isParsing, isCalibrating, ingestDatSprPair]);

  const handleConfirmReplaceDataset = useCallback(() => {
    void defaultDbBackend.delete('datasets', 'active');
    setDatFile(null);
    setSprFile(null);
    setDataset(null);
    setActiveSplit(null);
    setCalibration(null);
    setDiagnosticReport(null);
    setSpriteIndex(null);
    setGeneratedSprites([]);
    setApprovals([]);
    setPromptHistory([]);
    setChampionsByCategory({});
    setAgents([]);
    setRetrainTelemetry(null);
    setT3Vocabulary(null);
    setT3ModelPackage(null);
    setGithubSourcePkg(null);
    setPersistenceNotice(null);
    setParseError(null);
  }, []);

  const handleExportBackup = useCallback(async () => {
    const json = await exportFullBackupJson(defaultDbBackend);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    try {
      const a = document.createElement('a');
      a.href = url;
      a.download = `atelier-sprite-backup-${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } finally {
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
  }, []);

  const handleImportBackupFile = useCallback(
    async (file: File): Promise<BackupImportResult> => {
      const rawJson = await file.text();
      const result = await importFullBackupJson(defaultDbBackend, rawJson);
      await refreshLessonsFromDb();
      const restored = await loadActiveDataset(defaultDbBackend);
      if (restored) {
        await hydrateAndCalibrateRecord(restored, true);
      } else {
        setDataset(null);
        setActiveSplit(null);
        setCalibration(null);
        setDiagnosticReport(null);
        setSpriteIndex(null);
        setGeneratedSprites([]);
        setApprovals([]);
        setPromptHistory([]);
        setChampionsByCategory({});
        setAgents([]);
        setRetrainTelemetry(null);
        setT3Vocabulary(null);
        setT3ModelPackage(null);
      }
      return result;
    },
    [hydrateAndCalibrateRecord, refreshLessonsFromDb],
  );

  // --- Phase 2: Real Lessons Database handlers (all read-back verified) ---

  const handleMineLessonsFromDataset = useCallback(async () => {
    if (!calibration) return;
    setIsLessonsBusy(true);
    setLessonsStatusMsg(null);
    setLessonsErrorMsg(null);
    try {
      const mined = mineLessonsFromCalibration({
        calibration,
        split: activeSplit,
      });
      const res = await syncMinedLessonsVerified(defaultDbBackend, mined);
      setLessons(res.lessons);
      setLessonConflicts(res.conflicts);
      setLessonsStatusMsg(
        `${res.writtenCount} lição(ões) minerada(s) do split de treino, gravada(s) e verificada(s) (read-back) no IndexedDB.${
          res.conflicts.length > 0
            ? ` ${res.conflicts.length} conflito(s) detectado(s) com regras existentes.`
            : ''
        }`,
      );
    } catch (err) {
      setLessonsErrorMsg(
        err instanceof Error ? err.message : 'Falha ao minerar e salvar lições.',
      );
    } finally {
      setIsLessonsBusy(false);
    }
  }, [calibration, activeSplit]);

  const handleCreateLesson = useCallback(
    async (payload: NewLessonFormPayload) => {
      setIsLessonsBusy(true);
      setLessonsStatusMsg(null);
      setLessonsErrorMsg(null);
      try {
        const datasetKey =
          dataset ? buildDatasetKey(dataset.datSha256, dataset.sprSha256) : null;
        const targetRange =
          payload.minVal !== undefined || payload.maxVal !== undefined
            ? { min: payload.minVal, max: payload.maxVal }
            : undefined;

        const res = await createLessonVerified(defaultDbBackend, {
          datasetKey,
          category: payload.category,
          scopeKey: payload.scopeKey,
          title: payload.title,
          source: 'human_feedback',
          polarity: payload.polarity,
          directive: payload.directive,
          validationCheck: payload.validationCheck,
          targetRange,
        });
        setLessons(res.lessons);
        setLessonConflicts(res.conflicts);
        setLessonsStatusMsg(
          `Lição "${res.saved.title}" salva e verificada (read-back) no IndexedDB.${
            res.conflicts.length > 0
              ? ` Atenção: ${res.conflicts.length} conflito(s) aberto(s).`
              : ''
          }`,
        );
      } catch (err) {
        setLessonsErrorMsg(
          err instanceof Error ? err.message : 'Falha ao registrar lição.',
        );
        throw err;
      } finally {
        setIsLessonsBusy(false);
      }
    },
    [dataset],
  );

  const handleRecordLessonEvidence = useCallback(
    async (lessonId: string, outcome: 'confirm' | 'contest') => {
      setIsLessonsBusy(true);
      setLessonsStatusMsg(null);
      setLessonsErrorMsg(null);
      try {
        const res = await recordLessonEvidenceVerified(defaultDbBackend, lessonId, outcome);
        setLessons(res.lessons);
        setLessonConflicts(res.conflicts);
        setLessonsStatusMsg(
          `Evidência atualizada (${outcome === 'confirm' ? '+1 confirmação' : '−1 contestação'}) e verificada (read-back): confiança Wilson ${(
            res.saved.evidence.confidence * 100
          ).toFixed(1)}%.`,
        );
      } catch (err) {
        setLessonsErrorMsg(
          err instanceof Error ? err.message : 'Falha ao atualizar evidência da lição.',
        );
      } finally {
        setIsLessonsBusy(false);
      }
    },
    [],
  );

  const handleToggleLessonArchive = useCallback(async (lessonId: string, archived: boolean) => {
    setIsLessonsBusy(true);
    setLessonsStatusMsg(null);
    setLessonsErrorMsg(null);
    try {
      const res = await setLessonArchiveStateVerified(defaultDbBackend, lessonId, archived);
      setLessons(res.lessons);
      setLessonConflicts(res.conflicts);
      setLessonsStatusMsg(
        archived
          ? `Lição "${res.saved.title}" arquivada e verificada (read-back).`
          : `Lição "${res.saved.title}" reativada e verificada (read-back).`,
      );
    } catch (err) {
      setLessonsErrorMsg(
        err instanceof Error ? err.message : 'Falha ao atualizar estado da lição.',
      );
    } finally {
      setIsLessonsBusy(false);
    }
  }, []);

  const handleResolveConflict = useCallback(
    async (
      conflict: LessonConflict,
      strategy: ConflictResolutionStrategy,
      preferredWinnerId?: string,
    ) => {
      setIsLessonsBusy(true);
      setLessonsStatusMsg(null);
      setLessonsErrorMsg(null);
      try {
        const res = await resolveConflictVerified(
          defaultDbBackend,
          conflict,
          strategy,
          preferredWinnerId,
        );
        setLessons(res.lessons);
        setLessonConflicts(res.conflicts);
        setLessonsStatusMsg(
          `Conflito resolvido e verificado (read-back): "${res.resolution.winner.title}" mantida ativa. ${res.resolution.rationalePt}`,
        );
      } catch (err) {
        setLessonsErrorMsg(
          err instanceof Error ? err.message : 'Falha ao resolver conflito.',
        );
      } finally {
        setIsLessonsBusy(false);
      }
    },
    [],
  );

  const handleResolveAllConflictsAuto = useCallback(async () => {
    setIsLessonsBusy(true);
    setLessonsStatusMsg(null);
    setLessonsErrorMsg(null);
    try {
      const res = await resolveAllConflictsAutoVerified(defaultDbBackend);
      setLessons(res.lessons);
      setLessonConflicts(res.conflicts);
      setLessonsStatusMsg(
        `${res.resolutions.length} conflito(s) resolvido(s) automaticamente por precedência e verificado(s) (read-back) no IndexedDB.`,
      );
    } catch (err) {
      setLessonsErrorMsg(
        err instanceof Error ? err.message : 'Falha ao resolver conflitos automaticamente.',
      );
    } finally {
      setIsLessonsBusy(false);
    }
  }, []);

  const handleRegisterChampionLesson = useCallback(
    async (payload: CriticLoopLessonPayload) => {
      const datasetKey =
        dataset ? buildDatasetKey(dataset.datSha256, dataset.sprSha256) : null;
      const res = await createLessonVerified(defaultDbBackend, {
        datasetKey,
        category: payload.category,
        scopeKey: 'cluster_hygiene',
        title: payload.title,
        source: 'critic_loop',
        polarity: 'enforce',
        directive: payload.directive,
        validationCheck: payload.validationCheck,
        criticDeltaMean: payload.criticDeltaMean,
      });
      setLessons(res.lessons);
      setLessonConflicts(res.conflicts);
    },
    [dataset],
  );

  // --- Phase 4: Indexing, Generation Persistence, & Approve/Reject Handlers ---

  const handleRunIndexing = useCallback(async (): Promise<SpriteIndexState | null> => {
    if (!dataset || !activeSplit || !calibration) return null;
    setIsIndexing(true);
    setIndexProgress({ done: 0, total: 1 });
    try {
      await new Promise((r) => window.setTimeout(r, 15));
      const idx = runSpriteIndexingJob({
        datasetKey: activeSplit.datasetKey,
        dat: dataset.dat,
        spr: dataset.spr,
        split: activeSplit,
        calibration,
        onProgress: (done, total) => setIndexProgress({ done, total }),
      });
      setSpriteIndex(idx);
      return idx;
    } finally {
      setIsIndexing(false);
      setIndexProgress(null);
    }
  }, [dataset, activeSplit, calibration]);

  const handleSavePromptVersion = useCallback(
    async (record: PromptHistoryRecord): Promise<PromptHistoryRecord> => {
      const saved = await savePromptVersionVerified(defaultDbBackend, record);
      const reloaded = await loadPromptHistory(defaultDbBackend, saved.datasetKey);
      setPromptHistory(reloaded);
      return saved;
    },
    [],
  );

  const handleSaveGeneratedBatch = useCallback(
    async (records: GeneratedSpriteRecord[]) => {
      if (!activeSplit) return;
      for (const rec of records) {
        await saveGeneratedSpriteVerified(defaultDbBackend, rec);
      }
      const reloaded = await loadGeneratedSprites(defaultDbBackend, activeSplit.datasetKey);
      setGeneratedSprites(reloaded);
    },
    [activeSplit],
  );

  const runCriticAutoRetrain = useCallback(
    async (latestApprovals: ApprovalLabelRecord[]) => {
      if (!dataset || !activeSplit) return;
      setIsRetrainingCritic(true);
      try {
        await new Promise((r) => window.setTimeout(r, 10));
        const res = retrainCriticWithHumanApprovals({
          datasetKey: activeSplit.datasetKey,
          dat: dataset.dat,
          spr: dataset.spr,
          split: activeSplit,
          approvals: latestApprovals,
          previousCalibration: calibration,
          seed: activeSplit.seed,
        });

        await saveCriticState(defaultDbBackend, {
          id: activeSplit.datasetKey,
          datasetKey: activeSplit.datasetKey,
          calibratedAt: res.calibration.calibratedAt,
          seed: res.calibration.seed,
          payload: res.calibration,
        });

        setCalibration(res.calibration);
        setDiagnosticReport(res.diagnostics);
        setRetrainTelemetry((prev) => ({
          retrainCount: (prev?.retrainCount ?? 0) + 1,
          lastRetrainedAt: res.calibration.calibratedAt,
          approvedUsed: res.approvedUsed,
          rejectedUsed: res.rejectedUsed,
          previousValidationAuc: res.previousValidationAuc,
          currentValidationAuc: res.currentValidationAuc,
          humanPreferenceActive: Boolean(res.calibration.humanPreferences),
        }));
      } finally {
        setIsRetrainingCritic(false);
      }
    },
    [dataset, activeSplit, calibration],
  );

  const handleRecordApproval = useCallback(
    async (sprite: GeneratedSpriteRecord, decision: 'approved' | 'rejected') => {
      const res = await recordApprovalLabelVerified(defaultDbBackend, { sprite, decision });
      setApprovals(res.allApprovals);
      setGeneratedSprites((prev) =>
        prev.map((item) => (item.id === res.updatedSprite.id ? res.updatedSprite : item)),
      );
      // Automatic Critic v1 retraining on every new approval/rejection label!
      await runCriticAutoRetrain(res.allApprovals);
    },
    [runCriticAutoRetrain],
  );

  const handleManualRetrainCritic = useCallback(async () => {
    await runCriticAutoRetrain(approvals);
  }, [runCriticAutoRetrain, approvals]);

  const handleDeleteGenerated = useCallback(
    async (id: string) => {
      await deleteGeneratedSprite(defaultDbBackend, id);
      setGeneratedSprites((prev) => prev.filter((item) => item.id !== id));
    },
    [],
  );

  // --- Phase 5: Blind A/B Elo, "Tornar ativo", & Comment-to-Lesson Handlers ---

  const handleRecordEloMatch = useCallback(
    async (params: {
      agentAId: string;
      agentBId: string;
      winner: 'A' | 'B' | 'draw';
      prompt: string;
    }) => {
      if (!activeSplit) throw new Error('Nenhum dataset ativo.');
      const res = await recordBlindEloMatchVerified(defaultDbBackend, {
        datasetKey: activeSplit.datasetKey,
        ...params,
      });
      setAgents(res.allAgents);
      return { deltaA: res.deltaA, deltaB: res.deltaB };
    },
    [activeSplit],
  );

  const handleSetActiveAgent = useCallback(
    async (agentId: string) => {
      if (!activeSplit) throw new Error('Nenhum dataset ativo.');
      const res = await setActiveAgentVerified(defaultDbBackend, activeSplit.datasetKey, agentId);
      setAgents(res.allAgents);
    },
    [activeSplit],
  );

  const handleCreateLessonFromComment = useCallback(
    async (draft: CommentToLessonDraft) => {
      const datasetKey =
        dataset ? buildDatasetKey(dataset.datSha256, dataset.sprSha256) : null;
      const res = await createLessonVerified(defaultDbBackend, {
        datasetKey,
        category: draft.category,
        scopeKey: draft.scopeKey,
        title: draft.title,
        source: 'human_feedback',
        polarity: draft.polarity,
        directive: draft.directive,
        validationCheck: draft.validationCheck,
        relatedIssueCodes: draft.relatedIssueCodes,
      });
      setLessons(res.lessons);
      setLessonConflicts(res.conflicts);
    },
    [dataset],
  );

  // --- Phase 7: Trained Indexed Model (T3) Handlers ---

  const handleUpdateT3VocabRole = useCallback(
    async (tokenId: number, role: T3MaterialRole) => {
      if (!t3Vocabulary) return;
      const updated = applyVocabularyRoleCorrection(t3Vocabulary, tokenId, role);
      const saved = await saveT3VocabularyVerified(defaultDbBackend, updated);
      setT3Vocabulary(saved);
    },
    [t3Vocabulary],
  );

  const handleRebuildT3Vocab = useCallback(async () => {
    if (!dataset || !activeSplit) return;
    const overrides = new Map<string, T3MaterialRole>();
    if (t3Vocabulary) {
      for (const entry of t3Vocabulary.entries) {
        if (entry.userOverridden) {
          overrides.set(entry.hex.toUpperCase(), entry.role);
        }
      }
    }
    const rebuilt = buildRampRoleVocabularyFromTrainSplit({
      datasetKey: activeSplit.datasetKey,
      dat: dataset.dat,
      spr: dataset.spr,
      split: activeSplit,
      existingOverrides: overrides,
    });
    const saved = await saveT3VocabularyVerified(defaultDbBackend, rebuilt);
    setT3Vocabulary(saved);
  }, [dataset, activeSplit, t3Vocabulary]);

  const handleImportT3ModelPackage = useCallback(
    async (params: {
      modelCardJson: string;
      generatorOnnx: ArrayBuffer;
      rerankerOnnx: ArrayBuffer | null;
      generatorFileName: string;
      rerankerFileName: string | null;
    }) => {
      if (!activeSplit) throw new Error('Nenhum dataset ativo para associar ao modelo T3.');
      const saved = await importT3ModelPackageVerified(defaultDbBackend, {
        datasetKey: activeSplit.datasetKey,
        ...params,
      });
      setT3ModelPackage(saved);
      return saved;
    },
    [activeSplit],
  );

  const handleSaveT3TestEvaluation = useCallback(
    async (report: T3TestSplitComparisonReport) => {
      if (!t3ModelPackage) return;
      const nextPkg: T3ModelPackageRecord = {
        ...t3ModelPackage,
        enabled: report.recommendedEnabled,
        disabledReasonPt: report.t3IsWorse ? report.verdictSummaryPt : null,
        lastTestEvaluation: report,
      };
      const saved = await saveT3ModelPackageVerified(defaultDbBackend, nextPkg);
      setT3ModelPackage(saved);
    },
    [t3ModelPackage],
  );

  const handleToggleT3Enabled = useCallback(
    async (enabled: boolean) => {
      if (!t3ModelPackage) return;
      if (
        enabled &&
        (!t3ModelPackage.lastTestEvaluation || t3ModelPackage.lastTestEvaluation.t3IsWorse)
      ) {
        return;
      }
      const saved = await saveT3ModelPackageVerified(defaultDbBackend, {
        ...t3ModelPackage,
        enabled,
      });
      setT3ModelPackage(saved);
    },
    [t3ModelPackage],
  );

  return (
    <div className="min-h-screen bg-[#0D0F13] text-[#F1F3F7] flex flex-col">
      {/* Top Bar */}
      <header className="sticky top-0 z-30 bg-[#0D0F13]/95 backdrop-blur border-b border-[#1E232D]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-6">
          <button
            type="button"
            onClick={() => setActiveModule('upload')}
            className="text-base font-semibold tracking-tight text-[#F3F5F8] hover:text-[#E5A93C] transition-colors whitespace-nowrap shrink-0"
          >
            Atelier Sprite
          </button>

          <nav
            aria-label="Módulos do estúdio"
            className="hidden md:flex items-center gap-6 text-sm font-medium"
          >
            {NAV_MODULES.map((mod) => {
              const active = activeModule === mod.id;
              return (
                <button
                  key={mod.id}
                  type="button"
                  onClick={() => setActiveModule(mod.id)}
                  aria-current={active ? 'page' : undefined}
                  className={`inline-flex items-center gap-1.5 py-1 border-b-2 transition-colors whitespace-nowrap shrink-0 ${
                    active
                      ? 'border-[#E5A93C] text-[#F3F5F8]'
                      : 'border-transparent text-[#8C97AB] hover:text-[#E6EBF3]'
                  }`}
                >
                  <span>{mod.label}</span>
                  {mod.id === 'lessons' && lessonConflicts.length > 0 && (
                    <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded bg-[#5E2328] text-[#FCA5A5]">
                      {lessonConflicts.length}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>

          <button
            type="button"
            onClick={() => setActiveModule('upload')}
            data-testid="critic-status-button"
            className={`inline-flex items-center gap-2 px-3 py-2 text-xs font-medium rounded-lg border transition-colors whitespace-nowrap shrink-0 ${
              calibration && diagnosticReport
                ? 'bg-[#111E17] border-[#1F4A31] text-[#86EFAC] hover:bg-[#15281E]'
                : 'bg-[#151921] border-[#29303E] text-[#CBD3E1] hover:border-[#3D475B]'
            }`}
          >
            {calibration && diagnosticReport ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-[#22C55E] shrink-0" aria-hidden="true" />
            ) : (
              <Database className="w-3.5 h-3.5 text-[#E5A93C] shrink-0" aria-hidden="true" />
            )}
            <span className="hidden sm:inline">
              {calibration && diagnosticReport
                ? `Crítico v1 · Real ${diagnosticReport.realCeiling.median.toFixed(1)} vs Lixo ${diagnosticReport.junkFloor.median.toFixed(1)}`
                : 'Crítico não calibrado'}
            </span>
            <span className="sm:hidden">
              {calibration && diagnosticReport ? 'Calibrado' : 'Sem .spr'}
            </span>
          </button>
        </div>

        {/* Mobile navigation strip */}
        <nav
          aria-label="Módulos do estúdio (mobile)"
          className="md:hidden border-t border-[#1A1F28] px-4 flex items-center justify-between gap-2 h-11 overflow-x-auto"
        >
          {NAV_MODULES.map((mod) => {
            const active = activeModule === mod.id;
            return (
              <button
                key={mod.id}
                type="button"
                onClick={() => setActiveModule(mod.id)}
                aria-current={active ? 'page' : undefined}
                className={`text-xs font-medium py-1 border-b-2 whitespace-nowrap ${
                  active
                    ? 'border-[#E5A93C] text-[#F3F5F8]'
                    : 'border-transparent text-[#8C97AB]'
                }`}
              >
                {mod.mobileLabel}
              </button>
            );
          })}
        </nav>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6">
        {activeModule === 'upload' && (
          <ModuleUpload
            datFile={datFile}
            sprFile={sprFile}
            dataset={dataset}
            diagnosticReport={diagnosticReport}
            isRestoring={isRestoring}
            isParsing={isParsing}
            isCalibrating={isCalibrating}
            isFetchingGithub={isFetchingGithub}
            githubFetchProgress={githubFetchProgress}
            githubSourcePkg={githubSourcePkg}
            storageStatus={storageStatus}
            persistenceNotice={persistenceNotice}
            error={parseError}
            onFilesSelected={handleFilesSelected}
            onFetchPeonsoFromGithub={handleFetchPeonsoFromGithub}
            onConfirmReplaceDataset={handleConfirmReplaceDataset}
            onExportBackup={handleExportBackup}
            onImportBackupFile={handleImportBackupFile}
          />
        )}

        {activeModule === 'refinements' && (
          <ModuleRefinements
            dataset={dataset}
            activeSplit={activeSplit}
            calibration={calibration}
            diagnosticReport={diagnosticReport}
            onGoToUpload={() => setActiveModule('upload')}
            onRegisterChampionLesson={handleRegisterChampionLesson}
          />
        )}

        {activeModule === 'lessons' && (
          <ModuleLessons
            dataset={dataset}
            calibration={calibration}
            diagnosticReport={diagnosticReport}
            lessons={lessons}
            conflicts={lessonConflicts}
            isBusy={isLessonsBusy}
            statusMessage={lessonsStatusMsg}
            errorMessage={lessonsErrorMsg}
            onGoToUpload={() => setActiveModule('upload')}
            onMineFromDataset={handleMineLessonsFromDataset}
            onCreateLesson={handleCreateLesson}
            onRecordEvidence={handleRecordLessonEvidence}
            onToggleArchive={handleToggleLessonArchive}
            onResolveConflict={handleResolveConflict}
            onResolveAllConflictsAuto={handleResolveAllConflictsAuto}
          />
        )}

        {activeModule === 'comparator' && (
          <ModuleComparator
            dataset={dataset}
            activeSplit={activeSplit}
            calibration={calibration}
            diagnosticReport={diagnosticReport}
            spriteIndex={spriteIndex}
            isIndexing={isIndexing}
            agents={agents}
            lessons={lessons}
            championsByCategory={championsByCategory}
            generatedSprites={generatedSprites}
            approvals={approvals}
            retrainTelemetry={retrainTelemetry}
            isRetrainingCritic={isRetrainingCritic}
            onGoToUpload={() => setActiveModule('upload')}
            onGoToStudio={() => setActiveModule('studio')}
            onRunIndexing={handleRunIndexing}
            onRecordApproval={handleRecordApproval}
            onSaveGeneratedBatch={handleSaveGeneratedBatch}
            onRecordEloMatch={handleRecordEloMatch}
            onSetActiveAgent={handleSetActiveAgent}
            onCreateLessonFromComment={handleCreateLessonFromComment}
            onManualRetrainCritic={handleManualRetrainCritic}
          />
        )}

        {activeModule === 'studio' && (
          <ModulePromptStudio
            dataset={dataset}
            activeSplit={activeSplit}
            calibration={calibration}
            diagnosticReport={diagnosticReport}
            spriteIndex={spriteIndex}
            isIndexing={isIndexing}
            indexProgress={indexProgress}
            lessons={lessons}
            activeAgent={agents.find((a) => a.isActive) ?? agents[0] ?? null}
            championsByCategory={championsByCategory}
            generatedSprites={generatedSprites}
            approvals={approvals}
            promptHistory={promptHistory}
            t3Vocabulary={t3Vocabulary}
            t3ModelPackage={t3ModelPackage}
            onGoToUpload={() => setActiveModule('upload')}
            onRunIndexing={handleRunIndexing}
            onSavePromptVersion={handleSavePromptVersion}
            onSaveGeneratedBatch={handleSaveGeneratedBatch}
            onRecordApproval={handleRecordApproval}
            onDeleteGenerated={handleDeleteGenerated}
            onCreateLessonFromComment={handleCreateLessonFromComment}
            onUpdateT3VocabRole={handleUpdateT3VocabRole}
            onRebuildT3Vocab={handleRebuildT3Vocab}
            onImportT3ModelPackage={handleImportT3ModelPackage}
            onSaveT3TestEvaluation={handleSaveT3TestEvaluation}
            onToggleT3Enabled={handleToggleT3Enabled}
          />
        )}
      </main>
    </div>
  );
}
