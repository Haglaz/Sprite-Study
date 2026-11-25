import { useMemo, useState, type ChangeEvent } from 'react';
import {
  AlertTriangle,
  Archive,
  CheckCircle2,
  Cpu,
  Download,
  FileCode2,
  FlaskConical,
  Loader2,
  Palette,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Upload,
} from 'lucide-react';
import type { CriticCalibration } from '../critic/types';
import type { StoredDatasetSplit } from '../db/schema';
import { buildTrainingDatasetExportZip } from '../export/trainingDatasetExport';
import type {
  ApprovalLabelRecord,
  GeneratedSpriteRecord,
  SpriteIndexState,
} from '../generation/types';
import type { LessonRecord } from '../lessons/types';
import { buildT3ColabNotebook } from '../t3/colabNotebook';
import { runT3SamplingAndRanking } from '../t3/sampler';
import { evaluateT3OnTestSplit } from '../t3/testEvaluation';
import {
  T3_MATERIAL_ROLES,
  T3_ROLE_LABELS_PT,
  type T3MaterialRole,
  type T3ModelPackageRecord,
  type T3RampVocabulary,
  type T3SamplingBatchResult,
  type T3SamplingCandidate,
  type T3TestSplitComparisonReport,
} from '../t3/types';
import { createWorkerT3Client } from '../t3/workerClient';
import { ALL_ITEM_CATEGORIES, type ItemCategory, type LocalDatasetSummary } from '../types/studio';
import { SPRITE_SIZE } from '../utils/tibiaParser';
import { SpriteThumb, type ThumbBackdrop } from './SpriteThumb';

interface T3WorkbenchPanelProps {
  dataset: LocalDatasetSummary;
  activeSplit: StoredDatasetSplit;
  calibration: CriticCalibration;
  spriteIndex: SpriteIndexState | null;
  lessons: LessonRecord[];
  generatedSprites: GeneratedSpriteRecord[];
  approvals: ApprovalLabelRecord[];
  vocabulary: T3RampVocabulary | null;
  modelPackage: T3ModelPackageRecord | null;
  backdrop: ThumbBackdrop;
  onEnsureIndexReady: () => Promise<SpriteIndexState | null>;
  onUpdateVocabRole: (tokenId: number, role: T3MaterialRole) => Promise<void>;
  onRebuildVocab: () => Promise<void>;
  onImportModelPackage: (params: {
    modelCardJson: string;
    generatorOnnx: ArrayBuffer;
    rerankerOnnx: ArrayBuffer | null;
    generatorFileName: string;
    rerankerFileName: string | null;
  }) => Promise<T3ModelPackageRecord>;
  onSaveTestEvaluation: (report: T3TestSplitComparisonReport) => Promise<void>;
  onToggleT3Enabled: (enabled: boolean) => Promise<void>;
  onSaveGeneratedBatch: (records: GeneratedSpriteRecord[]) => Promise<void>;
}

const ROLE_BADGE_CLASS: Record<T3MaterialRole, string> = {
  outline: 'bg-[#19141C] text-[#D8B4FE] border-[#3F294D]',
  shadow: 'bg-[#131926] text-[#93C5FD] border-[#253657]',
  base: 'bg-[#16201D] text-[#86EFAC] border-[#23473B]',
  light: 'bg-[#262014] text-[#FCD34D] border-[#5B451B]',
  highlight: 'bg-[#291C16] text-[#FDBA74] border-[#663A24]',
};

function triggerTextOrBytesDownload(
  content: string | Uint8Array,
  mimeType: string,
  filename: string,
): void {
  let blob: Blob;
  if (typeof content === 'string') {
    blob = new Blob([content], { type: mimeType });
  } else {
    const copy = new Uint8Array(content.length);
    copy.set(content);
    blob = new Blob([copy.buffer], { type: mimeType });
  }
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

function extractPaletteHexFromRgba(rgba: Uint8ClampedArray): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (let p = 0; p < SPRITE_SIZE * SPRITE_SIZE; p++) {
    const i = p * 4;
    if (rgba[i + 3] < 128) continue;
    const hex =
      '#' +
      [rgba[i], rgba[i + 1], rgba[i + 2]]
        .map((v) => v.toString(16).padStart(2, '0'))
        .join('')
        .toUpperCase();
    if (!seen.has(hex)) {
      seen.add(hex);
      out.push(hex);
    }
  }
  return out;
}

export function T3WorkbenchPanel({
  dataset,
  activeSplit,
  calibration,
  spriteIndex,
  lessons,
  generatedSprites,
  approvals,
  vocabulary,
  modelPackage,
  backdrop,
  onEnsureIndexReady,
  onUpdateVocabRole,
  onRebuildVocab,
  onImportModelPackage,
  onSaveTestEvaluation,
  onToggleT3Enabled,
  onSaveGeneratedBatch,
}: T3WorkbenchPanelProps) {
  const [selectedMaterial, setSelectedMaterial] = useState<string>('all');
  const [updatingTokenId, setUpdatingTokenId] = useState<number | null>(null);
  const [isRebuildingVocab, setIsRebuildingVocab] = useState(false);

  const [isExportingRetrain, setIsExportingRetrain] = useState(false);
  const [retrainNotice, setRetrainNotice] = useState<string | null>(null);

  const [isImportingModel, setIsImportingModel] = useState(false);
  const [importFilesSummary, setImportFilesSummary] = useState<string | null>(null);

  const [t3Category, setT3Category] = useState<ItemCategory>('Equipamentos');
  const [t3Prompt, setT3Prompt] = useState<string>(
    'armadura aco dourado paladino contorno limpo',
  );
  const [t3CandidateCount, setT3CandidateCount] = useState<number>(2);
  const [isSamplingT3, setIsSamplingT3] = useState(false);
  const [samplingBatch, setSamplingBatch] = useState<T3SamplingBatchResult | null>(null);
  const [savedCandidateIds, setSavedCandidateIds] = useState<Set<number>>(new Set());

  const [isEvaluatingTest, setIsEvaluatingTest] = useState(false);
  const [panelError, setPanelError] = useState<string | null>(null);
  const [panelNotice, setPanelNotice] = useState<string | null>(null);

  const approvedCount = useMemo(
    () => approvals.filter((a) => a.decision === 'approved').length,
    [approvals],
  );
  const rejectedCount = useMemo(
    () => approvals.filter((a) => a.decision === 'rejected').length,
    [approvals],
  );

  const filteredVocabEntries = useMemo(() => {
    if (!vocabulary) return [];
    if (selectedMaterial === 'all') return vocabulary.entries;
    return vocabulary.entries.filter((e) => e.materialFamily === selectedMaterial);
  }, [vocabulary, selectedMaterial]);

  const splitHashMatches = useMemo(() => {
    if (!vocabulary || !modelPackage) return null;
    return modelPackage.modelCard.splitHash === vocabulary.splitHash;
  }, [vocabulary, modelPackage]);

  const activeTestReport = modelPackage?.lastTestEvaluation ?? null;

  const handleRoleChange = async (tokenId: number, role: T3MaterialRole) => {
    if (updatingTokenId !== null) return;
    setUpdatingTokenId(tokenId);
    setPanelError(null);
    setPanelNotice(null);
    try {
      await onUpdateVocabRole(tokenId, role);
      setPanelNotice(
        `Papel do token #${tokenId} atualizado para "${role}" e verificado (read-back) no IndexedDB.`,
      );
    } catch (err) {
      setPanelError(err instanceof Error ? err.message : 'Falha ao atualizar papel no vocabulário.');
    } finally {
      setUpdatingTokenId(null);
    }
  };

  const handleDownloadColabNotebook = () => {
    if (!vocabulary) {
      setPanelError('Vocabulário de rampas ainda não foi construído a partir do split train.');
      return;
    }
    setPanelError(null);
    const nb = buildT3ColabNotebook({
      vocabulary,
      exportedHumanLabelsCount: {
        approved: approvedCount,
        rejected: rejectedCount,
      },
    });
    const json = JSON.stringify(nb, null, 2);
    triggerTextOrBytesDownload(
      json,
      'application/x-ipynb+json',
      `atelier_sprite_t3_${vocabulary.splitHash}.ipynb`,
    );
    setPanelNotice(
      `Notebook Google Colab gerado (${nb.cells.length} células, splitHash=${vocabulary.splitHash}, ~7.4M parâmetros, sem flips horizontais).`,
    );
  };

  const handleRetrainExportZip = async () => {
    if (isExportingRetrain) return;
    setIsExportingRetrain(true);
    setPanelError(null);
    setRetrainNotice(null);
    try {
      const idx = await onEnsureIndexReady();
      if (!idx) throw new Error('Indexe o dataset antes de exportar.');

      await new Promise((r) => window.setTimeout(r, 15));
      const result = buildTrainingDatasetExportZip({
        index: idx,
        split: activeSplit,
        vocabulary,
        approvals,
        generatedSprites,
      });

      triggerTextOrBytesDownload(
        result.zipBytes,
        'application/zip',
        `atelier-t3-retreino-${new Date().toISOString().slice(0, 10)}.zip`,
      );

      setRetrainNotice(
        `Pacote "Re-treinar com novos rótulos" exportado: ${result.exportedSpriteCount} sprites base (0 test), vocabulary.json corrigido, + ${result.humanApprovedCount} aprovados e ${result.humanRejectedCount} rejeitados em human_labels.csv.`,
      );
    } catch (err) {
      setPanelError(
        err instanceof Error ? err.message : 'Falha ao exportar pacote de re-treino T3.',
      );
    } finally {
      setIsExportingRetrain(false);
    }
  };

  const handleImportFiles = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    e.target.value = '';
    if (files.length === 0) return;

    setIsImportingModel(true);
    setPanelError(null);
    setPanelNotice(null);

    try {
      let modelCardFile: File | null = null;
      let generatorFile: File | null = null;
      let rerankerFile: File | null = null;

      for (const file of files) {
        const lower = file.name.toLowerCase();
        if (lower.endsWith('.json')) {
          modelCardFile = file;
        } else if (lower.endsWith('.onnx')) {
          if (lower.includes('rerank') || lower.includes('disc')) {
            rerankerFile = file;
          } else if (!generatorFile) {
            generatorFile = file;
          } else if (!rerankerFile) {
            rerankerFile = file;
          }
        }
      }

      if (!modelCardFile) {
        throw new Error(
          'Selecione também o arquivo JSON do model card (ex: t3_model_card.json) junto ao .onnx.',
        );
      }
      if (!generatorFile) {
        throw new Error(
          'Selecione o arquivo ONNX do gerador (ex: t3_generator.onnx). Opcionalmente inclua t3_reranker.onnx.',
        );
      }

      const [cardJson, genBuf, rrBuf] = await Promise.all([
        modelCardFile.text(),
        generatorFile.arrayBuffer(),
        rerankerFile ? rerankerFile.arrayBuffer() : Promise.resolve(null),
      ]);

      const saved = await onImportModelPackage({
        modelCardJson: cardJson,
        generatorOnnx: genBuf,
        rerankerOnnx: rrBuf,
        generatorFileName: generatorFile.name,
        rerankerFileName: rerankerFile ? rerankerFile.name : null,
      });

      setImportFilesSummary(
        `${generatorFile.name} (${(genBuf.byteLength / 1024).toFixed(1)} KB)` +
          (rerankerFile && rrBuf
            ? ` + ${rerankerFile.name} (${(rrBuf.byteLength / 1024).toFixed(1)} KB)`
            : ' (sem reranker)'),
      );
      setSamplingBatch(null);
      setSavedCandidateIds(new Set());
      setPanelNotice(
        `Modelo T3 "${saved.modelCard.version}" (${saved.modelCard.steps} passos) importado e verificado no IndexedDB. Mantido DESATIVADO por padrão até passar na avaliação no split test.`,
      );
    } catch (err) {
      setPanelError(
        err instanceof Error ? err.message : 'Falha ao importar pacote ONNX T3.',
      );
    } finally {
      setIsImportingModel(false);
    }
  };

  const handleRunWorkerSampling = async () => {
    if (!modelPackage || isSamplingT3) return;
    setIsSamplingT3(true);
    setPanelError(null);
    setPanelNotice(null);
    setSavedCandidateIds(new Set());

    let workerClient: Awaited<ReturnType<typeof createWorkerT3Client>> | null = null;
    try {
      const idx = await onEnsureIndexReady();
      if (!idx) throw new Error('Indexe o dataset antes de amostrar com T3.');

      workerClient = await createWorkerT3Client(modelPackage);
      const batch = await runT3SamplingAndRanking({
        runner: workerClient.runner,
        modelCard: modelPackage.modelCard,
        category: t3Category,
        prompt: t3Prompt,
        calibration,
        spriteIndex: idx,
        candidateCount: t3CandidateCount,
        seed: activeSplit.seed + Math.floor(Date.now() % 100000),
      });

      setSamplingBatch(batch);
      setPanelNotice(
        `Amostragem T3 concluída em Web Worker (${batch.executionProvider.toUpperCase()}) em ${batch.samplingMs} ms · ${batch.constraintFailuresResampled} falha(s) estrutural(is) reamostrada(s).`,
      );
    } catch (err) {
      setPanelError(
        err instanceof Error
          ? `Falha na execução ONNX do Worker: ${err.message}`
          : 'Falha ao amostrar T3 no Web Worker.',
      );
    } finally {
      if (workerClient) {
        await workerClient.dispose();
      }
      setIsSamplingT3(false);
    }
  };

  const handleSaveT3CandidateToGallery = async (cand: T3SamplingCandidate) => {
    if (!modelPackage || !cand.nearestReal) return;
    const rec: GeneratedSpriteRecord = {
      id: `t3-${Date.now()}-${cand.candidateIndex}`,
      datasetKey: activeSplit.datasetKey,
      prompt: `${t3Prompt} [T3 ${modelPackage.modelCard.version}]`,
      category: t3Category,
      createdAt: new Date().toISOString(),
      rgba: cand.rgba,
      paletteHex: extractPaletteHexFromRgba(cand.rgba),
      evaluation: cand.evaluation,
      nearestReal: cand.nearestReal,
      provenance: {
        source: 'onnx-t3',
        tier: 'T3',
        model: modelPackage.modelCard.version,
        fallbackWarning: cand.memorizedCopyWarning
          ? 'Aviso de memorização: cópia muito próxima do split de treino.'
          : null,
        donorSpriteIds: [cand.nearestReal.spriteId],
        diffPixelCount: Math.round(cand.nearestReal.pixelDiffRatio * 1024),
        diffPixelLimit: 1024,
        repaired: false,
        repairDelta: 0,
        repairMethod: 'none',
        appliedLessonIds: [],
        seed: activeSplit.seed + cand.candidateIndex,
      },
      approvalStatus: 'pending',
      readBackVerified: true,
    };

    await onSaveGeneratedBatch([rec]);
    setSavedCandidateIds((prev) => new Set(prev).add(cand.candidateIndex));
    setPanelNotice(
      `Candidato T3 #${cand.candidateIndex + 1} salvo no histórico do Estúdio para curadoria (Aprovar/Rejeitar).`,
    );
  };

  const handleEvaluateTestSplit = async () => {
    if (!modelPackage || isEvaluatingTest) return;
    setIsEvaluatingTest(true);
    setPanelError(null);
    setPanelNotice(null);

    let workerClient: Awaited<ReturnType<typeof createWorkerT3Client>> | null = null;
    try {
      const idx = await onEnsureIndexReady();
      if (!idx) throw new Error('Indexe o dataset antes da avaliação no split test.');

      workerClient = await createWorkerT3Client(modelPackage);
      const report = await evaluateT3OnTestSplit({
        runner: workerClient.runner,
        modelCard: modelPackage.modelCard,
        dat: dataset.dat,
        spr: dataset.spr,
        split: activeSplit,
        calibration,
        spriteIndex: idx,
        lessons,
        maxRealTestSprites: 36,
        maxPromptCount: 4,
      });

      await onSaveTestEvaluation(report);
      setPanelNotice(report.verdictSummaryPt);
    } catch (err) {
      setPanelError(
        err instanceof Error ? err.message : 'Falha ao avaliar T3 no split test.',
      );
    } finally {
      if (workerClient) {
        await workerClient.dispose();
      }
      setIsEvaluatingTest(false);
    }
  };

  return (
    <div className="space-y-6" data-testid="t3-workbench">
      {/* Feedback banners */}
      {panelError && (
        <div
          data-testid="t3-error-banner"
          className="flex items-start gap-2.5 px-4 py-3 rounded-lg bg-[#231417] border border-[#5E2328] text-sm text-[#FCA5A5]"
        >
          <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
          <span>{panelError}</span>
        </div>
      )}
      {panelNotice && (
        <div
          data-testid="t3-notice-banner"
          className="flex items-start gap-2.5 px-4 py-3 rounded-lg bg-[#111E17] border border-[#1F4A31] text-xs text-[#86EFAC]"
        >
          <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0" />
          <span>{panelNotice}</span>
        </div>
      )}

      {/* A) Material-clustered Ramp-Role Vocabulary */}
      <section
        data-testid="t3-vocab-section"
        className="bg-[#12151C] border border-[#1E232D] rounded-xl p-5 space-y-4"
      >
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Palette className="w-4 h-4 text-[#E5A93C]" />
              <h2 className="text-base font-semibold text-[#F3F5F8]">
                A. Vocabulário de Papéis de Rampa (Clusterizado por Material · apenas train)
              </h2>
            </div>
            <p className="text-xs text-[#98A2B3]">
              Agrupa cores reais do split <code className="font-mono text-[#CBD3E1]">train</code>{' '}
              por família de material e atribui{' '}
              <code className="font-mono text-[#CBD3E1]">
                outline · shadow · base · light · highlight
              </code>
              . Corrija papéis abaixo se algum cluster estiver trocado.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 shrink-0">
            {vocabulary && (
              <div className="px-3 py-1.5 rounded-lg bg-[#0D0F13] border border-[#242A36] text-[11px] font-mono text-[#CBD3E1] tabular-nums">
                {vocabulary.vocabSize} tokens (0=transp) · split{' '}
                <span className="text-[#E5A93C]">{vocabulary.splitHash}</span> ·{' '}
                <span data-testid="t3-vocab-override-count" className="text-[#86EFAC]">
                  {vocabulary.overrideCount} correção(ões)
                </span>
              </div>
            )}
            <button
              type="button"
              disabled={isRebuildingVocab}
              onClick={async () => {
                setIsRebuildingVocab(true);
                setPanelError(null);
                try {
                  await onRebuildVocab();
                  setPanelNotice('Vocabulário recalculado a partir do split train e salvo no IDB.');
                } finally {
                  setIsRebuildingVocab(false);
                }
              }}
              data-testid="t3-rebuild-vocab-btn"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-[#151921] border border-[#29303E] text-[#CBD3E1] hover:border-[#E5A93C] disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRebuildingVocab ? 'animate-spin' : ''}`} />
              <span>Recalcular (train)</span>
            </button>
          </div>
        </div>

        {vocabulary ? (
          <>
            {/* Material filter strip */}
            <div className="flex flex-wrap items-center gap-1.5 text-xs">
              <button
                type="button"
                onClick={() => setSelectedMaterial('all')}
                className={`px-2.5 py-1 rounded border transition-colors ${
                  selectedMaterial === 'all'
                    ? 'bg-[#1D2432] border-[#E5A93C] text-[#F3F5F8]'
                    : 'bg-[#0D0F13] border-[#242A36] text-[#8C97AB] hover:text-[#F3F5F8]'
                }`}
              >
                Todos os materiais ({vocabulary.entries.length})
              </button>
              {vocabulary.materialSummaries.map((mat) => (
                <button
                  key={mat.materialFamily}
                  type="button"
                  onClick={() => setSelectedMaterial(mat.materialFamily)}
                  className={`px-2.5 py-1 rounded border font-mono text-[11px] transition-colors ${
                    selectedMaterial === mat.materialFamily
                      ? 'bg-[#1D2432] border-[#E5A93C] text-[#F3F5F8]'
                      : 'bg-[#0D0F13] border-[#242A36] text-[#8C97AB] hover:text-[#F3F5F8]'
                  }`}
                >
                  {mat.materialFamily} ({mat.tokenCount})
                </button>
              ))}
            </div>

            {/* Token grid with editable role selectors */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-2.5 max-h-80 overflow-y-auto pr-1">
              {filteredVocabEntries.map((entry) => (
                <div
                  key={entry.tokenId}
                  data-testid={`t3-vocab-row-${entry.tokenId}`}
                  className="flex items-center justify-between gap-2.5 px-3 py-2 rounded-lg bg-[#0D0F13] border border-[#1E2430]"
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <span
                      className="w-6 h-6 rounded border border-[#343C4D] shrink-0"
                      style={{ backgroundColor: entry.hex }}
                      title={`${entry.hex} · RGB(${entry.rgb.join(',')})`}
                    />
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5 text-xs font-mono text-[#F3F5F8]">
                        <span>#{entry.tokenId}</span>
                        <span className="text-[#8C97AB]">{entry.hex}</span>
                        {entry.userOverridden && (
                          <span className="text-[10px] px-1 rounded bg-[#262014] text-[#FCD34D]">
                            editado
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] font-mono text-[#6F7A8F] truncate">
                        {entry.materialFamily} · L {(entry.luma * 100).toFixed(0)}% · borda{' '}
                        {Math.round(entry.perimeterRatio * 100)}%
                      </div>
                    </div>
                  </div>

                  <select
                    aria-label={`Papel do token ${entry.tokenId} (${entry.hex})`}
                    data-testid={`t3-role-select-${entry.tokenId}`}
                    value={entry.role}
                    disabled={updatingTokenId === entry.tokenId}
                    onChange={(ev) =>
                      void handleRoleChange(entry.tokenId, ev.target.value as T3MaterialRole)
                    }
                    title={T3_ROLE_LABELS_PT[entry.role]}
                    className={`px-2 py-1 text-[11px] font-mono rounded border shrink-0 focus:outline-none focus:border-[#E5A93C] ${ROLE_BADGE_CLASS[entry.role]}`}
                  >
                    {T3_MATERIAL_ROLES.map((role) => (
                      <option key={role} value={role} className="bg-[#0D0F13] text-[#F3F5F8]">
                        {role}
                      </option>
                    ))}
                  </select>
                </div>
              ))}
            </div>
          </>
        ) : (
          <p className="text-xs text-[#8C97AB]">Construindo vocabulário de rampas...</p>
        )}
      </section>

      {/* B, C & F) Colab Notebook Export & Retrain Export */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <section className="bg-[#12151C] border border-[#1E232D] rounded-xl p-5 space-y-3.5">
          <div className="flex items-start justify-between gap-3">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <FileCode2 className="w-4 h-4 text-[#60A5FA]" />
                <h2 className="text-base font-semibold text-[#F3F5F8]">
                  B / C. Notebook Google Colab (PyTorch · GPU T4 Gratuita)
                </h2>
              </div>
              <p className="text-xs text-[#98A2B3]">
                Transformer mascarado condicional (~7.4M params) + Discriminador/Reranker com
                relatório de AUC e exportação dupla em ONNX + <code className="font-mono">t3_model_card.json</code>.
              </p>
            </div>
          </div>

          <ul className="text-xs text-[#CBD3E1] space-y-1 font-mono bg-[#0D0F13] border border-[#1B202A] rounded-lg p-3">
            <li>• Grid 32×32 indexado ({vocabulary?.vocabSize ?? '—'} tokens c/ alpha) + CFG</li>
            <li>• Augs permitidas: translação inteira + troca de paleta preservando rampa</li>
            <li>• Proibido: flip horizontal (ALLOW_HORIZONTAL_FLIP = False · preserva luz NW)</li>
            <li>• Splits: treino apenas train, validação em validation, ZERO test</li>
            <li>• Logs por época (perda + higiene), contact sheets e checkpoint no Drive</li>
            <li>• Célula 1: passos exatos Colab, tempo (~18-27m) e recuperação de queda</li>
          </ul>

          <button
            type="button"
            onClick={handleDownloadColabNotebook}
            disabled={!vocabulary}
            data-testid="t3-download-notebook-btn"
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg bg-[#1D4ED8] text-white hover:bg-[#2563EB] disabled:opacity-50 transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Baixar Notebook Colab (.ipynb)</span>
          </button>
        </section>

        <section className="bg-[#12151C] border border-[#1E232D] rounded-xl p-5 space-y-3.5">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Archive className="w-4 h-4 text-[#E5A93C]" />
              <h2 className="text-base font-semibold text-[#F3F5F8]">
                F. Re-treinar com novos rótulos
              </h2>
            </div>
            <p className="text-xs text-[#98A2B3]">
              Empacota o dataset <code className="font-mono text-[#CBD3E1]">train/validation</code>{' '}
              (sem <code className="font-mono text-[#CBD3E1]">test</code>), o{' '}
              <code className="font-mono text-[#CBD3E1]">vocabulary.json</code> corrigido por você
              e todos os seus votos humanos para o próximo ciclo no Colab.
            </p>
          </div>

          <div className="grid grid-cols-3 gap-2.5 text-center bg-[#0D0F13] border border-[#1B202A] rounded-lg p-3 font-mono">
            <div>
              <div className="text-[11px] text-[#6F7A8F]">Aprovados</div>
              <div className="text-lg font-semibold text-[#22C55E] tabular-nums">
                {approvedCount}
              </div>
            </div>
            <div>
              <div className="text-[11px] text-[#6F7A8F]">Rejeitados</div>
              <div className="text-lg font-semibold text-[#EF4444] tabular-nums">
                {rejectedCount}
              </div>
            </div>
            <div>
              <div className="text-[11px] text-[#6F7A8F]">Correções Vocab</div>
              <div className="text-lg font-semibold text-[#FCD34D] tabular-nums">
                {vocabulary?.overrideCount ?? 0}
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <button
              type="button"
              onClick={() => void handleRetrainExportZip()}
              disabled={isExportingRetrain}
              data-testid="t3-retrain-export-btn"
              className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg bg-[#E5A93C] text-[#0D0F13] hover:bg-[#EEB955] disabled:opacity-50 transition-colors"
            >
              {isExportingRetrain ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Archive className="w-3.5 h-3.5" />
              )}
              <span>Re-treinar com novos rótulos (.zip)</span>
            </button>
            {retrainNotice && (
              <p className="text-xs text-[#86EFAC]" data-testid="t3-retrain-summary">
                {retrainNotice}
              </p>
            )}
          </div>
        </section>
      </div>

      {/* D) Importar modelo (T3) & Amostragem Web Worker ONNX */}
      <section className="bg-[#12151C] border border-[#1E232D] rounded-xl p-5 space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3 border-b border-[#1E232D] pb-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-[#22C55E]" />
              <h2 className="text-base font-semibold text-[#F3F5F8]">
                D. Importar modelo (T3) &amp; Amostragem em Web Worker (WebGPU / WASM fallback)
              </h2>
            </div>
            <p className="text-xs text-[#98A2B3]">
              Carregue <code className="font-mono text-[#CBD3E1]">t3_model_card.json</code> +{' '}
              <code className="font-mono text-[#CBD3E1]">t3_generator.onnx</code> (e opcionalmente{' '}
              <code className="font-mono text-[#CBD3E1]">t3_reranker.onnx</code>). Reamostra
              automaticamente se falhar alpha binário, contorno ou silhueta conectada.
            </p>
          </div>

          <label
            data-testid="t3-import-model-btn"
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg bg-[#1A202B] border border-[#323B4D] text-[#F3F5F8] hover:border-[#E5A93C] cursor-pointer shrink-0 transition-colors"
          >
            {isImportingModel ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin text-[#E5A93C]" />
            ) : (
              <Upload className="w-3.5 h-3.5 text-[#E5A93C]" />
            )}
            <span>Importar modelo (T3)</span>
            <input
              type="file"
              multiple
              accept=".onnx,.json"
              onChange={(e) => void handleImportFiles(e)}
              data-testid="t3-import-files-input"
              className="sr-only"
            />
          </label>
        </div>

        {modelPackage ? (
          <div className="space-y-4">
            {/* Active Model Card Summary */}
            <div
              data-testid="t3-model-card-summary"
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 bg-[#0D0F13] border border-[#1B202A] rounded-lg p-3.5 text-xs font-mono"
            >
              <div>
                <div className="text-[#6F7A8F]">Versão &amp; Passos</div>
                <div className="text-[#F3F5F8] font-semibold">
                  {modelPackage.modelCard.version} · {modelPackage.modelCard.steps} steps
                </div>
              </div>
              <div>
                <div className="text-[#6F7A8F]">Split Hash</div>
                <div className={splitHashMatches ? 'text-[#86EFAC]' : 'text-[#FBBF24]'}>
                  {modelPackage.modelCard.splitHash}{' '}
                  {splitHashMatches ? '(compatível)' : '(divergente)'}
                </div>
              </div>
              <div>
                <div className="text-[#6F7A8F]">Reranker AUC (val)</div>
                <div className="text-[#60A5FA] font-semibold">
                  {modelPackage.modelCard.rerankerMetrics?.heldOutValAuc !== undefined
                    ? modelPackage.modelCard.rerankerMetrics.heldOutValAuc.toFixed(3)
                    : '—'}
                </div>
              </div>
              <div>
                <div className="text-[#6F7A8F]">Estado Padrão</div>
                <div
                  data-testid="t3-enabled-status"
                  className={modelPackage.enabled ? 'text-[#22C55E] font-semibold' : 'text-[#FCD34D] font-semibold'}
                >
                  {modelPackage.enabled ? 'ATIVO (aprovado em test)' : 'DESATIVADO por padrão'}
                </div>
              </div>
            </div>

            {importFilesSummary && (
              <div className="text-[11px] font-mono text-[#8C97AB]">
                Arquivos carregados: {importFilesSummary}
              </div>
            )}

            {/* Status / Disable explanation + conditional toggle */}
            <div className="flex flex-wrap items-center justify-between gap-3 px-3.5 py-2.5 rounded-lg bg-[#161922] border border-[#262C38] text-xs">
              <span className="text-[#CBD3E1]">
                {modelPackage.enabled
                  ? 'T3 superou ou igualou os baselines T1/T2 no split test.'
                  : modelPackage.disabledReasonPt ??
                    'T3 mantido desativado por padrão até passar na avaliação contra T1/T2 no split test.'}
              </span>
              {activeTestReport && !activeTestReport.t3IsWorse && (
                <button
                  type="button"
                  onClick={() => void onToggleT3Enabled(!modelPackage.enabled)}
                  className="px-3 py-1 rounded border border-[#2F394B] text-xs font-medium text-[#F3F5F8] hover:border-[#E5A93C]"
                >
                  {modelPackage.enabled ? 'Desativar T3' : 'Ativar T3'}
                </button>
              )}
            </div>

            {/* Sampling Runner Controls */}
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 items-end">
              <label className="space-y-1">
                <span className="block text-xs text-[#98A2B3]">Categoria</span>
                <select
                  value={t3Category}
                  onChange={(e) => setT3Category(e.target.value as ItemCategory)}
                  className="w-full px-3 py-2 text-xs rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8]"
                >
                  {ALL_ITEM_CATEGORIES.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </label>

              <label className="sm:col-span-2 space-y-1">
                <span className="block text-xs text-[#98A2B3]">Tags / Prompt T3</span>
                <input
                  type="text"
                  value={t3Prompt}
                  onChange={(e) => setT3Prompt(e.target.value)}
                  className="w-full px-3 py-2 text-xs rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8]"
                />
              </label>

              <div className="flex items-center gap-2">
                <select
                  aria-label="Candidatos T3"
                  value={t3CandidateCount}
                  onChange={(e) => setT3CandidateCount(Number(e.target.value))}
                  className="px-2.5 py-2 text-xs rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8]"
                >
                  <option value={2}>2 cand.</option>
                  <option value={4}>4 cand.</option>
                </select>

                <button
                  type="button"
                  onClick={() => void handleRunWorkerSampling()}
                  disabled={isSamplingT3 || isEvaluatingTest}
                  data-testid="t3-sample-worker-btn"
                  className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-semibold rounded-lg bg-[#22C55E] text-[#09120D] hover:bg-[#34D36D] disabled:opacity-50"
                >
                  {isSamplingT3 ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Sparkles className="w-3.5 h-3.5" />
                  )}
                  <span>Amostrar Worker</span>
                </button>
              </div>
            </div>

            {/* Worker Sampling Results */}
            {samplingBatch && (
              <div className="space-y-3" data-testid="t3-sampling-results">
                <div className="flex flex-wrap items-center gap-3 text-xs font-mono text-[#98A2B3]">
                  <span data-testid="t3-worker-provider" className="text-[#86EFAC]">
                    Backend ONNX: {samplingBatch.executionProvider.toUpperCase()}
                  </span>
                  <span>·</span>
                  <span>{samplingBatch.samplingMs} ms</span>
                  <span>·</span>
                  <span>
                    Reamostragens por falha estrutural: {samplingBatch.constraintFailuresResampled}
                  </span>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                  {samplingBatch.candidates.map((cand) => {
                    const isSaved = savedCandidateIds.has(cand.candidateIndex);
                    return (
                      <article
                        key={cand.candidateIndex}
                        data-testid="t3-candidate-card"
                        className="bg-[#0D0F13] border border-[#1F2532] rounded-lg p-3.5 space-y-3"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex items-center gap-3">
                            <SpriteThumb
                              rgba={cand.rgba}
                              scale={3}
                              backdrop={backdrop}
                              alt={`Candidato T3 #${cand.candidateIndex + 1}`}
                            />
                            {cand.nearestReal && (
                              <SpriteThumb
                                rgba={cand.nearestReal.rgba}
                                scale={3}
                                backdrop={backdrop}
                                alt={`Real mais próximo #${cand.nearestReal.spriteId}`}
                              />
                            )}
                          </div>

                          <div className="text-right font-mono text-xs space-y-0.5 tabular-nums">
                            <div className="text-[#F3F5F8] font-semibold">
                              Rank {cand.combinedRankScore.toFixed(1)} · Crítico{' '}
                              {cand.evaluation.score.toFixed(1)}
                            </div>
                            <div className="text-[#60A5FA]">
                              Reranker:{' '}
                              {cand.rerankerProbability !== null
                                ? `${(cand.rerankerProbability * 100).toFixed(1)}%`
                                : 'não carregado'}
                            </div>
                            <div className="text-[#8C97AB]">
                              Tentativas: {cand.resampleAttemptsUsed}
                            </div>
                          </div>
                        </div>

                        <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono">
                          <span
                            className={
                              cand.constraints.passed ? 'text-[#86EFAC]' : 'text-[#FCA5A5]'
                            }
                          >
                            {cand.constraints.passed
                              ? 'Restrições OK (alpha bin · contorno · silhueta)'
                              : `Falhas: ${cand.constraints.failures.join(', ')}`}
                          </span>

                          {cand.memorizedCopyWarning ? (
                            <span
                              data-testid="t3-memorization-badge"
                              className="inline-flex items-center gap-1 text-[#FCA5A5]"
                            >
                              <ShieldAlert className="w-3.5 h-3.5" />
                              Cópia memorizada! ({cand.nearestReal?.originalityScore.toFixed(1)}%)
                            </span>
                          ) : (
                            <span
                              data-testid="t3-memorization-badge"
                              className="inline-flex items-center gap-1 text-[#86EFAC]"
                            >
                              <ShieldCheck className="w-3.5 h-3.5" />
                              Originalidade {cand.nearestReal?.originalityScore.toFixed(1) ?? '—'}%
                            </span>
                          )}
                        </div>

                        <div className="flex justify-end">
                          <button
                            type="button"
                            disabled={isSaved || !cand.nearestReal}
                            onClick={() => void handleSaveT3CandidateToGallery(cand)}
                            className="px-3 py-1 text-xs font-medium rounded border border-[#2B3548] text-[#CBD3E1] hover:border-[#E5A93C] disabled:opacity-50"
                          >
                            {isSaved ? 'Salvo no histórico ✓' : 'Enviar para curadoria (A/R)'}
                          </button>
                        </div>
                      </article>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="rounded-lg bg-[#0D0F13] border border-[#1B202A] p-4 text-xs text-[#8C97AB]">
            Nenhum modelo ONNX T3 importado ainda. Treine no Google Colab com o notebook acima e
            importe <code className="font-mono text-[#CBD3E1]">t3_model_card.json</code> +{' '}
            <code className="font-mono text-[#CBD3E1]">t3_generator.onnx</code> (opcional{' '}
            <code className="font-mono text-[#CBD3E1]">t3_reranker.onnx</code>).
          </div>
        )}
      </section>

      {/* E) Explicit Comparison Report on Held-out Test Split */}
      <section
        data-testid="t3-test-eval-section"
        className="bg-[#12151C] border border-[#1E232D] rounded-xl p-5 space-y-4"
      >
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <FlaskConical className="w-4 h-4 text-[#FBBF24]" />
              <h2 className="text-base font-semibold text-[#F3F5F8]">
                E. Comparação Honesta no Split <code className="font-mono">test</code> (T3 vs T1 vs
                T2 vs Piso Lixo)
              </h2>
            </div>
            <p className="text-xs text-[#98A2B3]">
              Avalia o modelo T3 importado no split <code className="font-mono">test</code>{' '}
              intocado. Se T3 for pior que T1/T2 ou ficar perto do piso lixo, permanece desativado
              por padrão.
            </p>
          </div>

          <button
            type="button"
            disabled={!modelPackage || isEvaluatingTest || isSamplingT3}
            onClick={() => void handleEvaluateTestSplit()}
            data-testid="t3-eval-test-btn"
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg bg-[#1A202B] border border-[#323B4D] text-[#F3F5F8] hover:border-[#E5A93C] disabled:opacity-50 shrink-0"
          >
            {isEvaluatingTest ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin text-[#E5A93C]" />
            ) : (
              <FlaskConical className="w-3.5 h-3.5 text-[#FBBF24]" />
            )}
            <span>Avaliar T3 vs T1/T2 no split test</span>
          </button>
        </div>

        {activeTestReport && (
          <div className="space-y-3">
            <div
              data-testid="t3-test-verdict-banner"
              className={`px-4 py-3 rounded-lg border text-xs font-medium ${
                activeTestReport.t3IsWorse
                  ? 'bg-[#221C11] border-[#5B451B] text-[#FCD34D]'
                  : 'bg-[#111E17] border-[#1F4A31] text-[#86EFAC]'
              }`}
            >
              {activeTestReport.verdictSummaryPt}
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono border-collapse tabular-nums">
                <thead>
                  <tr className="border-b border-[#1E232D] text-[#8C97AB]">
                    <th className="py-2 pr-3">Método / Referência</th>
                    <th className="py-2 px-2">N</th>
                    <th className="py-2 px-2">Mediana</th>
                    <th className="py-2 px-2">Média</th>
                    <th className="py-2 px-2">Alpha Bin</th>
                    <th className="py-2 px-2">Contorno</th>
                    <th className="py-2 px-2">Conectada</th>
                    <th className="py-2 px-2">Originalidade</th>
                    <th className="py-2 pl-2">Memorização</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1A1F28] text-[#E6EBF3]">
                  {[
                    activeTestReport.realTest,
                    activeTestReport.t1,
                    activeTestReport.t2,
                    activeTestReport.t3,
                    activeTestReport.junkFloor,
                  ].map((row) => (
                    <tr key={row.label}>
                      <td className="py-2 pr-3 font-sans font-medium">{row.label}</td>
                      <td className="py-2 px-2">{row.sampleCount}</td>
                      <td className="py-2 px-2 font-semibold">{row.medianScore.toFixed(1)}</td>
                      <td className="py-2 px-2">{row.meanScore.toFixed(1)}</td>
                      <td className="py-2 px-2">{Math.round(row.binaryAlphaRate * 100)}%</td>
                      <td className="py-2 px-2">{Math.round(row.outlineClosureRate * 100)}%</td>
                      <td className="py-2 px-2">
                        {Math.round(row.connectedSilhouetteRate * 100)}%
                      </td>
                      <td className="py-2 px-2">
                        {row.meanOriginality !== null ? `${row.meanOriginality.toFixed(1)}%` : '—'}
                      </td>
                      <td className="py-2 pl-2">
                        {row.memorizationRate !== null
                          ? `${Math.round(row.memorizationRate * 100)}%`
                          : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
