import { useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Database,
  Download,
  FileArchive,
  Github,
  HardDrive,
  Loader2,
  RefreshCw,
  ShieldCheck,
  Upload,
} from 'lucide-react';
import type { DiagnosticReport } from '../critic/types';
import type { BackupImportResult, PersistentStorageStatus } from '../db/persistence';
import type { LocalDatasetSummary } from '../types/studio';
import type {
  DiscoveredPeonsoPackage,
  PeonsoFetchProgress,
} from '../utils/peonsoGithubLoader';
import { formatFileSize } from '../utils/pixelImage';
import { DiagnosticPanel } from './DiagnosticPanel';

interface ModuleUploadProps {
  datFile: File | null;
  sprFile: File | null;
  dataset: LocalDatasetSummary | null;
  diagnosticReport: DiagnosticReport | null;
  isRestoring: boolean;
  isParsing: boolean;
  isCalibrating: boolean;
  isFetchingGithub: boolean;
  githubFetchProgress: PeonsoFetchProgress | null;
  githubSourcePkg: DiscoveredPeonsoPackage | null;
  storageStatus: PersistentStorageStatus | null;
  persistenceNotice: string | null;
  error: string | null;
  onFilesSelected: (files: File[]) => void;
  onFetchPeonsoFromGithub: () => void;
  onConfirmReplaceDataset: () => void;
  onExportBackup: () => Promise<void>;
  onImportBackupFile: (file: File) => Promise<BackupImportResult>;
}

function formatPtBrDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString('pt-BR', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

export function ModuleUpload({
  datFile,
  sprFile,
  dataset,
  diagnosticReport,
  isRestoring,
  isParsing,
  isCalibrating,
  isFetchingGithub,
  githubFetchProgress,
  githubSourcePkg,
  storageStatus,
  persistenceNotice,
  error,
  onFilesSelected,
  onFetchPeonsoFromGithub,
  onConfirmReplaceDataset,
  onExportBackup,
  onImportBackupFile,
}: ModuleUploadProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const backupInputRef = useRef<HTMLInputElement | null>(null);

  const [isDragging, setIsDragging] = useState(false);
  const [confirmingReplace, setConfirmingReplace] = useState(false);
  const [showUploadForReplace, setShowUploadForReplace] = useState(false);
  const [isBackupBusy, setIsBackupBusy] = useState(false);
  const [backupMessage, setBackupMessage] = useState<string | null>(null);
  const [backupError, setBackupError] = useState<string | null>(null);

  const busy = isRestoring || isParsing || isCalibrating || isFetchingGithub;
  const shouldShowUploadScreen = !dataset || showUploadForReplace;

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (busy) return;
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      setShowUploadForReplace(false);
      setConfirmingReplace(false);
      onFilesSelected(files);
    }
  };

  const handleExportClick = async () => {
    setIsBackupBusy(true);
    setBackupMessage(null);
    setBackupError(null);
    try {
      await onExportBackup();
      setBackupMessage('Backup exportado em JSON com todos os arquivos e estados do IndexedDB.');
    } catch (err) {
      setBackupError(err instanceof Error ? err.message : 'Falha ao exportar backup.');
    } finally {
      setIsBackupBusy(false);
    }
  };

  const handleImportFileChange = async (file: File | undefined) => {
    if (!file) return;
    setIsBackupBusy(true);
    setBackupMessage(null);
    setBackupError(null);
    try {
      const result = await onImportBackupFile(file);
      setShowUploadForReplace(false);
      setConfirmingReplace(false);
      const summaryParts: string[] = [];
      const c = result.importedCounts;
      if (c.datasets > 0) summaryParts.push(`${c.datasets} dataset`);
      if (c.splits > 0) summaryParts.push(`${c.splits} split`);
      if (c.lessons > 0) summaryParts.push(`${c.lessons} lições`);
      if (c.agents > 0) summaryParts.push(`${c.agents} agentes`);
      if (c.approvals > 0) summaryParts.push(`${c.approvals} aprovações`);
      if (c.generatedSprites > 0) summaryParts.push(`${c.generatedSprites} sprites gerados`);
      if (c.checkpoints > 0) summaryParts.push(`${c.checkpoints} checkpoints`);
      setBackupMessage(
        `Backup importado e verificado (read-back): ${result.totalRecords} registros restaurados${
          summaryParts.length > 0 ? ` (${summaryParts.join(', ')})` : ''
        }.`,
      );
    } catch (err) {
      setBackupError(err instanceof Error ? err.message : 'Falha ao importar backup.');
    } finally {
      setIsBackupBusy(false);
      if (backupInputRef.current) backupInputRef.current.value = '';
    }
  };

  if (isRestoring) {
    return (
      <div className="max-w-3xl mx-auto py-16 text-center space-y-3">
        <Loader2 className="w-7 h-7 animate-spin text-[#E5A93C] mx-auto" aria-hidden="true" />
        <p className="text-sm text-[#CBD5E1]">
          Verificando IndexedDB local e restaurando pacote .dat/.spr...
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Hidden backup import input */}
      <input
        ref={backupInputRef}
        type="file"
        accept=".json,application/json"
        data-testid="import-backup-input"
        className="hidden"
        aria-label="Importar arquivo de backup JSON"
        onChange={(e) => void handleImportFileChange(e.target.files?.[0])}
      />

      {/* Restored / Active Dataset Top Banner */}
      {dataset && (
        <section
          aria-label="Status do dataset armazenado"
          className="rounded-xl border border-[#232A38] bg-[#12161F] p-4 space-y-3"
        >
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="space-y-1.5 min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-[#22C55E] shrink-0" aria-hidden="true" />
                <span
                  className="text-sm font-semibold text-[#F3F5F8]"
                  data-testid="dataset-status-banner"
                >
                  {dataset.restoredFromDb
                    ? `Dataset restaurado: ${dataset.datFileName}, ${dataset.sprFileName} (${formatPtBrDate(dataset.savedAt)})`
                    : `Dataset ativo e verificado: ${dataset.datFileName}, ${dataset.sprFileName} (${formatPtBrDate(dataset.savedAt)})`}
                </span>
              </div>

              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-[#8C97AB]">
                <span>
                  Versão: <strong className="text-[#CBD5E1]">{dataset.dat.versionHint}</strong>
                </span>
                <span>
                  Itens: <strong className="font-mono text-[#CBD5E1]">{dataset.dat.items.length.toLocaleString('pt-BR')}</strong>
                </span>
                <span>
                  Sprites não-vazios:{' '}
                  <strong className="font-mono text-[#CBD5E1]">
                    {dataset.spr.nonEmptySprites.toLocaleString('pt-BR')} / {dataset.spr.spriteCount.toLocaleString('pt-BR')}
                  </strong>
                </span>
                <span className="font-mono text-[11px] text-[#737F95]">
                  SHA-256: {dataset.datSha256.slice(0, 8)}… / {dataset.sprSha256.slice(0, 8)}…
                </span>
              </div>

              <div className="flex flex-wrap items-center gap-3 text-[11px] text-[#7D889D]">
                <span className="inline-flex items-center gap-1 text-[#86EFAC]">
                  <ShieldCheck className="w-3.5 h-3.5" aria-hidden="true" />
                  {persistenceNotice ?? 'Gravado no IndexedDB com verificação read-back'}
                </span>
                {storageStatus && (
                  <span className="inline-flex items-center gap-1">
                    <HardDrive className="w-3.5 h-3.5" aria-hidden="true" />
                    {storageStatus.persisted
                      ? 'Armazenamento persistente concedido'
                      : 'Armazenamento local ativo (exporte backup se mudar de origem)'}
                  </span>
                )}
                {githubSourcePkg && (
                  <span>
                    Origem: <code className="text-[#CBD5E1]">{githubSourcePkg.repo}</code>
                  </span>
                )}
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 shrink-0">
              <button
                type="button"
                data-testid="export-backup-btn"
                onClick={() => void handleExportClick()}
                disabled={busy || isBackupBusy}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-[#2B3344] bg-[#161B26] text-[#D8DFEC] hover:border-[#46536D] disabled:opacity-50 transition-colors"
              >
                <Download className="w-3.5 h-3.5" aria-hidden="true" />
                Exportar backup
              </button>

              <button
                type="button"
                data-testid="import-backup-btn"
                onClick={() => backupInputRef.current?.click()}
                disabled={busy || isBackupBusy}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-[#2B3344] bg-[#161B26] text-[#D8DFEC] hover:border-[#46536D] disabled:opacity-50 transition-colors"
              >
                <Upload className="w-3.5 h-3.5" aria-hidden="true" />
                Importar backup
              </button>

              <button
                type="button"
                onClick={() => setConfirmingReplace(true)}
                disabled={busy || isBackupBusy}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-[#5B451B] bg-[#241D0F] text-[#FDE68A] hover:bg-[#2F2512] disabled:opacity-50 transition-colors"
              >
                <RefreshCw className="w-3.5 h-3.5" aria-hidden="true" />
                Trocar dataset
              </button>
            </div>
          </div>

          {/* Explicit inline confirmation for replacing the stored dataset */}
          {confirmingReplace && (
            <div
              role="alertdialog"
              aria-labelledby="confirm-replace-title"
              className="rounded-lg border border-[#5E2328] bg-[#211214] p-3.5 flex flex-wrap items-center justify-between gap-3"
            >
              <div className="space-y-0.5">
                <h3 id="confirm-replace-title" className="text-xs font-semibold text-[#FCA5A5]">
                  Confirmar troca de dataset
                </h3>
                <p className="text-xs text-[#E2C2C5]">
                  Substituir o par .dat/.spr atual limpará o dataset ativo e exigirá nova calibração do Crítico v1. Exporte um backup antes se desejar preservar este estado.
                </p>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  onClick={() => setConfirmingReplace(false)}
                  className="px-3 py-1.5 text-xs font-medium rounded-lg border border-[#343C4D] bg-[#171C26] text-[#CBD5E1] hover:bg-[#1E2533]"
                >
                  Cancelar
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setConfirmingReplace(false);
                    setShowUploadForReplace(true);
                    onConfirmReplaceDataset();
                  }}
                  className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-[#DC2626] text-white hover:bg-[#B91C1C]"
                >
                  Sim, trocar dataset
                </button>
              </div>
            </div>
          )}
        </section>
      )}

      {/* Alerts */}
      {error && (
        <div
          role="alert"
          className="rounded-xl border border-[#5E2328] bg-[#241315] p-3.5 flex items-start gap-2.5 text-xs text-[#FCA5A5]"
        >
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}

      {backupMessage && (
        <div
          data-testid="backup-status-alert"
          className="rounded-xl border border-[#1F4A31] bg-[#101C16] p-3 flex items-center gap-2 text-xs text-[#86EFAC]"
        >
          <CheckCircle2 className="w-4 h-4 shrink-0" aria-hidden="true" />
          <span>{backupMessage}</span>
        </div>
      )}

      {backupError && (
        <div
          role="alert"
          className="rounded-xl border border-[#5E2328] bg-[#241315] p-3 flex items-center gap-2 text-xs text-[#FCA5A5]"
        >
          <AlertCircle className="w-4 h-4 shrink-0" aria-hidden="true" />
          <span>{backupError}</span>
        </div>
      )}

      {/* Upload Dropzone — shown ONLY when nothing is stored OR after confirming "Trocar dataset" */}
      {shouldShowUploadScreen && (
        <section
          aria-labelledby="upload-section-heading"
          data-testid="dataset-upload-screen"
          className="rounded-xl border border-[#1E232D] bg-[#12151C] p-6 space-y-5"
        >
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <h1
                id="upload-section-heading"
                className="text-lg font-semibold text-[#F3F5F8] tracking-tight"
              >
                Carregar Pacote Tibia (.dat + .spr)
              </h1>
              <p className="text-xs text-[#8C97AB] mt-1 max-w-2xl">
                Os arquivos binários brutos são lidos localmente no navegador, salvos no IndexedDB com verificação read-back e divididos deterministicamente em 70% treino / 15% validação / 15% teste para calibrar o Crítico v1.
              </p>
            </div>

            <button
              type="button"
              onClick={() => backupInputRef.current?.click()}
              disabled={busy || isBackupBusy}
              className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg border border-[#2B3344] bg-[#161B26] text-[#D8DFEC] hover:border-[#46536D] disabled:opacity-50"
            >
              <Upload className="w-3.5 h-3.5" aria-hidden="true" />
              Importar backup JSON
            </button>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept=".dat,.spr"
            multiple
            className="hidden"
            aria-label="Selecionar arquivos .dat e .spr"
            onChange={(e) => {
              const files = Array.from(e.target.files ?? []);
              if (files.length > 0) {
                setShowUploadForReplace(false);
                onFilesSelected(files);
              }
              e.target.value = '';
            }}
          />

          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
              isDragging
                ? 'border-[#E5A93C] bg-[#1E1A12]'
                : 'border-[#29303E] bg-[#0F1218] hover:border-[#3D475B]'
            }`}
          >
            <FileArchive className="w-8 h-8 text-[#E5A93C] mx-auto mb-3" aria-hidden="true" />
            <p className="text-sm font-medium text-[#E8ECF4]">
              Arraste <code className="text-[#E5A93C]">Tibia.dat</code> e{' '}
              <code className="text-[#E5A93C]">Tibia.spr</code> aqui
            </p>
            <p className="text-xs text-[#7D889D] mt-1">
              Suporta formatos Clássico (7.1–8.60) e Extended (9.60–10.98+ / Object Builder)
            </p>

            <div className="mt-5 flex flex-wrap items-center justify-center gap-3">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={busy}
                className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg bg-[#E5A93C] text-[#111318] hover:bg-[#EEB952] disabled:opacity-50 transition-colors"
              >
                <Database className="w-4 h-4" aria-hidden="true" />
                Selecionar .dat e .spr
              </button>

              <button
                type="button"
                onClick={onFetchPeonsoFromGithub}
                disabled={busy}
                className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium rounded-lg border border-[#2E3749] bg-[#161B26] text-[#E8ECF4] hover:border-[#4A5773] disabled:opacity-50 transition-colors"
              >
                {isFetchingGithub ? (
                  <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" />
                ) : (
                  <Github className="w-4 h-4" aria-hidden="true" />
                )}
                Baixar pacote aberto (peonso/GitHub)
              </button>
            </div>

            {(datFile || sprFile) && !dataset && (
              <div className="mt-4 flex flex-wrap items-center justify-center gap-3 text-xs text-[#9AA4B8]">
                <span className={datFile ? 'text-[#86EFAC]' : 'text-[#7D889D]'}>
                  .dat: {datFile ? `${datFile.name} (${formatFileSize(datFile.size)})` : 'pendente'}
                </span>
                <span className={sprFile ? 'text-[#86EFAC]' : 'text-[#7D889D]'}>
                  .spr: {sprFile ? `${sprFile.name} (${formatFileSize(sprFile.size)})` : 'pendente'}
                </span>
              </div>
            )}
          </div>

          {githubFetchProgress && (
            <div className="rounded-lg border border-[#263044] bg-[#131823] p-3 flex items-center gap-2.5 text-xs text-[#CBD5E1]">
              <Loader2 className="w-4 h-4 animate-spin text-[#E5A93C] shrink-0" aria-hidden="true" />
              <span>{githubFetchProgress.message}</span>
            </div>
          )}
        </section>
      )}

      {/* Calibration Spinner */}
      {(isParsing || isCalibrating) && (
        <div className="rounded-xl border border-[#263044] bg-[#12161F] p-5 flex items-center gap-3">
          <Loader2 className="w-5 h-5 animate-spin text-[#E5A93C] shrink-0" aria-hidden="true" />
          <div className="text-xs text-[#CBD5E1]">
            {isParsing
              ? 'Decodificando cabeçalhos .dat/.spr, calculando SHA-256 e salvando no IndexedDB...'
              : 'Minerando distribuições reais do split de treino, treinando discriminador logístico e avaliando teto real vs piso lixo...'}
          </div>
        </div>
      )}

      {/* Honest Critic v1 Diagnostic Panel */}
      {dataset && diagnosticReport && !isCalibrating && (
        <DiagnosticPanel report={diagnosticReport} />
      )}
    </div>
  );
}
