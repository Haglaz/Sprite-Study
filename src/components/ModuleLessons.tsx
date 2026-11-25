import { useMemo, useState } from 'react';
import {
  AlertTriangle,
  Archive,
  BookOpen,
  CheckCircle2,
  Database,
  GitMerge,
  Loader2,
  Plus,
  RotateCcw,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
} from 'lucide-react';
import type { CriticCalibration, CriticIssueCode, DiagnosticReport } from '../critic/types';
import { retrieveLessonsForQuery } from '../lessons/retrieval';
import {
  LESSON_SCOPE_OPTIONS,
  type ConflictResolutionStrategy,
  type LessonCategory,
  type LessonConflict,
  type LessonPolarity,
  type LessonRecord,
  type LessonScopeKey,
  type LessonSource,
} from '../lessons/types';
import { ALL_ITEM_CATEGORIES, type ItemCategory, type LocalDatasetSummary } from '../types/studio';

export interface NewLessonFormPayload {
  category: LessonCategory;
  scopeKey: LessonScopeKey;
  polarity: LessonPolarity;
  title: string;
  directive: string;
  validationCheck: string;
  minVal?: number;
  maxVal?: number;
}

interface ModuleLessonsProps {
  dataset: LocalDatasetSummary | null;
  calibration: CriticCalibration | null;
  diagnosticReport: DiagnosticReport | null;
  lessons: LessonRecord[];
  conflicts: LessonConflict[];
  isBusy: boolean;
  statusMessage: string | null;
  errorMessage: string | null;
  onGoToUpload: () => void;
  onMineFromDataset: () => Promise<void>;
  onCreateLesson: (payload: NewLessonFormPayload) => Promise<void>;
  onRecordEvidence: (lessonId: string, outcome: 'confirm' | 'contest') => Promise<void>;
  onToggleArchive: (lessonId: string, archived: boolean) => Promise<void>;
  onResolveConflict: (
    conflict: LessonConflict,
    strategy: ConflictResolutionStrategy,
    preferredWinnerId?: string,
  ) => Promise<void>;
  onResolveAllConflictsAuto: () => Promise<void>;
}

const SIMULATED_ISSUE_OPTIONS: ReadonlyArray<{ code: CriticIssueCode | ''; labelPt: string }> = [
  { code: '', labelPt: 'Nenhum defeito específico (ranking geral)' },
  { code: 'COLOR_COUNT_OUT_OF_RANGE', labelPt: 'Defeito: Contagem de cores fora da faixa' },
  { code: 'NEAR_DUPLICATE_PALETTE', labelPt: 'Defeito: Cores quase-duplicadas (paleta suja)' },
  { code: 'MISSING_OUTLINE', labelPt: 'Defeito: Contorno escuro ausente' },
  { code: 'OUTLINE_DOUBLES', labelPt: 'Defeito: Cantos duplos ("doubles")' },
  { code: 'WEAK_NW_LIGHTING', labelPt: 'Defeito: Iluminação NW → SE invertida' },
  { code: 'PILLOW_SHADING', labelPt: 'Defeito: Pillow shading radial' },
  { code: 'DITHER_MISMATCH', labelPt: 'Defeito: Dithering fora do padrão real' },
  { code: 'MIXEL_RESAMPLE', labelPt: 'Defeito: Mixels / blocos 2×2 reamostrados' },
  { code: 'SPECKLE_NOISE', labelPt: 'Defeito: Ruído aleatório / speckle' },
  { code: 'TOUCHING_BORDER', labelPt: 'Defeito: Tocando a borda / fora de centro' },
];

function sourceBadgeMeta(source: LessonSource): { label: string; cls: string } {
  if (source === 'human_feedback') {
    return {
      label: 'Feedback Humano',
      cls: 'bg-[#1D2538] border-[#3B5284] text-[#93C5FD]',
    };
  }
  if (source === 'critic_loop') {
    return {
      label: 'Loop Crítico',
      cls: 'bg-[#261E0F] border-[#5B451B] text-[#FDE68A]',
    };
  }
  return {
    label: 'Minerada .spr',
    cls: 'bg-[#102218] border-[#1F4A31] text-[#86EFAC]',
  };
}

function scopeLabel(scopeKey: LessonScopeKey): string {
  const found = LESSON_SCOPE_OPTIONS.find((s) => s.key === scopeKey);
  if (found) return found.labelPt;
  return scopeKey.replace(/^custom:/, 'Personalizado: ');
}

export function ModuleLessons({
  dataset,
  calibration,
  diagnosticReport,
  lessons,
  conflicts,
  isBusy,
  statusMessage,
  errorMessage,
  onGoToUpload,
  onMineFromDataset,
  onCreateLesson,
  onRecordEvidence,
  onToggleArchive,
  onResolveConflict,
  onResolveAllConflictsAuto,
}: ModuleLessonsProps) {
  const [queryCategory, setQueryCategory] = useState<ItemCategory>('Equipamentos');
  const [queryIssue, setQueryIssue] = useState<CriticIssueCode | ''>('');
  const [searchText, setSearchText] = useState('');
  const [statusView, setStatusView] = useState<'ranked' | 'conflicted' | 'superseded' | 'archived'>(
    'ranked',
  );

  // Create lesson form state
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [formCategory, setFormCategory] = useState<LessonCategory>('Equipamentos');
  const [formScope, setFormScope] = useState<LessonScopeKey>('palette_size');
  const [formPolarity, setFormPolarity] = useState<LessonPolarity>('enforce');
  const [formTitle, setFormTitle] = useState('');
  const [formDirective, setFormDirective] = useState('');
  const [formValidation, setFormValidation] = useState('');
  const [formMin, setFormMin] = useState('');
  const [formMax, setFormMax] = useState('');
  const [formError, setFormError] = useState<string | null>(null);

  const counts = useMemo(() => {
    let active = 0;
    let conflicted = 0;
    let superseded = 0;
    let archived = 0;
    let mined = 0;
    let human = 0;
    for (const l of lessons) {
      if (l.status === 'active') active++;
      else if (l.status === 'conflicted') conflicted++;
      else if (l.status === 'superseded') superseded++;
      else if (l.status === 'archived') archived++;
      if (l.source === 'mined') mined++;
      if (l.source === 'human_feedback') human++;
    }
    return { active, conflicted, superseded, archived, mined, human };
  }, [lessons]);

  const retrievedRanked = useMemo(() => {
    return retrieveLessonsForQuery(lessons, {
      category: queryCategory,
      issueCodes: queryIssue ? [queryIssue] : undefined,
      searchText: searchText.trim() || undefined,
      includeConflicted: true,
    });
  }, [lessons, queryCategory, queryIssue, searchText]);

  const filteredNonRanked = useMemo(() => {
    if (statusView === 'ranked') return [];
    const needle = searchText.trim().toLowerCase();
    return lessons
      .filter((l) => l.status === statusView)
      .filter((l) => {
        if (!needle) return true;
        return `${l.title} ${l.directive} ${l.category} ${l.scopeKey}`
          .toLowerCase()
          .includes(needle);
      })
      .sort((a, b) => (Date.parse(b.updatedAt) || 0) - (Date.parse(a.updatedAt) || 0));
  }, [lessons, statusView, searchText]);

  const handleSubmitNewLesson = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormError(null);
    if (!formTitle.trim() || !formDirective.trim() || !formValidation.trim()) {
      setFormError('Preencha Título, Diretriz e Critério de Verificação.');
      return;
    }
    const minNum = formMin.trim() !== '' ? Number(formMin) : undefined;
    const maxNum = formMax.trim() !== '' ? Number(formMax) : undefined;
    if (
      (minNum !== undefined && Number.isNaN(minNum)) ||
      (maxNum !== undefined && Number.isNaN(maxNum))
    ) {
      setFormError('Os valores de faixa mínima e máxima devem ser numéricos.');
      return;
    }
    if (minNum !== undefined && maxNum !== undefined && minNum > maxNum) {
      setFormError('O valor mínimo da faixa não pode ser maior que o máximo.');
      return;
    }

    try {
      await onCreateLesson({
        category: formCategory,
        scopeKey: formScope,
        polarity: formPolarity,
        title: formTitle.trim(),
        directive: formDirective.trim(),
        validationCheck: formValidation.trim(),
        minVal: minNum,
        maxVal: maxNum,
      });
      setFormTitle('');
      setFormDirective('');
      setFormValidation('');
      setFormMin('');
      setFormMax('');
      setShowCreateForm(false);
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Erro ao salvar lição no IndexedDB.');
    }
  };

  return (
    <div className="space-y-6">
      {/* Header & Actions */}
      <section className="rounded-xl border border-[#1E232D] bg-[#12151C] p-5 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex items-start gap-3">
            <BookOpen className="w-5 h-5 text-[#E5A93C] mt-0.5 shrink-0" aria-hidden="true" />
            <div>
              <h1 className="text-lg font-semibold text-[#F3F5F8] tracking-tight">
                Lições Aprendidas (Banco Real em IndexedDB)
              </h1>
              <p className="text-xs text-[#8C97AB] mt-0.5 max-w-2xl">
                Todas as regras são mineradas dos quantis reais do split de treino ou registradas por feedback humano, ranqueadas por especificidade, autoridade, relevância de defeito e confiança Wilson — e só confirmadas após verificação read-back.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => void onMineFromDataset()}
              disabled={!calibration || isBusy}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold rounded-lg bg-[#E5A93C] text-[#111318] hover:bg-[#EEB952] disabled:opacity-50 transition-colors"
            >
              {isBusy ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden="true" />
              ) : (
                <Sparkles className="w-3.5 h-3.5" aria-hidden="true" />
              )}
              Minerar lições do treino (.spr)
            </button>

            <button
              type="button"
              onClick={() => setShowCreateForm((v) => !v)}
              disabled={isBusy}
              className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg border border-[#2E3749] bg-[#161B26] text-[#E8ECF4] hover:border-[#4A5773] disabled:opacity-50 transition-colors"
            >
              <Plus className="w-3.5 h-3.5" aria-hidden="true" />
              Nova lição
            </button>

            {conflicts.length > 0 && (
              <button
                type="button"
                onClick={() => void onResolveAllConflictsAuto()}
                disabled={isBusy}
                className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-semibold rounded-lg border border-[#5B451B] bg-[#261E0F] text-[#FDE68A] hover:bg-[#322712] disabled:opacity-50 transition-colors"
              >
                <GitMerge className="w-3.5 h-3.5" aria-hidden="true" />
                Resolver conflitos automaticamente ({conflicts.length})
              </button>
            )}
          </div>
        </div>

        {/* Summary Counters */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div
            className="rounded-lg border border-[#232A38] bg-[#151922] p-3"
            data-testid="lessons-stat-active"
          >
            <div className="text-[#7D889D]">Lições Ativas</div>
            <div className="mt-1 text-lg font-mono font-semibold text-[#86EFAC]">
              {counts.active}
            </div>
          </div>
          <div
            className={`rounded-lg border p-3 ${
              conflicts.length > 0
                ? 'border-[#5E2328] bg-[#211214]'
                : 'border-[#232A38] bg-[#151922]'
            }`}
            data-testid="lessons-stat-conflicts"
          >
            <div className={conflicts.length > 0 ? 'text-[#FCA5A5]' : 'text-[#7D889D]'}>
              Conflitos Abertos
            </div>
            <div
              className={`mt-1 text-lg font-mono font-semibold ${
                conflicts.length > 0 ? 'text-[#FCA5A5]' : 'text-[#F3F5F8]'
              }`}
            >
              {conflicts.length}
            </div>
          </div>
          <div className="rounded-lg border border-[#232A38] bg-[#151922] p-3">
            <div className="text-[#7D889D]">Mineradas / Humanas</div>
            <div className="mt-1 text-lg font-mono font-semibold text-[#CBD5E1]">
              {counts.mined} / {counts.human}
            </div>
          </div>
          <div className="rounded-lg border border-[#232A38] bg-[#151922] p-3">
            <div className="text-[#7D889D]">Substituídas / Arquivadas</div>
            <div className="mt-1 text-lg font-mono font-semibold text-[#9AA4B8]">
              {counts.superseded} / {counts.archived}
            </div>
          </div>
        </div>

        {!dataset && (
          <div className="rounded-lg border border-[#2B3344] bg-[#161B26] p-3.5 flex flex-wrap items-center justify-between gap-3 text-xs text-[#CBD5E1]">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-[#E5A93C]" aria-hidden="true" />
              <span>
                Nenhum pacote .dat/.spr carregado. Carregue um dataset para minerar regras reais do split de treino, ou registre lições manuais globais.
              </span>
            </div>
            <button
              type="button"
              onClick={onGoToUpload}
              className="px-3 py-1.5 font-semibold rounded-lg bg-[#E5A93C] text-[#111318] hover:bg-[#EEB952]"
            >
              Ir para Pacote .dat/.spr
            </button>
          </div>
        )}

        {statusMessage && (
          <div
            data-testid="lessons-status-notice"
            className="rounded-lg border border-[#1F4A31] bg-[#101C16] p-3 flex items-center gap-2 text-xs text-[#86EFAC]"
          >
            <ShieldCheck className="w-4 h-4 shrink-0" aria-hidden="true" />
            <span>{statusMessage}</span>
          </div>
        )}

        {errorMessage && (
          <div
            role="alert"
            className="rounded-lg border border-[#5E2328] bg-[#241315] p-3 flex items-center gap-2 text-xs text-[#FCA5A5]"
          >
            <AlertTriangle className="w-4 h-4 shrink-0" aria-hidden="true" />
            <span>{errorMessage}</span>
          </div>
        )}
      </section>

      {/* Inline Create Lesson Form */}
      {showCreateForm && (
        <section
          aria-labelledby="new-lesson-heading"
          className="rounded-xl border border-[#2B3548] bg-[#131823] p-5"
        >
          <form onSubmit={(e) => void handleSubmitNewLesson(e)} className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 id="new-lesson-heading" className="text-sm font-semibold text-[#F3F5F8]">
                Registrar Nova Lição (Feedback Humano · Read-Back Verificado)
              </h2>
              <button
                type="button"
                onClick={() => setShowCreateForm(false)}
                className="text-xs text-[#8C97AB] hover:text-[#F1F3F7]"
              >
                Fechar
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <label className="block space-y-1 text-xs">
                <span className="text-[#9AA4B8]">Categoria</span>
                <select
                  value={formCategory}
                  onChange={(e) => setFormCategory(e.target.value as LessonCategory)}
                  className="w-full rounded-lg border border-[#2B3344] bg-[#0F131B] px-3 py-2 text-[#F1F3F7]"
                >
                  {ALL_ITEM_CATEGORIES.map((cat) => (
                    <option key={cat} value={cat}>
                      {cat}
                    </option>
                  ))}
                  <option value="Global">Global (Todas)</option>
                </select>
              </label>

              <label className="block space-y-1 text-xs">
                <span className="text-[#9AA4B8]">Domínio (detecção de conflito)</span>
                <select
                  value={formScope}
                  onChange={(e) => setFormScope(e.target.value as LessonScopeKey)}
                  className="w-full rounded-lg border border-[#2B3344] bg-[#0F131B] px-3 py-2 text-[#F1F3F7]"
                >
                  {LESSON_SCOPE_OPTIONS.map((opt) => (
                    <option key={opt.key} value={opt.key}>
                      {opt.labelPt}
                    </option>
                  ))}
                </select>
              </label>

              <label className="block space-y-1 text-xs">
                <span className="text-[#9AA4B8]">Polaridade</span>
                <select
                  value={formPolarity}
                  onChange={(e) => setFormPolarity(e.target.value as LessonPolarity)}
                  className="w-full rounded-lg border border-[#2B3344] bg-[#0F131B] px-3 py-2 text-[#F1F3F7]"
                >
                  <option value="enforce">EXIGIR (padrão obrigatório)</option>
                  <option value="avoid">EVITAR (defeito proibido)</option>
                </select>
              </label>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <label className="block space-y-1 text-xs">
                <span className="text-[#9AA4B8]">Título curto</span>
                <input
                  type="text"
                  value={formTitle}
                  onChange={(e) => setFormTitle(e.target.value)}
                  placeholder="Ex.: Paleta enxuta em armaduras clássicas"
                  className="w-full rounded-lg border border-[#2B3344] bg-[#0F131B] px-3 py-2 text-[#F1F3F7] placeholder:text-[#5E697D]"
                />
              </label>

              <label className="block space-y-1 text-xs">
                <span className="text-[#9AA4B8]">Critério de verificação no Crítico</span>
                <input
                  type="text"
                  value={formValidation}
                  onChange={(e) => setFormValidation(e.target.value)}
                  placeholder="Ex.: distinctColors <= 12 e outlineCoverage >= 80%"
                  className="w-full rounded-lg border border-[#2B3344] bg-[#0F131B] px-3 py-2 text-[#F1F3F7] placeholder:text-[#5E697D]"
                />
              </label>
            </div>

            <label className="block space-y-1 text-xs">
              <span className="text-[#9AA4B8]">Diretriz prática (pixel-a-pixel)</span>
              <textarea
                rows={2}
                value={formDirective}
                onChange={(e) => setFormDirective(e.target.value)}
                placeholder="Descreva a regra exata de construção de pixels, rampa de cor ou contorno."
                className="w-full rounded-lg border border-[#2B3344] bg-[#0F131B] px-3 py-2 text-[#F1F3F7] placeholder:text-[#5E697D]"
              />
            </label>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <label className="block space-y-1 text-xs">
                <span className="text-[#9AA4B8]">Faixa mín. (opcional)</span>
                <input
                  type="number"
                  step="any"
                  value={formMin}
                  onChange={(e) => setFormMin(e.target.value)}
                  placeholder="Ex.: 6"
                  className="w-full rounded-lg border border-[#2B3344] bg-[#0F131B] px-3 py-1.5 text-[#F1F3F7] font-mono"
                />
              </label>
              <label className="block space-y-1 text-xs">
                <span className="text-[#9AA4B8]">Faixa máx. (opcional)</span>
                <input
                  type="number"
                  step="any"
                  value={formMax}
                  onChange={(e) => setFormMax(e.target.value)}
                  placeholder="Ex.: 14"
                  className="w-full rounded-lg border border-[#2B3344] bg-[#0F131B] px-3 py-1.5 text-[#F1F3F7] font-mono"
                />
              </label>
            </div>

            {formError && (
              <div role="alert" className="text-xs text-[#FCA5A5]">
                {formError}
              </div>
            )}

            <div className="flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowCreateForm(false)}
                className="px-3 py-1.5 text-xs rounded-lg border border-[#2B3344] text-[#CBD5E1] hover:bg-[#19202E]"
              >
                Cancelar
              </button>
              <button
                type="submit"
                disabled={isBusy}
                className="px-4 py-1.5 text-xs font-semibold rounded-lg bg-[#E5A93C] text-[#111318] hover:bg-[#EEB952] disabled:opacity-50"
              >
                Salvar e verificar read-back
              </button>
            </div>
          </form>
        </section>
      )}

      {/* Active Conflicts Panel */}
      {conflicts.length > 0 && (
        <section
          aria-labelledby="conflicts-heading"
          data-testid="lessons-conflicts-panel"
          className="rounded-xl border border-[#5E2328] bg-[#1B1113] p-5 space-y-4"
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-5 h-5 text-[#FCA5A5] shrink-0" aria-hidden="true" />
              <div>
                <h2 id="conflicts-heading" className="text-sm font-semibold text-[#FCA5A5]">
                  Conflitos Detectados entre Lições ({conflicts.length})
                </h2>
                <p className="text-xs text-[#DDBAC0]">
                  Regras concorrentes no mesmo domínio e categoria precisam ser resolvidas para manter o ranking determinístico.
                </p>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            {conflicts.map((conflict) => {
              const canMergeRanges =
                conflict.lessonA.polarity === conflict.lessonB.polarity &&
                Boolean(conflict.lessonA.targetRange && conflict.lessonB.targetRange);

              return (
                <div
                  key={conflict.id}
                  className="rounded-lg border border-[#492025] bg-[#140E10] p-4 space-y-3"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-semibold text-[#FCA5A5]">
                        {conflict.category} · {scopeLabel(conflict.scopeKey)}
                      </span>
                      <span className="text-[#C8A4AA]">— {conflict.reasonLabelPt}</span>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      {canMergeRanges && (
                        <button
                          type="button"
                          disabled={isBusy}
                          onClick={() => void onResolveConflict(conflict, 'merge_range')}
                          className="px-2.5 py-1 rounded border border-[#3B5284] bg-[#182133] text-[#93C5FD] hover:bg-[#1E2B44] text-[11px] font-medium"
                        >
                          Mesclar faixas
                        </button>
                      )}
                      <button
                        type="button"
                        disabled={isBusy}
                        onClick={() => void onResolveConflict(conflict, 'auto_precedence')}
                        className="px-2.5 py-1 rounded border border-[#5B451B] bg-[#261E0F] text-[#FDE68A] hover:bg-[#322712] text-[11px] font-semibold"
                      >
                        Aplicar precedência automática
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {[conflict.lessonA, conflict.lessonB].map((item) => {
                      const src = sourceBadgeMeta(item.source);
                      return (
                        <div
                          key={item.id}
                          className="rounded-lg border border-[#292329] bg-[#12151C] p-3 flex flex-col justify-between gap-2.5"
                        >
                          <div className="space-y-1.5">
                            <div className="flex flex-wrap items-center gap-1.5 text-[10px]">
                              <span className={`px-1.5 py-0.5 rounded border ${src.cls}`}>
                                {src.label}
                              </span>
                              <span
                                className={`px-1.5 py-0.5 rounded border font-semibold ${
                                  item.polarity === 'enforce'
                                    ? 'bg-[#102218] border-[#1F4A31] text-[#86EFAC]'
                                    : 'bg-[#2A1416] border-[#5E2328] text-[#FCA5A5]'
                                }`}
                              >
                                {item.polarity === 'enforce' ? 'EXIGIR' : 'EVITAR'}
                              </span>
                              <span className="font-mono text-[#9AA4B8]">
                                conf {(item.evidence.confidence * 100).toFixed(1)}% · n=
                                {item.evidence.sampleSize}
                              </span>
                            </div>
                            <h3 className="text-xs font-semibold text-[#F3F5F8]">{item.title}</h3>
                            <p className="text-xs text-[#B8C2D4] leading-relaxed">
                              {item.directive}
                            </p>
                          </div>

                          <div className="flex justify-end">
                            <button
                              type="button"
                              disabled={isBusy}
                              onClick={() =>
                                void onResolveConflict(conflict, 'keep_lesson', item.id)
                              }
                              className="px-3 py-1 text-[11px] font-semibold rounded bg-[#E5A93C] text-[#111318] hover:bg-[#EEB952]"
                            >
                              Manter esta regra
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Retrieval Query & Ranked Lessons Explorer */}
      <section className="rounded-xl border border-[#1E232D] bg-[#101319] p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-semibold text-[#F3F5F8]">
              Consulta e Ordenação de Recuperação (Retrieval)
            </h2>
            <p className="text-xs text-[#8C97AB]">
              Simule exatamente quais lições e em qual ordem entram no Crítico e nos geradores para uma categoria e defeito específicos.
            </p>
          </div>

          <div className="inline-flex rounded-lg border border-[#262B35] bg-[#151922] p-0.5 text-xs">
            {(
              [
                { id: 'ranked', label: `Ranking Ativo (${retrievedRanked.length})` },
                { id: 'conflicted', label: `Conflitos (${counts.conflicted})` },
                { id: 'superseded', label: `Substituídas (${counts.superseded})` },
                { id: 'archived', label: `Arquivadas (${counts.archived})` },
              ] as const
            ).map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setStatusView(tab.id)}
                className={`px-2.5 py-1 rounded-md transition-colors ${
                  statusView === tab.id
                    ? 'bg-[#E5A93C] text-[#111318] font-medium'
                    : 'text-[#9AA4B8] hover:text-[#F1F3F7]'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Query Filters */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <label className="block space-y-1 text-xs">
            <span className="text-[#8C97AB]">Categoria alvo da consulta</span>
            <select
              value={queryCategory}
              onChange={(e) => setQueryCategory(e.target.value as ItemCategory)}
              disabled={statusView !== 'ranked'}
              className="w-full rounded-lg border border-[#262B35] bg-[#141821] px-3 py-2 text-[#F1F3F7] disabled:opacity-50"
            >
              {ALL_ITEM_CATEGORIES.map((cat) => (
                <option key={cat} value={cat}>
                  {cat}
                </option>
              ))}
            </select>
          </label>

          <label className="block space-y-1 text-xs">
            <span className="text-[#8C97AB]">Defeito ativo do Crítico v1 (boost)</span>
            <select
              value={queryIssue}
              onChange={(e) => setQueryIssue(e.target.value as CriticIssueCode | '')}
              disabled={statusView !== 'ranked'}
              className="w-full rounded-lg border border-[#262B35] bg-[#141821] px-3 py-2 text-[#F1F3F7] disabled:opacity-50"
            >
              {SIMULATED_ISSUE_OPTIONS.map((opt) => (
                <option key={opt.code || 'none'} value={opt.code}>
                  {opt.labelPt}
                </option>
              ))}
            </select>
          </label>

          <label className="block space-y-1 text-xs">
            <span className="text-[#8C97AB]">Filtrar por palavra-chave</span>
            <div className="relative">
              <Search
                className="w-3.5 h-3.5 text-[#6D788D] absolute left-3 top-1/2 -translate-y-1/2"
                aria-hidden="true"
              />
              <input
                type="text"
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                placeholder="Ex.: contorno, paleta, mixel, NW..."
                className="w-full rounded-lg border border-[#262B35] bg-[#141821] pl-8 pr-3 py-2 text-[#F1F3F7] placeholder:text-[#5E697D]"
              />
            </div>
          </label>
        </div>

        {/* Results List */}
        {statusView === 'ranked' ? (
          retrievedRanked.length === 0 ? (
            <div className="rounded-lg border border-dashed border-[#262B35] p-8 text-center text-xs text-[#8C97AB] space-y-2">
              <p>
                Nenhuma lição ativa encontrada para esta consulta em{' '}
                <strong className="text-[#E8ECF4]">{queryCategory}</strong>.
              </p>
              {calibration && (
                <p>
                  Clique em <strong className="text-[#E5A93C]">Minerar lições do treino (.spr)</strong>{' '}
                  para gerar regras estatísticas diretamente dos{' '}
                  {diagnosticReport?.splitTotals.train ?? 0} sprites reais do split de treino.
                </p>
              )}
            </div>
          ) : (
            <ol className="space-y-2.5" data-testid="ranked-lessons-list">
              {retrievedRanked.map(({ lesson, score }, index) => {
                const src = sourceBadgeMeta(lesson.source);
                return (
                  <li
                    key={lesson.id}
                    data-testid="lesson-card"
                    className={`rounded-lg border p-4 space-y-2.5 ${
                      lesson.status === 'conflicted'
                        ? 'border-[#5E2328] bg-[#191214]'
                        : 'border-[#1E232D] bg-[#12151C]'
                    }`}
                  >
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="space-y-1 min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-1.5 text-[10px]">
                          <span className="font-mono font-semibold px-1.5 py-0.5 rounded bg-[#1E2433] text-[#E5A93C]">
                            #{index + 1} · {score.total.toFixed(1)} pts
                          </span>
                          <span className={`px-1.5 py-0.5 rounded border ${src.cls}`}>
                            {src.label}
                          </span>
                          <span
                            className={`px-1.5 py-0.5 rounded border font-semibold ${
                              lesson.polarity === 'enforce'
                                ? 'bg-[#102218] border-[#1F4A31] text-[#86EFAC]'
                                : 'bg-[#2A1416] border-[#5E2328] text-[#FCA5A5]'
                            }`}
                          >
                            {lesson.polarity === 'enforce' ? 'EXIGIR' : 'EVITAR'}
                          </span>
                          <span className="px-1.5 py-0.5 rounded bg-[#171C26] text-[#CBD5E1]">
                            {lesson.category}
                          </span>
                          <span className="text-[#7D889D]">{scopeLabel(lesson.scopeKey)}</span>
                          {lesson.readBackVerified && (
                            <span className="inline-flex items-center gap-1 text-[#86EFAC]">
                              <CheckCircle2 className="w-3 h-3" aria-hidden="true" />
                              read-back
                            </span>
                          )}
                        </div>
                        <h3 className="text-sm font-medium text-[#F3F5F8]">{lesson.title}</h3>
                      </div>

                      {/* Evidence & Archive controls */}
                      <div className="flex items-center gap-1.5 shrink-0">
                        <button
                          type="button"
                          disabled={isBusy}
                          onClick={() => void onRecordEvidence(lesson.id, 'confirm')}
                          aria-label={`Confirmar lição ${lesson.title}`}
                          className="inline-flex items-center gap-1 px-2 py-1 text-[11px] rounded border border-[#1F4A31] bg-[#101C16] text-[#86EFAC] hover:bg-[#14271E] disabled:opacity-50"
                        >
                          <ThumbsUp className="w-3 h-3" aria-hidden="true" />
                          +1
                        </button>
                        <button
                          type="button"
                          disabled={isBusy}
                          onClick={() => void onRecordEvidence(lesson.id, 'contest')}
                          aria-label={`Contestar lição ${lesson.title}`}
                          className="inline-flex items-center gap-1 px-2 py-1 text-[11px] rounded border border-[#5E2328] bg-[#211214] text-[#FCA5A5] hover:bg-[#2B1619] disabled:opacity-50"
                        >
                          <ThumbsDown className="w-3 h-3" aria-hidden="true" />
                          −1
                        </button>
                        <button
                          type="button"
                          disabled={isBusy}
                          onClick={() => void onToggleArchive(lesson.id, true)}
                          aria-label={`Arquivar lição ${lesson.title}`}
                          className="inline-flex items-center gap-1 px-2 py-1 text-[11px] rounded border border-[#2B3344] bg-[#161B26] text-[#9AA4B8] hover:text-[#F1F3F7] disabled:opacity-50"
                        >
                          <Archive className="w-3 h-3" aria-hidden="true" />
                          Arquivar
                        </button>
                      </div>
                    </div>

                    <p className="text-xs text-[#D5DCE9] leading-relaxed">{lesson.directive}</p>

                    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-[#8C97AB]">
                      <span>
                        Verificação: <code className="text-[#CBD5E1]">{lesson.validationCheck}</code>
                      </span>
                      {lesson.targetRange && (
                        <span className="font-mono text-[#E5A93C]">
                          Faixa: [{lesson.targetRange.min ?? '—'}, {lesson.targetRange.max ?? '—'}]
                          {lesson.targetRange.unit ? ` ${lesson.targetRange.unit}` : ''}
                        </span>
                      )}
                    </div>

                    {/* Honest Empirical Metrics & Score Breakdown */}
                    <div className="pt-2 border-t border-[#1B202A] flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono text-[#8C97AB]">
                      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                        <span>
                          Confiança Wilson:{' '}
                          <strong className="text-[#E8ECF4]">
                            {(lesson.evidence.confidence * 100).toFixed(1)}%
                          </strong>
                        </span>
                        <span>
                          n={lesson.evidence.sampleSize} (+{lesson.evidence.positiveCount} / −
                          {lesson.evidence.negativeCount})
                        </span>
                        <span>
                          Delta Crítico:{' '}
                          {lesson.evidence.criticDeltaMean !== null
                            ? `${lesson.evidence.criticDeltaMean >= 0 ? '+' : ''}${lesson.evidence.criticDeltaMean.toFixed(1)}`
                            : 'não medido'}
                        </span>
                        {lesson.evidence.referenceSpriteIds.length > 0 && (
                          <span>
                            IDs .spr: {lesson.evidence.referenceSpriteIds.slice(0, 4).join(', ')}
                          </span>
                        )}
                      </div>

                      <div className="text-[#727E94]">
                        Score = Cat {score.categoryMatch} + Fonte {score.sourceAuthority} + Defeito{' '}
                        {score.issueRelevance} + Evid {score.evidenceStrength} + Rec {score.recency}
                        {score.textMatch > 0 ? ` + Texto ${score.textMatch}` : ''}
                      </div>
                    </div>
                  </li>
                );
              })}
            </ol>
          )
        ) : filteredNonRanked.length === 0 ? (
          <div className="rounded-lg border border-dashed border-[#262B35] p-6 text-center text-xs text-[#8C97AB]">
            Nenhuma lição no estado selecionado.
          </div>
        ) : (
          <ul className="space-y-2.5">
            {filteredNonRanked.map((lesson) => {
              const src = sourceBadgeMeta(lesson.source);
              return (
                <li
                  key={lesson.id}
                  className="rounded-lg border border-[#1E232D] bg-[#12151C] p-4 flex flex-wrap items-start justify-between gap-3"
                >
                  <div className="space-y-1 min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-1.5 text-[10px]">
                      <span className={`px-1.5 py-0.5 rounded border ${src.cls}`}>{src.label}</span>
                      <span className="px-1.5 py-0.5 rounded bg-[#171C26] text-[#CBD5E1]">
                        {lesson.category}
                      </span>
                      <span className="text-[#7D889D]">{scopeLabel(lesson.scopeKey)}</span>
                      {lesson.supersededById && (
                        <span className="font-mono text-[#FDE68A]">
                          substituída por {lesson.supersededById}
                        </span>
                      )}
                    </div>
                    <h3 className="text-sm font-medium text-[#E8ECF4]">{lesson.title}</h3>
                    <p className="text-xs text-[#9AA4B8]">{lesson.directive}</p>
                  </div>

                  {(lesson.status === 'archived' || lesson.status === 'superseded') && (
                    <button
                      type="button"
                      disabled={isBusy}
                      onClick={() => void onToggleArchive(lesson.id, false)}
                      className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs rounded border border-[#2B3344] bg-[#161B26] text-[#D8DFEC] hover:border-[#46536D]"
                    >
                      <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
                      Reativar
                    </button>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </section>
    </div>
  );
}
