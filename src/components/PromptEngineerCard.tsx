import { useEffect, useState } from 'react';
import {
  Check,
  CheckCircle2,
  Edit3,
  GitCompare,
  History,
  Loader2,
  Plus,
  RefreshCw,
  Sparkles,
  X,
} from 'lucide-react';
import type {
  EngineeredPromptSpec,
  IndexedSpriteEntry,
  PaletteRampSpec,
  PromptDiffChange,
  PromptHistoryRecord,
} from '../generation/types';
import { ALL_ITEM_CATEGORIES, type ItemCategory } from '../types/studio';
import { SpriteThumb, type ThumbBackdrop } from './SpriteThumb';

interface PromptEngineerCardProps {
  activeRecord: PromptHistoryRecord;
  requestVersions: PromptHistoryRecord[];
  diffChanges: PromptDiffChange[];
  donorLookup: Map<number, IndexedSpriteEntry>;
  availableDonors: IndexedSpriteEntry[];
  backdrop: ThumbBackdrop;
  isEngineering: boolean;
  isGenerating: boolean;
  candidateCount: number;
  onSelectVersion: (record: PromptHistoryRecord) => void;
  onRegeneratePrompt: () => void;
  onSaveEditedPrompt: (edited: EngineeredPromptSpec) => Promise<void>;
  onApproveAndGenerate: (spec: EngineeredPromptSpec) => Promise<void>;
}

const HEX_RE = /^#[0-9A-Fa-f]{6}$/;

export function PromptEngineerCard({
  activeRecord,
  requestVersions,
  diffChanges,
  donorLookup,
  availableDonors,
  backdrop,
  isEngineering,
  isGenerating,
  candidateCount,
  onSelectVersion,
  onRegeneratePrompt,
  onSaveEditedPrompt,
  onApproveAndGenerate,
}: PromptEngineerCardProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [draft, setDraft] = useState<EngineeredPromptSpec>(activeRecord.engineeredPrompt);
  const [negativeText, setNegativeText] = useState<string>(
    activeRecord.engineeredPrompt.negativePrompt.join('\n'),
  );
  const [editError, setEditError] = useState<string | null>(null);
  const [isSavingEdit, setIsSavingEdit] = useState(false);

  useEffect(() => {
    setDraft(activeRecord.engineeredPrompt);
    setNegativeText(activeRecord.engineeredPrompt.negativePrompt.join('\n'));
    setEditError(null);
  }, [activeRecord.id, activeRecord.engineeredPrompt]);

  const buildSanitizedDraft = (): EngineeredPromptSpec => {
    const negatives = negativeText
      .split(/\n|;/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    return {
      ...draft,
      itemName: draft.itemName.trim(),
      visualBrief: draft.visualBrief.trim(),
      negativePrompt: negatives.length >= 2 ? negatives : draft.negativePrompt,
    };
  };

  const handleSaveEditClick = async () => {
    setEditError(null);
    const sanitized = buildSanitizedDraft();
    if (sanitized.itemName.length < 2) {
      setEditError('Informe um nome de item com pelo menos 2 caracteres.');
      return;
    }
    if (sanitized.visualBrief.length < 15) {
      setEditError('O brief visual precisa ter pelo menos 15 caracteres.');
      return;
    }
    for (const r of sanitized.paletteRamps) {
      if (r.colors.length < 3 || r.colors.length > 5) {
        setEditError(`A rampa "${r.material}" precisa ter entre 3 e 5 cores.`);
        return;
      }
      if (r.colors.some((c) => !HEX_RE.test(c))) {
        setEditError(`A rampa "${r.material}" contém um código hex inválido (use #RRGGBB).`);
        return;
      }
    }
    if (sanitized.donorSuggestions.length < 3 || sanitized.donorSuggestions.length > 6) {
      setEditError('Selecione entre 3 e 6 sprites doadores reais (train/validation).');
      return;
    }

    setIsSavingEdit(true);
    try {
      await onSaveEditedPrompt(sanitized);
      setIsEditing(false);
    } catch (err) {
      setEditError(err instanceof Error ? err.message : 'Falha ao salvar edição do prompt.');
    } finally {
      setIsSavingEdit(false);
    }
  };

  const handleUpdateRampColor = (rampIdx: number, colorIdx: number, nextHex: string) => {
    const normalized = nextHex.startsWith('#') ? nextHex.toUpperCase() : `#${nextHex.toUpperCase()}`;
    setDraft((prev) => {
      const ramps: PaletteRampSpec[] = prev.paletteRamps.map((r, ri) => {
        if (ri !== rampIdx) return r;
        const nextColors = r.colors.map((c, ci) => (ci === colorIdx ? normalized : c));
        return { ...r, colors: nextColors };
      });
      return { ...prev, paletteRamps: ramps };
    });
  };

  const handleAddRampStop = (rampIdx: number) => {
    setDraft((prev) => {
      const ramps = prev.paletteRamps.map((r, ri) => {
        if (ri !== rampIdx || r.colors.length >= 5) return r;
        return { ...r, colors: [...r.colors, r.colors[r.colors.length - 1] ?? '#B8C4D8'] };
      });
      return { ...prev, paletteRamps: ramps };
    });
  };

  const handleRemoveRampStop = (rampIdx: number, colorIdx: number) => {
    setDraft((prev) => {
      const ramps = prev.paletteRamps.map((r, ri) => {
        if (ri !== rampIdx || r.colors.length <= 3) return r;
        return { ...r, colors: r.colors.filter((_, ci) => ci !== colorIdx) };
      });
      return { ...prev, paletteRamps: ramps };
    });
  };

  const handleToggleDonor = (spriteId: number) => {
    setDraft((prev) => {
      const exists = prev.donorSuggestions.includes(spriteId);
      if (exists) {
        if (prev.donorSuggestions.length <= 3) return prev;
        return {
          ...prev,
          donorSuggestions: prev.donorSuggestions.filter((id) => id !== spriteId),
        };
      }
      if (prev.donorSuggestions.length >= 6) return prev;
      return {
        ...prev,
        donorSuggestions: [...prev.donorSuggestions, spriteId],
      };
    });
  };

  const spec = isEditing ? draft : activeRecord.engineeredPrompt;

  return (
    <div
      data-testid="prompt-engineer-card"
      className="scroll-mt-20 bg-[#12151C] border border-[#2A3242] rounded-xl p-5 space-y-4"
    >
      {/* Top Bar: Step 1 Header, Provenance, Version History */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3 border-b border-[#1E232D] pb-3.5">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
            <span className="text-[#E5A93C] font-semibold">ETAPA 1 · PROMPT ENGENHEIRADO</span>
            <span className="text-[#4B5568]">·</span>
            <span
              data-testid="prompt-source-label"
              className={
                activeRecord.source === 'gemini' ? 'text-[#86EFAC]' : 'text-[#FCD34D]'
              }
            >
              {activeRecord.source === 'gemini'
                ? `Gemini (${activeRecord.model ?? 'servidor'})`
                : 'Rascunho manual (sem Gemini)'}
            </span>
            <span className="text-[#4B5568]">·</span>
            <span className="text-[#98A2B3]">v{activeRecord.version}</span>
            {activeRecord.approved && (
              <>
                <span className="text-[#4B5568]">·</span>
                <span className="inline-flex items-center gap-1 text-[#86EFAC]">
                  <CheckCircle2 className="w-3 h-3" />
                  Aprovado
                </span>
              </>
            )}
            {activeRecord.readBackVerified && (
              <>
                <span className="text-[#4B5568]">·</span>
                <span className="inline-flex items-center gap-0.5 text-[#93C5FD]">
                  <Check className="w-3 h-3" />
                  IDB ok
                </span>
              </>
            )}
          </div>
          <h2 className="text-base font-semibold text-[#F3F5F8]">
            {spec.itemName} <span className="text-xs font-normal text-[#8C97AB]">({spec.category})</span>
          </h2>
          {activeRecord.adjustmentComment && (
            <p className="text-xs text-[#93C5FD]">
              Revisado a partir do ajuste: &ldquo;{activeRecord.adjustmentComment}&rdquo;
            </p>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          {requestVersions.length > 1 && (
            <label className="inline-flex items-center gap-1.5 text-xs text-[#98A2B3]">
              <History className="w-3.5 h-3.5" />
              <select
                value={activeRecord.id}
                onChange={(e) => {
                  const found = requestVersions.find((v) => v.id === e.target.value);
                  if (found) onSelectVersion(found);
                }}
                data-testid="prompt-version-select"
                className="px-2 py-1 text-xs font-mono rounded bg-[#0D0F13] border border-[#262C38] text-[#E6EBF3]"
              >
                {requestVersions.map((v) => (
                  <option key={v.id} value={v.id}>
                    v{v.version} ({v.source}
                    {v.approved ? ' · OK' : ''})
                  </option>
                ))}
              </select>
            </label>
          )}

          <button
            type="button"
            onClick={onRegeneratePrompt}
            disabled={isEngineering || isGenerating}
            data-testid="regenerate-prompt-btn"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-[#151921] border border-[#29303E] text-[#E6EBF3] hover:border-[#3D475B] disabled:opacity-50 transition-colors"
          >
            {isEngineering ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin text-[#E5A93C]" />
            ) : (
              <RefreshCw className="w-3.5 h-3.5" />
            )}
            <span>Regenerar prompt</span>
          </button>

          {!isEditing ? (
            <button
              type="button"
              onClick={() => setIsEditing(true)}
              disabled={isEngineering || isGenerating}
              data-testid="edit-prompt-btn"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-[#151921] border border-[#29303E] text-[#E6EBF3] hover:border-[#E5A93C] disabled:opacity-50 transition-colors"
            >
              <Edit3 className="w-3.5 h-3.5" />
              <span>Editar</span>
            </button>
          ) : (
            <button
              type="button"
              onClick={() => void handleSaveEditClick()}
              disabled={isSavingEdit}
              data-testid="save-edited-prompt-btn"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-[#1E3A5F] border border-[#3B82F6] text-[#E6EBF3] hover:bg-[#254775] disabled:opacity-50 transition-colors"
            >
              {isSavingEdit ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Check className="w-3.5 h-3.5" />
              )}
              <span>Salvar revisão</span>
            </button>
          )}

          <button
            type="button"
            onClick={() => void onApproveAndGenerate(isEditing ? buildSanitizedDraft() : spec)}
            disabled={isEngineering || isGenerating}
            data-testid="approve-and-generate-btn"
            className="inline-flex items-center gap-1.5 px-4 py-1.5 text-xs font-semibold rounded-lg bg-[#E5A93C] text-[#0D0F13] hover:bg-[#EEB955] disabled:opacity-50 transition-colors"
          >
            {isGenerating ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Sparkles className="w-3.5 h-3.5" />
            )}
            <span>OK, gerar ({candidateCount})</span>
          </button>
        </div>
      </div>

      {/* Diff Banner when revised from a previous prompt version */}
      {diffChanges.length > 0 && (
        <div
          data-testid="prompt-diff-panel"
          className="bg-[#0E1622] border border-[#284366] rounded-lg p-3.5 space-y-2"
        >
          <div className="flex items-center gap-2 text-xs font-semibold text-[#93C5FD]">
            <GitCompare className="w-3.5 h-3.5" />
            <span>
              Diferenças em relação à versão anterior ({diffChanges.length} alteração(ões)) — revise
              e clique em &ldquo;OK, gerar&rdquo; para confirmar
            </span>
          </div>
          <div className="space-y-1.5 text-xs">
            {diffChanges.map((ch) => (
              <div
                key={ch.field}
                className="grid grid-cols-1 md:grid-cols-[140px_1fr_1fr] gap-2 bg-[#0B0F17] border border-[#1B2638] rounded p-2 font-mono text-[11px]"
              >
                <span className="text-[#CBD3E1] font-semibold">{ch.labelPt}</span>
                <span className="text-[#FCA5A5] line-through break-words">{ch.before}</span>
                <span className="text-[#86EFAC] break-words">{ch.after}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {editError && (
        <div className="px-3 py-2 rounded-lg bg-[#231417] border border-[#5E2328] text-xs text-[#FCA5A5]">
          {editError}
        </div>
      )}

      {/* Main Editable / Structured Body */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left: Brief + Constraints + Negatives */}
        <div className="lg:col-span-7 space-y-3.5">
          {isEditing && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <label className="space-y-1">
                <span className="text-[11px] font-medium text-[#98A2B3]">Nome do item</span>
                <input
                  type="text"
                  value={draft.itemName}
                  onChange={(e) => setDraft((p) => ({ ...p, itemName: e.target.value }))}
                  className="w-full px-3 py-1.5 text-xs rounded bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8]"
                />
              </label>
              <label className="space-y-1">
                <span className="text-[11px] font-medium text-[#98A2B3]">Categoria</span>
                <select
                  value={draft.category}
                  onChange={(e) =>
                    setDraft((p) => ({ ...p, category: e.target.value as ItemCategory }))
                  }
                  className="w-full px-3 py-1.5 text-xs rounded bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8]"
                >
                  {ALL_ITEM_CATEGORIES.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          )}

          {/* Visual Brief */}
          <div className="space-y-1">
            <span className="block text-[11px] font-medium uppercase tracking-wider text-[#8C97AB]">
              Brief Visual (linguagem Tibia · silhueta · materiais · luz NW)
            </span>
            {isEditing ? (
              <textarea
                rows={4}
                value={draft.visualBrief}
                onChange={(e) => setDraft((p) => ({ ...p, visualBrief: e.target.value }))}
                data-testid="edit-visual-brief-textarea"
                className="w-full px-3 py-2 text-xs rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#F3F5F8]"
              />
            ) : (
              <p className="text-xs leading-relaxed text-[#E6EBF3] bg-[#0D0F13] border border-[#1E232D] rounded-lg p-3">
                {spec.visualBrief}
              </p>
            )}
          </div>

          {/* Constraints */}
          <div className="space-y-1.5">
            <span className="block text-[11px] font-medium uppercase tracking-wider text-[#8C97AB]">
              Restrições Técnicas 32×32
            </span>
            {isEditing ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 bg-[#0D0F13] border border-[#1E232D] rounded-lg p-3 text-xs">
                <label className="space-y-1">
                  <span className="text-[11px] text-[#8C97AB]">Largura (px)</span>
                  <input
                    type="number"
                    min={4}
                    max={32}
                    value={draft.constraints.footprintPx.width}
                    onChange={(e) =>
                      setDraft((p) => ({
                        ...p,
                        constraints: {
                          ...p.constraints,
                          footprintPx: {
                            ...p.constraints.footprintPx,
                            width: Math.max(4, Math.min(32, Number(e.target.value) || 20)),
                          },
                        },
                      }))
                    }
                    className="w-full px-2 py-1 rounded bg-[#12151C] border border-[#262C38] text-[#F3F5F8] font-mono"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-[11px] text-[#8C97AB]">Altura (px)</span>
                  <input
                    type="number"
                    min={4}
                    max={32}
                    value={draft.constraints.footprintPx.height}
                    onChange={(e) =>
                      setDraft((p) => ({
                        ...p,
                        constraints: {
                          ...p.constraints,
                          footprintPx: {
                            ...p.constraints.footprintPx,
                            height: Math.max(4, Math.min(32, Number(e.target.value) || 20)),
                          },
                        },
                      }))
                    }
                    className="w-full px-2 py-1 rounded bg-[#12151C] border border-[#262C38] text-[#F3F5F8] font-mono"
                  />
                </label>
                <label className="space-y-1">
                  <span className="text-[11px] text-[#8C97AB]">Máx. cores</span>
                  <input
                    type="number"
                    min={4}
                    max={64}
                    value={draft.constraints.maxColors}
                    onChange={(e) =>
                      setDraft((p) => ({
                        ...p,
                        constraints: {
                          ...p.constraints,
                          maxColors: Math.max(4, Math.min(64, Number(e.target.value) || 16)),
                        },
                      }))
                    }
                    className="w-full px-2 py-1 rounded bg-[#12151C] border border-[#262C38] text-[#F3F5F8] font-mono"
                  />
                </label>
                <label className="col-span-2 sm:col-span-3 space-y-1">
                  <span className="text-[11px] text-[#8C97AB]">Regra de contorno</span>
                  <input
                    type="text"
                    value={draft.constraints.outlineRule}
                    onChange={(e) =>
                      setDraft((p) => ({
                        ...p,
                        constraints: { ...p.constraints, outlineRule: e.target.value },
                      }))
                    }
                    className="w-full px-2 py-1 rounded bg-[#12151C] border border-[#262C38] text-[#F3F5F8]"
                  />
                </label>
                <label className="inline-flex items-center gap-2 text-[11px] text-[#CBD3E1]">
                  <input
                    type="checkbox"
                    checked={draft.constraints.noDitheringOrAA}
                    onChange={(e) =>
                      setDraft((p) => ({
                        ...p,
                        constraints: { ...p.constraints, noDitheringOrAA: e.target.checked },
                      }))
                    }
                  />
                  Sem dithering / AA
                </label>
                <label className="inline-flex items-center gap-2 text-[11px] text-[#CBD3E1]">
                  <input
                    type="checkbox"
                    checked={draft.constraints.binaryAlpha}
                    onChange={(e) =>
                      setDraft((p) => ({
                        ...p,
                        constraints: { ...p.constraints, binaryAlpha: e.target.checked },
                      }))
                    }
                  />
                  Alpha binário 0/255
                </label>
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 bg-[#0D0F13] border border-[#1E232D] rounded-lg p-3 text-xs font-mono tabular-nums">
                <div>
                  <span className="block text-[10px] text-[#8C97AB]">Footprint</span>
                  <span className="text-[#F3F5F8]">
                    {spec.constraints.footprintPx.width}×{spec.constraints.footprintPx.height} px
                  </span>
                </div>
                <div>
                  <span className="block text-[10px] text-[#8C97AB]">Máx. cores</span>
                  <span className="text-[#F3F5F8]">{spec.constraints.maxColors} cores</span>
                </div>
                <div>
                  <span className="block text-[10px] text-[#8C97AB]">Alpha binário</span>
                  <span className="text-[#86EFAC]">
                    {spec.constraints.binaryAlpha ? 'Sim (0/255)' : 'Não'}
                  </span>
                </div>
                <div>
                  <span className="block text-[10px] text-[#8C97AB]">Sem dithering/AA</span>
                  <span className="text-[#CBD3E1]">
                    {spec.constraints.noDitheringOrAA ? 'Estrito' : 'Moderado'}
                  </span>
                </div>
                <div className="col-span-2 sm:col-span-4 border-t border-[#1B202A] pt-2 text-[11px] font-sans text-[#CBD3E1]">
                  <span className="text-[#8C97AB]">Contorno:</span> {spec.constraints.outlineRule}
                </div>
              </div>
            )}
          </div>

          {/* Negative Prompt */}
          <div className="space-y-1.5">
            <span className="block text-[11px] font-medium uppercase tracking-wider text-[#8C97AB]">
              Prompt Negativo (artefatos de IA a evitar)
            </span>
            {isEditing ? (
              <textarea
                rows={3}
                value={negativeText}
                onChange={(e) => setNegativeText(e.target.value)}
                placeholder="Um item por linha"
                className="w-full px-3 py-2 text-xs font-mono rounded-lg bg-[#0D0F13] border border-[#262C38] text-[#FCA5A5]"
              />
            ) : (
              <div className="flex flex-wrap gap-1.5 bg-[#0D0F13] border border-[#1E232D] rounded-lg p-2.5">
                {spec.negativePrompt.map((neg, i) => (
                  <span
                    key={`${neg}-${i}`}
                    className="px-2 py-0.5 text-[11px] rounded bg-[#1F1417] border border-[#452126] text-[#FCA5A5]"
                  >
                    {neg}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right: Palette Ramps + Real Donor Thumbnails */}
        <div className="lg:col-span-5 space-y-4">
          {/* Palette Ramps */}
          <div className="space-y-2">
            <span className="block text-[11px] font-medium uppercase tracking-wider text-[#8C97AB]">
              Rampas de Paleta (outline → shadow → base → light → highlight)
            </span>
            <div className="space-y-2.5 bg-[#0D0F13] border border-[#1E232D] rounded-lg p-3">
              {spec.paletteRamps.map((ramp, rIdx) => (
                <div key={`${ramp.material}-${rIdx}`} className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-[#E6EBF3]">{ramp.material}</span>
                    {isEditing && ramp.colors.length < 5 && (
                      <button
                        type="button"
                        onClick={() => handleAddRampStop(rIdx)}
                        className="inline-flex items-center gap-0.5 text-[11px] text-[#93C5FD] hover:underline"
                      >
                        <Plus className="w-3 h-3" /> cor
                      </button>
                    )}
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5">
                    {ramp.colors.map((hex, cIdx) => (
                      <div
                        key={`${rIdx}-${cIdx}`}
                        className="flex flex-col items-center gap-1 bg-[#12151C] border border-[#252B36] rounded p-1.5"
                      >
                        <span
                          className="w-7 h-7 rounded border border-white/15 block"
                          style={{ backgroundColor: HEX_RE.test(hex) ? hex : '#222222' }}
                          title={hex}
                        />
                        {isEditing ? (
                          <div className="flex items-center gap-0.5">
                            <input
                              type="text"
                              value={hex}
                              onChange={(e) => handleUpdateRampColor(rIdx, cIdx, e.target.value)}
                              className="w-16 px-1 py-0.5 text-[10px] font-mono rounded bg-[#0D0F13] border border-[#2E3544] text-[#F3F5F8]"
                            />
                            {ramp.colors.length > 3 && (
                              <button
                                type="button"
                                onClick={() => handleRemoveRampStop(rIdx, cIdx)}
                                className="text-[#8C97AB] hover:text-[#FCA5A5]"
                                title="Remover stop"
                              >
                                <X className="w-3 h-3" />
                              </button>
                            )}
                          </div>
                        ) : (
                          <span className="text-[10px] font-mono text-[#98A2B3]">{hex}</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Donor Suggestions Thumbnails */}
          <div className="space-y-2">
            <span className="block text-[11px] font-medium uppercase tracking-wider text-[#8C97AB]">
              Doadores Reais Sugeridos ({spec.donorSuggestions.length} · train/val · 0 test)
            </span>
            <div
              data-testid="prompt-donors-grid"
              className="grid grid-cols-3 gap-2 bg-[#0D0F13] border border-[#1E232D] rounded-lg p-3"
            >
              {spec.donorSuggestions.map((sid) => {
                const entry = donorLookup.get(sid);
                return (
                  <div
                    key={sid}
                    className="flex flex-col items-center gap-1 bg-[#12151C] border border-[#222834] rounded p-2 relative"
                  >
                    {isEditing && spec.donorSuggestions.length > 3 && (
                      <button
                        type="button"
                        onClick={() => handleToggleDonor(sid)}
                        className="absolute top-1 right-1 text-[#8C97AB] hover:text-[#FCA5A5]"
                        title="Remover doador"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    )}
                    {entry ? (
                      <SpriteThumb
                        rgba={entry.rgba}
                        scale={2}
                        backdrop={backdrop}
                        alt={`Doador real #${sid}`}
                      />
                    ) : (
                      <div className="w-16 h-16 rounded border border-[#262B35] flex items-center justify-center text-[10px] font-mono text-[#8C97AB]">
                        #{sid}
                      </div>
                    )}
                    <div className="text-[10px] font-mono text-[#CBD3E1]">
                      #{sid} {entry ? `(${entry.split === 'train' ? 'tr' : 'val'})` : ''}
                    </div>
                    {entry && (
                      <div className="text-[10px] font-mono text-[#8C97AB]">
                        Crítico {entry.criticScore.toFixed(0)}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {isEditing && availableDonors.length > 0 && (
              <div className="space-y-1">
                <span className="text-[10px] text-[#8C97AB]">
                  Adicionar doador do split train (máx. 6):
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {availableDonors
                    .filter((d) => !draft.donorSuggestions.includes(d.spriteId))
                    .slice(0, 6)
                    .map((d) => (
                      <button
                        key={d.spriteId}
                        type="button"
                        onClick={() => handleToggleDonor(d.spriteId)}
                        disabled={draft.donorSuggestions.length >= 6}
                        className="inline-flex items-center gap-1 px-2 py-1 text-[10px] font-mono rounded bg-[#151921] border border-[#29303E] text-[#CBD3E1] hover:border-[#E5A93C] disabled:opacity-40"
                      >
                        <Plus className="w-2.5 h-2.5" /> #{d.spriteId}
                      </button>
                    ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
