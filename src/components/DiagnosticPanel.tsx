import { useState } from 'react';
import { AlertTriangle, CheckCircle2, Info, ShieldAlert, SlidersHorizontal } from 'lucide-react';
import type { DiagnosticReport, DiagnosticSpriteSample, ProfileFeatureKey } from '../critic/types';
import { ALL_ITEM_CATEGORIES, type ItemCategory } from '../types/studio';
import { SpriteThumb, type ThumbBackdrop } from './SpriteThumb';

interface DiagnosticPanelProps {
  report: DiagnosticReport;
}

const FEATURE_META: Record<
  ProfileFeatureKey,
  { labelPt: string; format: (v: number) => string }
> = {
  coverage: { labelPt: 'Cobertura opaca', format: (v) => `${(v * 100).toFixed(1)}%` },
  distinctColors: { labelPt: 'Cores exatas distintas', format: (v) => v.toFixed(1) },
  lumaLevels: { labelPt: 'Níveis de luminância', format: (v) => v.toFixed(1) },
  nearDuplicateRatio: {
    labelPt: 'Cores quase-duplicadas (1–6/canal)',
    format: (v) => `${(v * 100).toFixed(1)}%`,
  },
  mixelScore: { labelPt: 'Índice mixel / blocos 2×2', format: (v) => `${(v * 100).toFixed(1)}%` },
  smoothGradientRatio: {
    labelPt: 'Assinatura de gradiente suave',
    format: (v) => `${(v * 100).toFixed(1)}%`,
  },
  speckleNoiseScore: {
    labelPt: 'Ruído / speckle',
    format: (v) => `${(v * 100).toFixed(1)}%`,
  },
  strayColorPixelRatio: {
    labelPt: 'Pixels de cor isolados (confete)',
    format: (v) => `${(v * 100).toFixed(1)}%`,
  },
  meanClusterSize: { labelPt: 'Tamanho médio de cluster', format: (v) => `${v.toFixed(1)} px` },
  outlineCoverage: {
    labelPt: 'Cobertura de contorno escuro',
    format: (v) => `${(v * 100).toFixed(1)}%`,
  },
  outlineDarknessDelta: {
    labelPt: 'Contraste interior − contorno',
    format: (v) => v.toFixed(2),
  },
  outlineDoublesRatio: {
    labelPt: 'Cantos duplos ("doubles")',
    format: (v) => `${(v * 100).toFixed(1)}%`,
  },
  outlineHueShiftedRatio: {
    labelPt: 'Contorno com matiz (não preto puro)',
    format: (v) => `${(v * 100).toFixed(1)}%`,
  },
  nwLightGradient: {
    labelPt: 'Direção de luz NW − SE',
    format: (v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}`,
  },
  pillowShadingScore: {
    labelPt: 'Pillow shading radial',
    format: (v) => `${(v * 100).toFixed(1)}%`,
  },
  bandingScore: { labelPt: 'Banding concêntrico', format: (v) => `${(v * 100).toFixed(1)}%` },
  ditherShare: { labelPt: 'Proporção de dithering', format: (v) => `${(v * 100).toFixed(1)}%` },
  hueShiftDegrees: { labelPt: 'Hue shift sombra → luz', format: (v) => `${v.toFixed(1)}°` },
};

const DISPLAY_FEATURE_KEYS: ProfileFeatureKey[] = [
  'coverage',
  'distinctColors',
  'lumaLevels',
  'nearDuplicateRatio',
  'mixelScore',
  'smoothGradientRatio',
  'speckleNoiseScore',
  'meanClusterSize',
  'outlineCoverage',
  'outlineDarknessDelta',
  'outlineDoublesRatio',
  'nwLightGradient',
  'pillowShadingScore',
  'bandingScore',
  'ditherShare',
  'hueShiftDegrees',
];

function scoreToneClass(score: number): string {
  if (score >= 70) return 'text-[#86EFAC] bg-[#102218] border-[#1F4A31]';
  if (score >= 40) return 'text-[#FDE68A] bg-[#261E0F] border-[#5B451B]';
  return 'text-[#FCA5A5] bg-[#2A1416] border-[#5E2328]';
}

function SampleDiagnosticCard({
  sample,
  backdrop,
}: {
  sample: DiagnosticSpriteSample;
  backdrop: ThumbBackdrop;
}) {
  const { evaluation } = sample;
  const hardCount = evaluation.issues.filter((i) => i.severity === 'hard').length;

  return (
    <article
      className="rounded-lg border border-[#1E232D] bg-[#12151C] p-3.5 flex flex-col gap-3"
      data-testid={`diag-sample-${sample.kind}`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <span
              className={`text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded border ${
                sample.kind === 'real'
                  ? 'bg-[#102218] border-[#1F4A31] text-[#86EFAC]'
                  : 'bg-[#2A1416] border-[#5E2328] text-[#FCA5A5]'
              }`}
            >
              {sample.kind === 'real' ? 'Real · Validação' : 'Lixo · Piso'}
            </span>
            <span className="text-[11px] text-[#7D889D] truncate">{sample.category}</span>
          </div>
          <h4 className="text-xs font-medium text-[#E8ECF4] mt-1 truncate" title={sample.label}>
            {sample.label}
          </h4>
        </div>

        <span
          className={`shrink-0 text-sm font-mono font-semibold px-2 py-0.5 rounded border ${scoreToneClass(
            evaluation.score,
          )}`}
        >
          {evaluation.score.toFixed(1)}
        </span>
      </div>

      <div className="flex items-end gap-3">
        <SpriteThumb rgba={sample.rgba} scale={3} backdrop={backdrop} alt={sample.label} />
        <div className="flex flex-col gap-1.5 text-[11px] text-[#9AA4B8] min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <SpriteThumb rgba={sample.rgba} scale={1} backdrop={backdrop} alt={`${sample.label} 1x`} />
            <span className="text-[10px] text-[#7D889D]">1× nativo (32×32)</span>
          </div>
          <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 font-mono text-[10px] text-[#A8B2C5]">
            <span>Higiene: {evaluation.hygieneScore.toFixed(1)}</span>
            <span>Estat.: {evaluation.statisticalScore.toFixed(1)}</span>
            <span>
              Discr.:{' '}
              {evaluation.discriminatorScore !== null
                ? evaluation.discriminatorScore.toFixed(1)
                : '—'}
            </span>
            <span>Teto: {evaluation.hardRuleCap.toFixed(0)}</span>
          </div>
          <div className="text-[10px] text-[#7D889D]">
            {evaluation.features.distinctColors} cores · {Math.round(evaluation.features.coverage * 100)}% cob.
            {evaluation.isTile ? ' · Tile' : ''}
          </div>
        </div>
      </div>

      {evaluation.issues.length === 0 ? (
        <div className="flex items-center gap-1.5 text-[11px] text-[#86EFAC] bg-[#0F1B14] border border-[#1A3826] rounded px-2 py-1.5">
          <CheckCircle2 className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
          <span>Nenhuma violação dura ou de higiene detectada.</span>
        </div>
      ) : (
        <ul className="space-y-1">
          {evaluation.issues.slice(0, 3).map((issue, idx) => (
            <li
              key={`${issue.code}-${idx}`}
              className={`text-[11px] rounded px-2 py-1 border flex items-start gap-1.5 ${
                issue.severity === 'hard'
                  ? 'bg-[#241315] border-[#4B1E23] text-[#FCA5A5]'
                  : issue.severity === 'hygiene'
                    ? 'bg-[#231B0E] border-[#4D3B18] text-[#FDE68A]'
                    : 'bg-[#151A24] border-[#263044] text-[#B8C4DA]'
              }`}
            >
              {issue.severity === 'hard' ? (
                <ShieldAlert className="w-3.5 h-3.5 shrink-0 mt-0.5" aria-hidden="true" />
              ) : (
                <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" aria-hidden="true" />
              )}
              <span className="leading-snug">
                <strong className="font-medium">{issue.labelPt}</strong>
                {issue.measured !== undefined ? ` (${issue.measured})` : ''}
              </span>
            </li>
          ))}
          {evaluation.issues.length > 3 && (
            <li className="text-[10px] text-[#7D889D] pl-1">
              +{evaluation.issues.length - 3} outro(s) alerta(s) ({hardCount} regra(s) dura(s))
            </li>
          )}
        </ul>
      )}
    </article>
  );
}

export function DiagnosticPanel({ report }: DiagnosticPanelProps) {
  const [sampleFilter, setSampleFilter] = useState<'both' | 'real' | 'junk'>('both');
  const [backdrop, setBackdrop] = useState<ThumbBackdrop>('transparent');
  const [selectedCategory, setSelectedCategory] = useState<ItemCategory>('Equipamentos');

  const activeProfile = report.categoryProfiles[selectedCategory];
  const separationGap = report.realCeiling.median - report.junkFloor.median;

  const visibleSamples: DiagnosticSpriteSample[] =
    sampleFilter === 'real'
      ? report.realSamples
      : sampleFilter === 'junk'
        ? report.junkSamples
        : [...report.realSamples.slice(0, 6), ...report.junkSamples.slice(0, 6)];

  return (
    <section
      aria-labelledby="diagnostic-panel-heading"
      data-testid="diagnostic-panel"
      className="space-y-6"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2
            id="diagnostic-panel-heading"
            className="text-lg font-semibold text-[#F3F5F8] tracking-tight"
          >
            Diagnóstico do Crítico v1
          </h2>
          <p className="text-xs text-[#8C97AB] mt-0.5">
            Todas as notas são calculadas em tempo de execução a partir dos pixels reais, sem pisos artificiais, sem fórmulas por rodada e sem 100% fixo.
          </p>
        </div>
        <span className="text-[11px] font-mono text-[#8C97AB] bg-[#141821] border border-[#232A38] rounded px-2.5 py-1">
          Seed fixa: {report.seed}
        </span>
      </div>

      {/* Honest Baselines Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        <div
          className="rounded-xl border border-[#1F4A31] bg-[#101C16] p-4"
          data-testid="baseline-real-ceiling"
        >
          <div className="text-[11px] font-medium uppercase tracking-wider text-[#86EFAC]">
            Teto Real (Validação)
          </div>
          <div className="mt-1.5 flex items-baseline gap-2">
            <span className="text-2xl font-mono font-semibold text-[#F3F5F8]">
              {report.realCeiling.median.toFixed(1)}
            </span>
            <span className="text-xs text-[#9AA4B8]">mediana</span>
          </div>
          <div className="mt-2 space-y-0.5 text-[11px] font-mono text-[#A6B3C7]">
            <div>
              IC 95%: [{report.realCeiling.ciLow.toFixed(1)}, {report.realCeiling.ciHigh.toFixed(1)}]
            </div>
            <div>
              Média: {report.realCeiling.mean.toFixed(1)} · n={report.realCeiling.n}
            </div>
            <div className="text-[#7E8EA6]">
              Faixa: {report.realCeiling.min.toFixed(1)} – {report.realCeiling.max.toFixed(1)}
            </div>
          </div>
        </div>

        <div
          className="rounded-xl border border-[#5E2328] bg-[#1E1214] p-4"
          data-testid="baseline-junk-floor"
        >
          <div className="text-[11px] font-medium uppercase tracking-wider text-[#FCA5A5]">
            Piso Lixo (Defeitos)
          </div>
          <div className="mt-1.5 flex items-baseline gap-2">
            <span className="text-2xl font-mono font-semibold text-[#F3F5F8]">
              {report.junkFloor.median.toFixed(1)}
            </span>
            <span className="text-xs text-[#9AA4B8]">mediana</span>
          </div>
          <div className="mt-2 space-y-0.5 text-[11px] font-mono text-[#A6B3C7]">
            <div>
              IC 95%: [{report.junkFloor.ciLow.toFixed(1)}, {report.junkFloor.ciHigh.toFixed(1)}]
            </div>
            <div>
              Média: {report.junkFloor.mean.toFixed(1)} · n={report.junkFloor.n}
            </div>
            <div className="text-[#7E8EA6]">
              Separação real−lixo: {separationGap >= 0 ? '+' : ''}
              {separationGap.toFixed(1)} pts
            </div>
          </div>
        </div>

        <div
          className="rounded-xl border border-[#263044] bg-[#12161F] p-4"
          data-testid="baseline-discriminator-auc"
        >
          <div className="text-[11px] font-medium uppercase tracking-wider text-[#93C5FD]">
            Discriminador (ROC-AUC)
          </div>
          <div className="mt-1.5 flex items-baseline gap-2">
            <span className="text-2xl font-mono font-semibold text-[#F3F5F8]">
              {(report.discriminatorValidationAuc * 100).toFixed(1)}%
            </span>
            <span className="text-xs text-[#9AA4B8]">held-out</span>
          </div>
          <div className="mt-2 space-y-0.5 text-[11px] font-mono text-[#A6B3C7]">
            <div>AUC: {report.discriminatorValidationAuc.toFixed(3)}</div>
            <div>
              Val+: {report.discriminatorValPosCount} · Val−: {report.discriminatorValNegCount}
            </div>
            <div className="text-[#7E8EA6]">Regressão logística L2</div>
          </div>
        </div>

        <div
          className="rounded-xl border border-[#262B35] bg-[#12151C] p-4"
          data-testid="baseline-split-counts"
        >
          <div className="text-[11px] font-medium uppercase tracking-wider text-[#E5A93C]">
            Divisão 70 / 15 / 15
          </div>
          <div className="mt-1.5 flex items-baseline gap-2">
            <span className="text-2xl font-mono font-semibold text-[#F3F5F8]">
              {report.splitTotals.total.toLocaleString('pt-BR')}
            </span>
            <span className="text-xs text-[#9AA4B8]">sprites únicos</span>
          </div>
          <div className="mt-2 space-y-0.5 text-[11px] font-mono text-[#A6B3C7]">
            <div>Treino: {report.splitTotals.train.toLocaleString('pt-BR')}</div>
            <div>Validação: {report.splitTotals.validation.toLocaleString('pt-BR')}</div>
            <div className="text-[#7E8EA6]">
              Teste (isolado): {report.splitTotals.test.toLocaleString('pt-BR')}
            </div>
          </div>
        </div>
      </div>

      {/* Real vs Junk Samples Grid */}
      <div className="rounded-xl border border-[#1E232D] bg-[#101319] p-4 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-[#F3F5F8]">
              Sprites Reais (Validação) vs Piso Lixo
            </h3>
            <p className="text-xs text-[#8C97AB]">
              Avaliados exatamente pela mesma função do Crítico v1 — sprites reais nunca vistos no treino vs ruído, chapado, gradiente, blur, mixel e pillow shading.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="inline-flex rounded-lg border border-[#262B35] bg-[#151922] p-0.5 text-xs">
              {(
                [
                  { id: 'both', label: 'Ambos' },
                  { id: 'real', label: `Reais (${report.realSamples.length})` },
                  { id: 'junk', label: `Lixo (${report.junkSamples.length})` },
                ] as const
              ).map((opt) => (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => setSampleFilter(opt.id)}
                  className={`px-2.5 py-1 rounded-md transition-colors ${
                    sampleFilter === opt.id
                      ? 'bg-[#E5A93C] text-[#111318] font-medium'
                      : 'text-[#9AA4B8] hover:text-[#F1F3F7]'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>

            <div className="inline-flex rounded-lg border border-[#262B35] bg-[#151922] p-0.5 text-xs">
              {(
                [
                  { id: 'transparent', label: 'Grade' },
                  { id: 'magenta', label: 'Magenta' },
                  { id: 'dark', label: 'Escuro' },
                ] as const
              ).map((bg) => (
                <button
                  key={bg.id}
                  type="button"
                  onClick={() => setBackdrop(bg.id)}
                  className={`px-2 py-1 rounded-md transition-colors ${
                    backdrop === bg.id
                      ? 'bg-[#262E3E] text-[#F3F5F8] font-medium'
                      : 'text-[#8C97AB] hover:text-[#E8ECF4]'
                  }`}
                >
                  {bg.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
          {visibleSamples.map((sample) => (
            <SampleDiagnosticCard key={sample.id} sample={sample} backdrop={backdrop} />
          ))}
        </div>
      </div>

      {/* Per-Category Mined Feature Distributions */}
      <div className="rounded-xl border border-[#1E232D] bg-[#101319] p-4 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-start gap-2">
            <SlidersHorizontal className="w-4 h-4 text-[#E5A93C] mt-0.5 shrink-0" aria-hidden="true" />
            <div>
              <h3 className="text-sm font-semibold text-[#F3F5F8]">
                Distribuições Mineradas do Treino por Categoria
              </h3>
              <p className="text-xs text-[#8C97AB]">
                Estatísticas reais extraídas apenas do split <code className="text-[#CBD5E1]">train</code> (nunca da validação ou teste).
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-1.5" role="tablist" aria-label="Categorias mineradas">
            {ALL_ITEM_CATEGORIES.map((cat) => {
              const count = report.categoryProfiles[cat]?.sampleCount ?? 0;
              const active = selectedCategory === cat;
              return (
                <button
                  key={cat}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  onClick={() => setSelectedCategory(cat)}
                  className={`px-3 py-1.5 rounded-lg text-xs border transition-colors ${
                    active
                      ? 'bg-[#1D2330] border-[#E5A93C] text-[#F3F5F8] font-medium'
                      : 'bg-[#141821] border-[#232A38] text-[#8C97AB] hover:text-[#E8ECF4]'
                  }`}
                >
                  {cat} <span className="font-mono text-[10px] opacity-80">(n={count})</span>
                </button>
              );
            })}
          </div>
        </div>

        {activeProfile.sampleCount === 0 ? (
          <div className="flex items-center gap-2 text-xs text-[#9AA4B8] bg-[#141821] border border-[#232A38] rounded-lg p-3">
            <Info className="w-4 h-4 text-[#E5A93C]" aria-hidden="true" />
            <span>Nenhum sprite nesta categoria no split de treino; o crítico utiliza o perfil global como referência.</span>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-[#1E232D] text-[11px] uppercase tracking-wider text-[#7D889D]">
                  <th className="py-2 pr-3 font-medium">Métrica Pixel-Exata</th>
                  <th className="py-2 px-3 font-medium text-right">Média ± DP</th>
                  <th className="py-2 px-3 font-medium text-right">p10</th>
                  <th className="py-2 px-3 font-medium text-right">p50 (Mediana)</th>
                  <th className="py-2 pl-3 font-medium text-right">p90</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#191E28] font-mono">
                {DISPLAY_FEATURE_KEYS.map((key) => {
                  const dist = activeProfile.features[key];
                  const meta = FEATURE_META[key];
                  return (
                    <tr key={key} className="hover:bg-[#141822]">
                      <td className="py-2 pr-3 font-sans text-[#D8DFEC]">{meta.labelPt}</td>
                      <td className="py-2 px-3 text-right text-[#CBD5E1]">
                        {meta.format(dist.mean)} ± {meta.format(dist.std)}
                      </td>
                      <td className="py-2 px-3 text-right text-[#8C97AB]">{meta.format(dist.p10)}</td>
                      <td className="py-2 px-3 text-right text-[#E5A93C] font-medium">
                        {meta.format(dist.p50)}
                      </td>
                      <td className="py-2 pl-3 text-right text-[#8C97AB]">{meta.format(dist.p90)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}
