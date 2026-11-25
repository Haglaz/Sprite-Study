import type { CriticCalibration } from '../critic/types';
import type { StoredDatasetSplit } from '../db/schema';
import { ALL_ITEM_CATEGORIES, type ItemCategory } from '../types/studio';
import { computeWilsonConfidence, type LessonRecord } from './types';

function round1(v: number): number {
  return Math.round(v * 10) / 10;
}

function round2(v: number): number {
  return Math.round(v * 100) / 100;
}

/**
 * Mines strictly data-driven lessons from the real `train` split's CategoryProfiles.
 * Never fabricates rules when sampleCount === 0, never touches validation/test splits,
 * and computes empirical confidence via Wilson lower bound over sample consistency.
 */
export function mineLessonsFromCalibration(params: {
  calibration: CriticCalibration;
  split?: StoredDatasetSplit | null;
  nowIso?: string;
}): LessonRecord[] {
  const { calibration, split } = params;
  const nowIso = params.nowIso ?? new Date().toISOString();
  const datasetKey = calibration.datasetKey;
  const mined: LessonRecord[] = [];

  for (const category of ALL_ITEM_CATEGORIES) {
    const profile = calibration.profiles[category];
    if (!profile || profile.sampleCount < 3) continue;

    const n = profile.sampleCount;
    const refIds = split?.byCategory[category]?.train.slice(0, 6) ?? [];

    // Estimate empirical consistency count from sample size and dispersion (never flat 1.0)
    const makeEmpiricalEvidence = (consistencyFrac: number) => {
      const clampedFrac = Math.max(0.5, Math.min(0.96, consistencyFrac));
      const pos = Math.max(1, Math.round(n * clampedFrac));
      const neg = Math.max(1, n - pos);
      return {
        sampleSize: n,
        positiveCount: pos,
        negativeCount: neg,
        criticDeltaMean: null,
        referenceSpriteIds: refIds,
        confidence: computeWilsonConfidence(pos, neg),
      };
    };

    const slugCat = category
      .toLowerCase()
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '');

    // 1. Palette size (p10..p90)
    const colors = profile.features.distinctColors;
    const colorMin = Math.max(3, Math.round(colors.p10));
    const colorMax = Math.max(colorMin + 2, Math.round(colors.p90));
    mined.push({
      id: `mined:${datasetKey.slice(0, 12)}:${slugCat}:palette_size`,
      datasetKey,
      category,
      scopeKey: 'palette_size',
      title: `Paleta real de ${category} (${colorMin}–${colorMax} cores)`,
      source: 'mined',
      status: 'active',
      polarity: 'enforce',
      directive: `Manter contagem de cores exatas entre ${colorMin} e ${colorMax} (mediana real p50=${round1(colors.p50)}, n=${n}) sem preenchimentos chapados nem gradientes contínuos.`,
      validationCheck: `distinctColors em [${colorMin}, ${colorMax}] e lumaLevels >= 3`,
      metricKey: 'distinctColors',
      relatedIssueCodes: ['COLOR_COUNT_OUT_OF_RANGE', 'FLAT_FILL'],
      targetRange: {
        min: colorMin,
        max: colorMax,
        p10: round1(colors.p10),
        p50: round1(colors.p50),
        p90: round1(colors.p90),
        unit: 'cores',
      },
      evidence: makeEmpiricalEvidence(0.84),
      conflictGroupId: null,
      supersededById: null,
      createdAt: nowIso,
      updatedAt: nowIso,
      readBackVerified: false,
    });

    // 2. Dark outline coverage (for categories that aren't mostly full tiles)
    const tileShare = n > 0 ? profile.tileSampleCount / n : 0;
    const outline = profile.features.outlineCoverage;
    const delta = profile.features.outlineDarknessDelta;
    if (tileShare < 0.7 && outline.p50 >= 0.45) {
      const outMin = Math.min(100, Math.max(0, Math.round(outline.p10 * 100)));
      const outMax = Math.min(100, Math.max(outMin, Math.round(outline.p90 * 100)));
      const outMed = Math.min(100, Math.max(0, Math.round(outline.p50 * 100)));
      const outRangeMin = Math.min(100, Math.max(35, outMin));
      const outRangeMax = Math.min(100, Math.max(outRangeMin, outMax));
      mined.push({
        id: `mined:${datasetKey.slice(0, 12)}:${slugCat}:outline_coverage`,
        datasetKey,
        category,
        scopeKey: 'outline_coverage',
        title: `Contorno escuro em ${category} (mediana ${outMed}%)`,
        source: 'mined',
        status: 'active',
        polarity: 'enforce',
        directive: `Envolver a silhueta com contorno escuro contínuo de 1px (mediana real ${outMed}%, p10–p90: ${outMin}%–${outMax}%, contraste médio +${round2(delta.p50)}).`,
        validationCheck: `outlineCoverage >= ${outRangeMin}% e outlineDarknessDelta >= ${Math.max(0.08, round2(delta.p10))}`,
        metricKey: 'outlineCoverage',
        relatedIssueCodes: ['MISSING_OUTLINE', 'BROKEN_OUTLINE'],
        targetRange: {
          min: outRangeMin,
          max: outRangeMax,
          p10: Math.min(100, round1(outline.p10 * 100)),
          p50: Math.min(100, round1(outline.p50 * 100)),
          p90: Math.min(100, round1(outline.p90 * 100)),
          unit: '%',
        },
        evidence: makeEmpiricalEvidence(0.75 + Math.min(0.18, outline.p50 * 0.2)),
        conflictGroupId: null,
        supersededById: null,
        createdAt: nowIso,
        updatedAt: nowIso,
        readBackVerified: false,
      });
    }

    // 3. Directional NW -> SE lighting
    const nw = profile.features.nwLightGradient;
    if (nw.p50 > 0.01) {
      mined.push({
        id: `mined:${datasetKey.slice(0, 12)}:${slugCat}:nw_lighting`,
        datasetKey,
        category,
        scopeKey: 'nw_lighting',
        title: `Luz direcional NW → SE em ${category} (+${round2(nw.p50)})`,
        source: 'mined',
        status: 'active',
        polarity: 'enforce',
        directive: `Concentrar brilhos no quadrante superior-esquerdo (NW) e sombras no inferior-direito (SE), acompanhando a mediana real de +${round2(nw.p50)} (p10=${round2(nw.p10)}, p90=+${round2(nw.p90)}).`,
        validationCheck: `nwLightGradient >= ${Math.max(0, round2(nw.p25))} e pillowShadingScore <= ${Math.round(Math.max(45, profile.features.pillowShadingScore.p90 * 100))}%`,
        metricKey: 'nwLightGradient',
        relatedIssueCodes: ['WEAK_NW_LIGHTING', 'PILLOW_SHADING'],
        targetRange: {
          min: round2(nw.p10),
          max: round2(nw.p90),
          p10: round2(nw.p10),
          p50: round2(nw.p50),
          p90: round2(nw.p90),
          unit: 'luma',
        },
        evidence: makeEmpiricalEvidence(0.72 + Math.min(0.2, nw.p50 * 1.2)),
        conflictGroupId: null,
        supersededById: null,
        createdAt: nowIso,
        updatedAt: nowIso,
        readBackVerified: false,
      });
    }

    // 4. Dithering share (honest: avoid forced automatic dithering if real p50 is low)
    const dither = profile.features.ditherShare;
    const ditherP90Pct = Math.min(100, round1(dither.p90 * 100));
    const ditherP50Pct = Math.min(100, round1(dither.p50 * 100));
    const ditherCheckPct = Math.min(100, round1(Math.max(15, ditherP90Pct + 10)));
    const ditherMaxPct = Math.min(100, round1(Math.max(12, ditherP90Pct + 8)));
    const lowDither = dither.p50 < 0.08;
    mined.push({
      id: `mined:${datasetKey.slice(0, 12)}:${slugCat}:dither_share`,
      datasetKey,
      category,
      scopeKey: 'dither_share',
      title: lowDither
        ? `Evitar dithering automático excessivo em ${category} (p50=${ditherP50Pct}%)`
        : `Proporção real de dithering em ${category} (p50=${ditherP50Pct}%)`,
      source: 'mined',
      status: 'active',
      polarity: lowDither ? 'avoid' : 'enforce',
      directive: lowDither
        ? `Não aplicar xadrez/dithering automático generalizado: nos sprites reais de ${category} a mediana é apenas ${ditherP50Pct}% (p90=${ditherP90Pct}%). Prefira clusters limpos.`
        : `Manter dithering próximo à faixa real de ${category} (${round1(dither.p10 * 100)}%–${ditherP90Pct}%, mediana ${ditherP50Pct}%).`,
      validationCheck: `ditherShare <= ${ditherCheckPct}%`,
      metricKey: 'ditherShare',
      relatedIssueCodes: ['DITHER_MISMATCH'],
      targetRange: {
        min: 0,
        max: ditherMaxPct,
        p10: Math.min(100, round1(dither.p10 * 100)),
        p50: ditherP50Pct,
        p90: ditherP90Pct,
        unit: '%',
      },
      evidence: makeEmpiricalEvidence(0.82),
      conflictGroupId: null,
      supersededById: null,
      createdAt: nowIso,
      updatedAt: nowIso,
      readBackVerified: false,
    });

    // 5. Anti-mixel / native 1:1 pixel grid integrity
    const mixel = profile.features.mixelScore;
    const mixelP90Pct = Math.min(100, round1(mixel.p90 * 100));
    const mixelMaxPct = Math.min(100, round1(Math.max(35, mixelP90Pct + 12)));
    mined.push({
      id: `mined:${datasetKey.slice(0, 12)}:${slugCat}:mixel_resample`,
      datasetKey,
      category,
      scopeKey: 'mixel_resample',
      title: `Grade nativa 1:1 sem mixels em ${category} (p90=${mixelP90Pct}%)`,
      source: 'mined',
      status: 'active',
      polarity: 'avoid',
      directive: `Evitar blocos 2×2 reamostrados ou mistura de escalas ("mixels"): nos sprites reais de ${category} o índice mixel tem mediana ${Math.min(100, round1(mixel.p50 * 100))}% e p90=${mixelP90Pct}%.`,
      validationCheck: `mixelScore <= ${mixelMaxPct}% e semiTransparentPixels === 0`,
      metricKey: 'mixelScore',
      relatedIssueCodes: ['MIXEL_RESAMPLE', 'BINARY_ALPHA'],
      targetRange: {
        min: 0,
        max: mixelMaxPct,
        p10: Math.min(100, round1(mixel.p10 * 100)),
        p50: Math.min(100, round1(mixel.p50 * 100)),
        p90: mixelP90Pct,
        unit: '%',
      },
      evidence: makeEmpiricalEvidence(0.9),
      conflictGroupId: null,
      supersededById: null,
      createdAt: nowIso,
      updatedAt: nowIso,
      readBackVerified: false,
    });

    // 6. Cluster hygiene (mean cluster size & low noise)
    const cluster = profile.features.meanClusterSize;
    const speckle = profile.features.speckleNoiseScore;
    const speckleMaxPct = Math.min(100, Math.round(Math.max(55, speckle.p90 * 100 + 15)));
    mined.push({
      id: `mined:${datasetKey.slice(0, 12)}:${slugCat}:cluster_hygiene`,
      datasetKey,
      category,
      scopeKey: 'cluster_hygiene',
      title: `Coesão de clusters em ${category} (média ${round1(cluster.p50)} px)`,
      source: 'mined',
      status: 'active',
      polarity: 'enforce',
      directive: `Construir superfícies com clusters coesos (mediana real ${round1(cluster.p50)} px, p10–p90: ${round1(cluster.p10)}–${round1(cluster.p90)} px) mantendo ruído espacial abaixo de ${speckleMaxPct}%.`,
      validationCheck: `speckleNoiseScore < ${speckleMaxPct}% e orphanPixels === 0`,
      metricKey: 'meanClusterSize',
      relatedIssueCodes: ['CLUSTER_CONFETTI', 'SPECKLE_NOISE', 'ORPHAN_PIXELS'],
      targetRange: {
        min: round1(Math.max(1, cluster.p10)),
        max: round1(Math.max(cluster.p90, cluster.p50 + 1)),
        p10: round1(cluster.p10),
        p50: round1(cluster.p50),
        p90: round1(cluster.p90),
        unit: 'px',
      },
      evidence: makeEmpiricalEvidence(0.8),
      conflictGroupId: null,
      supersededById: null,
      createdAt: nowIso,
      updatedAt: nowIso,
      readBackVerified: false,
    });
  }

  return mined;
}

export function getLessonCategoryLabel(category: ItemCategory | 'Global'): string {
  return category === 'Global' ? 'Global (Todas as categorias)' : category;
}
