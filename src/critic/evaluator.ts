import { predictDiscriminatorProbability } from './discriminator';
import { extractProfileNumericArray, extractSpriteFeatures } from './features';
import {
  PROFILE_FEATURE_KEYS,
  type CategoryProfile,
  type CriticEvaluation,
  type CriticEvaluationContext,
  type CriticIssue,
  type ProfileFeatureKey,
  type SpriteFeatureVector,
} from './types';

function clamp(v: number, min: number, max: number): number {
  if (Number.isNaN(v) || !Number.isFinite(v)) return min;
  return Math.max(min, Math.min(max, v));
}

function round1(v: number): number {
  return Math.round(v * 10) / 10;
}

const STAT_WEIGHTS: Record<ProfileFeatureKey, number> = {
  coverage: 1.0,
  distinctColors: 1.1,
  lumaLevels: 1.0,
  nearDuplicateRatio: 1.5,
  mixelScore: 1.4,
  smoothGradientRatio: 1.4,
  speckleNoiseScore: 1.5,
  strayColorPixelRatio: 1.2,
  meanClusterSize: 0.9,
  outlineCoverage: 1.2,
  outlineDarknessDelta: 1.1,
  outlineDoublesRatio: 0.8,
  outlineHueShiftedRatio: 0.6,
  nwLightGradient: 1.0,
  pillowShadingScore: 1.1,
  bandingScore: 0.9,
  ditherShare: 0.7,
  hueShiftDegrees: 0.6,
};

const TILE_SKIP_STAT_KEYS = new Set<ProfileFeatureKey>([
  'outlineCoverage',
  'outlineDarknessDelta',
  'outlineDoublesRatio',
  'outlineHueShiftedRatio',
]);

function computeStatisticalRealismScore(
  features: SpriteFeatureVector,
  profile: CategoryProfile,
  isTile: boolean,
): { score: number; outlierKeys: ProfileFeatureKey[] } {
  let weightedPenalty = 0;
  let weightSum = 0;
  const outlierKeys: ProfileFeatureKey[] = [];

  for (const key of PROFILE_FEATURE_KEYS) {
    if (isTile && TILE_SKIP_STAT_KEYS.has(key)) continue;
    const dist = profile.features[key];
    if (!dist || dist.n === 0) continue;

    const val = features[key];
    const iqr = Math.max(0, dist.p75 - dist.p25);
    const robustScale = Math.max(
      dist.std,
      iqr / 1.349,
      Math.abs(dist.p50) * 0.12,
      key === 'distinctColors' ? 3.5 : key === 'hueShiftDegrees' ? 12 : 0.06,
    );

    // Distance from the central interquartile band [p25, p75] is zero-penalty; outside grows smoothly
    let dev = 0;
    if (val < dist.p25) {
      dev = (dist.p25 - val) / robustScale;
    } else if (val > dist.p75) {
      dev = (val - dist.p75) / robustScale;
    }

    const cappedZ = Math.min(5.0, dev);
    const w = STAT_WEIGHTS[key];
    weightedPenalty += w * cappedZ;
    weightSum += w;

    if (dev >= 2.8) {
      outlierKeys.push(key);
    }
  }

  if (weightSum === 0) {
    return { score: 50, outlierKeys: [] };
  }

  const meanZ = weightedPenalty / weightSum;
  // Smooth exponential decay from 98 -> 0 as average z-deviation grows; never flat 100
  const rawScore = 98 * Math.exp(-0.48 * meanZ);
  return {
    score: clamp(rawScore, 0, 98.5),
    outlierKeys,
  };
}

function computeHumanPreferenceScore(
  features: SpriteFeatureVector,
  ctx: CriticEvaluationContext,
): number | null {
  const prefs = ctx.calibration?.humanPreferences;
  if (!prefs || prefs.approvedCount + prefs.rejectedCount < 6) {
    return null;
  }
  const vec = extractProfileNumericArray(features, PROFILE_FEATURE_KEYS);
  const stds = ctx.calibration?.discriminator.stds ?? vec.map(() => 1);

  let dApproved = 0;
  let dRejected = 0;
  for (let i = 0; i < vec.length; i++) {
    const s = stds[i] > 1e-4 ? stds[i] : 1;
    const da = (vec[i] - (prefs.approvedMean[i] ?? 0)) / s;
    const dr = (vec[i] - (prefs.rejectedMean[i] ?? 0)) / s;
    dApproved += da * da;
    dRejected += dr * dr;
  }
  const distA = Math.sqrt(dApproved / vec.length);
  const distR = Math.sqrt(dRejected / vec.length);
  const diff = distR - distA; // positive = closer to approved than rejected
  return clamp(50 + diff * 22, 5, 95);
}

/**
 * Critic v1 — Single source of truth.
 * Evaluates a 32x32 RGBA sprite deterministically using:
 * 1) Hard-rule validators (binary alpha, color range, orphans, smooth gradient, speckle, flat fill, non-tile silhouette/outline/border/centering)
 * 2) Pixel-exact hygiene (near-duplicate palette, mixels, line quality/doubles/jaggies, cluster hygiene, outline consistency, NW light, pillow shading, banding, dithering, hue shift)
 * 3) Statistical realism against per-category real `train` distributions
 * 4) Learned logistic discriminator trained on real vs corrupted negatives
 * 5) Optional human preference term (when Comparator labels exist)
 *
 * Honesty rules:
 * - Zero artificial floors, zero round-based formulas, never returns a flat 100%.
 * - Uncalibrated evaluations are explicitly capped and labeled "não calibrado".
 */
export function evaluateSprite(
  rgba: Uint8ClampedArray,
  ctx: CriticEvaluationContext,
): CriticEvaluation {
  const features = extractSpriteFeatures(rgba);
  const issues: CriticIssue[] = [];

  const isTile = Boolean(
    ctx.isTileOverride || ctx.flags?.ground || ctx.flags?.groundBorder || features.coverage > 0.9,
  );

  const calibration = ctx.calibration ?? null;
  const activeProfile: CategoryProfile | null = calibration
    ? calibration.profiles[ctx.category]?.sampleCount >= 5
      ? calibration.profiles[ctx.category]
      : calibration.globalProfile
    : null;
  const calibrated = Boolean(calibration && activeProfile && activeProfile.sampleCount > 0);

  if (features.opaquePixels === 0) {
    issues.push({
      code: 'EMPTY_SPRITE',
      severity: 'hard',
      labelPt: 'Sprite vazio',
      detailPt: 'Nenhum pixel opaco encontrado na grade 32×32.',
      measured: 0,
      expected: '>= 16 pixels opacos',
    });
    return {
      score: 0,
      calibrated,
      calibrationLabel: calibrated ? `calibrado (${ctx.category}, n=${activeProfile!.sampleCount})` : 'não calibrado',
      isTile,
      hardRuleCap: 0,
      hygieneScore: 0,
      statisticalScore: 0,
      discriminatorScore: calibrated ? 0 : null,
      humanPreferenceScore: null,
      features,
      issues,
    };
  }

  // --------------------------------------------------------------------------
  // 1. HARD-RULE VALIDATORS (any violation applies a concrete cap)
  // --------------------------------------------------------------------------
  let hardRuleCap = 99.5;
  const applyHardCap = (cap: number, issue: CriticIssue) => {
    hardRuleCap = Math.min(hardRuleCap, cap);
    issues.push(issue);
  };

  // Rule 1a: Binary alpha only (0 or 255)
  if (features.semiTransparentPixels > 0) {
    applyHardCap(34, {
      code: 'BINARY_ALPHA',
      severity: 'hard',
      labelPt: 'Alpha semitransparente / anti-aliasing',
      detailPt: `${features.semiTransparentPixels} pixels com alpha intermediário (1–254). Sprites 7.72 exigem alpha estritamente binário (0 ou 255).`,
      measured: features.semiTransparentPixels,
      expected: '0 pixels semitransparentes',
    });
  }

  // Rule 1b: Flat fill (< 3 luminance levels)
  if (features.lumaLevels < 3) {
    applyHardCap(26, {
      code: 'FLAT_FILL',
      severity: 'hard',
      labelPt: 'Preenchimento chapado (< 3 níveis de luminância)',
      detailPt: `Apenas ${features.lumaLevels} nível(is) de luminância detectado(s). Falta contraste de luz e sombra.`,
      measured: features.lumaLevels,
      expected: '>= 3 níveis de luminância',
    });
  }

  // Rule 1c: Distinct colors within real per-category range
  const colorMin = calibrated
    ? Math.max(3, Math.floor(activeProfile!.features.distinctColors.min))
    : 4;
  const colorMax = calibrated
    ? Math.max(48, Math.ceil(activeProfile!.features.distinctColors.max * 1.15))
    : 56;

  if (features.distinctColors < colorMin || features.distinctColors > colorMax) {
    const cap = features.distinctColors <= 2 || features.distinctColors > colorMax * 1.8 ? 28 : 42;
    applyHardCap(cap, {
      code: 'COLOR_COUNT_OUT_OF_RANGE',
      severity: 'hard',
      labelPt: 'Contagem de cores fora da faixa real',
      detailPt: `${features.distinctColors} cores exatas (faixa esperada para ${ctx.category}: ${colorMin}–${colorMax}).`,
      measured: features.distinctColors,
      expected: `${colorMin}–${colorMax} cores`,
    });
  }

  // Rule 1d: Smooth gradient signature
  if (features.smoothGradientRatio >= 0.42 && features.distinctColors >= 20) {
    applyHardCap(30, {
      code: 'SMOOTH_GRADIENT',
      severity: 'hard',
      labelPt: 'Assinatura de gradiente suave',
      detailPt: `${Math.round(features.smoothGradientRatio * 100)}% das transições adjacentes são passos suaves contínuos com ${features.distinctColors} cores.`,
      measured: round1(features.smoothGradientRatio * 100),
      expected: '< 42% (paleta discreta)',
    });
  }

  // Rule 1e: Speckle / random noise (calibrated against real category p90 when available)
  const speckleHardThreshold = calibrated
    ? Math.max(0.62, activeProfile!.features.speckleNoiseScore.p90 + 0.18)
    : 0.62;
  if (features.speckleNoiseScore >= speckleHardThreshold && features.distinctColors >= 24) {
    applyHardCap(24, {
      code: 'SPECKLE_NOISE',
      severity: 'hard',
      labelPt: 'Ruído aleatório / speckle',
      detailPt: `Índice de ruído espacial não correlacionado em ${Math.round(features.speckleNoiseScore * 100)}% (tamanho médio de cluster ${features.meanClusterSize.toFixed(1)} px).`,
      measured: round1(features.speckleNoiseScore * 100),
      expected: `< ${Math.round(speckleHardThreshold * 100)}%`,
    });
  }

  // Rule 1f: Detached orphan pixels
  if (!isTile && (features.orphanPixels >= 3 || (features.orphanPixels >= 2 && features.orphanRatio >= 0.015))) {
    applyHardCap(38, {
      code: 'ORPHAN_PIXELS',
      severity: 'hard',
      labelPt: 'Pixels órfãos soltos',
      detailPt: `${features.orphanPixels} pixel(is) isolado(s) fora da silhueta principal.`,
      measured: features.orphanPixels,
      expected: '0 pixels órfãos',
    });
  }

  // Rules 1g–1j: Non-tile silhouette / outline / border / centering rules
  if (!isTile) {
    // Missing dark outline
    if (features.outlineCoverage < 0.35 && features.outlineDarknessDelta < 0.08) {
      applyHardCap(35, {
        code: 'MISSING_OUTLINE',
        severity: 'hard',
        labelPt: 'Contorno escuro ausente',
        detailPt: `Apenas ${Math.round(features.outlineCoverage * 100)}% do perímetro tem contorno escuro (delta de luminância ${features.outlineDarknessDelta.toFixed(2)}).`,
        measured: round1(features.outlineCoverage * 100),
        expected: '>= 35% contorno escuro',
      });
    }

    // Disconnected silhouette (more than 1 substantial component)
    const majorDisconnected =
      features.componentCount >= 2 && features.orphanPixels < features.opaquePixels * 0.5 && features.componentCount - features.orphanPixels >= 2;
    if (majorDisconnected) {
      applyHardCap(36, {
        code: 'DISCONNECTED_SILHOUETTE',
        severity: 'hard',
        labelPt: 'Silhueta desconectada',
        detailPt: `${features.componentCount} componentes separados detectados em um item não-tile.`,
        measured: features.componentCount,
        expected: '1 componente coeso',
      });
    }

    // Touching frame border: heavy clipping (>= 10 px or off-center) is a hard cap; minor grazing is a mild hygiene note
    if (features.borderTouchCount >= 10 || (features.borderTouchCount >= 4 && features.centerOffsetNorm > 0.25)) {
      applyHardCap(38, {
        code: 'TOUCHING_BORDER',
        severity: 'hard',
        labelPt: 'Tocando a borda do quadro 32×32',
        detailPt: `${features.borderTouchCount} pixels opacos cortados no limite do canvas 32×32.`,
        measured: features.borderTouchCount,
        expected: '0 pixels na borda (margem limpa)',
      });
    } else if (features.borderTouchCount > 0) {
      issues.push({
        code: 'TOUCHING_BORDER',
        severity: 'hygiene',
        labelPt: 'Encostando levemente na borda 32×32',
        detailPt: `${features.borderTouchCount} pixel(is) encostando na extremidade do quadro.`,
        measured: features.borderTouchCount,
        expected: 'margem de 1–2 px',
      });
    }

    // Off-center centroid
    if (features.centerOffsetNorm > 0.38) {
      applyHardCap(39, {
        code: 'OFF_CENTER',
        severity: 'hard',
        labelPt: 'Item desalinhado do centro',
        detailPt: `Centroide deslocado ${(features.centerOffsetNorm * 16).toFixed(1)} px do centro (15.5, 15.5).`,
        measured: round1(features.centerOffsetNorm * 16),
        expected: '<= 6.0 px do centro',
      });
    }
  }

  // --------------------------------------------------------------------------
  // 2. PIXEL-EXACT HYGIENE (scored 0..99, calibrated by category when available)
  // --------------------------------------------------------------------------
  let hygienePenalty = 0;

  // 2a. Exact-palette integrity: near-duplicate colors (1..6 per channel)
  const nearDupThreshold = calibrated
    ? Math.max(0.18, activeProfile!.features.nearDuplicateRatio.p75 + 0.12)
    : 0.22;
  if (features.nearDuplicateRatio > nearDupThreshold && features.nearDuplicateColorPairs >= 3) {
    const excess = features.nearDuplicateRatio - nearDupThreshold;
    hygienePenalty += Math.min(48, 18 + excess * 70);
    issues.push({
      code: 'NEAR_DUPLICATE_PALETTE',
      severity: 'hygiene',
      labelPt: 'Cores quase-duplicadas (paleta suja)',
      detailPt: `${features.nearDuplicateColorPairs} pares de cores separados por apenas 1–6 níveis por canal (${Math.round(features.nearDuplicateRatio * 100)}% da paleta).`,
      measured: features.nearDuplicateColorPairs,
      expected: `<= ${Math.round(nearDupThreshold * 100)}%`,
    });
  }

  // 2b. Resample / mixel detection
  const mixelThreshold = calibrated
    ? Math.max(0.45, activeProfile!.features.mixelScore.p90 + 0.12)
    : 0.48;
  if (features.mixelScore > mixelThreshold) {
    hygienePenalty += Math.min(54, 24 + (features.mixelScore - mixelThreshold) * 85);
    issues.push({
      code: 'MIXEL_RESAMPLE',
      severity: 'hygiene',
      labelPt: 'Mixels / blocos 2×2 reamostrados',
      detailPt: `Sinal de escala mista/reamostragem em ${Math.round(features.mixelScore * 100)}% (blocos 2×2 repetidos em ambos os eixos).`,
      measured: round1(features.mixelScore * 100),
      expected: `<= ${Math.round(mixelThreshold * 100)}%`,
    });
  }

  // 2c. Line quality (doubles, broken outline, staircase jaggies) for non-tiles
  if (!isTile) {
    const doublesThreshold = calibrated
      ? Math.max(0.42, activeProfile!.features.outlineDoublesRatio.p90 + 0.12)
      : 0.45;
    if (features.outlineDoublesRatio > doublesThreshold) {
      hygienePenalty += Math.min(20, 8 + (features.outlineDoublesRatio - doublesThreshold) * 35);
      issues.push({
        code: 'OUTLINE_DOUBLES',
        severity: 'hygiene',
        labelPt: 'Cantos duplos ("doubles") no contorno',
        detailPt: `Proporção de cantos L espessos no contorno: ${Math.round(features.outlineDoublesRatio * 100)}%.`,
        measured: round1(features.outlineDoublesRatio * 100),
        expected: `<= ${Math.round(doublesThreshold * 100)}%`,
      });
    }

    if (features.brokenOutlineRatio > 0.45) {
      hygienePenalty += Math.min(22, (features.brokenOutlineRatio - 0.45) * 45);
      issues.push({
        code: 'BROKEN_OUTLINE',
        severity: 'hygiene',
        labelPt: 'Quebras no contorno externo',
        detailPt: `${Math.round(features.brokenOutlineRatio * 100)}% do perímetro expõe pixels claros diretamente contra a transparência.`,
        measured: round1(features.brokenOutlineRatio * 100),
        expected: '<= 45%',
      });
    }

    if (features.staircaseIrregularity > 0.72) {
      hygienePenalty += Math.min(12, (features.staircaseIrregularity - 0.72) * 30);
      issues.push({
        code: 'STAIRCASE_JAGGIES',
        severity: 'hygiene',
        labelPt: 'Degraus irregulares ("jaggies")',
        detailPt: `Irregularidade de escada no traço: ${Math.round(features.staircaseIrregularity * 100)}%.`,
        measured: round1(features.staircaseIrregularity * 100),
        expected: '<= 72%',
      });
    }
  }

  // 2d. Cluster hygiene (confetti / stray 1px color islands)
  const strayThreshold = calibrated
    ? Math.max(0.35, activeProfile!.features.strayColorPixelRatio.p90 + 0.12)
    : 0.38;
  if (features.strayColorPixelRatio > strayThreshold) {
    hygienePenalty += Math.min(26, 8 + (features.strayColorPixelRatio - strayThreshold) * 45);
    issues.push({
      code: 'CLUSTER_CONFETTI',
      severity: 'hygiene',
      labelPt: 'Excesso de pixels isolados de cor (confete)',
      detailPt: `${Math.round(features.strayColorPixelRatio * 100)}% dos pixels opacos não têm vizinho ortogonal da mesma cor.`,
      measured: round1(features.strayColorPixelRatio * 100),
      expected: `<= ${Math.round(strayThreshold * 100)}%`,
    });
  }

  // 2e. Shading logic: pillow shading, banding, NW lighting, dithering mismatch, hue shift
  const pillowThreshold = calibrated
    ? Math.max(0.45, activeProfile!.features.pillowShadingScore.p90 + 0.1)
    : 0.46;
  if (!isTile && features.pillowShadingScore > pillowThreshold) {
    hygienePenalty += Math.min(54, 24 + (features.pillowShadingScore - pillowThreshold) * 80);
    issues.push({
      code: 'PILLOW_SHADING',
      severity: 'hygiene',
      labelPt: 'Pillow shading radial',
      detailPt: `Sombreamento concêntrico simétrico (centro claro, borda escura inclusive no quadrante NW): ${Math.round(features.pillowShadingScore * 100)}%.`,
      measured: round1(features.pillowShadingScore * 100),
      expected: `<= ${Math.round(pillowThreshold * 100)}%`,
    });
  }

  const bandingThreshold = calibrated
    ? Math.max(0.42, activeProfile!.features.bandingScore.p90 + 0.12)
    : 0.45;
  if (!isTile && features.bandingScore > bandingThreshold) {
    hygienePenalty += Math.min(18, 6 + (features.bandingScore - bandingThreshold) * 30);
    issues.push({
      code: 'BANDING_DETECTED',
      severity: 'hygiene',
      labelPt: 'Faixas concêntricas ("banding")',
      detailPt: `Índice de banding paralelo ao contorno: ${Math.round(features.bandingScore * 100)}%.`,
      measured: round1(features.bandingScore * 100),
      expected: `<= ${Math.round(bandingThreshold * 100)}%`,
    });
  }

  if (calibrated) {
    const ditherDist = activeProfile!.features.ditherShare;
    if (features.ditherShare > Math.max(0.28, ditherDist.p90 + 0.18)) {
      hygienePenalty += 10;
      issues.push({
        code: 'DITHER_MISMATCH',
        severity: 'hygiene',
        labelPt: 'Dithering acima do padrão real da categoria',
        detailPt: `Dithering em ${Math.round(features.ditherShare * 100)}% vs p90 real de ${Math.round(ditherDist.p90 * 100)}%.`,
        measured: round1(features.ditherShare * 100),
        expected: `<= ${Math.round(Math.max(0.28, ditherDist.p90 + 0.18) * 100)}%`,
      });
    }

    const nwDist = activeProfile!.features.nwLightGradient;
    if (!isTile && nwDist.p50 > 0.04 && features.nwLightGradient < -0.06) {
      hygienePenalty += 12;
      issues.push({
        code: 'WEAK_NW_LIGHTING',
        severity: 'hygiene',
        labelPt: 'Iluminação invertida (SE mais claro que NW)',
        detailPt: `Gradiente NW−SE medido em ${features.nwLightGradient.toFixed(2)} (mediana real da categoria: +${nwDist.p50.toFixed(2)}).`,
        measured: round1(features.nwLightGradient * 100),
        expected: `>= 0 (luz NW)`,
      });
    }
  }

  const hygieneScore = clamp(98.5 - hygienePenalty, 0, 98.5);

  // --------------------------------------------------------------------------
  // 3. STATISTICAL REALISM + 4. LEARNED DISCRIMINATOR + 5. HUMAN PREFERENCE
  // --------------------------------------------------------------------------
  let statisticalScore = 50;
  let discriminatorScore: number | null = null;
  let humanPreferenceScore: number | null = null;
  let combinedScore: number;

  if (!calibrated || !activeProfile || !calibration) {
    issues.push({
      code: 'UNCALIBRATED',
      severity: 'statistical',
      labelPt: 'Crítico não calibrado',
      detailPt: 'Nenhum perfil real de treino carregado. Pontuação de estilo limitada até calibrar com um .dat/.spr.',
    });
    // Uncalibrated: cap score at 55 so uncalibrated estimates can never masquerade as high fidelity
    combinedScore = Math.min(55, hygieneScore * 0.55);
  } else {
    const statRes = computeStatisticalRealismScore(features, activeProfile, isTile);
    statisticalScore = statRes.score;
    if (statRes.outlierKeys.length >= 3) {
      issues.push({
        code: 'STATISTICAL_OUTLIER',
        severity: 'statistical',
        labelPt: 'Fora da distribuição estatística da categoria',
        detailPt: `${statRes.outlierKeys.length} métricas fora do padrão real de ${ctx.category}: ${statRes.outlierKeys.slice(0, 4).join(', ')}.`,
        measured: statRes.outlierKeys.length,
        expected: '< 3 desvios extremos',
      });
    }

    const discProb = predictDiscriminatorProbability(calibration.discriminator, features);
    discriminatorScore = clamp(discProb * 98.5, 0, 98.5);
    humanPreferenceScore = computeHumanPreferenceScore(features, ctx);

    if (humanPreferenceScore !== null) {
      combinedScore =
        hygieneScore * 0.34 +
        statisticalScore * 0.28 +
        discriminatorScore * 0.26 +
        humanPreferenceScore * 0.12;
    } else {
      combinedScore =
        hygieneScore * 0.38 +
        statisticalScore * 0.32 +
        discriminatorScore * 0.3;
    }
  }

  const finalScore = round1(clamp(Math.min(combinedScore, hardRuleCap), 0, 99.5));

  return {
    score: finalScore,
    calibrated,
    calibrationLabel: calibrated
      ? `calibrado (${ctx.category}, n=${activeProfile!.sampleCount})`
      : 'não calibrado',
    isTile,
    hardRuleCap: round1(hardRuleCap),
    hygieneScore: round1(hygieneScore),
    statisticalScore: round1(statisticalScore),
    discriminatorScore: discriminatorScore !== null ? round1(discriminatorScore) : null,
    humanPreferenceScore: humanPreferenceScore !== null ? round1(humanPreferenceScore) : null,
    features,
    issues,
  };
}
