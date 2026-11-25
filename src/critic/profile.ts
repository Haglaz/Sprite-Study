import { ALL_ITEM_CATEGORIES, type ItemCategory } from '../types/studio';
import { extractSpriteFeatures } from './features';
import {
  PROFILE_FEATURE_KEYS,
  type CategoryProfile,
  type FeatureDistribution,
  type ProfileFeatureKey,
  type SpriteFeatureVector,
} from './types';

function percentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const idx = Math.max(0, Math.min(sorted.length - 1, p * (sorted.length - 1)));
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const frac = idx - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

export function computeFeatureDistribution(values: readonly number[]): FeatureDistribution {
  const n = values.length;
  if (n === 0) {
    return { mean: 0, std: 0, min: 0, max: 0, p10: 0, p25: 0, p50: 0, p75: 0, p90: 0, n: 0 };
  }
  const sorted = values.slice().sort((a, b) => a - b);
  let sum = 0;
  for (let i = 0; i < n; i++) sum += sorted[i];
  const mean = sum / n;
  let sqSum = 0;
  for (let i = 0; i < n; i++) {
    const d = sorted[i] - mean;
    sqSum += d * d;
  }
  const std = Math.sqrt(sqSum / Math.max(1, n));
  return {
    mean,
    std,
    min: sorted[0],
    max: sorted[n - 1],
    p10: percentile(sorted, 0.1),
    p25: percentile(sorted, 0.25),
    p50: percentile(sorted, 0.5),
    p75: percentile(sorted, 0.75),
    p90: percentile(sorted, 0.9),
    n,
  };
}

export interface LabeledSpriteSample {
  spriteId: number;
  category: ItemCategory;
  rgba: Uint8ClampedArray;
  isTile?: boolean;
}

export function buildCategoryProfileFromVectors(
  category: ItemCategory,
  vectors: readonly SpriteFeatureVector[],
  tileCount: number,
): CategoryProfile {
  const features = {} as Record<ProfileFeatureKey, FeatureDistribution>;
  for (const key of PROFILE_FEATURE_KEYS) {
    const vals = vectors.map((v) => v[key]);
    features[key] = computeFeatureDistribution(vals);
  }
  return {
    category,
    sampleCount: vectors.length,
    tileSampleCount: tileCount,
    features,
  };
}

/**
 * Mines per-category statistical distributions strictly from the real `train` split sprites.
 * Never touches validation or test splits.
 */
export function mineCategoryProfiles(samples: readonly LabeledSpriteSample[]): {
  profiles: Record<ItemCategory, CategoryProfile>;
  globalProfile: CategoryProfile;
  featureVectors: Array<{ sample: LabeledSpriteSample; features: SpriteFeatureVector }>;
} {
  const grouped: Record<ItemCategory, { vectors: SpriteFeatureVector[]; tileCount: number }> = {
    Equipamentos: { vectors: [], tileCount: 0 },
    'Consumíveis': { vectors: [], tileCount: 0 },
    Containers: { vectors: [], tileCount: 0 },
    'Decoração': { vectors: [], tileCount: 0 },
  };

  const allVectors: SpriteFeatureVector[] = [];
  let allTiles = 0;
  const featureVectors: Array<{ sample: LabeledSpriteSample; features: SpriteFeatureVector }> = [];

  for (const sample of samples) {
    const feat = extractSpriteFeatures(sample.rgba);
    if (feat.opaquePixels === 0) continue;
    const isTile = Boolean(sample.isTile || feat.coverage > 0.9);
    grouped[sample.category].vectors.push(feat);
    if (isTile) {
      grouped[sample.category].tileCount++;
      allTiles++;
    }
    allVectors.push(feat);
    featureVectors.push({ sample, features: feat });
  }

  const globalProfile = buildCategoryProfileFromVectors('Equipamentos', allVectors, allTiles);
  const profiles = {} as Record<ItemCategory, CategoryProfile>;

  for (const cat of ALL_ITEM_CATEGORIES) {
    const g = grouped[cat];
    profiles[cat] =
      g.vectors.length > 0
        ? buildCategoryProfileFromVectors(cat, g.vectors, g.tileCount)
        : { ...globalProfile, category: cat, sampleCount: 0, tileSampleCount: 0 };
  }

  return { profiles, globalProfile, featureVectors };
}
