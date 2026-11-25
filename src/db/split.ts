import { ALL_ITEM_CATEGORIES, type ItemCategory, type ParsedItemMetadata, type ParsedSprArchive } from '../types/studio';
import type { SplitCategoryPartition, StoredDatasetSplit } from './schema';

export const DEFAULT_SPLIT_SEED = 77202026;
export const SPLIT_RATIOS = {
  train: 0.7,
  validation: 0.15,
  test: 0.15,
} as const;

/**
 * Deterministic Mulberry32 PRNG returning values in [0, 1).
 */
export function createSeededRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function seededShuffle<T>(input: readonly T[], rng: () => number): T[] {
  const arr = input.slice();
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}

export interface SpriteSplitCandidate {
  spriteId: number;
  category: ItemCategory;
}

/**
 * Extracts unique non-empty primary sprite IDs from parsed `.dat` items and `.spr` offset table.
 * Each spriteId is assigned to its first encountered category so no spriteId can ever appear
 * in two categories or leak across train/validation/test splits.
 */
export function collectUniqueNonEmptySprites(
  items: readonly ParsedItemMetadata[],
  spr: Pick<ParsedSprArchive, 'spriteCount' | 'offsets'>,
): SpriteSplitCandidate[] {
  const seen = new Set<number>();
  const out: SpriteSplitCandidate[] = [];

  for (const item of items) {
    const sid = item.primarySpriteId;
    if (sid <= 0 || sid > spr.spriteCount) continue;
    if (spr.offsets[sid - 1] === 0) continue;
    if (seen.has(sid)) continue;
    seen.add(sid);
    out.push({ spriteId: sid, category: item.category });
  }

  return out;
}

/**
 * Builds a deterministic, category-stratified 70/15/15 (train / validation / test) split
 * keyed by `spriteId`. Test split is strictly separated for explicit final reports.
 */
export function createStratifiedDatasetSplit(params: {
  datasetKey: string;
  candidates: readonly SpriteSplitCandidate[];
  seed?: number;
  createdAt?: string;
}): StoredDatasetSplit {
  const seed = params.seed ?? DEFAULT_SPLIT_SEED;
  const rng = createSeededRng(seed);

  // Deduplicate by spriteId deterministically (ascending sort before stratified shuffle)
  const uniqueById = new Map<number, ItemCategory>();
  for (const c of params.candidates) {
    if (!uniqueById.has(c.spriteId)) {
      uniqueById.set(c.spriteId, c.category);
    }
  }

  const grouped: Record<ItemCategory, number[]> = {
    Equipamentos: [],
    'Consumíveis': [],
    Containers: [],
    'Decoração': [],
  };

  for (const [spriteId, category] of uniqueById.entries()) {
    grouped[category].push(spriteId);
  }

  const byCategory: Record<ItemCategory, SplitCategoryPartition> = {
    Equipamentos: { train: [], validation: [], test: [] },
    'Consumíveis': { train: [], validation: [], test: [] },
    Containers: { train: [], validation: [], test: [] },
    'Decoração': { train: [], validation: [], test: [] },
  };

  let totalTrain = 0;
  let totalValidation = 0;
  let totalTest = 0;

  for (const cat of ALL_ITEM_CATEGORIES) {
    const sortedIds = grouped[cat].slice().sort((a, b) => a - b);
    const shuffled = seededShuffle(sortedIds, rng);
    const n = shuffled.length;

    let trainCount: number;
    let valCount: number;

    if (n === 0) {
      trainCount = 0;
      valCount = 0;
    } else if (n === 1) {
      trainCount = 1;
      valCount = 0;
    } else if (n === 2) {
      trainCount = 1;
      valCount = 1;
    } else {
      trainCount = Math.max(1, Math.round(n * SPLIT_RATIOS.train));
      valCount = Math.max(1, Math.round(n * SPLIT_RATIOS.validation));
      while (trainCount + valCount >= n) {
        if (trainCount > 1) trainCount--;
        else if (valCount > 1) valCount--;
        else break;
      }
    }

    const train = shuffled.slice(0, trainCount);
    const validation = shuffled.slice(trainCount, trainCount + valCount);
    const test = shuffled.slice(trainCount + valCount);

    byCategory[cat] = { train, validation, test };
    totalTrain += train.length;
    totalValidation += validation.length;
    totalTest += test.length;
  }

  return {
    id: params.datasetKey,
    datasetKey: params.datasetKey,
    seed,
    ratios: { ...SPLIT_RATIOS },
    byCategory,
    totals: {
      train: totalTrain,
      validation: totalValidation,
      test: totalTest,
      total: totalTrain + totalValidation + totalTest,
    },
    createdAt: params.createdAt ?? new Date().toISOString(),
  };
}

export function getSplitSpriteIds(
  split: StoredDatasetSplit,
  partition: 'train' | 'validation' | 'test',
  category?: ItemCategory,
): number[] {
  if (category) {
    return split.byCategory[category][partition].slice();
  }
  const all: number[] = [];
  for (const cat of ALL_ITEM_CATEGORIES) {
    all.push(...split.byCategory[cat][partition]);
  }
  return all;
}
