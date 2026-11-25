import type { ItemCategory } from '../types/studio';

export const ATELIER_DB_NAME = 'atelier_sprite_v2';
export const ATELIER_DB_VERSION = 1;

export const STORE_NAMES = [
  'datasets',
  'splits',
  'criticState',
  'neatPopulations',
  'gaPopulations',
  'eliteArchive',
  'lessons',
  'agents',
  'approvals',
  'generatedSprites',
  'checkpoints',
  'onnxModels',
] as const;

export type AtelierStoreName = (typeof STORE_NAMES)[number];

export interface StoredDatasetRecord {
  id: 'active';
  datFileName: string;
  sprFileName: string;
  datSizeBytes: number;
  sprSizeBytes: number;
  datSha256: string;
  sprSha256: string;
  datBuffer: ArrayBuffer;
  sprBuffer: ArrayBuffer;
  itemCount: number;
  spriteCount: number;
  nonEmptySprites: number;
  versionHint: string;
  savedAt: string;
}

export type SplitPartition = 'train' | 'validation' | 'test';

export interface SplitCategoryPartition {
  train: number[];
  validation: number[];
  test: number[];
}

export interface StoredDatasetSplit {
  id: string; // matches datasetKey `${datSha256}:${sprSha256}`
  datasetKey: string;
  seed: number;
  ratios: { train: number; validation: number; test: number };
  byCategory: Record<ItemCategory, SplitCategoryPartition>;
  totals: { train: number; validation: number; test: number; total: number };
  createdAt: string;
}

export interface StoredCriticState {
  id: string; // matches datasetKey
  datasetKey: string;
  calibratedAt: string;
  seed: number;
  payload: unknown;
}

export interface GenericStoreRecord {
  id: string;
  updatedAt: string;
  data: unknown;
}

export interface BackupEnvelope {
  schema: 'atelier-sprite-backup-v2';
  exportedAt: string;
  stores: Record<AtelierStoreName, unknown[]>;
}
