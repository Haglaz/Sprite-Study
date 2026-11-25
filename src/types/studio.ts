export type StudioModuleId = 'upload' | 'refinements' | 'lessons' | 'comparator' | 'studio';

export type ItemCategory = 'Equipamentos' | 'Consumíveis' | 'Containers' | 'Decoração';

export const ALL_ITEM_CATEGORIES: readonly ItemCategory[] = [
  'Equipamentos',
  'Consumíveis',
  'Containers',
  'Decoração',
] as const;

export type ZoomLevel = 1 | 4 | 8;

export type BackdropMode = 'magenta' | 'dirt' | 'stone' | 'transparent' | 'dark' | 'light';

export interface RgbColor {
  r: number;
  g: number;
  b: number;
}

export interface ParsedItemFlags {
  ground: boolean;
  groundBorder: boolean;
  onBottom: boolean;
  onTop: boolean;
  container: boolean;
  stackable: boolean;
  forceUse: boolean;
  multiUse: boolean;
  usable: boolean;
  writable: boolean;
  pickupable: boolean;
  movable: boolean;
  unpassable: boolean;
  fluidContainer: boolean;
  splash: boolean;
  light: boolean;
  hangable: boolean;
  equipable: boolean;
  clothSlot?: number;
  marketName?: string;
}

export interface ParsedItemMetadata {
  id: number;
  category: ItemCategory;
  width: number;
  height: number;
  layers: number;
  xdiv: number;
  ydiv: number;
  zdiv: number;
  animCount: number;
  spriteIds: number[];
  primarySpriteId: number;
  flags: ParsedItemFlags;
}

export interface ParsedSprArchive {
  signature: number;
  signatureHex: string;
  spriteCount: number;
  isExtended: boolean;
  offsets: Uint32Array;
  buffer: ArrayBuffer;
  nonEmptySprites: number;
  hasAlpha?: boolean;
}

export interface ParsedDatArchive {
  signature: number;
  signatureHex: string;
  versionHint: string;
  itemCount: number;
  outfitCount: number;
  effectCount: number;
  missileCount: number;
  items: ParsedItemMetadata[];
  categoryCounts: Record<ItemCategory, number>;
}

export interface LocalDatasetSummary {
  datFileName: string;
  sprFileName: string;
  datSizeBytes: number;
  sprSizeBytes: number;
  datSha256: string;
  sprSha256: string;
  dat: ParsedDatArchive;
  spr: ParsedSprArchive;
  dominantPaletteHex: string[];
  savedAt: string;
  restoredFromDb: boolean;
}
