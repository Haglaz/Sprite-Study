import type {
  ItemCategory,
  ParsedDatArchive,
  ParsedItemMetadata,
  ParsedSprArchive,
} from '../types/studio';

export const SPRITE_SIZE = 32;
export const SPRITE_PIXEL_COUNT = SPRITE_SIZE * SPRITE_SIZE;

export class TibiaParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TibiaParseError';
  }
}

const KNOWN_SIGNATURES: Record<number, string> = {
  0x3e3b00a9: 'Tibia 7.1',
  0x3f8f0f4c: 'Tibia 7.2',
  0x406c0866: 'Tibia 7.26',
  0x40f258a0: 'Tibia 7.3',
  0x41bf619c: 'Tibia 7.4',
  0x42f81973: 'Tibia 7.5',
  0x439d5a33: 'Tibia 7.6',
  0x446b0e57: 'Tibia 7.7',
  0x439d5a34: 'Tibia 7.72',
  0x4475a7d6: 'Tibia 7.72',
  0x4c2b108c: 'Tibia 8.60',
  0x5383504e: 'Tibia 10.41 (Extended)',
  0x53835077: 'Tibia 10.41 (Extended)',
};

function toHexSignature(value: number): string {
  return `0x${(value >>> 0).toString(16).toUpperCase().padStart(8, '0')}`;
}

/**
 * Checks whether an RLE chunk at `offset` cleanly consumes `dataSize` bytes
 * with `bytesPerPixel` (3 for classic RGB, 4 for alpha RGBA).
 */
function testSpriteRleLayout(
  view: DataView,
  byteLength: number,
  offset: number,
  bytesPerPixel: 3 | 4,
): boolean {
  if (offset <= 0 || offset + 5 > byteLength) return false;
  const dataSize = view.getUint16(offset + 3, true);
  let cursor = offset + 5;
  const endCursor = cursor + dataSize;
  if (endCursor > byteLength) return false;

  let currentPixel = 0;
  while (cursor + 4 <= endCursor && currentPixel <= SPRITE_PIXEL_COUNT) {
    const transparentPixels = view.getUint16(cursor, true);
    const coloredPixels = view.getUint16(cursor + 2, true);
    cursor += 4;
    currentPixel += transparentPixels + coloredPixels;
    if (currentPixel > SPRITE_PIXEL_COUNT) return false;
    cursor += coloredPixels * bytesPerPixel;
    if (cursor > endCursor) return false;
  }

  return cursor === endCursor;
}

interface SprCandidateResult {
  valid: boolean;
  spriteCount: number;
  tableStart: number;
  isExtended: boolean;
  validOffsets: number;
  invalidOffsets: number;
  hasAlpha: boolean;
}

function evaluateSprCandidate(
  view: DataView,
  byteLength: number,
  isExtended: boolean,
): SprCandidateResult {
  const tableStart = isExtended ? 8 : 6;
  if (byteLength < tableStart) {
    return {
      valid: false,
      spriteCount: 0,
      tableStart,
      isExtended,
      validOffsets: 0,
      invalidOffsets: 1,
      hasAlpha: false,
    };
  }

  const spriteCount = isExtended ? view.getUint32(4, true) : view.getUint16(4, true);
  const tableEnd = tableStart + spriteCount * 4;
  if (spriteCount === 0 || spriteCount > 2_000_000 || tableEnd > byteLength) {
    return {
      valid: false,
      spriteCount,
      tableStart,
      isExtended,
      validOffsets: 0,
      invalidOffsets: 1,
      hasAlpha: false,
    };
  }

  let validOffsets = 0;
  let invalidOffsets = 0;
  let rgbMatches = 0;
  let rgbaMatches = 0;
  const sampleLimit = Math.min(spriteCount, 256);

  for (let i = 0; i < sampleLimit; i++) {
    const off = view.getUint32(tableStart + i * 4, true);
    if (off === 0) continue;
    if (off >= tableEnd && off + 5 <= byteLength) {
      validOffsets += 1;
      if (rgbMatches + rgbaMatches < 24) {
        if (testSpriteRleLayout(view, byteLength, off, 3)) rgbMatches += 1;
        if (testSpriteRleLayout(view, byteLength, off, 4)) rgbaMatches += 1;
      }
    } else {
      invalidOffsets += 1;
    }
  }

  const hasValidStructure =
    invalidOffsets === 0 && (validOffsets > 0 || byteLength === tableEnd);
  const hasAlpha = rgbaMatches > rgbMatches;

  return {
    valid: hasValidStructure,
    spriteCount,
    tableStart,
    isExtended,
    validOffsets,
    invalidOffsets,
    hasAlpha,
  };
}

/**
 * Parses both Classic (Tibia 7.1–9.54, uint16 spriteCount) and Extended
 * (Tibia 9.60–10.x / Object Builder, uint32 spriteCount) `.spr` binary archives.
 * Automatically validates offset tables and detects 3-byte RGB vs 4-byte RGBA RLE.
 */
export function parseTibiaSpr(buffer: ArrayBuffer): ParsedSprArchive & { hasAlpha?: boolean } {
  if (buffer.byteLength < 6) {
    throw new TibiaParseError(
      'Arquivo .spr inválido ou corrompido: cabeçalho menor que 6 bytes.',
    );
  }

  const view = new DataView(buffer);
  const signature = view.getUint32(0, true);

  const extendedCandidate = evaluateSprCandidate(view, buffer.byteLength, true);
  const classicCandidate = evaluateSprCandidate(view, buffer.byteLength, false);

  let chosen: SprCandidateResult | null = null;
  if (extendedCandidate.valid && !classicCandidate.valid) {
    chosen = extendedCandidate;
  } else if (classicCandidate.valid && !extendedCandidate.valid) {
    chosen = classicCandidate;
  } else if (extendedCandidate.valid && classicCandidate.valid) {
    // Prefer whichever had more verified valid offsets, breaking ties by classic for small files
    chosen =
      extendedCandidate.validOffsets > classicCandidate.validOffsets
        ? extendedCandidate
        : classicCandidate;
  }

  if (!chosen) {
    throw new TibiaParseError(
      'Não foi possível validar a tabela de offsets do arquivo .spr (formato clássico ou Extended).',
    );
  }

  const { spriteCount, tableStart, isExtended, hasAlpha } = chosen;
  const tableEnd = tableStart + spriteCount * 4;
  const offsets = new Uint32Array(spriteCount);
  let nonEmptySprites = 0;

  for (let i = 0; i < spriteCount; i++) {
    const offset = view.getUint32(tableStart + i * 4, true);
    if (offset >= tableEnd && offset + 5 <= buffer.byteLength) {
      offsets[i] = offset;
      nonEmptySprites += 1;
    } else {
      offsets[i] = 0;
    }
  }

  return {
    signature,
    signatureHex: toHexSignature(signature),
    spriteCount,
    isExtended,
    offsets,
    buffer,
    nonEmptySprites,
    hasAlpha,
  };
}

/**
 * Decodes a single 1-based sprite ID from a parsed `.spr` archive via RLE.
 * Supports both classic 3-byte RGB (`#FF00FF` key) and 4-byte RGBA alpha sprites.
 */
export function decodeSpritePixels(
  spr: ParsedSprArchive & { hasAlpha?: boolean },
  spriteId: number,
): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(SPRITE_PIXEL_COUNT * 4);

  if (spriteId <= 0 || spriteId > spr.spriteCount) {
    return rgba;
  }

  const offset = spr.offsets[spriteId - 1];
  if (!offset || offset + 5 > spr.buffer.byteLength) {
    return rgba;
  }

  const view = new DataView(spr.buffer);
  const bytes = new Uint8Array(spr.buffer);

  const keyR = bytes[offset];
  const keyG = bytes[offset + 1];
  const keyB = bytes[offset + 2];
  const dataSize = view.getUint16(offset + 3, true);
  const bytesPerPixel: 3 | 4 = spr.hasAlpha ? 4 : 3;

  let cursor = offset + 5;
  const endCursor = Math.min(cursor + dataSize, spr.buffer.byteLength);
  let currentPixel = 0;

  while (cursor + 4 <= endCursor && currentPixel < SPRITE_PIXEL_COUNT) {
    const transparentPixels = view.getUint16(cursor, true);
    cursor += 2;
    const coloredPixels = view.getUint16(cursor, true);
    cursor += 2;

    currentPixel += transparentPixels;
    if (currentPixel >= SPRITE_PIXEL_COUNT) {
      break;
    }

    for (let i = 0; i < coloredPixels && currentPixel < SPRITE_PIXEL_COUNT; i++) {
      if (cursor + bytesPerPixel > endCursor) {
        break;
      }
      const r = bytes[cursor];
      const g = bytes[cursor + 1];
      const b = bytes[cursor + 2];
      const a = bytesPerPixel === 4 ? bytes[cursor + 3] : 255;
      cursor += bytesPerPixel;

      const isKeyTransparent =
        bytesPerPixel === 3 &&
        ((r === 0xff && g === 0x00 && b === 0xff) ||
          (r === keyR &&
            g === keyG &&
            b === keyB &&
            keyR === 0xff &&
            keyG === 0x00 &&
            keyB === 0xff));

      if (!isKeyTransparent && a > 0) {
        const idx = currentPixel * 4;
        rgba[idx] = r;
        rgba[idx + 1] = g;
        rgba[idx + 2] = b;
        rgba[idx + 3] = a;
      }
      currentPixel += 1;
    }
  }

  return rgba;
}

export function decodeSpriteToImageData(
  spr: ParsedSprArchive,
  spriteId: number,
): ImageData {
  const rgba = decodeSpritePixels(spr, spriteId);
  return new ImageData(new Uint8ClampedArray(rgba), SPRITE_SIZE, SPRITE_SIZE);
}

export function classifyItemCategory(flags: ParsedItemMetadata['flags']): ItemCategory {
  if (flags.container) {
    return 'Containers';
  }
  if (flags.stackable || flags.fluidContainer || flags.splash) {
    return 'Consumíveis';
  }
  if (flags.equipable || (flags.pickupable && !flags.ground && !flags.hangable)) {
    return 'Equipamentos';
  }
  return 'Decoração';
}

interface DatProfileSpec {
  label: string;
  hasZDiv: boolean;
  extendedSpriteIds: boolean;
  improvedAnimations: boolean;
  frameGroups: boolean;
  marketAttr?: number;
  maxKnownAttr: number;
  flagPayloadBytes: Record<number, number>;
  flagMap: {
    ground: number;
    groundBorder: number;
    onBottom: number;
    onTop: number;
    container: number;
    stackable: number;
    forceUse: number;
    multiUse: number;
    usable: number;
    writable: number;
    writableOnce: number;
    pickupable: number;
    notMovable: number;
    unpassable: number;
    fluidContainer: number;
    splash: number;
    light: number;
    hangable: number;
    equipable: number;
  };
}

// Tibia 10.10–10.49 (with 0x10 NoMoveAnimation, 0x22 Market, 0x23 DefaultAction, 0xFE Usable)
const BASE_ATTRS_1010 = {
  marketAttr: 0x22,
  maxKnownAttr: 0x26,
  flagPayloadBytes: {
    0x00: 2, // Ground speed
    0x08: 2, // Writable
    0x09: 2, // WritableOnce
    0x16: 4, // Light (level + color)
    0x19: 4, // Displacement (x + y)
    0x1a: 2, // Elevation
    0x1d: 2, // MinimapColor
    0x1e: 2, // LensHelp
    0x21: 2, // Cloth slot
    0x23: 2, // DefaultAction
  },
  flagMap: {
    ground: 0x00,
    groundBorder: 0x01,
    onBottom: 0x02,
    onTop: 0x03,
    container: 0x04,
    stackable: 0x05,
    forceUse: 0x06,
    multiUse: 0x07,
    writable: 0x08,
    writableOnce: 0x09,
    usable: 0xfe,
    fluidContainer: 0x0a,
    splash: 0x0b,
    unpassable: 0x0c,
    notMovable: 0x0d,
    pickupable: 0x11,
    hangable: 0x12,
    light: 0x16,
    equipable: 0x21,
  },
};

// Tibia 7.55–10.00 (without 0x10 NoMoveAnimation; 0x10 = Pickupable, 0x15 = Light, 0x18 = Displacement, 0x1C = MinimapColor)
const BASE_ATTRS_755_1000 = {
  marketAttr: 0x21,
  maxKnownAttr: 0x25,
  flagPayloadBytes: {
    0x00: 2, // Ground speed
    0x08: 2, // Writable
    0x09: 2, // WritableOnce
    0x15: 4, // Light (level + color)
    0x18: 4, // Displacement (x + y)
    0x19: 2, // Elevation
    0x1c: 2, // MinimapColor
    0x1d: 2, // LensHelp
    0x20: 2, // Cloth slot
    0x22: 2, // DefaultAction
  },
  flagMap: {
    ground: 0x00,
    groundBorder: 0x01,
    onBottom: 0x02,
    onTop: 0x03,
    container: 0x04,
    stackable: 0x05,
    forceUse: 0x06,
    multiUse: 0x07,
    usable: 0x07,
    writable: 0x08,
    writableOnce: 0x09,
    fluidContainer: 0x0a,
    splash: 0x0b,
    unpassable: 0x0c,
    notMovable: 0x0d,
    pickupable: 0x10,
    hangable: 0x11,
    light: 0x15,
    equipable: 0x20,
  },
};

// Tibia 7.10–7.50 (no zdiv byte; 0x03 Container, 0x04 Stackable, 0x07 Writable, 0x0F Pickupable, 0x10 Light)
const BASE_ATTRS_710_750 = {
  maxKnownAttr: 0x1f,
  flagPayloadBytes: {
    0x00: 2, // Ground speed
    0x07: 2, // Writable
    0x08: 2, // WritableOnce
    0x10: 4, // Light (level + color)
    0x13: 2, // Elevation
    0x16: 2, // MinimapColor
    0x1d: 2, // LensHelp
  },
  flagMap: {
    ground: 0x00,
    groundBorder: 0xfe,
    onBottom: 0x01,
    onTop: 0x02,
    container: 0x03,
    stackable: 0x04,
    forceUse: 0x05,
    multiUse: 0x06,
    usable: 0x05,
    writable: 0x07,
    writableOnce: 0x08,
    fluidContainer: 0x09,
    splash: 0x0a,
    unpassable: 0x0b,
    notMovable: 0x0c,
    pickupable: 0x0f,
    hangable: 0x19,
    light: 0x10,
    equipable: 0x0f,
  },
};

const DAT_PROFILES: DatProfileSpec[] = [
  // Tibia 10.10–10.49 Extended (uint32 sprite IDs) — e.g. 10.41 Object Builder
  {
    label: 'Tibia 10.10–10.49 (Extended)',
    hasZDiv: true,
    extendedSpriteIds: true,
    improvedAnimations: false,
    frameGroups: false,
    ...BASE_ATTRS_1010,
  },
  // Tibia 10.50–10.56 Extended + Improved Animations
  {
    label: 'Tibia 10.50–10.56 (Extended)',
    hasZDiv: true,
    extendedSpriteIds: true,
    improvedAnimations: true,
    frameGroups: false,
    ...BASE_ATTRS_1010,
  },
  // Tibia 10.57–10.98+ Extended + Improved Animations + Outfit Frame Groups
  {
    label: 'Tibia 10.57+ (Extended)',
    hasZDiv: true,
    extendedSpriteIds: true,
    improvedAnimations: true,
    frameGroups: true,
    ...BASE_ATTRS_1010,
  },
  // Tibia 7.55–10.00 Classic (uint16 sprite IDs) — covers 7.6, 7.72, 8.54, 8.60
  {
    label: 'Tibia 7.55–8.60',
    hasZDiv: true,
    extendedSpriteIds: false,
    improvedAnimations: false,
    frameGroups: false,
    ...BASE_ATTRS_755_1000,
  },
  // Tibia 7.55–10.00 Extended (uint32 sprite IDs, custom OTClient / Object Builder)
  {
    label: 'Tibia 7.6–9.8 (Extended)',
    hasZDiv: true,
    extendedSpriteIds: true,
    improvedAnimations: false,
    frameGroups: false,
    ...BASE_ATTRS_755_1000,
  },
  // Tibia 10.10–10.49 Classic (uint16 sprite IDs)
  {
    label: 'Tibia 10.10–10.49',
    hasZDiv: true,
    extendedSpriteIds: false,
    improvedAnimations: false,
    frameGroups: false,
    ...BASE_ATTRS_1010,
  },
  // Tibia 7.10–7.50 Classic (no zdiv)
  {
    label: 'Tibia 7.1–7.5',
    hasZDiv: false,
    extendedSpriteIds: false,
    improvedAnimations: false,
    frameGroups: false,
    ...BASE_ATTRS_710_750,
  },
];

interface DatAttemptResult {
  items: ParsedItemMetadata[];
  exactEof: boolean;
  bytesRemaining: number;
}

function tryParseDatWithProfile(
  view: DataView,
  itemCount: number,
  outfitCount: number,
  effectCount: number,
  missileCount: number,
  spec: DatProfileSpec,
  maxSpriteId?: number,
): DatAttemptResult | null {
  const byteLength = view.byteLength;
  let cursor = 12;
  const items: ParsedItemMetadata[] = [];

  const categories: Array<{
    kind: 'items' | 'outfits' | 'effects' | 'missiles';
    startId: number;
    endId: number;
  }> = [
    { kind: 'items', startId: 100, endId: itemCount },
    { kind: 'outfits', startId: 1, endId: outfitCount },
    { kind: 'effects', startId: 1, endId: effectCount },
    { kind: 'missiles', startId: 1, endId: missileCount },
  ];

  for (const cat of categories) {
    for (let id = cat.startId; id <= cat.endId; id++) {
      const flags: ParsedItemMetadata['flags'] = {
        ground: false,
        groundBorder: false,
        onBottom: false,
        onTop: false,
        container: false,
        stackable: false,
        forceUse: false,
        multiUse: false,
        usable: false,
        writable: false,
        pickupable: false,
        movable: true,
        unpassable: false,
        fluidContainer: false,
        splash: false,
        light: false,
        hangable: false,
        equipable: false,
      };

      let guard = 0;
      while (true) {
        if (cursor >= byteLength || guard > 64) {
          return null;
        }
        const attr = view.getUint8(cursor);
        cursor += 1;
        guard += 1;

        if (attr === 0xff) {
          break;
        }

        if (attr > spec.maxKnownAttr && attr !== 0xfe) {
          return null;
        }

        if (cat.kind === 'items') {
          if (attr === spec.flagMap.ground) flags.ground = true;
          if (attr === spec.flagMap.groundBorder) flags.groundBorder = true;
          if (attr === spec.flagMap.onBottom) flags.onBottom = true;
          if (attr === spec.flagMap.onTop) flags.onTop = true;
          if (attr === spec.flagMap.container) flags.container = true;
          if (attr === spec.flagMap.stackable) flags.stackable = true;
          if (attr === spec.flagMap.forceUse) flags.forceUse = true;
          if (attr === spec.flagMap.multiUse) flags.multiUse = true;
          if (attr === spec.flagMap.usable) flags.usable = true;
          if (attr === spec.flagMap.writable || attr === spec.flagMap.writableOnce) {
            flags.writable = true;
          }
          if (attr === spec.flagMap.pickupable) flags.pickupable = true;
          if (attr === spec.flagMap.notMovable) flags.movable = false;
          if (attr === spec.flagMap.unpassable) flags.unpassable = true;
          if (attr === spec.flagMap.fluidContainer) flags.fluidContainer = true;
          if (attr === spec.flagMap.splash) flags.splash = true;
          if (attr === spec.flagMap.light) flags.light = true;
          if (attr === spec.flagMap.hangable) flags.hangable = true;
          if (attr === spec.flagMap.equipable) {
            flags.equipable = true;
            if (spec.flagPayloadBytes[attr] === 2 && cursor + 2 <= byteLength) {
              flags.clothSlot = view.getUint16(cursor, true);
            }
          }
        }

        if (spec.marketAttr !== undefined && attr === spec.marketAttr) {
          // Market: category(u16) + tradeAs(u16) + showAs(u16) + nameLen(u16) + name + restrictVoc(u16) + reqLevel(u16)
          if (cursor + 8 > byteLength) return null;
          cursor += 6;
          const nameLen = view.getUint16(cursor, true);
          cursor += 2;
          if (nameLen > 256 || cursor + nameLen + 4 > byteLength) return null;
          if (cat.kind === 'items' && nameLen > 0) {
            const nameBytes = new Uint8Array(view.buffer, view.byteOffset + cursor, nameLen);
            let decoded = '';
            for (let b = 0; b < nameBytes.length; b++) {
              decoded += String.fromCharCode(nameBytes[b]);
            }
            flags.marketName = decoded.trim();
          }
          cursor += nameLen + 4;
        } else {
          const skip = spec.flagPayloadBytes[attr] ?? 0;
          if (cursor + skip > byteLength) {
            return null;
          }
          cursor += skip;
        }
      }

      // Frame groups exist on outfits in Tibia 10.57+
      let groupCount = 1;
      if (spec.frameGroups && cat.kind === 'outfits') {
        if (cursor >= byteLength) return null;
        groupCount = view.getUint8(cursor++);
        if (groupCount === 0 || groupCount > 4) return null;
      }

      let itemWidth = 1;
      let itemHeight = 1;
      let itemLayers = 1;
      let itemXDiv = 1;
      let itemYDiv = 1;
      let itemZDiv = 1;
      let itemAnim = 1;
      let itemSpriteIds: number[] = [];
      let itemPrimarySpriteId = 0;

      for (let g = 0; g < groupCount; g++) {
        if (spec.frameGroups && cat.kind === 'outfits') {
          if (cursor >= byteLength) return null;
          cursor += 1; // frameGroupType
        }

        if (cursor + 2 > byteLength) return null;
        const width = view.getUint8(cursor++);
        const height = view.getUint8(cursor++);
        if (width === 0 || height === 0 || width > 16 || height > 16) return null;

        if (width > 1 || height > 1) {
          if (cursor >= byteLength) return null;
          cursor += 1; // exactSize
        }

        const remainingMetaBytes = spec.hasZDiv ? 5 : 4;
        if (cursor + remainingMetaBytes > byteLength) return null;

        const layers = view.getUint8(cursor++);
        const xdiv = view.getUint8(cursor++);
        const ydiv = view.getUint8(cursor++);
        const zdiv = spec.hasZDiv ? view.getUint8(cursor++) : 1;
        const animCount = view.getUint8(cursor++);

        if (
          layers === 0 ||
          xdiv === 0 ||
          ydiv === 0 ||
          zdiv === 0 ||
          animCount === 0 ||
          layers > 16 ||
          animCount > 128
        ) {
          return null;
        }

        if (spec.improvedAnimations && animCount > 1) {
          // async(1) + loopCount(4) + startPhase(1) + animCount * (min(4) + max(4))
          const animBytes = 6 + animCount * 8;
          if (cursor + animBytes > byteLength) return null;
          cursor += animBytes;
        }

        const totalSprites = width * height * layers * xdiv * ydiv * zdiv * animCount;
        const spriteBytes = spec.extendedSpriteIds ? 4 : 2;
        if (totalSprites > 16384 || cursor + totalSprites * spriteBytes > byteLength) {
          return null;
        }

        const spriteIds: number[] = new Array(totalSprites);
        let primarySpriteId = 0;

        for (let s = 0; s < totalSprites; s++) {
          const spriteId = spec.extendedSpriteIds
            ? view.getUint32(cursor, true)
            : view.getUint16(cursor, true);
          cursor += spriteBytes;

          if (maxSpriteId !== undefined && spriteId > maxSpriteId) {
            return null;
          }
          spriteIds[s] = spriteId;
          if (primarySpriteId === 0 && spriteId > 0) {
            primarySpriteId = spriteId;
          }
        }

        if (g === 0) {
          itemWidth = width;
          itemHeight = height;
          itemLayers = layers;
          itemXDiv = xdiv;
          itemYDiv = ydiv;
          itemZDiv = zdiv;
          itemAnim = animCount;
          itemSpriteIds = spriteIds;
          itemPrimarySpriteId = primarySpriteId;
        }
      }

      if (cat.kind === 'items') {
        items.push({
          id,
          category: classifyItemCategory(flags),
          width: itemWidth,
          height: itemHeight,
          layers: itemLayers,
          xdiv: itemXDiv,
          ydiv: itemYDiv,
          zdiv: itemZDiv,
          animCount: itemAnim,
          spriteIds: itemSpriteIds,
          primarySpriteId: itemPrimarySpriteId,
          flags,
        });
      }
    }
  }

  return {
    items,
    exactEof: cursor === byteLength,
    bytesRemaining: byteLength - cursor,
  };
}

/**
 * Parses Classic (Tibia 7.1–8.60) and Extended (Tibia 9.60–10.98+ / Object Builder) `.dat` files,
 * automatically validating against `.spr` extended mode and exact EOF alignment across all categories.
 */
export function parseTibiaDat(
  buffer: ArrayBuffer,
  maxSpriteId?: number,
  preferExtended?: boolean,
): ParsedDatArchive {
  if (buffer.byteLength < 12) {
    throw new TibiaParseError(
      'Arquivo .dat inválido ou corrompido: cabeçalho menor que 12 bytes.',
    );
  }

  const view = new DataView(buffer);
  const signature = view.getUint32(0, true);
  const itemCount = view.getUint16(4, true);
  const outfitCount = view.getUint16(6, true);
  const effectCount = view.getUint16(8, true);
  const missileCount = view.getUint16(10, true);

  if (itemCount < 100) {
    throw new TibiaParseError(
      `Contagem de itens inválida no cabeçalho .dat (${itemCount}). Esperado >= 100.`,
    );
  }

  // Order candidate profiles so matching extended/classic format is checked first
  const orderedProfiles = [...DAT_PROFILES].sort((a, b) => {
    if (preferExtended !== undefined) {
      const aMatch = a.extendedSpriteIds === preferExtended ? 0 : 1;
      const bMatch = b.extendedSpriteIds === preferExtended ? 0 : 1;
      if (aMatch !== bMatch) return aMatch - bMatch;
    }
    return 0;
  });

  let bestMatch: { result: DatAttemptResult; spec: DatProfileSpec } | null = null;

  for (const spec of orderedProfiles) {
    const result = tryParseDatWithProfile(
      view,
      itemCount,
      outfitCount,
      effectCount,
      missileCount,
      spec,
      maxSpriteId,
    );
    if (!result || result.items.length === 0) continue;

    if (result.exactEof) {
      bestMatch = { result, spec };
      break;
    }
    if (!bestMatch || Math.abs(result.bytesRemaining) < Math.abs(bestMatch.result.bytesRemaining)) {
      bestMatch = { result, spec };
    }
  }

  if (!bestMatch) {
    throw new TibiaParseError(
      `Não foi possível decodificar o arquivo .dat (assinatura ${toHexSignature(signature)}, ${itemCount} itens). Verifique se o par .dat e .spr pertence à mesma versão.`,
    );
  }

  const parsedItems = bestMatch.result.items;
  const detectedLabel = KNOWN_SIGNATURES[signature] ?? bestMatch.spec.label;

  const categoryCounts: Record<ItemCategory, number> = {
    Equipamentos: 0,
    Consumíveis: 0,
    Containers: 0,
    Decoração: 0,
  };

  for (const item of parsedItems) {
    categoryCounts[item.category] += 1;
  }

  return {
    signature,
    signatureHex: toHexSignature(signature),
    versionHint: detectedLabel,
    itemCount,
    outfitCount,
    effectCount,
    missileCount,
    items: parsedItems,
    categoryCounts,
  };
}

/**
 * Extracts up to `maxColors` dominant hex colors directly from decoded `.spr` sprites
 * (excluding transparent pixels and pure black outline `#000000`).
 */
export function extractDominantPalette(
  spr: ParsedSprArchive,
  items: ParsedItemMetadata[],
  maxColors = 16,
): string[] {
  const counts = new Map<number, number>();
  const sampleLimit = Math.min(items.length, 240);

  for (let i = 0; i < sampleLimit; i++) {
    const spriteId = items[i].primarySpriteId;
    if (!spriteId) continue;
    const rgba = decodeSpritePixels(spr, spriteId);
    for (let p = 0; p < rgba.length; p += 4) {
      const a = rgba[p + 3];
      if (a < 128) continue;
      const r = rgba[p];
      const g = rgba[p + 1];
      const b = rgba[p + 2];
      if (r === 0 && g === 0 && b === 0) continue;
      const packed = (r << 16) | (g << 8) | b;
      counts.set(packed, (counts.get(packed) ?? 0) + 1);
    }
  }

  return Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxColors)
    .map(([packed]) => `#${packed.toString(16).toUpperCase().padStart(6, '0')}`);
}
