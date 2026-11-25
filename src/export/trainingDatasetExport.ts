import type { StoredDatasetSplit } from '../db/schema';
import { encodeRgbaToVocabGrid } from '../t3/vocabulary';
import type { T3RampVocabulary } from '../t3/types';
import type {
  ApprovalLabelRecord,
  GeneratedSpriteRecord,
  IndexedSpriteEntry,
  SpriteIndexState,
} from '../generation/types';
import { RAMP_ROLE_NAMES } from '../generation/types';
import { SPRITE_SIZE } from '../utils/tibiaParser';

// ============================================================================
// 1. PURE SYNCHRONOUS PNG ENCODER (TRUE 32-BIT RGBA ALPHA, ZERO DEPENDENCIES)
// ============================================================================

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) !== 0 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function adler32(bytes: Uint8Array): number {
  let s1 = 1;
  let s2 = 0;
  for (let i = 0; i < bytes.length; i++) {
    s1 = (s1 + bytes[i]) % 65521;
    s2 = (s2 + s1) % 65521;
  }
  return ((s2 << 16) | s1) >>> 0;
}

function writeU32BE(buf: Uint8Array, offset: number, val: number): void {
  buf[offset] = (val >>> 24) & 0xff;
  buf[offset + 1] = (val >>> 16) & 0xff;
  buf[offset + 2] = (val >>> 8) & 0xff;
  buf[offset + 3] = val & 0xff;
}

function makePngChunk(type: string, data: Uint8Array): Uint8Array {
  const chunk = new Uint8Array(12 + data.length);
  writeU32BE(chunk, 0, data.length);
  for (let i = 0; i < 4; i++) chunk[4 + i] = type.charCodeAt(i);
  chunk.set(data, 8);
  const crc = crc32(chunk.subarray(4, 8 + data.length));
  writeU32BE(chunk, 8 + data.length, crc);
  return chunk;
}

/**
 * Encodes a native-size (32x32) RGBA pixel buffer into a 100% valid PNG with true 8-bit alpha
 * using spec-compliant uncompressed zlib deflate blocks (works identically in Node and browsers).
 */
export function encodeRgbaToPngBytes(
  rgba: Uint8ClampedArray,
  width = SPRITE_SIZE,
  height = SPRITE_SIZE,
): Uint8Array {
  if (rgba.length !== width * height * 4) {
    throw new Error(`Invalid RGBA length: expected ${width * height * 4}, got ${rgba.length}`);
  }

  // Raw PNG scanlines: 1 filter byte (0=None) + width*4 RGBA bytes per row
  const rowBytes = width * 4;
  const raw = new Uint8Array((rowBytes + 1) * height);
  for (let y = 0; y < height; y++) {
    const dstRow = y * (rowBytes + 1);
    raw[dstRow] = 0; // filter None
    const srcRow = y * rowBytes;
    raw.set(rgba.subarray(srcRow, srcRow + rowBytes), dstRow + 1);
  }

  // Wrap raw scanlines in valid ZLIB stream (CMF=0x78, FLG=0x01, stored DEFLATE block, Adler-32)
  // 32x32 raw is 4,128 bytes (< 65,535), fits cleanly in a single stored block!
  const len = raw.length;
  const nlen = (~len) & 0xffff;
  const zlib = new Uint8Array(2 + 5 + len + 4);
  zlib[0] = 0x78;
  zlib[1] = 0x01;
  zlib[2] = 0x01; // BFINAL=1, BTYPE=00 (uncompressed)
  zlib[3] = len & 0xff;
  zlib[4] = (len >>> 8) & 0xff;
  zlib[5] = nlen & 0xff;
  zlib[6] = (nlen >>> 8) & 0xff;
  zlib.set(raw, 7);
  writeU32BE(zlib, 7 + len, adler32(raw));

  const ihdr = new Uint8Array(13);
  writeU32BE(ihdr, 0, width);
  writeU32BE(ihdr, 4, height);
  ihdr[8] = 8; // bit depth 8
  ihdr[9] = 6; // color type 6 = RGBA (true alpha)
  ihdr[10] = 0; // compression
  ihdr[11] = 0; // filter
  ihdr[12] = 0; // interlace

  const signature = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdrChunk = makePngChunk('IHDR', ihdr);
  const idatChunk = makePngChunk('IDAT', zlib);
  const iendChunk = makePngChunk('IEND', new Uint8Array(0));

  const png = new Uint8Array(
    signature.length + ihdrChunk.length + idatChunk.length + iendChunk.length,
  );
  let pos = 0;
  png.set(signature, pos);
  pos += signature.length;
  png.set(ihdrChunk, pos);
  pos += ihdrChunk.length;
  png.set(idatChunk, pos);
  pos += idatChunk.length;
  png.set(iendChunk, pos);
  return png;
}

// ============================================================================
// 2. PURE SYNCHRONOUS ZIP ARCHIVE BUILDER (STORE METHOD, VALID CRC-32)
// ============================================================================

export interface ZipEntryInput {
  path: string;
  data: Uint8Array;
}

function writeU16LE(buf: Uint8Array, offset: number, val: number): void {
  buf[offset] = val & 0xff;
  buf[offset + 1] = (val >>> 8) & 0xff;
}

function writeU32LE(buf: Uint8Array, offset: number, val: number): void {
  buf[offset] = val & 0xff;
  buf[offset + 1] = (val >>> 8) & 0xff;
  buf[offset + 2] = (val >>> 16) & 0xff;
  buf[offset + 3] = (val >>> 24) & 0xff;
}

export function buildZipArchive(entries: ZipEntryInput[]): Uint8Array {
  const encoder = new TextEncoder();
  const localParts: Uint8Array[] = [];
  const centralParts: Uint8Array[] = [];
  let offset = 0;

  for (const entry of entries) {
    const nameBytes = encoder.encode(entry.path);
    const data = entry.data;
    const crc = crc32(data);

    // Local file header (30 bytes + name)
    const localHeader = new Uint8Array(30 + nameBytes.length);
    writeU32LE(localHeader, 0, 0x04034b50);
    writeU16LE(localHeader, 4, 20); // version needed
    writeU16LE(localHeader, 6, 0x0800); // UTF-8 flag
    writeU16LE(localHeader, 8, 0); // method 0 (store)
    writeU16LE(localHeader, 10, 0);
    writeU16LE(localHeader, 12, 0);
    writeU32LE(localHeader, 14, crc);
    writeU32LE(localHeader, 18, data.length);
    writeU32LE(localHeader, 22, data.length);
    writeU16LE(localHeader, 26, nameBytes.length);
    writeU16LE(localHeader, 28, 0);
    localHeader.set(nameBytes, 30);

    localParts.push(localHeader, data);

    // Central directory header (46 bytes + name)
    const central = new Uint8Array(46 + nameBytes.length);
    writeU32LE(central, 0, 0x02014b50);
    writeU16LE(central, 4, 20);
    writeU16LE(central, 6, 20);
    writeU16LE(central, 8, 0x0800);
    writeU16LE(central, 10, 0);
    writeU16LE(central, 12, 0);
    writeU16LE(central, 14, 0);
    writeU32LE(central, 16, crc);
    writeU32LE(central, 20, data.length);
    writeU32LE(central, 24, data.length);
    writeU16LE(central, 28, nameBytes.length);
    writeU16LE(central, 30, 0);
    writeU16LE(central, 32, 0);
    writeU16LE(central, 34, 0);
    writeU16LE(central, 36, 0);
    writeU32LE(central, 38, 0);
    writeU32LE(central, 42, offset);
    central.set(nameBytes, 46);

    centralParts.push(central);
    offset += localHeader.length + data.length;
  }

  const centralSize = centralParts.reduce((s, p) => s + p.length, 0);
  const eocd = new Uint8Array(22);
  writeU32LE(eocd, 0, 0x06054b50);
  writeU16LE(eocd, 4, 0);
  writeU16LE(eocd, 6, 0);
  writeU16LE(eocd, 8, entries.length);
  writeU16LE(eocd, 10, entries.length);
  writeU32LE(eocd, 12, centralSize);
  writeU32LE(eocd, 16, offset);
  writeU16LE(eocd, 20, 0);

  const totalSize = offset + centralSize + eocd.length;
  const archive = new Uint8Array(totalSize);
  let pos = 0;
  for (const part of localParts) {
    archive.set(part, pos);
    pos += part.length;
  }
  for (const part of centralParts) {
    archive.set(part, pos);
    pos += part.length;
  }
  archive.set(eocd, pos);
  return archive;
}

// ============================================================================
// 3. TRAINING DATASET EXPORT ("Exportar dataset de treino") — TEST EXCLUDED!
// ============================================================================

function toMatrix32(flat: Uint8Array): number[][] {
  const rows: number[][] = [];
  for (let y = 0; y < SPRITE_SIZE; y++) {
    const row: number[] = [];
    for (let x = 0; x < SPRITE_SIZE; x++) {
      row.push(flat[y * SPRITE_SIZE + x]);
    }
    rows.push(row);
  }
  return rows;
}

function escapeCsvCell(val: string): string {
  if (val.includes(',') || val.includes('"') || val.includes('\n')) {
    return `"${val.replace(/"/g, '""')}"`;
  }
  return val;
}

function serializeFlags(entry: IndexedSpriteEntry): string {
  const active: string[] = [];
  if (entry.flags.ground) active.push('ground');
  if (entry.flags.groundBorder) active.push('groundBorder');
  if (entry.flags.container) active.push('container');
  if (entry.flags.stackable) active.push('stackable');
  if (entry.flags.pickupable) active.push('pickupable');
  if (entry.flags.fluidContainer) active.push('fluidContainer');
  if (entry.flags.splash) active.push('splash');
  if (entry.flags.light) active.push('light');
  if (entry.flags.unpassable) active.push('unpassable');
  if (entry.flags.clothSlot !== undefined) active.push(`clothSlot:${entry.flags.clothSlot}`);
  return active.join('|');
}

export interface TrainingDatasetExportResult {
  zipBytes: Uint8Array;
  fileCount: number;
  exportedSpriteCount: number;
  trainCount: number;
  validationCount: number;
  excludedTestCount: number;
  humanApprovedCount: number;
  humanRejectedCount: number;
  hasVocabularyJson: boolean;
  labelsCsv: string;
  humanLabelsCsv: string | null;
  filePaths: string[];
}

/**
 * Builds the complete "Exportar dataset de treino" or "Re-treinar com novos rótulos" ZIP bundle:
 *  - Native-size 32x32 PNGs with true binary alpha (`pngs/<split>/<id>.png`)
 *  - Indexed palette grids (`grids/<id>.json`)
 *  - Per-sprite exact palettes (`palettes/<id>.json`)
 *  - Ramp-role maps (`ramp_roles/<id>.json`)
 *  - `labels.csv` with id, category, flags, tags, split
 *  - Optional `vocabulary.json` with user-corrected material ramp roles
 *  - Optional `human_labels.csv` + `human_labels/{approved,rejected}/<id>.png` + `human_labels/grids/<id>.json`
 *  - Fixed deterministic split (`split.json`) with `test` partition strictly excluded!
 */
export function buildTrainingDatasetExportZip(params: {
  index: SpriteIndexState;
  split: StoredDatasetSplit;
  vocabulary?: T3RampVocabulary | null;
  approvals?: ApprovalLabelRecord[];
  generatedSprites?: GeneratedSpriteRecord[];
  exportedAt?: string;
}): TrainingDatasetExportResult {
  const {
    index,
    split,
    vocabulary = null,
    approvals = [],
    generatedSprites = [],
  } = params;
  const exportedAt = params.exportedAt ?? new Date().toISOString();
  const encoder = new TextEncoder();

  // Defensive double-check: collect test sprite IDs so ZERO test sprites can ever leak into export
  const testIdSet = new Set<number>();
  for (const part of Object.values(split.byCategory)) {
    for (const sid of part.test) testIdSet.add(sid);
  }

  const exportEntries = index.entries.filter(
    (e) => (e.split === 'train' || e.split === 'validation') && !testIdSet.has(e.spriteId),
  );

  const zipFiles: ZipEntryInput[] = [];
  const csvLines: string[] = [
    'id,item_id,category,flags,tags,split,is_tile,opaque_pixels,distinct_colors,critic_score,png_path,grid_path,palette_path,ramp_role_path',
  ];

  let trainCount = 0;
  let validationCount = 0;

  for (const entry of exportEntries) {
    if (entry.split === 'train') trainCount++;
    else validationCount++;

    const idStr = `spr_${entry.spriteId}`;
    const pngPath = `pngs/${entry.split}/${idStr}.png`;
    const gridPath = `grids/${idStr}.json`;
    const palettePath = `palettes/${idStr}.json`;
    const rampRolePath = `ramp_roles/${idStr}.json`;

    // 1. Native 32x32 PNG with true RGBA alpha
    const pngBytes = encodeRgbaToPngBytes(entry.rgba, SPRITE_SIZE, SPRITE_SIZE);
    zipFiles.push({ path: pngPath, data: pngBytes });

    // 2. 32x32 indexed grid (0 = transparent, 1..N = palette index)
    zipFiles.push({
      path: gridPath,
      data: encoder.encode(
        JSON.stringify({
          spriteId: entry.spriteId,
          width: SPRITE_SIZE,
          height: SPRITE_SIZE,
          transparentIndex: 0,
          grid: toMatrix32(entry.indexedGrid),
        }),
      ),
    });

    // 3. Per-sprite exact palette ordered by luminance
    zipFiles.push({
      path: palettePath,
      data: encoder.encode(
        JSON.stringify({
          spriteId: entry.spriteId,
          colors: entry.paletteHex.map((hex, i) => ({
            index: i + 1,
            hex,
            rgb: entry.paletteRgb[i],
          })),
        }),
      ),
    });

    // 4. 32x32 ramp-role map
    zipFiles.push({
      path: rampRolePath,
      data: encoder.encode(
        JSON.stringify({
          spriteId: entry.spriteId,
          legend: RAMP_ROLE_NAMES,
          map: toMatrix32(entry.rampRoles),
        }),
      ),
    });

    // 5. Row in labels.csv
    csvLines.push(
      [
        idStr,
        String(entry.itemId),
        entry.category,
        serializeFlags(entry),
        entry.tags.join('|'),
        entry.split,
        entry.isTile ? '1' : '0',
        String(entry.opaqueCount),
        String(entry.paletteHex.length),
        entry.criticScore.toFixed(1),
        pngPath,
        gridPath,
        palettePath,
        rampRolePath,
      ]
        .map(escapeCsvCell)
        .join(','),
    );
  }

  const labelsCsv = csvLines.join('\n') + '\n';
  zipFiles.push({ path: 'labels.csv', data: encoder.encode(labelsCsv) });

  // Fixed split JSON with `test` explicitly excluded
  const sanitizedSplitByCategory: Record<string, { train: number[]; validation: number[] }> = {};
  for (const [cat, part] of Object.entries(split.byCategory)) {
    sanitizedSplitByCategory[cat] = {
      train: [...part.train],
      validation: [...part.validation],
    };
  }
  zipFiles.push({
    path: 'split.json',
    data: encoder.encode(
      JSON.stringify(
        {
          datasetKey: split.datasetKey,
          seed: split.seed,
          ratios: split.ratios,
          testSplitExcluded: true,
          excludedTestCount: testIdSet.size,
          byCategory: sanitizedSplitByCategory,
        },
        null,
        2,
      ),
    ),
  });

  if (vocabulary) {
    zipFiles.push({
      path: 'vocabulary.json',
      data: encoder.encode(
        JSON.stringify(
          {
            splitHash: vocabulary.splitHash,
            vocabSize: vocabulary.vocabSize,
            overrideCount: vocabulary.overrideCount,
            entries: vocabulary.entries,
          },
          null,
          2,
        ),
      ),
    });
  }

  // Optional Part F: Human approvals & rejections + token grids for next Colab retrain
  let humanApprovedCount = 0;
  let humanRejectedCount = 0;
  let humanLabelsCsv: string | null = null;
  const spriteById = new Map(generatedSprites.map((s) => [s.id, s]));

  if (approvals.length > 0) {
    const hLines: string[] = [
      'id,category,prompt,decision,critic_score,originality_score,source,issues,created_at,png_path,grid_path',
    ];
    for (const label of approvals) {
      const gen = spriteById.get(label.spriteRecordId);
      if (!gen) continue;
      const safeId = label.id.replace(/[^a-zA-Z0-9_-]/g, '_');
      const pngPath = `human_labels/${label.decision}/${safeId}.png`;
      const gridPath = `human_labels/grids/${safeId}.json`;
      zipFiles.push({
        path: pngPath,
        data: encodeRgbaToPngBytes(gen.rgba, SPRITE_SIZE, SPRITE_SIZE),
      });

      const gridFlat = vocabulary
        ? encodeRgbaToVocabGrid(gen.rgba, vocabulary.entries)
        : new Uint8Array(SPRITE_SIZE * SPRITE_SIZE);
      zipFiles.push({
        path: gridPath,
        data: encoder.encode(
          JSON.stringify({
            id: label.id,
            decision: label.decision,
            width: SPRITE_SIZE,
            height: SPRITE_SIZE,
            grid: toMatrix32(gridFlat),
          }),
        ),
      });

      if (label.decision === 'approved') humanApprovedCount++;
      else humanRejectedCount++;

      hLines.push(
        [
          label.id,
          label.category,
          gen.prompt,
          label.decision,
          label.criticScore.toFixed(1),
          label.originalityScore.toFixed(1),
          label.source,
          label.issues.join('|'),
          label.createdAt,
          pngPath,
          gridPath,
        ]
          .map(escapeCsvCell)
          .join(','),
      );
    }
    humanLabelsCsv = hLines.join('\n') + '\n';
    zipFiles.push({ path: 'human_labels.csv', data: encoder.encode(humanLabelsCsv) });
  }

  zipFiles.push({
    path: 'manifest.json',
    data: encoder.encode(
      JSON.stringify(
        {
          schema: 'atelier-sprite-training-dataset-v1',
          exportedAt,
          datasetKey: index.datasetKey,
          nativeResolution: `${SPRITE_SIZE}x${SPRITE_SIZE}`,
          alphaMode: 'binary-rgba-true-alpha',
          testSplitExcluded: true,
          counts: {
            exportedTotal: exportEntries.length,
            train: trainCount,
            validation: validationCount,
            excludedTest: testIdSet.size,
            humanApproved: humanApprovedCount,
            humanRejected: humanRejectedCount,
          },
          includesHumanLabels: approvals.length > 0,
          includesVocabulary: Boolean(vocabulary),
        },
        null,
        2,
      ),
    ),
  });

  const zipBytes = buildZipArchive(zipFiles);

  return {
    zipBytes,
    fileCount: zipFiles.length,
    exportedSpriteCount: exportEntries.length,
    trainCount,
    validationCount,
    excludedTestCount: testIdSet.size,
    humanApprovedCount,
    humanRejectedCount,
    hasVocabularyJson: Boolean(vocabulary),
    labelsCsv,
    humanLabelsCsv,
    filePaths: zipFiles.map((f) => f.path),
  };
}
