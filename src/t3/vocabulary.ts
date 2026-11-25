import type { StoredDatasetSplit } from '../db/schema';
import { getSplitSpriteIds } from '../db/split';
import type { ParsedDatArchive, ParsedSprArchive } from '../types/studio';
import { rgbToHex } from '../utils/pixelImage';
import { decodeSpritePixels, SPRITE_SIZE } from '../utils/tibiaParser';
import type {
  MaterialClusterSummary,
  RampRoleVocabularyEntry,
  T3MaterialRole,
  T3RampVocabulary,
} from './types';

function calcLuma(r: number, g: number, b: number): number {
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

function rgbToHsl(r: number, g: number, b: number): { h: number; s: number; l: number } {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const l = (max + min) / 2;
  const d = max - min;
  if (d < 1e-5) return { h: 0, s: 0, l };
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h = 0;
  if (max === rn) h = (gn - bn) / d + (gn < bn ? 6 : 0);
  else if (max === gn) h = (bn - rn) / d + 2;
  else h = (rn - gn) / d + 4;
  return { h: h * 60, s, l };
}

/**
 * Computes a deterministic FNV-1a / 64-bit hex signature for a dataset split
 * so Colab-trained T3 model cards can be verified against the exact train/val/test split.
 */
export function computeSplitHash(split: StoredDatasetSplit): string {
  const payload = JSON.stringify({
    datasetKey: split.datasetKey,
    seed: split.seed,
    totals: split.totals,
    byCategory: split.byCategory,
  });
  let h1 = 0x811c9dc5;
  let h2 = 0x9e3779b9;
  for (let i = 0; i < payload.length; i++) {
    const ch = payload.charCodeAt(i);
    h1 ^= ch;
    h1 = Math.imul(h1, 0x01000193) >>> 0;
    h2 ^= ch + (i & 0xff);
    h2 = Math.imul(h2, 0x85ebca6b) >>> 0;
  }
  return `${h1.toString(16).padStart(8, '0')}${h2.toString(16).padStart(8, '0')}`;
}

export function classifyMaterialFamily(r: number, g: number, b: number): string {
  const { h, s, l } = rgbToHsl(r, g, b);
  if (l < 0.14 || (s < 0.12 && l < 0.22)) return 'Contorno / Sombra Escura';
  if (s < 0.16) return 'Aço / Neutro';
  if (h >= 20 && h < 58) return 'Ouro / Bronze / Madeira';
  if (h < 20 || h >= 332) return 'Rubi / Carmesim';
  if (h >= 58 && h < 165) return 'Esmeralda / Folhagem';
  if (h >= 165 && h < 252) return 'Safira / Gelo / Arcano';
  return 'Ametista / Místico';
}

interface RawColorSample {
  r: number;
  g: number;
  b: number;
  luma: number;
  pixelCount: number;
  perimeterCount: number;
}

function assignRolesForMaterialCluster(
  entries: Array<{
    rgb: [number, number, number];
    luma: number;
    hueDeg: number;
    saturation: number;
    materialFamily: string;
    pixelCount: number;
    perimeterRatio: number;
  }>,
): T3MaterialRole[] {
  if (entries.length === 0) return [];
  const sortedIndices = entries
    .map((_, idx) => idx)
    .sort((a, b) => entries[a].luma - entries[b].luma);

  const roles = new Array<T3MaterialRole>(entries.length).fill('base');
  const n = sortedIndices.length;

  for (let rank = 0; rank < n; rank++) {
    const idx = sortedIndices[rank];
    const item = entries[idx];
    const q = n === 1 ? 0.5 : rank / (n - 1);

    if (
      item.luma <= 0.17 ||
      (item.perimeterRatio >= 0.45 && item.luma <= 0.28) ||
      (item.materialFamily === 'Contorno / Sombra Escura' && q <= 0.5)
    ) {
      roles[idx] = 'outline';
    } else if (q < 0.28 || item.luma < 0.30) {
      roles[idx] = 'shadow';
    } else if (q < 0.62 || item.luma < 0.56) {
      roles[idx] = 'base';
    } else if (q < 0.85 || item.luma < 0.78) {
      roles[idx] = 'light';
    } else {
      roles[idx] = 'highlight';
    }
  }

  return roles;
}

export function recomputeMaterialSummaries(
  entries: RampRoleVocabularyEntry[],
): MaterialClusterSummary[] {
  const byMat = new Map<string, MaterialClusterSummary>();
  for (const e of entries) {
    let summary = byMat.get(e.materialFamily);
    if (!summary) {
      summary = {
        materialFamily: e.materialFamily,
        tokenCount: 0,
        totalPixels: 0,
        roleCounts: { outline: 0, shadow: 0, base: 0, light: 0, highlight: 0 },
      };
      byMat.set(e.materialFamily, summary);
    }
    summary.tokenCount += 1;
    summary.totalPixels += e.pixelCount;
    summary.roleCounts[e.role] += 1;
  }
  return Array.from(byMat.values()).sort((a, b) => b.totalPixels - a.totalPixels);
}

/**
 * Deterministic k-means color clustering weighted by pixel support inside one material family.
 */
function clusterMaterialColors(
  colors: RawColorSample[],
  targetClusters: number,
): Array<{
  rgb: [number, number, number];
  luma: number;
  pixelCount: number;
  perimeterRatio: number;
}> {
  if (colors.length === 0) return [];
  const k = Math.max(1, Math.min(targetClusters, colors.length));
  const sorted = [...colors].sort((a, b) =>
    b.pixelCount !== a.pixelCount ? b.pixelCount - a.pixelCount : a.luma - b.luma,
  );

  // Initialize centroids spread deterministically across luminance + support
  const centroids: Array<[number, number, number]> = [];
  if (k === sorted.length) {
    return sorted.map((c) => ({
      rgb: [c.r, c.g, c.b],
      luma: c.luma,
      pixelCount: c.pixelCount,
      perimeterRatio: c.pixelCount > 0 ? c.perimeterCount / c.pixelCount : 0,
    }));
  }

  const byLuma = [...sorted].sort((a, b) => a.luma - b.luma);
  for (let i = 0; i < k; i++) {
    const pick = byLuma[Math.min(byLuma.length - 1, Math.floor((i * (byLuma.length - 1)) / Math.max(1, k - 1)))];
    centroids.push([pick.r, pick.g, pick.b]);
  }

  const assign = new Int32Array(sorted.length);
  for (let iter = 0; iter < 8; iter++) {
    let changed = false;
    for (let i = 0; i < sorted.length; i++) {
      const c = sorted[i];
      let bestIdx = 0;
      let bestDist = Infinity;
      for (let m = 0; m < k; m++) {
        const dr = c.r - centroids[m][0];
        const dg = c.g - centroids[m][1];
        const db = c.b - centroids[m][2];
        const dl = (c.luma - calcLuma(centroids[m][0], centroids[m][1], centroids[m][2])) * 180;
        const d = dr * dr + dg * dg + db * db + dl * dl;
        if (d < bestDist) {
          bestDist = d;
          bestIdx = m;
        }
      }
      if (assign[i] !== bestIdx) {
        assign[i] = bestIdx;
        changed = true;
      }
    }

    const sumR = new Float64Array(k);
    const sumG = new Float64Array(k);
    const sumB = new Float64Array(k);
    const sumW = new Float64Array(k);

    for (let i = 0; i < sorted.length; i++) {
      const m = assign[i];
      const w = Math.sqrt(Math.max(1, sorted[i].pixelCount));
      sumR[m] += sorted[i].r * w;
      sumG[m] += sorted[i].g * w;
      sumB[m] += sorted[i].b * w;
      sumW[m] += w;
    }

    for (let m = 0; m < k; m++) {
      if (sumW[m] > 0) {
        centroids[m] = [
          Math.round(sumR[m] / sumW[m]),
          Math.round(sumG[m] / sumW[m]),
          Math.round(sumB[m] / sumW[m]),
        ];
      }
    }
    if (!changed) break;
  }

  // Snap each cluster center to nearest real observed color (or weighted medoid) so palette colors remain crisp pixel colors
  const clusters: Array<{
    rgb: [number, number, number];
    luma: number;
    pixelCount: number;
    perimeterRatio: number;
  }> = [];

  for (let m = 0; m < k; m++) {
    const members: RawColorSample[] = [];
    for (let i = 0; i < sorted.length; i++) {
      if (assign[i] === m) members.push(sorted[i]);
    }
    if (members.length === 0) continue;

    let totalPixels = 0;
    let totalPerim = 0;
    let bestMedoid = members[0];
    let bestScore = Infinity;

    for (const mem of members) {
      totalPixels += mem.pixelCount;
      totalPerim += mem.perimeterCount;
      const dr = mem.r - centroids[m][0];
      const dg = mem.g - centroids[m][1];
      const db = mem.b - centroids[m][2];
      const dist = (dr * dr + dg * dg + db * db) / Math.log2(mem.pixelCount + 2);
      if (dist < bestScore) {
        bestScore = dist;
        bestMedoid = mem;
      }
    }

    clusters.push({
      rgb: [bestMedoid.r, bestMedoid.g, bestMedoid.b],
      luma: bestMedoid.luma,
      pixelCount: totalPixels,
      perimeterRatio: totalPixels > 0 ? totalPerim / totalPixels : 0,
    });
  }

  return clusters;
}

/**
 * Builds a material-clustered Ramp-Role Vocabulary strictly from `train` split sprites.
 * Never reads `validation` or `test` splits.
 */
export function buildRampRoleVocabularyFromTrainSplit(params: {
  datasetKey: string;
  spr: ParsedSprArchive;
  dat: ParsedDatArchive;
  split: StoredDatasetSplit;
  maxPaletteTokens?: number;
  existingOverrides?: Map<string, T3MaterialRole>;
}): T3RampVocabulary {
  const { datasetKey, spr, split, maxPaletteTokens = 45, existingOverrides } = params;
  const trainIds = getSplitSpriteIds(split, 'train');
  const colorMap = new Map<number, RawColorSample>();
  let trainSpriteCount = 0;
  for (const spriteId of trainIds) {
    const rgbaBuf = decodeSpritePixels(spr, spriteId);
    let hasOpaque = false;
    for (let p = 3; p < rgbaBuf.length; p += 4) {
      if (rgbaBuf[p] >= 128) {
        hasOpaque = true;
        break;
      }
    }
    if (!hasOpaque) continue;
    trainSpriteCount++;

    for (let y = 0; y < SPRITE_SIZE; y++) {
      for (let x = 0; x < SPRITE_SIZE; x++) {
        const p = y * SPRITE_SIZE + x;
        const i = p * 4;
        if (rgbaBuf[i + 3] < 128) continue;

        const isPerimeter =
          x === 0 ||
          x === SPRITE_SIZE - 1 ||
          y === 0 ||
          y === SPRITE_SIZE - 1 ||
          rgbaBuf[(p - 1) * 4 + 3] < 128 ||
          rgbaBuf[(p + 1) * 4 + 3] < 128 ||
          rgbaBuf[(p - SPRITE_SIZE) * 4 + 3] < 128 ||
          rgbaBuf[(p + SPRITE_SIZE) * 4 + 3] < 128;

        const r = rgbaBuf[i];
        const g = rgbaBuf[i + 1];
        const b = rgbaBuf[i + 2];
        const key = (r << 16) | (g << 8) | b;
        let sample = colorMap.get(key);
        if (!sample) {
          sample = {
            r,
            g,
            b,
            luma: calcLuma(r, g, b),
            pixelCount: 0,
            perimeterCount: 0,
          };
          colorMap.set(key, sample);
        }
        sample.pixelCount += 1;
        if (isPerimeter) sample.perimeterCount += 1;
      }
    }
  }

  // Group observed colors by material family
  const byMaterial = new Map<string, RawColorSample[]>();
  let totalPixelsAll = 0;
  for (const sample of colorMap.values()) {
    totalPixelsAll += sample.pixelCount;
    const mat = classifyMaterialFamily(sample.r, sample.g, sample.b);
    const list = byMaterial.get(mat);
    if (list) list.push(sample);
    else byMaterial.set(mat, [sample]);
  }

  // Allocate cluster quota proportionally across material families (ensuring >= 3 tokens if available)
  const matKeys = Array.from(byMaterial.keys()).sort();
  const rawClusters: Array<{
    rgb: [number, number, number];
    luma: number;
    hueDeg: number;
    saturation: number;
    materialFamily: string;
    pixelCount: number;
    perimeterRatio: number;
    autoRole: T3MaterialRole;
  }> = [];

  for (const mat of matKeys) {
    const samples = byMaterial.get(mat) ?? [];
    const matPixels = samples.reduce((s, x) => s + x.pixelCount, 0);
    const share = totalPixelsAll > 0 ? matPixels / totalPixelsAll : 1 / Math.max(1, matKeys.length);
    const quota = Math.max(
      Math.min(samples.length, 4),
      Math.min(samples.length, Math.round(share * maxPaletteTokens)),
    );
    const clustered = clusterMaterialColors(samples, quota);
    const matEntries = clustered.map((c) => {
      const { h, s } = rgbToHsl(c.rgb[0], c.rgb[1], c.rgb[2]);
      return {
        ...c,
        hueDeg: Math.round(h),
        saturation: Math.round(s * 1000) / 1000,
        materialFamily: mat,
      };
    });
    const assignedRoles = assignRolesForMaterialCluster(matEntries);
    matEntries.forEach((me, idx) => {
      rawClusters.push({
        ...me,
        autoRole: assignedRoles[idx],
      });
    });
  }

  // Deduplicate exact hexes and trim cleanly to maxPaletteTokens if slightly over quota
  const seenHex = new Set<string>();
  const deduped = rawClusters
    .sort((a, b) => b.pixelCount - a.pixelCount)
    .filter((item) => {
      const hex = rgbToHex(item.rgb[0], item.rgb[1], item.rgb[2]);
      if (seenHex.has(hex)) return false;
      seenHex.add(hex);
      return true;
    })
    .slice(0, maxPaletteTokens)
    .sort((a, b) =>
      a.materialFamily !== b.materialFamily
        ? a.materialFamily.localeCompare(b.materialFamily)
        : a.luma - b.luma,
    );

  const entries: RampRoleVocabularyEntry[] = deduped.map((item, idx) => {
    const hex = rgbToHex(item.rgb[0], item.rgb[1], item.rgb[2]);
    const overriddenRole = existingOverrides?.get(hex);
    return {
      tokenId: idx + 1,
      rgb: item.rgb,
      hex,
      luma: Math.round(item.luma * 1000) / 1000,
      hueDeg: item.hueDeg,
      saturation: item.saturation,
      materialFamily: item.materialFamily,
      role: overriddenRole ?? item.autoRole,
      autoRole: item.autoRole,
      pixelCount: item.pixelCount,
      perimeterRatio: Math.round(item.perimeterRatio * 1000) / 1000,
      userOverridden: Boolean(overriddenRole && overriddenRole !== item.autoRole),
    };
  });

  const now = new Date().toISOString();
  return {
    id: `t3-vocab:${datasetKey}`,
    datasetKey,
    splitHash: computeSplitHash(split),
    builtAt: now,
    updatedAt: now,
    trainSpriteCount,
    vocabSize: entries.length + 1,
    entries,
    materialSummaries: recomputeMaterialSummaries(entries),
    overrideCount: entries.filter((e) => e.userOverridden).length,
    readBackVerified: false,
  };
}

/**
 * Pure update: corrects a mis-assigned ramp role on a single vocabulary entry.
 */
export function applyVocabularyRoleCorrection(
  vocab: T3RampVocabulary,
  tokenId: number,
  nextRole: T3MaterialRole,
): T3RampVocabulary {
  const entries = vocab.entries.map((entry) => {
    if (entry.tokenId !== tokenId) return entry;
    return {
      ...entry,
      role: nextRole,
      userOverridden: nextRole !== entry.autoRole,
    };
  });
  return {
    ...vocab,
    updatedAt: new Date().toISOString(),
    entries,
    materialSummaries: recomputeMaterialSummaries(entries),
    overrideCount: entries.filter((e) => e.userOverridden).length,
  };
}

/**
 * Encodes a 32x32 RGBA buffer into discrete vocabulary token IDs (`0=transparent`, `1..K=nearest vocab entry`).
 */
export function encodeRgbaToVocabGrid(
  rgba: Uint8ClampedArray,
  entries: ReadonlyArray<Pick<RampRoleVocabularyEntry, 'tokenId' | 'rgb'>>,
): Uint8Array {
  const grid = new Uint8Array(SPRITE_SIZE * SPRITE_SIZE);
  if (entries.length === 0) return grid;

  const exactLookup = new Map<number, number>();
  for (const e of entries) {
    exactLookup.set((e.rgb[0] << 16) | (e.rgb[1] << 8) | e.rgb[2], e.tokenId);
  }

  for (let p = 0; p < SPRITE_SIZE * SPRITE_SIZE; p++) {
    const i = p * 4;
    if (rgba[i + 3] < 128) {
      grid[p] = 0;
      continue;
    }
    const r = rgba[i];
    const g = rgba[i + 1];
    const b = rgba[i + 2];
    const key = (r << 16) | (g << 8) | b;
    const exact = exactLookup.get(key);
    if (exact !== undefined) {
      grid[p] = exact;
      continue;
    }
    let bestId = entries[0].tokenId;
    let bestDist = Infinity;
    for (const e of entries) {
      const dr = r - e.rgb[0];
      const dg = g - e.rgb[1];
      const db = b - e.rgb[2];
      const d = dr * dr + dg * dg + db * db;
      if (d < bestDist) {
        bestDist = d;
        bestId = e.tokenId;
      }
    }
    grid[p] = bestId;
  }

  return grid;
}

/**
 * Decodes a 32x32 discrete vocabulary grid (`0=transparent`, `1..K=exact palette RGB`) into RGBA with binary alpha.
 */
export function decodeVocabGridToRgba(
  tokenGrid: Uint8Array | Int32Array | number[],
  entries: ReadonlyArray<Pick<RampRoleVocabularyEntry, 'tokenId' | 'rgb'>>,
): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(SPRITE_SIZE * SPRITE_SIZE * 4);
  const byToken = new Map<number, [number, number, number]>();
  for (const e of entries) {
    byToken.set(e.tokenId, e.rgb);
  }
  const fallbackRgb: [number, number, number] = entries[0]?.rgb ?? [24, 24, 30];

  const len = Math.min(SPRITE_SIZE * SPRITE_SIZE, tokenGrid.length);
  for (let p = 0; p < len; p++) {
    const tok = Number(tokenGrid[p]);
    if (tok <= 0) continue; // transparent
    const rgb = byToken.get(tok) ?? fallbackRgb;
    const i = p * 4;
    rgba[i] = rgb[0];
    rgba[i + 1] = rgb[1];
    rgba[i + 2] = rgb[2];
    rgba[i + 3] = 255;
  }

  return rgba;
}
