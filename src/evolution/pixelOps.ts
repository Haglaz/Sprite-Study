import { SPRITE_SIZE } from '../utils/tibiaParser';
import { evaluateNeatGenome } from './neat';
import type { NeatGenome, PixelOpId, RecipeStepGene } from './types';

export interface RgbColor {
  r: number;
  g: number;
  b: number;
  luma: number;
  count: number;
}

export const ALL_PIXEL_OP_IDS: readonly PixelOpId[] = [
  'snap_binary_alpha',
  'remove_orphans',
  'snap_near_duplicates',
  'quantize_palette',
  'enforce_dark_outline',
  'trim_outline_doubles',
  'despeckle_clusters',
  'shift_nw_lighting',
  'cleanup_dither',
  'center_and_pad',
] as const;

export const PIXEL_OP_LABELS_PT: Record<PixelOpId, string> = {
  snap_binary_alpha: 'Alpha binário (0/255)',
  remove_orphans: 'Remover pixels órfãos',
  snap_near_duplicates: 'Fundir cores quase-duplicadas',
  quantize_palette: 'Quantizar paleta discreta',
  enforce_dark_outline: 'Contorno escuro 1px',
  trim_outline_doubles: 'Aparar cantos duplos (doubles)',
  despeckle_clusters: 'Limpar confete / speckle',
  shift_nw_lighting: 'Ajustar luz direcional NW→SE',
  cleanup_dither: 'Limpar dithering excessivo',
  center_and_pad: 'Centralizar com margem',
};

function calcLuma(r: number, g: number, b: number): number {
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

function colorDistSq(r1: number, g1: number, b1: number, r2: number, g2: number, b2: number): number {
  const dr = r1 - r2;
  const dg = g1 - g2;
  const db = b1 - b2;
  return dr * dr + dg * dg + db * db;
}

function pixelIdx(x: number, y: number): number {
  return (y * SPRITE_SIZE + x) * 4;
}

/**
 * Extracts exact distinct RGB colors from opaque pixels, sorted darkest -> brightest.
 */
export function extractExactPalette(rgba: Uint8ClampedArray): RgbColor[] {
  const counts = new Map<number, number>();
  for (let i = 0; i < rgba.length; i += 4) {
    if (rgba[i + 3] < 128) continue;
    const key = (rgba[i] << 16) | (rgba[i + 1] << 8) | rgba[i + 2];
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const palette: RgbColor[] = [];
  for (const [key, count] of counts.entries()) {
    const r = (key >> 16) & 255;
    const g = (key >> 8) & 255;
    const b = key & 255;
    palette.push({ r, g, b, luma: calcLuma(r, g, b), count });
  }
  palette.sort((a, b) => a.luma - b.luma);
  return palette;
}

function findNearestPaletteColor(palette: RgbColor[], r: number, g: number, b: number): number {
  let bestIdx = 0;
  let bestDist = Infinity;
  for (let i = 0; i < palette.length; i++) {
    const c = palette[i];
    const d = colorDistSq(r, g, b, c.r, c.g, c.b);
    if (d < bestDist) {
      bestDist = d;
      bestIdx = i;
    }
  }
  return bestIdx;
}

// ============================================================================
// 10 DISCRETE PIXEL-EDITOR OPERATIONS (NO BLUR, NO RESAMPLING, EXACT PALETTE)
// ============================================================================

/** 1. Hard binary alpha snap (eliminates semi-transparent AA halos). */
export function opSnapBinaryAlpha(rgba: Uint8ClampedArray, threshold = 0.5): Uint8ClampedArray {
  const out = new Uint8ClampedArray(rgba);
  const cutoff = Math.max(32, Math.min(224, Math.round(threshold * 255)));
  for (let i = 0; i < out.length; i += 4) {
    const a = out[i + 3];
    if (a === 0 || a === 255) continue;
    if (a >= cutoff) {
      out[i + 3] = 255;
    } else {
      out[i] = 0;
      out[i + 1] = 0;
      out[i + 2] = 0;
      out[i + 3] = 0;
    }
  }
  return out;
}

/** 2. Removes detached orphan components (<= maxComponentSize pixels outside main body). */
export function opRemoveOrphans(rgba: Uint8ClampedArray, threshold = 0.35): Uint8ClampedArray {
  const out = opSnapBinaryAlpha(rgba, 0.5);
  const visited = new Uint8Array(SPRITE_SIZE * SPRITE_SIZE);
  const components: number[][] = [];

  for (let p = 0; p < SPRITE_SIZE * SPRITE_SIZE; p++) {
    if (visited[p] || out[p * 4 + 3] === 0) continue;
    const comp: number[] = [];
    const queue = [p];
    visited[p] = 1;
    while (queue.length > 0) {
      const cur = queue.shift()!;
      comp.push(cur);
      const cx = cur % SPRITE_SIZE;
      const cy = Math.floor(cur / SPRITE_SIZE);
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = cx + dx;
          const ny = cy + dy;
          if (nx < 0 || nx >= SPRITE_SIZE || ny < 0 || ny >= SPRITE_SIZE) continue;
          const np = ny * SPRITE_SIZE + nx;
          if (!visited[np] && out[np * 4 + 3] > 0) {
            visited[np] = 1;
            queue.push(np);
          }
        }
      }
    }
    components.push(comp);
  }

  if (components.length <= 1) return out;
  components.sort((a, b) => b.length - a.length);
  const maxOrphanSize = Math.max(1, Math.min(4, Math.round(1 + threshold * 3)));

  for (let c = 1; c < components.length; c++) {
    if (components[c].length <= maxOrphanSize) {
      for (const p of components[c]) {
        out[p * 4] = 0;
        out[p * 4 + 1] = 0;
        out[p * 4 + 2] = 0;
        out[p * 4 + 3] = 0;
      }
    }
  }
  return out;
}

/** 3. Merges near-duplicate colors (per-channel drift) into the higher-frequency exact color. */
export function opSnapNearDuplicates(rgba: Uint8ClampedArray, threshold = 0.45): Uint8ClampedArray {
  const out = opSnapBinaryAlpha(rgba, 0.5);
  const palette = extractExactPalette(out);
  if (palette.length <= 3) return out;

  // Sort by frequency descending so frequent anchor colors absorb noisy neighbors
  const byFreq = [...palette].sort((a, b) => b.count - a.count);
  const maxChannelDiff = Math.round(6 + threshold * 14); // 6..20
  const maxDistSq = maxChannelDiff * maxChannelDiff * 2.2;
  const canonical: RgbColor[] = [];
  const remap = new Map<number, RgbColor>();

  for (const col of byFreq) {
    const key = (col.r << 16) | (col.g << 8) | col.b;
    let absorbed: RgbColor | null = null;
    for (const anchor of canonical) {
      if (
        Math.abs(col.r - anchor.r) <= maxChannelDiff &&
        Math.abs(col.g - anchor.g) <= maxChannelDiff &&
        Math.abs(col.b - anchor.b) <= maxChannelDiff &&
        colorDistSq(col.r, col.g, col.b, anchor.r, anchor.g, anchor.b) <= maxDistSq
      ) {
        absorbed = anchor;
        break;
      }
    }
    if (absorbed) {
      remap.set(key, absorbed);
    } else {
      canonical.push(col);
      remap.set(key, col);
    }
  }

  for (let i = 0; i < out.length; i += 4) {
    if (out[i + 3] === 0) continue;
    const key = (out[i] << 16) | (out[i + 1] << 8) | out[i + 2];
    const target = remap.get(key);
    if (target) {
      out[i] = target.r;
      out[i + 1] = target.g;
      out[i + 2] = target.b;
    }
  }
  return out;
}

/** 4. Quantizes excessive palette colors to a crisp discrete subset of exact existing colors. */
export function opQuantizePalette(rgba: Uint8ClampedArray, intensity = 0.5): Uint8ClampedArray {
  const out = opSnapBinaryAlpha(rgba, 0.5);
  const palette = extractExactPalette(out);
  const targetMax = Math.max(6, Math.min(28, Math.round(24 - intensity * 14)));
  if (palette.length <= targetMax) return out;

  // Always preserve darkest outline anchor + brightest highlight anchor + most frequent well-spaced colors
  const chosen: RgbColor[] = [palette[0], palette[palette.length - 1]];
  const byFreq = [...palette].sort((a, b) => b.count - a.count);

  for (const cand of byFreq) {
    if (chosen.length >= targetMax) break;
    const alreadyClose = chosen.some(
      (c) => colorDistSq(c.r, c.g, c.b, cand.r, cand.g, cand.b) < 280,
    );
    if (!alreadyClose) chosen.push(cand);
  }
  // Fill remaining slots if needed
  for (const cand of byFreq) {
    if (chosen.length >= targetMax) break;
    if (!chosen.includes(cand)) chosen.push(cand);
  }

  for (let i = 0; i < out.length; i += 4) {
    if (out[i + 3] === 0) continue;
    const nearest = chosen[findNearestPaletteColor(chosen, out[i], out[i + 1], out[i + 2])];
    out[i] = nearest.r;
    out[i + 1] = nearest.g;
    out[i + 2] = nearest.b;
  }
  return out;
}

/** 5. Enforces a crisp 1px dark outline on outer perimeter pixels (for non-tiles). */
export function opEnforceDarkOutline(
  rgba: Uint8ClampedArray,
  threshold = 0.45,
  isTile = false,
): Uint8ClampedArray {
  const out = opSnapBinaryAlpha(rgba, 0.5);
  if (isTile) return out;

  const palette = extractExactPalette(out);
  if (palette.length < 2) return out;

  // Choose darkest existing color if genuinely dark, else derive a hue-shifted dark outline color
  const darkest = palette[0];
  let outlineR = darkest.r;
  let outlineG = darkest.g;
  let outlineB = darkest.b;
  if (darkest.luma > 0.24) {
    outlineR = Math.max(12, Math.round(darkest.r * 0.28));
    outlineG = Math.max(12, Math.round(darkest.g * 0.28));
    outlineB = Math.max(18, Math.round(darkest.b * 0.34));
  }

  const maxOutlineLuma = 0.18 + threshold * 0.22; // pixels brighter than this on perimeter become outline
  const copy = new Uint8ClampedArray(out);

  for (let y = 0; y < SPRITE_SIZE; y++) {
    for (let x = 0; x < SPRITE_SIZE; x++) {
      const idx = pixelIdx(x, y);
      if (copy[idx + 3] === 0) continue;

      const isPerimeter =
        x === 0 ||
        x === SPRITE_SIZE - 1 ||
        y === 0 ||
        y === SPRITE_SIZE - 1 ||
        copy[pixelIdx(x - 1, y) + 3] === 0 ||
        copy[pixelIdx(x + 1, y) + 3] === 0 ||
        copy[pixelIdx(x, y - 1) + 3] === 0 ||
        copy[pixelIdx(x, y + 1) + 3] === 0;

      if (!isPerimeter) continue;
      const lum = calcLuma(copy[idx], copy[idx + 1], copy[idx + 2]);
      if (lum > maxOutlineLuma) {
        out[idx] = outlineR;
        out[idx + 1] = outlineG;
        out[idx + 2] = outlineB;
        out[idx + 3] = 255;
      }
    }
  }
  return out;
}

/** 6. Trims redundant inner L-corner outline doubles without breaking silhouette enclosure. */
export function opTrimOutlineDoubles(rgba: Uint8ClampedArray, isTile = false): Uint8ClampedArray {
  const out = opSnapBinaryAlpha(rgba, 0.5);
  if (isTile) return out;

  const palette = extractExactPalette(out);
  if (palette.length < 3) return out;
  const darkCutoff = Math.min(0.28, palette[0].luma + 0.08);
  const copy = new Uint8ClampedArray(out);

  const isDark = (x: number, y: number): boolean => {
    if (x < 0 || x >= SPRITE_SIZE || y < 0 || y >= SPRITE_SIZE) return false;
    const i = pixelIdx(x, y);
    return copy[i + 3] > 0 && calcLuma(copy[i], copy[i + 1], copy[i + 2]) <= darkCutoff;
  };
  const isOpaque = (x: number, y: number): boolean => {
    if (x < 0 || x >= SPRITE_SIZE || y < 0 || y >= SPRITE_SIZE) return false;
    return copy[pixelIdx(x, y) + 3] > 0;
  };

  for (let y = 1; y < SPRITE_SIZE - 1; y++) {
    for (let x = 1; x < SPRITE_SIZE - 1; x++) {
      if (!isDark(x, y)) continue;
      // Only consider interior dark pixel adjacent to 2 orthogonal dark pixels (an L double)
      const touchesTrans =
        !isOpaque(x - 1, y) || !isOpaque(x + 1, y) || !isOpaque(x, y - 1) || !isOpaque(x, y + 1);
      if (touchesTrans) continue;

      const lCorner =
        (isDark(x - 1, y) && isDark(x, y - 1) && !isDark(x + 1, y) && !isDark(x, y + 1)) ||
        (isDark(x + 1, y) && isDark(x, y - 1) && !isDark(x - 1, y) && !isDark(x, y + 1)) ||
        (isDark(x - 1, y) && isDark(x, y + 1) && !isDark(x + 1, y) && !isDark(x, y - 1)) ||
        (isDark(x + 1, y) && isDark(x, y + 1) && !isDark(x - 1, y) && !isDark(x, y - 1));

      if (!lCorner) continue;

      // Replace with nearest brighter orthogonal interior neighbor (exact palette color)
      const neighbors: [number, number][] = [
        [x - 1, y],
        [x + 1, y],
        [x, y - 1],
        [x, y + 1],
      ];
      let best: [number, number, number] | null = null;
      let bestLum = -1;
      for (const [nx, ny] of neighbors) {
        if (!isOpaque(nx, ny) || isDark(nx, ny)) continue;
        const ni = pixelIdx(nx, ny);
        const nl = calcLuma(copy[ni], copy[ni + 1], copy[ni + 2]);
        if (nl > bestLum) {
          bestLum = nl;
          best = [copy[ni], copy[ni + 1], copy[ni + 2]];
        }
      }
      if (best) {
        const idx = pixelIdx(x, y);
        out[idx] = best[0];
        out[idx + 1] = best[1];
        out[idx + 2] = best[2];
      }
    }
  }
  return out;
}

/** 7. Removes 1px stray color confetti inside the silhouette by snapping to modal 4-neighbor exact color. */
export function opDespeckleClusters(rgba: Uint8ClampedArray, intensity = 0.5): Uint8ClampedArray {
  const out = opSnapBinaryAlpha(rgba, 0.5);
  const copy = new Uint8ClampedArray(out);
  const minNeighborAgreement = intensity > 0.65 ? 2 : 3;

  for (let y = 1; y < SPRITE_SIZE - 1; y++) {
    for (let x = 1; x < SPRITE_SIZE - 1; x++) {
      const idx = pixelIdx(x, y);
      if (copy[idx + 3] === 0) continue;

      // Skip perimeter outline pixels
      const isPerimeter =
        copy[pixelIdx(x - 1, y) + 3] === 0 ||
        copy[pixelIdx(x + 1, y) + 3] === 0 ||
        copy[pixelIdx(x, y - 1) + 3] === 0 ||
        copy[pixelIdx(x, y + 1) + 3] === 0;
      if (isPerimeter) continue;

      const curKey = (copy[idx] << 16) | (copy[idx + 1] << 8) | copy[idx + 2];
      const nCoords: [number, number][] = [
        [x - 1, y],
        [x + 1, y],
        [x, y - 1],
        [x, y + 1],
      ];

      let sameCount = 0;
      const freq = new Map<number, number>();
      for (const [nx, ny] of nCoords) {
        const ni = pixelIdx(nx, ny);
        if (copy[ni + 3] === 0) continue;
        const k = (copy[ni] << 16) | (copy[ni + 1] << 8) | copy[ni + 2];
        if (k === curKey) sameCount++;
        freq.set(k, (freq.get(k) ?? 0) + 1);
      }

      if (sameCount > 0) continue;

      let bestKey = -1;
      let bestCount = 0;
      for (const [k, c] of freq.entries()) {
        if (c > bestCount) {
          bestCount = c;
          bestKey = k;
        }
      }
      if (bestKey >= 0 && bestCount >= minNeighborAgreement) {
        out[idx] = (bestKey >> 16) & 255;
        out[idx + 1] = (bestKey >> 8) & 255;
        out[idx + 2] = bestKey & 255;
      }
    }
  }
  return out;
}

/** 8. Shifts interior pixels along existing luminance ramp to restore classic NW -> SE lighting. */
export function opShiftNwLighting(
  rgba: Uint8ClampedArray,
  intensity = 0.5,
  isTile = false,
): Uint8ClampedArray {
  const out = opSnapBinaryAlpha(rgba, 0.5);
  if (isTile) return out;

  const palette = extractExactPalette(out);
  if (palette.length < 4) return out;

  // Compute opaque bounding box
  let minX = SPRITE_SIZE;
  let maxX = 0;
  let minY = SPRITE_SIZE;
  let maxY = 0;
  for (let y = 0; y < SPRITE_SIZE; y++) {
    for (let x = 0; x < SPRITE_SIZE; x++) {
      if (out[pixelIdx(x, y) + 3] > 0) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  const w = Math.max(1, maxX - minX);
  const h = Math.max(1, maxY - minY);
  const darkLimit = palette[0].luma + 0.05;
  const copy = new Uint8ClampedArray(out);
  const nwMargin = 0.32 - intensity * 0.14; // stricter when intensity is low

  for (let y = 1; y < SPRITE_SIZE - 1; y++) {
    for (let x = 1; x < SPRITE_SIZE - 1; x++) {
      const idx = pixelIdx(x, y);
      if (copy[idx + 3] === 0) continue;
      // Leave perimeter untouched
      if (
        copy[pixelIdx(x - 1, y) + 3] === 0 ||
        copy[pixelIdx(x + 1, y) + 3] === 0 ||
        copy[pixelIdx(x, y - 1) + 3] === 0 ||
        copy[pixelIdx(x, y + 1) + 3] === 0
      ) {
        continue;
      }

      const pIdx = findNearestPaletteColor(palette, copy[idx], copy[idx + 1], copy[idx + 2]);
      if (palette[pIdx].luma <= darkLimit) continue;

      const nx = (x - minX) / w - 0.5; // -0.5..0.5
      const ny = (y - minY) / h - 0.5;
      const nwFactor = -(nx + ny) * 0.5; // positive in NW, negative in SE

      if (nwFactor > nwMargin && pIdx < palette.length - 1) {
        const brighter = palette[pIdx + 1];
        out[idx] = brighter.r;
        out[idx + 1] = brighter.g;
        out[idx + 2] = brighter.b;
      } else if (nwFactor < -nwMargin && pIdx > 1) {
        const darker = palette[pIdx - 1];
        out[idx] = darker.r;
        out[idx + 1] = darker.g;
        out[idx + 2] = darker.b;
      }
    }
  }
  return out;
}

/** 9. Cleans up isolated 2x2 checkerboard dithering pixels into cleaner clusters. */
export function opCleanupDither(rgba: Uint8ClampedArray, intensity = 0.5): Uint8ClampedArray {
  const out = opSnapBinaryAlpha(rgba, 0.5);
  const copy = new Uint8ClampedArray(out);
  const skipStep = intensity > 0.6 ? 1 : 2;

  for (let y = 1; y < SPRITE_SIZE - 1; y += skipStep) {
    for (let x = 1; x < SPRITE_SIZE - 1; x++) {
      const idx = pixelIdx(x, y);
      if (copy[idx + 3] === 0) continue;
      const kC = (copy[idx] << 16) | (copy[idx + 1] << 8) | copy[idx + 2];
      const iL = pixelIdx(x - 1, y);
      const iR = pixelIdx(x + 1, y);
      const iU = pixelIdx(x, y - 1);
      const iD = pixelIdx(x, y + 1);
      if (copy[iL + 3] === 0 || copy[iR + 3] === 0 || copy[iU + 3] === 0 || copy[iD + 3] === 0) {
        continue;
      }
      const kL = (copy[iL] << 16) | (copy[iL + 1] << 8) | copy[iL + 2];
      const kR = (copy[iR] << 16) | (copy[iR + 1] << 8) | copy[iR + 2];
      const kU = (copy[iU] << 16) | (copy[iU + 1] << 8) | copy[iU + 2];
      const kD = (copy[iD] << 16) | (copy[iD + 1] << 8) | copy[iD + 2];

      if (kL === kR && kU === kD && kL === kU && kC !== kL) {
        out[idx] = (kL >> 16) & 255;
        out[idx + 1] = (kL >> 8) & 255;
        out[idx + 2] = kL & 255;
      }
    }
  }
  return out;
}

/** 10. Translates off-center or border-touching non-tile silhouettes back to center with clean margin. */
export function opCenterAndPad(rgba: Uint8ClampedArray, isTile = false): Uint8ClampedArray {
  const out = opSnapBinaryAlpha(rgba, 0.5);
  if (isTile) return out;

  let minX = SPRITE_SIZE;
  let maxX = -1;
  let minY = SPRITE_SIZE;
  let maxY = -1;
  let borderTouches = 0;

  for (let y = 0; y < SPRITE_SIZE; y++) {
    for (let x = 0; x < SPRITE_SIZE; x++) {
      if (out[pixelIdx(x, y) + 3] > 0) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        if (x === 0 || x === SPRITE_SIZE - 1 || y === 0 || y === SPRITE_SIZE - 1) {
          borderTouches++;
        }
      }
    }
  }
  if (maxX < minX) return out;

  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const offCenter = Math.hypot(cx - 15.5, cy - 15.5);
  if (borderTouches === 0 && offCenter < 3.5) return out;

  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;
  if (bw >= SPRITE_SIZE - 1 || bh >= SPRITE_SIZE - 1) return out;

  const targetMinX = Math.max(1, Math.min(SPRITE_SIZE - 1 - bw, Math.round((SPRITE_SIZE - bw) / 2)));
  const targetMinY = Math.max(1, Math.min(SPRITE_SIZE - 1 - bh, Math.round((SPRITE_SIZE - bh) / 2)));
  const dx = targetMinX - minX;
  const dy = targetMinY - minY;
  if (dx === 0 && dy === 0) return out;

  const shifted = new Uint8ClampedArray(SPRITE_SIZE * SPRITE_SIZE * 4);
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const src = pixelIdx(x, y);
      if (out[src + 3] === 0) continue;
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || nx >= SPRITE_SIZE || ny < 0 || ny >= SPRITE_SIZE) continue;
      const dst = pixelIdx(nx, ny);
      shifted[dst] = out[src];
      shifted[dst + 1] = out[src + 1];
      shifted[dst + 2] = out[src + 2];
      shifted[dst + 3] = 255;
    }
  }
  return shifted;
}

export function applySinglePixelOp(
  rgba: Uint8ClampedArray,
  step: RecipeStepGene,
  isTile = false,
): Uint8ClampedArray {
  if (!step.enabled) return rgba;
  switch (step.op) {
    case 'snap_binary_alpha':
      return opSnapBinaryAlpha(rgba, step.threshold);
    case 'remove_orphans':
      return opRemoveOrphans(rgba, step.threshold);
    case 'snap_near_duplicates':
      return opSnapNearDuplicates(rgba, step.threshold);
    case 'quantize_palette':
      return opQuantizePalette(rgba, step.intensity);
    case 'enforce_dark_outline':
      return opEnforceDarkOutline(rgba, step.threshold, isTile);
    case 'trim_outline_doubles':
      return opTrimOutlineDoubles(rgba, isTile);
    case 'despeckle_clusters':
      return opDespeckleClusters(rgba, step.intensity);
    case 'shift_nw_lighting':
      return opShiftNwLighting(rgba, step.intensity, isTile);
    case 'cleanup_dither':
      return opCleanupDither(rgba, step.intensity);
    case 'center_and_pad':
      return opCenterAndPad(rgba, isTile);
    default:
      return rgba;
  }
}

export function applyRecipePipeline(
  rgba: Uint8ClampedArray,
  steps: RecipeStepGene[],
  isTile = false,
): Uint8ClampedArray {
  let current: Uint8ClampedArray = new Uint8ClampedArray(rgba);
  for (const step of steps) {
    if (!step.enabled) continue;
    current = applySinglePixelOp(current, step, isTile);
  }
  return current;
}

// ============================================================================
// NEAT LOCAL PIXEL-BRUSH REFINEMENT PASS (STRICT EXACT-PALETTE DISCRETE EDITS)
// ============================================================================

export const NEAT_PIXEL_INPUT_COUNT = 8;
export const NEAT_PIXEL_OUTPUT_COUNT = 5;

/**
 * Runs a NEAT policy over candidate pixels (perimeter, stray 1px islands, or high-contrast transitions)
 * making strictly discrete palette decisions:
 *   0: keep unchanged
 *   1: snap to darkest outline color (if perimeter)
 *   2: snap to modal orthogonal neighbor color (exact palette)
 *   3: step +1 brighter on existing palette luminance ramp
 *   4: step -1 darker on existing palette luminance ramp
 *
 * Guarantees: zero anti-aliasing, zero off-palette colors, binary alpha preserved.
 */
export function applyNeatPixelRefinementPass(
  rgba: Uint8ClampedArray,
  genome: NeatGenome,
  isTile = false,
): Uint8ClampedArray {
  const out = opSnapBinaryAlpha(rgba, 0.5);
  const palette = extractExactPalette(out);
  if (palette.length < 3) return out;

  const copy = new Uint8ClampedArray(out);
  const pCount = palette.length;

  // Compute bounding box for normalized NW coordinates
  let minX = SPRITE_SIZE;
  let maxX = 0;
  let minY = SPRITE_SIZE;
  let maxY = 0;
  for (let y = 0; y < SPRITE_SIZE; y++) {
    for (let x = 0; x < SPRITE_SIZE; x++) {
      if (copy[pixelIdx(x, y) + 3] > 0) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX <= minX || maxY <= minY) return out;
  const bw = maxX - minX;
  const bh = maxY - minY;

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const idx = pixelIdx(x, y);
      if (copy[idx + 3] === 0) continue;

      const curKey = (copy[idx] << 16) | (copy[idx + 1] << 8) | copy[idx + 2];
      const curLuma = calcLuma(copy[idx], copy[idx + 1], copy[idx + 2]);

      const nCoords: [number, number][] = [
        [x - 1, y],
        [x + 1, y],
        [x, y - 1],
        [x, y + 1],
      ];

      let opaqueNeighbors = 0;
      let sameColorNeighbors = 0;
      let neighborLumaSum = 0;
      const freq = new Map<number, number>();

      for (const [nx, ny] of nCoords) {
        if (nx < 0 || nx >= SPRITE_SIZE || ny < 0 || ny >= SPRITE_SIZE) continue;
        const ni = pixelIdx(nx, ny);
        if (copy[ni + 3] === 0) continue;
        opaqueNeighbors++;
        const k = (copy[ni] << 16) | (copy[ni + 1] << 8) | copy[ni + 2];
        if (k === curKey) sameColorNeighbors++;
        neighborLumaSum += calcLuma(copy[ni], copy[ni + 1], copy[ni + 2]);
        freq.set(k, (freq.get(k) ?? 0) + 1);
      }

      const isPerimeter = opaqueNeighbors < 4;
      const isStray = sameColorNeighbors === 0;

      // Only invoke NEAT on perimeter or non-uniform transition pixels for speed & focus
      if (!isPerimeter && sameColorNeighbors >= 3) continue;

      const meanNeighborLuma = opaqueNeighbors > 0 ? neighborLumaSum / opaqueNeighbors : curLuma;
      const normX = ((x - minX) / bw) * 2 - 1; // -1..1
      const normY = ((y - minY) / bh) * 2 - 1;
      const nwBias = -(normX + normY) * 0.5;
      const pIdx = findNearestPaletteColor(palette, copy[idx], copy[idx + 1], copy[idx + 2]);

      const inputs = [
        curLuma * 2 - 1,
        meanNeighborLuma * 2 - 1,
        (curLuma - meanNeighborLuma) * 2,
        isPerimeter ? 1 : -1,
        isStray ? 1 : -1,
        sameColorNeighbors / 4,
        nwBias,
        pIdx / Math.max(1, pCount - 1),
      ];

      const outputs = evaluateNeatGenome(genome, inputs);
      // Argmax with conservative bias toward "keep" (outputs[0]) so NEAT doesn't destroy clean sprites
      let bestAction = 0;
      let bestVal = (outputs[0] ?? 0) + 0.25;
      for (let a = 1; a < NEAT_PIXEL_OUTPUT_COUNT; a++) {
        if ((outputs[a] ?? -Infinity) > bestVal) {
          bestVal = outputs[a];
          bestAction = a;
        }
      }

      if (bestAction === 0) continue;

      if (bestAction === 1 && isPerimeter && !isTile) {
        const dark = palette[0];
        out[idx] = dark.r;
        out[idx + 1] = dark.g;
        out[idx + 2] = dark.b;
      } else if (bestAction === 2 && freq.size > 0) {
        let modalKey = curKey;
        let modalCount = 0;
        for (const [k, c] of freq.entries()) {
          if (c > modalCount) {
            modalCount = c;
            modalKey = k;
          }
        }
        out[idx] = (modalKey >> 16) & 255;
        out[idx + 1] = (modalKey >> 8) & 255;
        out[idx + 2] = modalKey & 255;
      } else if (bestAction === 3 && !isPerimeter && pIdx < pCount - 1) {
        const up = palette[pIdx + 1];
        out[idx] = up.r;
        out[idx + 1] = up.g;
        out[idx + 2] = up.b;
      } else if (bestAction === 4 && !isPerimeter && pIdx > 1) {
        const down = palette[pIdx - 1];
        out[idx] = down.r;
        out[idx + 1] = down.g;
        out[idx + 2] = down.b;
      }
    }
  }

  return out;
}
