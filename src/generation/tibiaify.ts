import type { CategoryProfile } from '../critic/types';
import { SPRITE_SIZE } from '../utils/tibiaParser';
import { calcRgbLuma, findNearestPaletteRgb, hexToRgb, rgbDistance } from './promptEngineer';
import type { PaletteRampSpec } from './types';

export interface TibiaifyOptions {
  rgba: Uint8ClampedArray;
  width?: number;
  height?: number;
  approvedRamps: readonly PaletteRampSpec[];
  datasetPaletteRgb?: ReadonlyArray<[number, number, number]>;
  categoryProfile?: CategoryProfile | null;
  maxColors?: number;
  isTile?: boolean;
  allowDithering?: boolean;
  targetCenterGround?: boolean;
}

export interface TibiaifyResult {
  rgba: Uint8ClampedArray;
  distinctColors: number;
  snappedPixels: number;
  removedOrphans: number;
  removedDitherPixels: number;
  shiftedBy: { dx: number; dy: number };
}

function pIdx(x: number, y: number): number {
  return (y * SPRITE_SIZE + x) * 4;
}

/**
 * Nearest-neighbor downsample to 32x32 ONLY when input dimensions differ from 32x32.
 * Never performs bilinear/bicubic blending.
 */
export function nearestNeighborTo32x32(
  src: Uint8ClampedArray,
  srcW: number,
  srcH: number,
): Uint8ClampedArray {
  if (srcW === SPRITE_SIZE && srcH === SPRITE_SIZE && src.length === SPRITE_SIZE * SPRITE_SIZE * 4) {
    return new Uint8ClampedArray(src);
  }
  const out = new Uint8ClampedArray(SPRITE_SIZE * SPRITE_SIZE * 4);
  for (let y = 0; y < SPRITE_SIZE; y++) {
    const sy = Math.min(srcH - 1, Math.floor((y * srcH) / SPRITE_SIZE));
    for (let x = 0; x < SPRITE_SIZE; x++) {
      const sx = Math.min(srcW - 1, Math.floor((x * srcW) / SPRITE_SIZE));
      const sOff = (sy * srcW + sx) * 4;
      const dOff = pIdx(x, y);
      out[dOff] = src[sOff] ?? 0;
      out[dOff + 1] = src[sOff + 1] ?? 0;
      out[dOff + 2] = src[sOff + 2] ?? 0;
      out[dOff + 3] = src[sOff + 3] ?? 0;
    }
  }
  return out;
}

function rgbKey(r: number, g: number, b: number): number {
  return (r << 16) | (g << 8) | b;
}

/**
 * Removes isolated orphan pixels and tiny detached fragments (< 4px) outside the dominant
 * 8-connected silhouette component on non-tile sprites.
 */
function removeOrphanPixels(rgba: Uint8ClampedArray): number {
  const visited = new Uint8Array(SPRITE_SIZE * SPRITE_SIZE);
  const components: number[][] = [];

  for (let y = 0; y < SPRITE_SIZE; y++) {
    for (let x = 0; x < SPRITE_SIZE; x++) {
      const p = y * SPRITE_SIZE + x;
      if (visited[p] || rgba[p * 4 + 3] < 128) continue;
      const queue = [p];
      visited[p] = 1;
      const comp: number[] = [];
      let head = 0;
      while (head < queue.length) {
        const cur = queue[head++];
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
            if (!visited[np] && rgba[np * 4 + 3] >= 128) {
              visited[np] = 1;
              queue.push(np);
            }
          }
        }
      }
      components.push(comp);
    }
  }

  if (components.length <= 1) return 0;
  components.sort((a, b) => b.length - a.length);
  const largest = components[0].length;
  let removed = 0;

  for (let c = 1; c < components.length; c++) {
    const comp = components[c];
    // Always remove 1px-3px orphans or fragments tiny relative to main body
    if (comp.length <= 3 || comp.length < largest * 0.12) {
      for (const p of comp) {
        const i = p * 4;
        rgba[i] = 0;
        rgba[i + 1] = 0;
        rgba[i + 2] = 0;
        rgba[i + 3] = 0;
        removed++;
      }
    }
  }
  return removed;
}

/**
 * Removes 1px checkerboard dithering pixels by replacing interior pixels that alternate
 * with all 4 orthogonal neighbors (matching a single neighbor color on >= 3 sides).
 */
function removeCheckerboardDithering(rgba: Uint8ClampedArray): number {
  const copy = new Uint8ClampedArray(rgba);
  let fixed = 0;

  for (let y = 1; y < SPRITE_SIZE - 1; y++) {
    for (let x = 1; x < SPRITE_SIZE - 1; x++) {
      const i = pIdx(x, y);
      if (copy[i + 3] < 128) continue;

      const neighbors = [
        pIdx(x - 1, y),
        pIdx(x + 1, y),
        pIdx(x, y - 1),
        pIdx(x, y + 1),
      ];
      // Skip outer perimeter pixels (keep outline intact)
      if (neighbors.some((n) => copy[n + 3] < 128)) continue;

      const selfKey = rgbKey(copy[i], copy[i + 1], copy[i + 2]);
      const counts = new Map<number, { count: number; rgb: [number, number, number] }>();
      let sameCount = 0;
      for (const n of neighbors) {
        const nk = rgbKey(copy[n], copy[n + 1], copy[n + 2]);
        if (nk === selfKey) sameCount++;
        const cur = counts.get(nk);
        if (cur) cur.count++;
        else counts.set(nk, { count: 1, rgb: [copy[n], copy[n + 1], copy[n + 2]] });
      }

      if (sameCount === 0) {
        let topColor: [number, number, number] | null = null;
        let topCount = 0;
        for (const v of counts.values()) {
          if (v.count > topCount) {
            topCount = v.count;
            topColor = v.rgb;
          }
        }
        // Only collapse true isolated/checkerboard 1px dither (3 or 4 orthogonal neighbors share same color)
        if (topColor && topCount >= 3) {
          rgba[i] = topColor[0];
          rgba[i + 1] = topColor[1];
          rgba[i + 2] = topColor[2];
          fixed++;
        }
      }
    }
  }
  return fixed;
}

/**
 * Deterministic "Tibia-ification" post-processor:
 * 1) Nearest-neighbor downsample to 32x32 ONLY if needed
 * 2) Binary alpha (0 | 255) — zero semi-transparent pixels
 * 3) Snap every opaque pixel to the approved ramps / dataset palette (zero blending/AA)
 * 4) Remove orphan pixels
 * 5) Cap distinct color count to category median + tolerance
 * 6) Remove checkerboard dithering unless category profile uses it
 * 7) Enforce 1px dark hue-shifted outline around silhouette perimeter (non-tiles)
 * 8) Enforce NW lighting order (monotonic luma along material ramps, NW brighter than SE)
 * 9) Center/ground the sprite within the 32x32 frame with clean border margin (non-tiles)
 */
export function tibiaifySprite(options: TibiaifyOptions): TibiaifyResult {
  const {
    rgba,
    width = SPRITE_SIZE,
    height = SPRITE_SIZE,
    approvedRamps,
    datasetPaletteRgb = [],
    categoryProfile = null,
    maxColors,
    isTile = false,
    allowDithering,
    targetCenterGround = true,
  } = options;

  // 1. Nearest-neighbor downsample only if needed
  const buf = nearestNeighborTo32x32(rgba, width, height);

  // 2. Strict binary alpha 0 | 255
  for (let p = 0; p < SPRITE_SIZE * SPRITE_SIZE; p++) {
    const i = p * 4;
    if (buf[i + 3] >= 128) {
      buf[i + 3] = 255;
    } else {
      buf[i] = 0;
      buf[i + 1] = 0;
      buf[i + 2] = 0;
      buf[i + 3] = 0;
    }
  }

  // Build ordered ramp RGB lists from approvedRamps (snapped to dataset palette if provided)
  const rampRgbLists: Array<Array<[number, number, number]>> = [];
  const allowedPalette: Array<[number, number, number]> = [];
  const seenAllowed = new Set<number>();

  const pushAllowed = (rgb: [number, number, number]) => {
    const k = rgbKey(rgb[0], rgb[1], rgb[2]);
    if (!seenAllowed.has(k)) {
      seenAllowed.add(k);
      allowedPalette.push(rgb);
    }
  };

  for (const ramp of approvedRamps) {
    const parsed: Array<[number, number, number]> = [];
    for (const hex of ramp.colors) {
      const rgb = hexToRgb(hex);
      if (!rgb) continue;
      const finalRgb =
        datasetPaletteRgb.length > 0 ? findNearestPaletteRgb(rgb, datasetPaletteRgb).rgb : rgb;
      parsed.push(finalRgb);
      pushAllowed(finalRgb);
    }
    if (parsed.length > 0) {
      parsed.sort((a, b) => calcRgbLuma(a[0], a[1], a[2]) - calcRgbLuma(b[0], b[1], b[2]));
      rampRgbLists.push(parsed);
    }
  }

  // If approvedRamps was empty (defensive), fall back to datasetPaletteRgb
  if (allowedPalette.length === 0 && datasetPaletteRgb.length > 0) {
    for (const rgb of datasetPaletteRgb.slice(0, 24)) {
      pushAllowed(rgb);
    }
  }

  // Identify canonical dark hue-shifted outline color (darkest stop with luma <= 0.18)
  let outlineColor: [number, number, number] = [20, 18, 28];
  if (allowedPalette.length > 0) {
    const sortedByLuma = allowedPalette
      .slice()
      .sort((a, b) => calcRgbLuma(a[0], a[1], a[2]) - calcRgbLuma(b[0], b[1], b[2]));
    const darkest = sortedByLuma[0];
    if (calcRgbLuma(darkest[0], darkest[1], darkest[2]) <= 0.2) {
      outlineColor = darkest;
    } else {
      // Hue-shift darken the darkest ramp color to guarantee crisp dark outline
      outlineColor = [
        Math.max(10, Math.round(darkest[0] * 0.35)),
        Math.max(10, Math.round(darkest[1] * 0.33)),
        Math.max(14, Math.round(darkest[2] * 0.42)),
      ];
      if (datasetPaletteRgb.length > 0) {
        const nearestDark = findNearestPaletteRgb(outlineColor, datasetPaletteRgb).rgb;
        if (calcRgbLuma(nearestDark[0], nearestDark[1], nearestDark[2]) <= 0.2) {
          outlineColor = nearestDark;
        }
      }
      pushAllowed(outlineColor);
    }
  }

  // 3. Snap every opaque pixel to the approved ramps / allowed palette (zero blending/AA)
  let snappedPixels = 0;
  if (allowedPalette.length > 0) {
    for (let p = 0; p < SPRITE_SIZE * SPRITE_SIZE; p++) {
      const i = p * 4;
      if (buf[i + 3] < 128) continue;
      const cur: [number, number, number] = [buf[i], buf[i + 1], buf[i + 2]];
      const nearest = findNearestPaletteRgb(cur, allowedPalette).rgb;
      if (nearest[0] !== cur[0] || nearest[1] !== cur[1] || nearest[2] !== cur[2]) {
        buf[i] = nearest[0];
        buf[i + 1] = nearest[1];
        buf[i + 2] = nearest[2];
        snappedPixels++;
      }
    }
  }

  // 4. Remove orphan pixels on non-tiles
  const removedOrphans = isTile ? 0 : removeOrphanPixels(buf);

  // 5. Cap distinct color count to category median + tolerance
  const categoryMedianColors = categoryProfile?.features.distinctColors.p50 ?? 14;
  const colorCap = Math.max(
    5,
    Math.min(maxColors ?? Math.round(categoryMedianColors + 6), Math.round(categoryMedianColors + 8)),
  );

  const colorCounts = new Map<number, { count: number; rgb: [number, number, number] }>();
  for (let p = 0; p < SPRITE_SIZE * SPRITE_SIZE; p++) {
    const i = p * 4;
    if (buf[i + 3] < 128) continue;
    const k = rgbKey(buf[i], buf[i + 1], buf[i + 2]);
    const cur = colorCounts.get(k);
    if (cur) cur.count++;
    else colorCounts.set(k, { count: 1, rgb: [buf[i], buf[i + 1], buf[i + 2]] });
  }

  if (colorCounts.size > colorCap) {
    const outlineK = rgbKey(outlineColor[0], outlineColor[1], outlineColor[2]);
    const sorted = Array.from(colorCounts.entries()).sort((a, b) => {
      if (a[0] === outlineK) return -1;
      if (b[0] === outlineK) return 1;
      return b[1].count - a[1].count;
    });
    const keptRgbs = sorted.slice(0, colorCap).map(([, v]) => v.rgb);
    const remap = new Map<number, [number, number, number]>();
    for (let idx = colorCap; idx < sorted.length; idx++) {
      const [k, v] = sorted[idx];
      remap.set(k, findNearestPaletteRgb(v.rgb, keptRgbs).rgb);
    }
    for (let p = 0; p < SPRITE_SIZE * SPRITE_SIZE; p++) {
      const i = p * 4;
      if (buf[i + 3] < 128) continue;
      const k = rgbKey(buf[i], buf[i + 1], buf[i + 2]);
      const mapped = remap.get(k);
      if (mapped) {
        buf[i] = mapped[0];
        buf[i + 1] = mapped[1];
        buf[i + 2] = mapped[2];
      }
    }
  }

  // 6. Remove checkerboard dithering unless category profile uses it
  const profileUsesDither = (categoryProfile?.features.ditherShare.p50 ?? 0) >= 0.08;
  const shouldKeepDither = allowDithering ?? profileUsesDither;
  const removedDitherPixels = shouldKeepDither ? 0 : removeCheckerboardDithering(buf);

  // 7. Enforce crisp 1px dark hue-shifted outline around silhouette perimeter (non-tiles)
  if (!isTile) {
    const perimeterMask = new Uint8Array(SPRITE_SIZE * SPRITE_SIZE);
    for (let y = 0; y < SPRITE_SIZE; y++) {
      for (let x = 0; x < SPRITE_SIZE; x++) {
        const i = pIdx(x, y);
        if (buf[i + 3] < 128) continue;
        const isEdge =
          x === 0 ||
          x === SPRITE_SIZE - 1 ||
          y === 0 ||
          y === SPRITE_SIZE - 1 ||
          buf[pIdx(x - 1, y) + 3] < 128 ||
          buf[pIdx(x + 1, y) + 3] < 128 ||
          buf[pIdx(x, y - 1) + 3] < 128 ||
          buf[pIdx(x, y + 1) + 3] < 128;
        if (isEdge) {
          perimeterMask[y * SPRITE_SIZE + x] = 1;
        }
      }
    }
    for (let p = 0; p < SPRITE_SIZE * SPRITE_SIZE; p++) {
      if (!perimeterMask[p]) continue;
      const i = p * 4;
      buf[i] = outlineColor[0];
      buf[i + 1] = outlineColor[1];
      buf[i + 2] = outlineColor[2];
      buf[i + 3] = 255;
    }

    // 8. Enforce NW lighting order along discrete material ramps (NW brighter than SE)
    if (rampRgbLists.length > 0) {
      let minX = SPRITE_SIZE;
      let maxX = -1;
      let minY = SPRITE_SIZE;
      let maxY = -1;
      for (let y = 0; y < SPRITE_SIZE; y++) {
        for (let x = 0; x < SPRITE_SIZE; x++) {
          if (buf[pIdx(x, y) + 3] >= 128) {
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
          }
        }
      }
      const cx = (minX + maxX) / 2;
      const cy = (minY + maxY) / 2;

      let nwSum = 0;
      let nwCnt = 0;
      let seSum = 0;
      let seCnt = 0;
      for (let y = 1; y < SPRITE_SIZE - 1; y++) {
        for (let x = 1; x < SPRITE_SIZE - 1; x++) {
          const p = y * SPRITE_SIZE + x;
          if (buf[p * 4 + 3] < 128 || perimeterMask[p]) continue;
          const lum = calcRgbLuma(buf[p * 4], buf[p * 4 + 1], buf[p * 4 + 2]);
          if (x <= cx && y <= cy) {
            nwSum += lum;
            nwCnt++;
          } else if (x >= cx && y >= cy) {
            seSum += lum;
            seCnt++;
          }
        }
      }

      const nwMean = nwCnt > 0 ? nwSum / nwCnt : 0.5;
      const seMean = seCnt > 0 ? seSum / seCnt : 0.5;

      // If NW lighting is weak or inverted, gently step NW interior pixels up 1 ramp stop and SE down 1 stop
      if (nwMean - seMean < 0.04) {
        const primaryRamp = rampRgbLists[0];
        if (primaryRamp.length >= 3) {
          for (let y = 1; y < SPRITE_SIZE - 1; y++) {
            for (let x = 1; x < SPRITE_SIZE - 1; x++) {
              const p = y * SPRITE_SIZE + x;
              if (buf[p * 4 + 3] < 128 || perimeterMask[p]) continue;
              const cur: [number, number, number] = [buf[p * 4], buf[p * 4 + 1], buf[p * 4 + 2]];

              // Find closest ramp and index
              let bestRamp = primaryRamp;
              let bestStopIdx = 0;
              let bestD = Infinity;
              for (const ramp of rampRgbLists) {
                for (let s = 0; s < ramp.length; s++) {
                  const d = rgbDistance(cur, ramp[s]);
                  if (d < bestD) {
                    bestD = d;
                    bestRamp = ramp;
                    bestStopIdx = s;
                  }
                }
              }

              const isNw = x < cx && y < cy;
              const isSe = x > cx && y > cy;
              let nextIdx = bestStopIdx;
              if (isNw && bestStopIdx < bestRamp.length - 1) {
                nextIdx = bestStopIdx + 1;
              } else if (isSe && bestStopIdx > 1) {
                nextIdx = bestStopIdx - 1;
              }
              if (nextIdx !== bestStopIdx) {
                const col = bestRamp[nextIdx];
                buf[p * 4] = col[0];
                buf[p * 4 + 1] = col[1];
                buf[p * 4 + 2] = col[2];
              }
            }
          }
        }
      }
    }
  }

  // 9. Center / ground the sprite cleanly on integer pixels (non-tiles)
  let shiftedBy = { dx: 0, dy: 0 };
  if (!isTile && targetCenterGround) {
    let minX = SPRITE_SIZE;
    let maxX = -1;
    let minY = SPRITE_SIZE;
    let maxY = -1;
    for (let y = 0; y < SPRITE_SIZE; y++) {
      for (let x = 0; x < SPRITE_SIZE; x++) {
        if (buf[pIdx(x, y) + 3] >= 128) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX >= minX && maxY >= minY) {
      const bboxW = maxX - minX + 1;
      const bboxH = maxY - minY + 1;
      const targetMinX = Math.floor((SPRITE_SIZE - bboxW) / 2);
      // Slight +1px grounding bias when plenty of vertical margin exists (classic Tibia item placement)
      const idealMinY = Math.floor((SPRITE_SIZE - bboxH) / 2);
      const targetMinY = bboxH <= 26 ? Math.min(SPRITE_SIZE - 2 - bboxH, idealMinY + 1) : idealMinY;

      const dx = targetMinX - minX;
      const dy = targetMinY - minY;
      if (dx !== 0 || dy !== 0) {
        const shifted = new Uint8ClampedArray(SPRITE_SIZE * SPRITE_SIZE * 4);
        for (let y = minY; y <= maxY; y++) {
          const ny = y + dy;
          if (ny < 0 || ny >= SPRITE_SIZE) continue;
          for (let x = minX; x <= maxX; x++) {
            const nx = x + dx;
            if (nx < 0 || nx >= SPRITE_SIZE) continue;
            const sOff = pIdx(x, y);
            const dOff = pIdx(nx, ny);
            shifted[dOff] = buf[sOff];
            shifted[dOff + 1] = buf[sOff + 1];
            shifted[dOff + 2] = buf[sOff + 2];
            shifted[dOff + 3] = buf[sOff + 3];
          }
        }
        buf.set(shifted);
        shiftedBy = { dx, dy };
      }
    }
  }

  // Count final exact distinct colors
  const finalColors = new Set<number>();
  for (let p = 0; p < SPRITE_SIZE * SPRITE_SIZE; p++) {
    const i = p * 4;
    if (buf[i + 3] >= 128) {
      finalColors.add(rgbKey(buf[i], buf[i + 1], buf[i + 2]));
    }
  }

  return {
    rgba: buf,
    distinctColors: finalColors.size,
    snappedPixels,
    removedOrphans,
    removedDitherPixels,
    shiftedBy,
  };
}
