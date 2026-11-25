import { SPRITE_SIZE } from '../utils/tibiaParser';
import type { ProfileFeatureKey, SpriteFeatureVector } from './types';

function rgbToLuma(r: number, g: number, b: number): number {
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

function rgbToHsv(r: number, g: number, b: number): { h: number; s: number; v: number } {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const d = max - min;
  const v = max;
  const s = max === 0 ? 0 : d / max;
  let h = 0;
  if (d > 0) {
    if (max === rn) h = ((gn - bn) / d) % 6;
    else if (max === gn) h = (bn - rn) / d + 2;
    else h = (rn - gn) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  return { h, s, v };
}

function circularHueMean(anglesDeg: number[]): number | null {
  if (anglesDeg.length === 0) return null;
  let sinSum = 0;
  let cosSum = 0;
  for (const deg of anglesDeg) {
    const rad = (deg * Math.PI) / 180;
    sinSum += Math.sin(rad);
    cosSum += Math.cos(rad);
  }
  if (Math.abs(sinSum) < 1e-6 && Math.abs(cosSum) < 1e-6) return null;
  let mean = (Math.atan2(sinSum, cosSum) * 180) / Math.PI;
  if (mean < 0) mean += 360;
  return mean;
}

function circularHueDistance(a: number, b: number): number {
  const diff = Math.abs(a - b) % 360;
  return diff > 180 ? 360 - diff : diff;
}

function clamp01(v: number): number {
  if (Number.isNaN(v) || !Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

/**
 * Extracts deterministic pixel-exact features from a 32x32 RGBA buffer.
 * Every pixel is treated as a deliberate palette decision.
 */
export function extractSpriteFeatures(rgba: Uint8ClampedArray): SpriteFeatureVector {
  const size = SPRITE_SIZE;
  const totalPixels = size * size;

  if (rgba.length !== totalPixels * 4) {
    throw new Error(`Expected ${totalPixels * 4} RGBA bytes, got ${rgba.length}.`);
  }

  const opaqueMask = new Uint8Array(totalPixels);
  const colorKeys = new Int32Array(totalPixels);
  const lumas = new Float32Array(totalPixels);
  const saturations = new Float32Array(totalPixels);

  let opaquePixels = 0;
  let semiTransparentPixels = 0;
  let borderTouchCount = 0;
  let sumX = 0;
  let sumY = 0;
  let minX = size;
  let maxX = -1;
  let minY = size;
  let maxY = -1;

  const uniqueColors = new Map<number, { r: number; g: number; b: number; count: number; luma: number; h: number; s: number }>();
  const lumaBuckets = new Set<number>();

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const p = y * size + x;
      const i = p * 4;
      const a = rgba[i + 3];
      if (a > 0 && a < 255) {
        semiTransparentPixels++;
      }
      if (a > 0) {
        opaqueMask[p] = 1;
        opaquePixels++;
        sumX += x;
        sumY += y;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        if (x === 0 || x === size - 1 || y === 0 || y === size - 1) {
          borderTouchCount++;
        }

        const r = rgba[i];
        const g = rgba[i + 1];
        const b = rgba[i + 2];
        const key = (r << 16) | (g << 8) | b;
        colorKeys[p] = key;

        const luma = rgbToLuma(r, g, b);
        const hsv = rgbToHsv(r, g, b);
        lumas[p] = luma;
        saturations[p] = hsv.s;
        lumaBuckets.add(Math.round(luma * 24));

        const existing = uniqueColors.get(key);
        if (existing) {
          existing.count++;
        } else {
          uniqueColors.set(key, { r, g, b, count: 1, luma, h: hsv.h, s: hsv.s });
        }
      } else {
        colorKeys[p] = -1;
      }
    }
  }

  if (opaquePixels === 0) {
    return {
      opaquePixels: 0,
      coverage: 0,
      semiTransparentPixels,
      distinctColors: 0,
      lumaLevels: 0,
      borderTouchCount: 0,
      centerOffsetNorm: 0,
      bboxWidth: 0,
      bboxHeight: 0,
      componentCount: 0,
      orphanPixels: 0,
      orphanRatio: 0,
      nearDuplicateColorPairs: 0,
      nearDuplicateRatio: 0,
      mixelScore: 0,
      smoothGradientRatio: 0,
      speckleNoiseScore: 0,
      strayColorPixelRatio: 0,
      meanClusterSize: 0,
      outlineCoverage: 0,
      outlineDarknessDelta: 0,
      outlineDoublesRatio: 0,
      outlineDoubleThickRatio: 0,
      outlineHueShiftedRatio: 0,
      brokenOutlineRatio: 0,
      staircaseIrregularity: 0,
      nwLightGradient: 0,
      pillowShadingScore: 0,
      bandingScore: 0,
      ditherShare: 0,
      hueShiftDegrees: 0,
    };
  }

  const coverage = opaquePixels / totalPixels;
  const cx = sumX / opaquePixels;
  const cy = sumY / opaquePixels;
  const centerOffsetNorm = clamp01(Math.hypot(cx - 15.5, cy - 15.5) / 16);
  const bboxWidth = maxX - minX + 1;
  const bboxHeight = maxY - minY + 1;

  // 1. Exact-palette integrity: near-duplicate color pairs (1..6 per channel apart)
  const paletteArr = Array.from(uniqueColors.values());
  const distinctColors = paletteArr.length;
  let nearDuplicateColorPairs = 0;
  let nearDupColorInvolved = 0;
  const involved = new Uint8Array(distinctColors);

  const maxColorsToCompare = Math.min(distinctColors, 128);
  for (let i = 0; i < maxColorsToCompare; i++) {
    const a = paletteArr[i];
    for (let j = i + 1; j < maxColorsToCompare; j++) {
      const b = paletteArr[j];
      const dr = Math.abs(a.r - b.r);
      const dg = Math.abs(a.g - b.g);
      const db = Math.abs(a.b - b.b);
      const cheb = Math.max(dr, dg, db);
      if (cheb >= 1 && cheb <= 6) {
        nearDuplicateColorPairs++;
        involved[i] = 1;
        involved[j] = 1;
      }
    }
  }
  for (let i = 0; i < maxColorsToCompare; i++) {
    if (involved[i]) nearDupColorInvolved++;
  }
  const nearDuplicateRatio = distinctColors > 1 ? clamp01(nearDupColorInvolved / maxColorsToCompare) : 0;

  // 2. 8-connected foreground silhouette components & 1px orphan detection
  const fgVisited = new Uint8Array(totalPixels);
  const componentSizes: number[] = [];
  const queue = new Int32Array(totalPixels);

  for (let p = 0; p < totalPixels; p++) {
    if (!opaqueMask[p] || fgVisited[p]) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = p;
    fgVisited[p] = 1;
    let compSize = 0;
    while (head < tail) {
      const cur = queue[head++];
      compSize++;
      const x = cur % size;
      const y = (cur - x) / size;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= size) continue;
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          if (nx < 0 || nx >= size) continue;
          const np = ny * size + nx;
          if (opaqueMask[np] && !fgVisited[np]) {
            fgVisited[np] = 1;
            queue[tail++] = np;
          }
        }
      }
    }
    componentSizes.push(compSize);
  }

  componentSizes.sort((a, b) => b - a);
  const componentCount = componentSizes.length;
  let orphanPixels = 0;
  for (let i = 1; i < componentSizes.length; i++) {
    if (componentSizes[i] <= 2) {
      orphanPixels += componentSizes[i];
    }
  }
  const orphanRatio = clamp01(orphanPixels / opaquePixels);

  // 3. 4-connected same-color clusters (cluster hygiene & stray color confetti)
  const colorVisited = new Uint8Array(totalPixels);
  let clusterCount = 0;
  let strayColorPixels = 0;

  for (let p = 0; p < totalPixels; p++) {
    if (!opaqueMask[p] || colorVisited[p]) continue;
    const targetColor = colorKeys[p];
    let head = 0;
    let tail = 0;
    queue[tail++] = p;
    colorVisited[p] = 1;
    let cSize = 0;

    while (head < tail) {
      const cur = queue[head++];
      cSize++;
      const x = cur % size;
      const y = (cur - x) / size;
      if (x > 0 && opaqueMask[cur - 1] && !colorVisited[cur - 1] && colorKeys[cur - 1] === targetColor) {
        colorVisited[cur - 1] = 1;
        queue[tail++] = cur - 1;
      }
      if (x + 1 < size && opaqueMask[cur + 1] && !colorVisited[cur + 1] && colorKeys[cur + 1] === targetColor) {
        colorVisited[cur + 1] = 1;
        queue[tail++] = cur + 1;
      }
      if (y > 0 && opaqueMask[cur - size] && !colorVisited[cur - size] && colorKeys[cur - size] === targetColor) {
        colorVisited[cur - size] = 1;
        queue[tail++] = cur - size;
      }
      if (y + 1 < size && opaqueMask[cur + size] && !colorVisited[cur + size] && colorKeys[cur + size] === targetColor) {
        colorVisited[cur + size] = 1;
        queue[tail++] = cur + size;
      }
    }
    clusterCount++;
    if (cSize === 1) {
      strayColorPixels++;
    }
  }

  const meanClusterSize = clusterCount > 0 ? opaquePixels / clusterCount : 0;
  const strayColorPixelRatio = clamp01(strayColorPixels / opaquePixels);

  // 4. Adjacent pixel transitions: smooth gradient & speckle/noise
  let adjPairs = 0;
  let diffAdjPairs = 0;
  let smoothStepPairs = 0;
  let harshStepPairs = 0;
  let sameAdjPairs = 0;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const p = y * size + x;
      if (!opaqueMask[p]) continue;
      const i = p * 4;
      const r1 = rgba[i];
      const g1 = rgba[i + 1];
      const b1 = rgba[i + 2];

      const neighbors = [
        x + 1 < size ? p + 1 : -1,
        y + 1 < size ? p + size : -1,
      ];
      for (const np of neighbors) {
        if (np < 0 || !opaqueMask[np]) continue;
        adjPairs++;
        const j = np * 4;
        const dr = Math.abs(r1 - rgba[j]);
        const dg = Math.abs(g1 - rgba[j + 1]);
        const db = Math.abs(b1 - rgba[j + 2]);
        const maxDiff = Math.max(dr, dg, db);
        if (maxDiff === 0) {
          sameAdjPairs++;
        } else {
          diffAdjPairs++;
          if (maxDiff <= 12) {
            smoothStepPairs++;
          } else if (maxDiff >= 45) {
            harshStepPairs++;
          }
        }
      }
    }
  }

  // Smooth gradient signature: many different adjacent pairs AND high proportion of tiny <=12 channel steps
  // AND many distinct colors (prevents false positives on small crisp palettes that happen to share one close ramp step)
  const rawSmoothFrac = diffAdjPairs > 0 ? smoothStepPairs / diffAdjPairs : 0;
  const colorExcessFactor = clamp01((distinctColors - 12) / 28);
  const smoothGradientRatio = clamp01(rawSmoothFrac * (0.45 + 0.55 * colorExcessFactor));

  // True speckle / random noise signature:
  // Measure how many interior/opaque pixels jump wildly (max channel diff >= 65) from EVERY 4-connected opaque neighbor,
  // combined with high harsh-transition ratio and extreme color uniqueness (distinctColors / opaquePixels).
  // Detailed Tibia pixel art has coherent surface ramps so neighbors stay within < 65 RGB; smooth gradients have tiny steps (<= 12);
  // true RGB noise has wild uncorrelated jumps on virtually every pixel.
  let wildIsolatedPixels = 0;
  let checkedForNoise = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const p = y * size + x;
      if (!opaqueMask[p]) continue;
      const i = p * 4;
      const r1 = rgba[i];
      const g1 = rgba[i + 1];
      const b1 = rgba[i + 2];
      let minNbDiff = 255;
      let opaqueNbCount = 0;
      const nbs = [
        x > 0 ? p - 1 : -1,
        x + 1 < size ? p + 1 : -1,
        y > 0 ? p - size : -1,
        y + 1 < size ? p + size : -1,
      ];
      for (const np of nbs) {
        if (np < 0 || !opaqueMask[np]) continue;
        opaqueNbCount++;
        const j = np * 4;
        const d = Math.max(
          Math.abs(r1 - rgba[j]),
          Math.abs(g1 - rgba[j + 1]),
          Math.abs(b1 - rgba[j + 2]),
        );
        if (d < minNbDiff) minNbDiff = d;
      }
      if (opaqueNbCount >= 2) {
        checkedForNoise++;
        if (minNbDiff >= 65) {
          wildIsolatedPixels++;
        }
      }
    }
  }
  const wildNoiseFrac = checkedForNoise > 0 ? wildIsolatedPixels / checkedForNoise : 0;
  const harshRatio = diffAdjPairs > 0 ? harshStepPairs / diffAdjPairs : 0;
  const uniquenessRatio = opaquePixels > 0 ? distinctColors / opaquePixels : 0;
  const speckleNoiseScore = clamp01(
    wildNoiseFrac * 0.6 + harshRatio * 0.25 + clamp01((uniquenessRatio - 0.35) / 0.55) * 0.15,
  );

  // 5. Resample / "mixel" detection:
  // Detect 2x2 uniform blocks across the sprite and repeated 2+ runs aligned on both axes
  let active2x2Blocks = 0;
  let uniform2x2Blocks = 0;
  let uniform2x2OffsetBlocks = 0;
  let active2x2OffsetBlocks = 0;

  for (let y = 0; y < size - 1; y += 2) {
    for (let x = 0; x < size - 1; x += 2) {
      const p00 = y * size + x;
      const p10 = p00 + 1;
      const p01 = p00 + size;
      const p11 = p01 + 1;
      const anyOpaque = opaqueMask[p00] | opaqueMask[p10] | opaqueMask[p01] | opaqueMask[p11];
      if (!anyOpaque) continue;
      active2x2Blocks++;
      if (
        colorKeys[p00] === colorKeys[p10] &&
        colorKeys[p00] === colorKeys[p01] &&
        colorKeys[p00] === colorKeys[p11]
      ) {
        uniform2x2Blocks++;
      }
    }
  }

  for (let y = 1; y < size - 1; y += 2) {
    for (let x = 1; x < size - 1; x += 2) {
      const p00 = y * size + x;
      const p10 = p00 + 1;
      const p01 = p00 + size;
      const p11 = p01 + 1;
      const anyOpaque = opaqueMask[p00] | opaqueMask[p10] | opaqueMask[p01] | opaqueMask[p11];
      if (!anyOpaque) continue;
      active2x2OffsetBlocks++;
      if (
        colorKeys[p00] === colorKeys[p10] &&
        colorKeys[p00] === colorKeys[p01] &&
        colorKeys[p00] === colorKeys[p11]
      ) {
        uniform2x2OffsetBlocks++;
      }
    }
  }

  const evenBlockUniformity = active2x2Blocks > 0 ? uniform2x2Blocks / active2x2Blocks : 0;
  const oddBlockUniformity = active2x2OffsetBlocks > 0 ? uniform2x2OffsetBlocks / active2x2OffsetBlocks : 0;
  // A flat fill has both even and odd uniformity ~1.0, whereas a 2x2 upscaled/mixel sprite has
  // high phase-locked difference between even-aligned vs odd-aligned 2x2 blocks, OR very high even uniformity with > 2 colors.
  const phaseLock = Math.abs(evenBlockUniformity - oddBlockUniformity);
  const multiColorFactor = clamp01((distinctColors - 2) / 6);
  const mixelScore = clamp01(
    (phaseLock * 1.45 + Math.max(0, Math.max(evenBlockUniformity, oddBlockUniformity) - 0.65) * 1.2) *
      multiColorFactor,
  );

  // 6. Silhouette perimeter, distance transform (for banding & pillow shading), and outline analysis
  const isPerimeter = new Uint8Array(totalPixels);
  const distFromEdge = new Int16Array(totalPixels);
  distFromEdge.fill(-1);
  let bfsHead = 0;
  let bfsTail = 0;

  let perimeterCount = 0;
  let interiorCount = 0;
  let perimeterLumaSum = 0;
  let interiorLumaSum = 0;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const p = y * size + x;
      if (!opaqueMask[p]) continue;
      const touchesTrans =
        x === 0 ||
        x === size - 1 ||
        y === 0 ||
        y === size - 1 ||
        !opaqueMask[p - 1] ||
        !opaqueMask[p + 1] ||
        !opaqueMask[p - size] ||
        !opaqueMask[p + size];

      if (touchesTrans) {
        isPerimeter[p] = 1;
        distFromEdge[p] = 0;
        queue[bfsTail++] = p;
        perimeterCount++;
        perimeterLumaSum += lumas[p];
      } else {
        interiorCount++;
        interiorLumaSum += lumas[p];
      }
    }
  }

  // BFS inward distance from silhouette edge
  let maxEdgeDist = 0;
  while (bfsHead < bfsTail) {
    const cur = queue[bfsHead++];
    const d = distFromEdge[cur];
    if (d > maxEdgeDist) maxEdgeDist = d;
    const x = cur % size;
    const y = (cur - x) / size;
    const nbs = [
      x > 0 ? cur - 1 : -1,
      x + 1 < size ? cur + 1 : -1,
      y > 0 ? cur - size : -1,
      y + 1 < size ? cur + size : -1,
    ];
    for (const np of nbs) {
      if (np >= 0 && opaqueMask[np] && distFromEdge[np] === -1) {
        distFromEdge[np] = d + 1;
        queue[bfsTail++] = np;
      }
    }
  }

  const meanPerimeterLuma = perimeterCount > 0 ? perimeterLumaSum / perimeterCount : 0;
  const meanInteriorLuma = interiorCount > 0 ? interiorLumaSum / interiorCount : meanPerimeterLuma;
  const outlineDarknessDelta = meanInteriorLuma - meanPerimeterLuma;

  // Outline pixel metrics
  const darkThreshold = Math.min(0.28, Math.max(0.14, meanInteriorLuma * 0.58));
  let darkPerimeterCount = 0;
  let brokenPerimeterCount = 0;
  let hueShiftedOutlineCount = 0;
  let doubleThickOutlineCount = 0;

  for (let p = 0; p < totalPixels; p++) {
    if (!isPerimeter[p]) continue;
    const l = lumas[p];
    if (l <= darkThreshold || l < meanInteriorLuma - 0.14) {
      darkPerimeterCount++;
      if (saturations[p] >= 0.14 && l > 0.02) {
        hueShiftedOutlineCount++;
      }
      // Check if an inward 4-neighbor is also dark (indicates >=2px thick outline)
      const x = p % size;
      const y = (p - x) / size;
      let darkInward = false;
      const nbs = [
        x > 0 ? p - 1 : -1,
        x + 1 < size ? p + 1 : -1,
        y > 0 ? p - size : -1,
        y + 1 < size ? p + size : -1,
      ];
      for (const np of nbs) {
        if (np >= 0 && opaqueMask[np] && !isPerimeter[np] && lumas[np] <= darkThreshold) {
          darkInward = true;
          break;
        }
      }
      if (darkInward) doubleThickOutlineCount++;
    } else if (l >= Math.max(0.38, meanInteriorLuma * 0.85)) {
      brokenPerimeterCount++;
    }
  }

  const outlineCoverage = perimeterCount > 0 ? clamp01(darkPerimeterCount / perimeterCount) : 0;
  const brokenOutlineRatio = perimeterCount > 0 ? clamp01(brokenPerimeterCount / perimeterCount) : 0;
  const outlineHueShiftedRatio =
    darkPerimeterCount > 0 ? clamp01(hueShiftedOutlineCount / darkPerimeterCount) : 0;
  const outlineDoubleThickRatio =
    darkPerimeterCount > 0 ? clamp01(doubleThickOutlineCount / darkPerimeterCount) : 0;

  // Outline "doubles" (L-corners where a perimeter pixel has 2 orthogonal perimeter neighbors creating a 1px corner jag)
  let outlineDoubles = 0;
  let outlineCornerCandidates = 0;
  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const p = y * size + x;
      if (!isPerimeter[p]) continue;
      const up = isPerimeter[p - size];
      const down = isPerimeter[p + size];
      const left = isPerimeter[p - 1];
      const right = isPerimeter[p + 1];
      const hasL =
        (up && left && !down && !right) ||
        (up && right && !down && !left) ||
        (down && left && !up && !right) ||
        (down && right && !up && !left);
      if (hasL) {
        outlineCornerCandidates++;
        // True "double" on a diagonal: the diagonal neighbor opposite the L is also perimeter,
        // meaning this pixel creates a redundant 90-degree step along a diagonal edge
        const diagOuter =
          (up && left && isPerimeter[p - size + 1] && isPerimeter[p + size - 1]) ||
          (up && right && isPerimeter[p - size - 1] && isPerimeter[p + size + 1]) ||
          (down && left && isPerimeter[p + size + 1] && isPerimeter[p - size - 1]) ||
          (down && right && isPerimeter[p + size - 1] && isPerimeter[p - size + 1]);
        // Or a 3-pixel L-clump inside a 2x2 box where all 3 are dark outline
        const isDarkL =
          lumas[p] <= darkThreshold &&
          ((up && left && opaqueMask[p - size - 1] && isPerimeter[p - size] && isPerimeter[p - 1]) ||
            (up && right && opaqueMask[p - size + 1] && isPerimeter[p - size] && isPerimeter[p + 1]) ||
            (down && left && opaqueMask[p + size - 1] && isPerimeter[p + size] && isPerimeter[p - 1]) ||
            (down && right && opaqueMask[p + size + 1] && isPerimeter[p + size] && isPerimeter[p + 1]));
        if (diagOuter || isDarkL) {
          outlineDoubles++;
        }
      }
    }
  }
  const outlineDoublesRatio =
    perimeterCount > 4 ? clamp01(outlineDoubles / Math.max(4, perimeterCount * 0.45)) : 0;

  // Staircase irregularity: scan horizontal and vertical perimeter segment lengths and compute variance
  const runLengths: number[] = [];
  for (let y = 0; y < size; y++) {
    let run = 0;
    for (let x = 0; x < size; x++) {
      if (isPerimeter[y * size + x]) {
        run++;
      } else if (run > 0) {
        if (run >= 2 && run <= 8) runLengths.push(run);
        run = 0;
      }
    }
    if (run >= 2 && run <= 8) runLengths.push(run);
  }
  let runChanges = 0;
  for (let i = 1; i < runLengths.length; i++) {
    if (Math.abs(runLengths[i] - runLengths[i - 1]) >= 2) {
      runChanges++;
    }
  }
  const staircaseIrregularity =
    runLengths.length > 2
      ? clamp01(runChanges / (runLengths.length - 1))
      : clamp01(outlineCornerCandidates > 0 ? outlineDoubles / (outlineCornerCandidates + 2) : 0);

  // 7. Shading logic: NW-SE light direction, pillow shading, banding, dithering, hue shift
  let nwSum = 0;
  let nwCount = 0;
  let seSum = 0;
  let seCount = 0;

  // For pillow shading: compare luminance at each edge distance ring AND check if NW edge is dark like SE edge
  const ringLumaSum = new Float32Array(16);
  const ringCount = new Int32Array(16);
  let nwRing1Sum = 0;
  let nwRing1Count = 0;
  let seRing1Sum = 0;
  let seRing1Count = 0;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const p = y * size + x;
      if (!opaqueMask[p] || isPerimeter[p]) continue;
      const l = lumas[p];
      const dx = x - cx;
      const dy = y - cy;
      const diagProj = dx + dy; // negative = NW, positive = SE
      if (diagProj < -0.5) {
        nwSum += l;
        nwCount++;
      } else if (diagProj > 0.5) {
        seSum += l;
        seCount++;
      }

      const d = Math.min(15, Math.max(0, distFromEdge[p]));
      ringLumaSum[d] += l;
      ringCount[d]++;
      if (d === 1) {
        if (diagProj < -0.5) {
          nwRing1Sum += l;
          nwRing1Count++;
        } else if (diagProj > 0.5) {
          seRing1Sum += l;
          seRing1Count++;
        }
      }
    }
  }

  const nwMean = nwCount > 0 ? nwSum / nwCount : meanInteriorLuma;
  const seMean = seCount > 0 ? seSum / seCount : meanInteriorLuma;
  const nwLightGradient = Math.max(-1, Math.min(1, nwMean - seMean));

  // Pillow shading: monotonic radial brightening from ring 1 -> deep core, while ring 1 NW and SE have nearly equal dark luminance
  let monotonicIncreases = 0;
  let ringPairs = 0;
  let ring1Mean = ringCount[1] > 0 ? ringLumaSum[1] / ringCount[1] : meanInteriorLuma;
  let coreLumaSum = 0;
  let coreCount = 0;
  for (let d = 1; d < Math.min(8, maxEdgeDist); d++) {
    if (ringCount[d] >= 3 && ringCount[d + 1] >= 2) {
      ringPairs++;
      const mCur = ringLumaSum[d] / ringCount[d];
      const mNext = ringLumaSum[d + 1] / ringCount[d + 1];
      if (mNext > mCur + 0.025) monotonicIncreases++;
    }
  }
  for (let d = Math.max(2, Math.floor(maxEdgeDist * 0.6)); d <= Math.min(15, maxEdgeDist); d++) {
    coreLumaSum += ringLumaSum[d];
    coreCount += ringCount[d];
  }
  const coreMean = coreCount > 0 ? coreLumaSum / coreCount : meanInteriorLuma;
  const radialBrightening = clamp01((coreMean - ring1Mean) * 2.5);
  const nwSeRingSymmetry =
    nwRing1Count > 0 && seRing1Count > 0
      ? clamp01(1 - Math.abs(nwRing1Sum / nwRing1Count - seRing1Sum / seRing1Count) * 4.5)
      : 0;
  const monoFrac = ringPairs > 0 ? monotonicIncreases / ringPairs : 0;
  const pillowShadingScore = clamp01(
    radialBrightening * 0.45 + nwSeRingSymmetry * 0.35 + monoFrac * 0.2 - Math.max(0, nwLightGradient) * 0.9,
  );

  // Banding: high same-color coherence along the same distance-to-edge ring while changing strictly between ring d and d+1
  let sameRingPairs = 0;
  let sameRingSameColor = 0;
  let crossRingPairs = 0;
  let crossRingDiffColor = 0;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const p = y * size + x;
      if (!opaqueMask[p] || distFromEdge[p] < 1) continue;
      const d1 = distFromEdge[p];
      const c1 = colorKeys[p];
      const nbs = [x + 1 < size ? p + 1 : -1, y + 1 < size ? p + size : -1];
      for (const np of nbs) {
        if (np < 0 || !opaqueMask[np] || distFromEdge[np] < 1) continue;
        const d2 = distFromEdge[np];
        if (d1 === d2) {
          sameRingPairs++;
          if (colorKeys[np] === c1) sameRingSameColor++;
        } else {
          crossRingPairs++;
          if (colorKeys[np] !== c1) crossRingDiffColor++;
        }
      }
    }
  }
  const sameRingHomogeneity = sameRingPairs > 4 ? sameRingSameColor / sameRingPairs : 0;
  const crossRingChange = crossRingPairs > 4 ? crossRingDiffColor / crossRingPairs : 0;
  const bandingScore = clamp01(
    Math.max(0, sameRingHomogeneity - 0.55) * 1.4 * Math.max(0, crossRingChange - 0.45) * 1.8,
  );

  // Dithering share: interior 2x2 alternating checkerboards (A B / B A where A != B)
  let interior2x2 = 0;
  let dither2x2 = 0;
  for (let y = 0; y < size - 1; y++) {
    for (let x = 0; x < size - 1; x++) {
      const p00 = y * size + x;
      const p10 = p00 + 1;
      const p01 = p00 + size;
      const p11 = p01 + 1;
      if (
        !opaqueMask[p00] ||
        !opaqueMask[p10] ||
        !opaqueMask[p01] ||
        !opaqueMask[p11] ||
        isPerimeter[p00] ||
        isPerimeter[p10] ||
        isPerimeter[p01] ||
        isPerimeter[p11]
      ) {
        continue;
      }
      interior2x2++;
      if (
        colorKeys[p00] === colorKeys[p11] &&
        colorKeys[p10] === colorKeys[p01] &&
        colorKeys[p00] !== colorKeys[p10]
      ) {
        dither2x2++;
      }
    }
  }
  const ditherShare = interior2x2 > 0 ? clamp01(dither2x2 / interior2x2) : 0;

  // Hue shift between dark shadow colors and bright highlight colors (among saturated colors)
  const saturatedColors = paletteArr.filter((c) => c.s >= 0.15).sort((a, b) => a.luma - b.luma);
  let hueShiftDegrees = 0;
  if (saturatedColors.length >= 2) {
    const splitIdx = Math.max(1, Math.floor(saturatedColors.length / 2));
    const darkHues = saturatedColors.slice(0, splitIdx).map((c) => c.h);
    const brightHues = saturatedColors.slice(splitIdx).map((c) => c.h);
    const darkMean = circularHueMean(darkHues);
    const brightMean = circularHueMean(brightHues);
    if (darkMean !== null && brightMean !== null) {
      hueShiftDegrees = circularHueDistance(darkMean, brightMean);
    }
  }

  return {
    opaquePixels,
    coverage,
    semiTransparentPixels,
    distinctColors,
    lumaLevels: lumaBuckets.size,
    borderTouchCount,
    centerOffsetNorm,
    bboxWidth,
    bboxHeight,
    componentCount,
    orphanPixels,
    orphanRatio,
    nearDuplicateColorPairs,
    nearDuplicateRatio,
    mixelScore,
    smoothGradientRatio,
    speckleNoiseScore,
    strayColorPixelRatio,
    meanClusterSize,
    outlineCoverage,
    outlineDarknessDelta,
    outlineDoublesRatio,
    outlineDoubleThickRatio,
    outlineHueShiftedRatio,
    brokenOutlineRatio,
    staircaseIrregularity,
    nwLightGradient,
    pillowShadingScore,
    bandingScore,
    ditherShare,
    hueShiftDegrees,
  };
}

export function extractProfileNumericArray(
  features: SpriteFeatureVector,
  keys: readonly ProfileFeatureKey[],
): number[] {
  return keys.map((k) => features[k]);
}
