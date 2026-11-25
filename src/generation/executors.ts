import { createSeededRng } from '../db/split';
import { opEnforceDarkOutline, opSnapBinaryAlpha } from '../evolution/pixelOps';
import { SPRITE_SIZE } from '../utils/tibiaParser';
import { decomposeSpriteToIndexedAndRampRoles, normalizePromptTokens } from './indexer';
import type { PixelDiffEdit } from './types';

function calcLuma(r: number, g: number, b: number): number {
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

function clampByte(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v)));
}

function pIdx(x: number, y: number): number {
  return (y * SPRITE_SIZE + x) * 4;
}

// ============================================================================
// CLASSIC TIBIA 7.72 HUE-SHIFTED MATERIAL RAMPS (STRICTLY LUMA-ORDERED)
// ============================================================================

export interface MaterialRampPreset {
  id: string;
  keywords: string[];
  primary: Array<[number, number, number]>; // 5 stops: outline -> shadow -> mid -> light -> specular
  accent: Array<[number, number, number]>; // 3 stops: dark -> mid -> bright
}

export const CLASSIC_MATERIAL_RAMPS: readonly MaterialRampPreset[] = [
  {
    id: 'royal_blue_gold',
    keywords: ['azul', 'safira', 'gelo', 'magic', 'mpa', 'real', 'celeste', 'arcano', 'blue', 'ice'],
    primary: [
      [18, 22, 38],
      [34, 52, 92],
      [58, 92, 148],
      [104, 148, 206],
      [176, 214, 246],
    ],
    accent: [
      [104, 68, 24],
      [184, 136, 42],
      [244, 216, 112],
    ],
  },
  {
    id: 'crimson_ruby',
    keywords: ['fogo', 'rubi', 'vermelho', 'sangue', 'dragao', 'demon', 'infernal', 'fire', 'red', 'crimson'],
    primary: [
      [34, 14, 22],
      [88, 24, 36],
      [148, 40, 44],
      [204, 78, 62],
      [246, 162, 116],
    ],
    accent: [
      [112, 62, 18],
      [204, 148, 44],
      [250, 224, 124],
    ],
  },
  {
    id: 'emerald_forest',
    keywords: ['verde', 'esmeralda', 'veneno', 'floresta', 'druida', 'jade', 'green', 'emerald', 'nature'],
    primary: [
      [16, 30, 26],
      [28, 68, 54],
      [48, 116, 74],
      [96, 168, 96],
      [172, 224, 144],
    ],
    accent: [
      [96, 62, 24],
      [172, 126, 48],
      [236, 204, 110],
    ],
  },
  {
    id: 'golden_paladin',
    keywords: ['ouro', 'dourado', 'paladino', 'solar', 'sagrado', 'luz', 'gold', 'golden', 'holy'],
    primary: [
      [34, 22, 14],
      [88, 54, 22],
      [148, 102, 34],
      [208, 162, 56],
      [248, 230, 134],
    ],
    accent: [
      [32, 52, 92],
      [68, 108, 168],
      [146, 192, 236],
    ],
  },
  {
    id: 'obsidian_amethyst',
    keywords: ['roxo', 'ametista', 'sombra', 'obsidiana', 'negro', 'warlock', 'necro', 'purple', 'dark', 'shadow'],
    primary: [
      [22, 16, 32],
      [54, 34, 78],
      [94, 58, 126],
      [146, 98, 178],
      [208, 168, 232],
    ],
    accent: [
      [116, 34, 48],
      [188, 58, 68],
      [242, 136, 124],
    ],
  },
  {
    id: 'forged_steel',
    keywords: ['aco', 'prata', 'ferro', 'cavaleiro', 'loyal', 'steel', 'silver', 'iron', 'knight'],
    primary: [
      [20, 24, 34],
      [54, 62, 82],
      [98, 108, 132],
      [154, 164, 186],
      [218, 226, 240],
    ],
    accent: [
      [108, 72, 26],
      [182, 134, 46],
      [242, 212, 114],
    ],
  },
  {
    id: 'ancient_bronze_oak',
    keywords: ['madeira', 'couro', 'bronze', 'antigo', 'rustico', 'terra', 'wood', 'leather', 'bronze', 'oak'],
    primary: [
      [30, 18, 14],
      [74, 44, 28],
      [122, 78, 44],
      [168, 118, 68],
      [214, 174, 118],
    ],
    accent: [
      [42, 84, 68],
      [78, 136, 102],
      [144, 196, 148],
    ],
  },
];

/**
 * Selects a classic hue-shifted Tibia material ramp from prompt tokens + deterministic variant offset.
 */
export function selectMaterialRampForPrompt(
  prompt: string,
  variantIndex: number,
): MaterialRampPreset {
  const tokens = normalizePromptTokens(prompt);
  let bestPreset: MaterialRampPreset | null = null;
  let bestHits = 0;

  for (const preset of CLASSIC_MATERIAL_RAMPS) {
    let hits = 0;
    for (const tok of tokens) {
      if (preset.keywords.some((kw) => tok.includes(kw) || kw.includes(tok))) {
        hits++;
      }
    }
    if (hits > bestHits) {
      bestHits = hits;
      bestPreset = preset;
    }
  }

  if (bestPreset && variantIndex % 3 === 0) {
    return bestPreset;
  }
  if (bestPreset) {
    // Subtle deterministic hue/brightness shift variant while keeping strict luma ordering
    const shift = variantIndex % 2 === 1 ? 8 : -8;
    const adjust = (stops: Array<[number, number, number]>): Array<[number, number, number]> =>
      stops
        .map(([r, g, b], idx): [number, number, number] => {
          if (idx === 0) return [r, g, b]; // keep dark outline anchor intact
          return [clampByte(r + shift), clampByte(g + Math.round(shift * 0.6)), clampByte(b - Math.round(shift * 0.5))];
        })
        .sort((a, b) => calcLuma(a[0], a[1], a[2]) - calcLuma(b[0], b[1], b[2]));
    return {
      ...bestPreset,
      id: `${bestPreset.id}_v${variantIndex}`,
      primary: adjust(bestPreset.primary),
      accent: adjust(bestPreset.accent),
    };
  }

  return CLASSIC_MATERIAL_RAMPS[variantIndex % CLASSIC_MATERIAL_RAMPS.length];
}

/**
 * Interpolates a target RGB ramp of `targetCount` discrete colors strictly ordered by luminance.
 */
export function buildLumaOrderedTargetPalette(
  stops: Array<[number, number, number]>,
  targetCount: number,
): Array<[number, number, number]> {
  const sortedStops = [...stops].sort(
    (a, b) => calcLuma(a[0], a[1], a[2]) - calcLuma(b[0], b[1], b[2]),
  );
  if (targetCount <= 0) return [];
  if (targetCount === 1) return [sortedStops[0]];
  if (sortedStops.length === 1) return Array.from({ length: targetCount }, () => sortedStops[0]);

  const out: Array<[number, number, number]> = [];
  for (let i = 0; i < targetCount; i++) {
    const t = i / (targetCount - 1);
    const scaled = t * (sortedStops.length - 1);
    const lo = Math.floor(scaled);
    const hi = Math.min(sortedStops.length - 1, lo + 1);
    const frac = scaled - lo;
    const r = clampByte(sortedStops[lo][0] + (sortedStops[hi][0] - sortedStops[lo][0]) * frac);
    const g = clampByte(sortedStops[lo][1] + (sortedStops[hi][1] - sortedStops[lo][1]) * frac);
    const b = clampByte(sortedStops[lo][2] + (sortedStops[hi][2] - sortedStops[lo][2]) * frac);
    out.push([r, g, b]);
  }

  // Enforce strict monotonic luma ordering even after integer rounding
  out.sort((a, b) => calcLuma(a[0], a[1], a[2]) - calcLuma(b[0], b[1], b[2]));
  return out;
}

/**
 * TIER T1 EXECUTOR — Semantic Ramp Recolor:
 * Recolors a source sprite using a strictly luma-ordered target ramp (+ optional accent ramp),
 * guaranteeing:
 *  1) Binary alpha (0 or 255) preserved identically.
 *  2) Dark outer outline remains the darkest luminance tier.
 *  3) Strict monotonic luminance ordering preserved for every source palette entry.
 */
export function executeRampRecolor(params: {
  sourceRgba: Uint8ClampedArray;
  primaryStops: Array<[number, number, number]>;
  accentStops?: Array<[number, number, number]>;
  isTile?: boolean;
}): {
  rgba: Uint8ClampedArray;
  sourcePaletteLumas: number[];
  mappedPaletteLumas: number[];
} {
  const { sourceRgba, primaryStops, accentStops, isTile = false } = params;
  const cleanSrc = opSnapBinaryAlpha(sourceRgba, 0.5);
  const dec = decomposeSpriteToIndexedAndRampRoles(cleanSrc);
  const nColors = dec.paletteRgb.length;

  if (nColors === 0) {
    return {
      rgba: new Uint8ClampedArray(cleanSrc),
      sourcePaletteLumas: [],
      mappedPaletteLumas: [],
    };
  }

  // Classic Tibia 7.72 pixel art uses discrete ramps (<=10 stops), never 200+ smooth interpolated shades
  const discretePrimaryCount = Math.min(nColors, Math.max(5, Math.min(10, primaryStops.length * 2)));
  const discretePrimary = buildLumaOrderedTargetPalette(primaryStops, discretePrimaryCount);
  const orderedPrimary: Array<[number, number, number]> = Array.from({ length: nColors }, (_, idx) => {
    const mappedIdx =
      nColors <= 1
        ? 0
        : Math.round((idx / (nColors - 1)) * (discretePrimary.length - 1));
    return discretePrimary[mappedIdx];
  });
  const orderedAccent = accentStops
    ? buildLumaOrderedTargetPalette(accentStops, Math.max(3, Math.min(4, accentStops.length)))
    : null;

  const out = new Uint8ClampedArray(SPRITE_SIZE * SPRITE_SIZE * 4);
  const sourcePaletteLumas = dec.paletteRgb.map(([r, g, b]) => calcLuma(r, g, b));
  const mappedPaletteLumas = orderedPrimary.map(([r, g, b]) => calcLuma(r, g, b));

  for (let p = 0; p < SPRITE_SIZE * SPRITE_SIZE; p++) {
    const gridIdx = dec.indexedGrid[p];
    if (gridIdx === 0) continue; // strict binary transparent

    const palIdx = gridIdx - 1;
    const role = dec.rampRoles[p];
    let rgb = orderedPrimary[palIdx];

    // Map accent-role pixels onto accent ramp ONLY if monotonic relative to outline
    if (role === 5 && orderedAccent && palIdx > 0) {
      const aIdx = Math.min(
        orderedAccent.length - 1,
        Math.round((palIdx / Math.max(1, nColors - 1)) * (orderedAccent.length - 1)),
      );
      const candidateAccent = orderedAccent[aIdx];
      // Preserve outline darkest invariant
      if (calcLuma(candidateAccent[0], candidateAccent[1], candidateAccent[2]) > mappedPaletteLumas[0] + 0.04) {
        rgb = candidateAccent;
      }
    }

    const i = p * 4;
    out[i] = rgb[0];
    out[i + 1] = rgb[1];
    out[i + 2] = rgb[2];
    out[i + 3] = 255;
  }

  // Ensure crisp outer dark outline is never lost for non-tiles
  const finalRgba = isTile ? out : opEnforceDarkOutline(out, 0.45, false);

  return {
    rgba: finalRgba,
    sourcePaletteLumas,
    mappedPaletteLumas,
  };
}

/**
 * BOUNDED PIXEL-DIFF EXECUTOR (used by both Tier T1 and Tier T2):
 * Applies at most `maxDiffPixels` discrete exact-pixel edits on the 32x32 grid.
 * Guarantees:
 *  - Actual changed pixel count NEVER exceeds `maxDiffPixels`.
 *  - Binary alpha strictly preserved (`0` or `255`, never semi-transparent).
 *  - Outer dark silhouette outline is protected when `preserveOutline` is true.
 */
export function executeBoundedPixelDiff(params: {
  baseRgba: Uint8ClampedArray;
  edits: PixelDiffEdit[];
  maxDiffPixels: number;
  preserveOutline?: boolean;
  isTile?: boolean;
}): {
  rgba: Uint8ClampedArray;
  appliedCount: number;
  rejectedCount: number;
} {
  const {
    baseRgba,
    edits,
    maxDiffPixels,
    preserveOutline = true,
    isTile = false,
  } = params;

  const limit = Math.max(0, Math.floor(maxDiffPixels));
  const out = opSnapBinaryAlpha(baseRgba, 0.5);

  // Identify protected outer outline pixels on non-tiles
  const protectedMask = new Uint8Array(SPRITE_SIZE * SPRITE_SIZE);
  if (preserveOutline && !isTile) {
    for (let y = 0; y < SPRITE_SIZE; y++) {
      for (let x = 0; x < SPRITE_SIZE; x++) {
        const idx = pIdx(x, y);
        if (out[idx + 3] === 0) continue;
        const isEdge =
          x === 0 ||
          x === SPRITE_SIZE - 1 ||
          y === 0 ||
          y === SPRITE_SIZE - 1 ||
          out[pIdx(x - 1, y) + 3] === 0 ||
          out[pIdx(x + 1, y) + 3] === 0 ||
          out[pIdx(x, y - 1) + 3] === 0 ||
          out[pIdx(x, y + 1) + 3] === 0;
        if (isEdge) {
          protectedMask[y * SPRITE_SIZE + x] = 1;
        }
      }
    }
  }

  const changedCoords = new Set<number>();
  let rejectedCount = 0;

  for (const edit of edits) {
    if (changedCoords.size >= limit) {
      rejectedCount++;
      continue;
    }
    const x = Math.round(edit.x);
    const y = Math.round(edit.y);
    if (x < 0 || x >= SPRITE_SIZE || y < 0 || y >= SPRITE_SIZE) {
      rejectedCount++;
      continue;
    }

    const p = y * SPRITE_SIZE + x;
    const i = p * 4;
    const nextA: 0 | 255 = edit.a >= 128 ? 255 : 0;
    const nextR = nextA === 255 ? clampByte(edit.r) : 0;
    const nextG = nextA === 255 ? clampByte(edit.g) : 0;
    const nextB = nextA === 255 ? clampByte(edit.b) : 0;

    if (preserveOutline && !isTile) {
      const wasOpaque = out[i + 3] === 255;
      // 1) Never erase or brighten an existing outer outline pixel
      if (protectedMask[p] && (nextA === 0 || calcLuma(nextR, nextG, nextB) > 0.3)) {
        rejectedCount++;
        continue;
      }
      // 2) Never paint a bright un-outlined pixel onto transparent exterior space
      if (!wasOpaque && nextA === 255 && calcLuma(nextR, nextG, nextB) > 0.3) {
        rejectedCount++;
        continue;
      }
      // 3) Never punch a transparent hole inside opaque body (would expose bright interior to alpha=0)
      if (wasOpaque && nextA === 0) {
        rejectedCount++;
        continue;
      }
    }

    if (out[i] === nextR && out[i + 1] === nextG && out[i + 2] === nextB && out[i + 3] === nextA) {
      continue;
    }

    out[i] = nextR;
    out[i + 1] = nextG;
    out[i + 2] = nextB;
    out[i + 3] = nextA;
    changedCoords.add(p);
  }

  return {
    rgba: out,
    appliedCount: changedCoords.size,
    rejectedCount,
  };
}

/**
 * Generates deterministic, pixel-art-clean bounded interior motif edits (gem, clasp, trim, specular cluster, rune crest)
 * using ONLY colors from the sprite's target primary/accent ramp, strictly within `maxDiffPixels`.
 */
export function buildDeterministicDiffEdits(params: {
  baseRgba: Uint8ClampedArray;
  accentStops: Array<[number, number, number]>;
  highlightStop: [number, number, number];
  maxDiffPixels: number;
  seed: number;
}): PixelDiffEdit[] {
  const { baseRgba, accentStops, highlightStop, maxDiffPixels, seed } = params;
  const rng = createSeededRng(seed);
  const edits: PixelDiffEdit[] = [];

  // Collect strictly interior opaque pixels (all 4 orthogonal neighbors opaque)
  const interior: Array<{ x: number; y: number }> = [];
  let sumX = 0;
  let sumY = 0;

  for (let y = 2; y < SPRITE_SIZE - 2; y++) {
    for (let x = 2; x < SPRITE_SIZE - 2; x++) {
      if (
        baseRgba[pIdx(x, y) + 3] === 255 &&
        baseRgba[pIdx(x - 1, y) + 3] === 255 &&
        baseRgba[pIdx(x + 1, y) + 3] === 255 &&
        baseRgba[pIdx(x, y - 1) + 3] === 255 &&
        baseRgba[pIdx(x, y + 1) + 3] === 255
      ) {
        interior.push({ x, y });
        sumX += x;
        sumY += y;
      }
    }
  }

  if (interior.length < 6 || maxDiffPixels <= 0) return [];
  const cx = Math.round(sumX / interior.length);
  const cy = Math.round(sumY / interior.length);
  const interiorSet = new Set(interior.map((pt) => pt.y * SPRITE_SIZE + pt.x));

  const accentDark = accentStops[0];
  const accentMid = accentStops[Math.min(1, accentStops.length - 1)];
  const accentBright = accentStops[accentStops.length - 1];

  const motifKind = Math.floor(rng() * 3);

  if (motifKind === 0) {
    // Classic central jeweled clasp / emblem (compact 2x2 gem + subtle bottom-right shade)
    const offsets: Array<[number, number, [number, number, number]]> = [
      [0, -1, accentBright],
      [-1, 0, accentBright],
      [0, 0, accentMid],
      [1, 0, accentDark],
      [0, 1, accentDark],
    ];
    for (const [dx, dy, col] of offsets) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (interiorSet.has(ny * SPRITE_SIZE + nx)) {
        edits.push({ x: nx, y: ny, r: col[0], g: col[1], b: col[2], a: 255 });
      }
    }
  } else if (motifKind === 1) {
    // Compact metallic buckle / trim rivets (5px horizontal accent)
    for (let dx = -2; dx <= 2; dx++) {
      const nx = cx + dx;
      const ny = cy;
      if (interiorSet.has(ny * SPRITE_SIZE + nx)) {
        const col = dx <= -1 ? accentBright : dx === 0 ? accentMid : accentDark;
        edits.push({ x: nx, y: ny, r: col[0], g: col[1], b: col[2], a: 255 });
      }
    }
  } else {
    // Twin vertical ornate studs (authentic Tibia armor/chest hardware)
    const studs: Array<[number, number, [number, number, number]]> = [
      [-1, -1, accentBright],
      [-1, 0, accentDark],
      [1, -1, accentMid],
      [1, 0, accentDark],
    ];
    for (const [dx, dy, col] of studs) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (interiorSet.has(ny * SPRITE_SIZE + nx)) {
        edits.push({ x: nx, y: ny, r: col[0], g: col[1], b: col[2], a: 255 });
      }
    }
  }

  // Add crisp 2px NW specular glint if within budget
  const nwCandidates = interior
    .filter((pt) => pt.x <= cx - 1 && pt.y <= cy - 1)
    .sort((a, b) => a.x + a.y - (b.x + b.y));
  for (const pt of nwCandidates.slice(0, 2)) {
    edits.push({
      x: pt.x,
      y: pt.y,
      r: highlightStop[0],
      g: highlightStop[1],
      b: highlightStop[2],
      a: 255,
    });
  }

  return edits.slice(0, maxDiffPixels);
}

/**
 * TIER T2 EXECUTOR — Composite Donor Parts + Palette Unification + Crisp Outline:
 * Splices an exact integer-grid interior/structural region from `secondaryRgba` onto `primaryRgba`
 * (aligned by centroid, zero resampling, zero anti-aliasing), unifies into the target luma-ordered
 * material ramp, and guarantees binary alpha + 1px dark outline integrity.
 */
export function executeDonorComposite(params: {
  primaryRgba: Uint8ClampedArray;
  secondaryRgba: Uint8ClampedArray;
  primaryStops: Array<[number, number, number]>;
  accentStops: Array<[number, number, number]>;
  spliceMode: 'center_core' | 'upper_crest' | 'lower_trim';
  isTile?: boolean;
}): Uint8ClampedArray {
  const { primaryRgba, secondaryRgba, primaryStops, accentStops, spliceMode, isTile = false } = params;

  // Step 1: Recolor primary into target primary ramp and secondary into complementary ramp
  const baseRecolored = executeRampRecolor({
    sourceRgba: primaryRgba,
    primaryStops,
    accentStops,
    isTile,
  }).rgba;

  const secRecolored = executeRampRecolor({
    sourceRgba: secondaryRgba,
    primaryStops: [...primaryStops.slice(0, 2), ...accentStops],
    accentStops: primaryStops.slice(2),
    isTile,
  }).rgba;

  const getCentroid = (buf: Uint8ClampedArray) => {
    let sx = 0;
    let sy = 0;
    let cnt = 0;
    let minY = SPRITE_SIZE;
    let maxY = 0;
    for (let y = 0; y < SPRITE_SIZE; y++) {
      for (let x = 0; x < SPRITE_SIZE; x++) {
        if (buf[pIdx(x, y) + 3] >= 128) {
          sx += x;
          sy += y;
          cnt++;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    return {
      cx: cnt > 0 ? Math.round(sx / cnt) : 16,
      cy: cnt > 0 ? Math.round(sy / cnt) : 16,
      minY,
      maxY,
    };
  };

  const c1 = getCentroid(baseRecolored);
  const c2 = getCentroid(secRecolored);
  const dx = c1.cx - c2.cx;
  const dy = c1.cy - c2.cy;
  const out = new Uint8ClampedArray(baseRecolored);

  for (let y = 1; y < SPRITE_SIZE - 1; y++) {
    for (let x = 1; x < SPRITE_SIZE - 1; x++) {
      // Only splice onto existing interior silhouette of primary so silhouette stays cohesive & non-disconnected
      const dstIdx = pIdx(x, y);
      if (baseRecolored[dstIdx + 3] < 128) continue;

      const isOuterEdge =
        baseRecolored[pIdx(x - 1, y) + 3] < 128 ||
        baseRecolored[pIdx(x + 1, y) + 3] < 128 ||
        baseRecolored[pIdx(x, y - 1) + 3] < 128 ||
        baseRecolored[pIdx(x, y + 1) + 3] < 128;
      if (!isTile && isOuterEdge) continue;

      let inSpliceZone = false;
      if (spliceMode === 'center_core') {
        inSpliceZone = Math.abs(x - c1.cx) <= 4 && Math.abs(y - c1.cy) <= 4;
      } else if (spliceMode === 'upper_crest') {
        inSpliceZone = y <= c1.cy && Math.abs(x - c1.cx) <= 5;
      } else {
        inSpliceZone = y >= c1.cy && Math.abs(x - c1.cx) <= 5;
      }
      if (!inSpliceZone) continue;

      const sx = x - dx;
      const sy = y - dy;
      if (sx < 0 || sx >= SPRITE_SIZE || sy < 0 || sy >= SPRITE_SIZE) continue;
      const srcIdx = pIdx(sx, sy);
      if (secRecolored[srcIdx + 3] < 128) continue;

      // Skip secondary dark outline inside primary interior to avoid noisy black blobs
      if (calcLuma(secRecolored[srcIdx], secRecolored[srcIdx + 1], secRecolored[srcIdx + 2]) <= 0.16) {
        continue;
      }

      out[dstIdx] = secRecolored[srcIdx];
      out[dstIdx + 1] = secRecolored[srcIdx + 1];
      out[dstIdx + 2] = secRecolored[srcIdx + 2];
      out[dstIdx + 3] = 255;
    }
  }

  return isTile ? opSnapBinaryAlpha(out, 0.5) : opEnforceDarkOutline(out, 0.45, false);
}
