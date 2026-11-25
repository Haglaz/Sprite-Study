import { createSeededRng } from '../db/split';
import type { ItemCategory } from '../types/studio';
import { SPRITE_SIZE } from '../utils/tibiaParser';

function createBlankRgba(): Uint8ClampedArray {
  return new Uint8ClampedArray(SPRITE_SIZE * SPRITE_SIZE * 4);
}

function setPixel(
  rgba: Uint8ClampedArray,
  x: number,
  y: number,
  r: number,
  g: number,
  b: number,
  a = 255,
): void {
  if (x < 0 || x >= SPRITE_SIZE || y < 0 || y >= SPRITE_SIZE) return;
  const i = (y * SPRITE_SIZE + x) * 4;
  rgba[i] = Math.max(0, Math.min(255, Math.round(r)));
  rgba[i + 1] = Math.max(0, Math.min(255, Math.round(g)));
  rgba[i + 2] = Math.max(0, Math.min(255, Math.round(b)));
  rgba[i + 3] = Math.max(0, Math.min(255, Math.round(a)));
}

function getPixel(rgba: Uint8ClampedArray, x: number, y: number): [number, number, number, number] {
  const i = (y * SPRITE_SIZE + x) * 4;
  return [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]];
}

function addCrispOutline(rgba: Uint8ClampedArray, outlineRgb: [number, number, number]): void {
  const mask = new Uint8Array(SPRITE_SIZE * SPRITE_SIZE);
  for (let p = 0; p < SPRITE_SIZE * SPRITE_SIZE; p++) {
    if (rgba[p * 4 + 3] > 0) mask[p] = 1;
  }
  for (let y = 1; y < SPRITE_SIZE - 1; y++) {
    for (let x = 1; x < SPRITE_SIZE - 1; x++) {
      const p = y * SPRITE_SIZE + x;
      if (mask[p]) continue;
      if (mask[p - 1] || mask[p + 1] || mask[p - SPRITE_SIZE] || mask[p + SPRITE_SIZE]) {
        setPixel(rgba, x, y, outlineRgb[0], outlineRgb[1], outlineRgb[2], 255);
      }
    }
  }
}

// ============================================================================
// 1. HAND-BUILT DEFECT / JUNK FIXTURES (for honest floor & validator tests)
// ============================================================================

export function makeRandomNoiseSprite(seed = 101): Uint8ClampedArray {
  const rng = createSeededRng(seed);
  const rgba = createBlankRgba();
  for (let y = 4; y < 28; y++) {
    for (let x = 4; x < 28; x++) {
      setPixel(
        rgba,
        x,
        y,
        Math.floor(rng() * 256),
        Math.floor(rng() * 256),
        Math.floor(rng() * 256),
        255,
      );
    }
  }
  return rgba;
}

export function makeFlatRectangleSprite(): Uint8ClampedArray {
  const rgba = createBlankRgba();
  for (let y = 6; y < 26; y++) {
    for (let x = 6; x < 26; x++) {
      setPixel(rgba, x, y, 128, 128, 128, 255);
    }
  }
  return rgba;
}

export function makeSmoothGradientSprite(): Uint8ClampedArray {
  const rgba = createBlankRgba();
  for (let y = 4; y < 28; y++) {
    for (let x = 4; x < 28; x++) {
      const t = ((x - 4) + (y - 4)) / 46;
      const r = Math.round(40 + t * 195);
      const g = Math.round(60 + t * 165);
      const b = Math.round(90 + t * 145);
      setPixel(rgba, x, y, r, g, b, 255);
    }
  }
  return rgba;
}

export function makeBlurredBlobSprite(): Uint8ClampedArray {
  const rgba = createBlankRgba();
  for (let y = 3; y < 29; y++) {
    for (let x = 3; x < 29; x++) {
      const dist = Math.hypot(x - 15.5, y - 15.5);
      if (dist <= 11.5) {
        const t = Math.max(0, 1 - dist / 11.5);
        const alpha = dist > 9.0 ? Math.round(((11.5 - dist) / 2.5) * 230) : 255;
        setPixel(
          rgba,
          x,
          y,
          Math.round(50 + t * 180),
          Math.round(55 + t * 175),
          Math.round(65 + t * 160),
          alpha,
        );
      }
    }
  }
  return rgba;
}

export function makeSemiTransparentSprite(): Uint8ClampedArray {
  const rgba = makeCleanPixelArtFixture('Equipamentos', 1);
  // Corrupt several outline pixels with anti-aliased semi-transparency
  setPixel(rgba, 10, 8, 40, 45, 60, 128);
  setPixel(rgba, 11, 8, 40, 45, 60, 180);
  setPixel(rgba, 12, 8, 40, 45, 60, 64);
  return rgba;
}

export function makeOrphanSpeckledSprite(): Uint8ClampedArray {
  const rgba = makeCleanPixelArtFixture('Equipamentos', 2);
  // Add detached 1px orphan pixels outside silhouette
  setPixel(rgba, 3, 3, 210, 180, 90, 255);
  setPixel(rgba, 28, 4, 210, 180, 90, 255);
  setPixel(rgba, 4, 27, 190, 150, 70, 255);
  setPixel(rgba, 27, 27, 190, 150, 70, 255);
  return rgba;
}

export function makeDisconnectedSprite(): Uint8ClampedArray {
  const rgba = createBlankRgba();
  // Two separate outlined blobs far apart
  for (let y = 5; y < 12; y++) {
    for (let x = 5; x < 12; x++) {
      const isEdge = x === 5 || x === 11 || y === 5 || y === 11;
      if (isEdge) setPixel(rgba, x, y, 22, 24, 34, 255);
      else setPixel(rgba, x, y, 170, 150, 110, 255);
    }
  }
  for (let y = 20; y < 27; y++) {
    for (let x = 20; x < 27; x++) {
      const isEdge = x === 20 || x === 26 || y === 20 || y === 26;
      if (isEdge) setPixel(rgba, x, y, 22, 24, 34, 255);
      else setPixel(rgba, x, y, 110, 140, 175, 255);
    }
  }
  return rgba;
}

export function makeBorderTouchingOffCenterSprite(): Uint8ClampedArray {
  const rgba = createBlankRgba();
  // Shoved into top-left corner touching x=0 and y=0
  for (let y = 0; y < 11; y++) {
    for (let x = 0; x < 11; x++) {
      const isEdge = x === 0 || x === 10 || y === 0 || y === 10;
      const l = x + y < 8 ? 205 : x + y < 12 ? 140 : 85;
      if (isEdge) setPixel(rgba, x, y, 24, 22, 30, 255);
      else setPixel(rgba, x, y, l, l - 10, l - 20, 255);
    }
  }
  return rgba;
}

export function makeUnoutlinedSprite(): Uint8ClampedArray {
  const rgba = createBlankRgba();
  // Bright pastel interior with zero dark outline
  for (let y = 7; y < 25; y++) {
    for (let x = 8; x < 24; x++) {
      const t = x + y < 28 ? 0 : x + y < 34 ? 1 : 2;
      const cols: Array<[number, number, number]> = [
        [240, 220, 175],
        [210, 180, 125],
        [175, 140, 95],
      ];
      setPixel(rgba, x, y, cols[t][0], cols[t][1], cols[t][2], 255);
    }
  }
  return rgba;
}

export function makeNearDuplicatePaletteSprite(): Uint8ClampedArray {
  const rgba = makeCleanPixelArtFixture('Containers', 3);
  // Perturb interior pixels by +-2 RGB channels creating near-duplicate colors
  let k = 0;
  for (let y = 8; y < 24; y++) {
    for (let x = 8; x < 24; x++) {
      const [r, g, b, a] = getPixel(rgba, x, y);
      if (a === 255 && r + g + b > 100) {
        const delta = (k % 5) - 2;
        setPixel(rgba, x, y, r + delta, g + (delta === 0 ? 1 : delta), b - delta, 255);
        k++;
      }
    }
  }
  return rgba;
}

export function makeMixelUpscaledSprite(): Uint8ClampedArray {
  // Build a 16x16 sprite and nearest-upscale 2x2 into 32x32
  const rgba = createBlankRgba();
  const palette: Array<[number, number, number]> = [
    [24, 22, 34],
    [82, 92, 124],
    [138, 152, 184],
    [206, 218, 236],
    [194, 146, 52],
  ];
  for (let gy = 2; gy < 14; gy++) {
    for (let gx = 3; gx < 13; gx++) {
      const isEdge = gx === 3 || gx === 12 || gy === 2 || gy === 13;
      let col = palette[2];
      if (isEdge) col = palette[0];
      else if (gx + gy < 12) col = palette[3];
      else if (gx === 7 || gx === 8) col = palette[4];
      else if (gx + gy > 18) col = palette[1];

      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          setPixel(rgba, gx * 2 + dx, gy * 2 + dy, col[0], col[1], col[2], 255);
        }
      }
    }
  }
  return rgba;
}

export function makePillowShadedSprite(): Uint8ClampedArray {
  const rgba = createBlankRgba();
  const ramp: Array<[number, number, number]> = [
    [20, 20, 26], // edge outline
    [60, 62, 74], // ring 1 (dark all around including NW)
    [115, 118, 130], // ring 2
    [175, 178, 190], // ring 3
    [235, 238, 245], // center hotspot
  ];
  for (let y = 5; y < 27; y++) {
    for (let x = 5; x < 27; x++) {
      const dist = Math.hypot(x - 15.5, y - 15.5);
      if (dist <= 10.5) {
        let idx = 4;
        if (dist >= 9.3) idx = 0;
        else if (dist >= 7.2) idx = 1;
        else if (dist >= 5.0) idx = 2;
        else if (dist >= 2.8) idx = 3;
        setPixel(rgba, x, y, ramp[idx][0], ramp[idx][1], ramp[idx][2], 255);
      }
    }
  }
  return rgba;
}

export interface NamedJunkFixture {
  id: string;
  label: string;
  category: ItemCategory;
  rgba: Uint8ClampedArray;
}

export function buildStandardJunkFloorFixtures(seed = 772): NamedJunkFixture[] {
  return [
    { id: 'junk-noise-1', label: 'Ruído RGB Aleatório #1', category: 'Equipamentos', rgba: makeRandomNoiseSprite(seed + 1) },
    { id: 'junk-noise-2', label: 'Ruído RGB Aleatório #2', category: 'Consumíveis', rgba: makeRandomNoiseSprite(seed + 2) },
    { id: 'junk-flat', label: 'Retângulo Chapado (1 cor)', category: 'Containers', rgba: makeFlatRectangleSprite() },
    { id: 'junk-gradient', label: 'Gradiente Suave 2D', category: 'Decoração', rgba: makeSmoothGradientSprite() },
    { id: 'junk-blur', label: 'Blob Desfocado (Alpha Suave)', category: 'Equipamentos', rgba: makeBlurredBlobSprite() },
    { id: 'junk-unoutlined', label: 'Silhueta Sem Contorno', category: 'Consumíveis', rgba: makeUnoutlinedSprite() },
    { id: 'junk-disconnected', label: 'Silhueta Desconectada', category: 'Containers', rgba: makeDisconnectedSprite() },
    { id: 'junk-border', label: 'Colado na Borda / Fora de Centro', category: 'Equipamentos', rgba: makeBorderTouchingOffCenterSprite() },
    { id: 'junk-mixel', label: 'Mixel / Reamostrado 2×2', category: 'Equipamentos', rgba: makeMixelUpscaledSprite() },
    { id: 'junk-pillow', label: 'Pillow Shading Radial', category: 'Decoração', rgba: makePillowShadedSprite() },
    { id: 'junk-neardup', label: 'Paleta Suja (Quase-Duplicadas)', category: 'Containers', rgba: makeNearDuplicatePaletteSprite() },
    { id: 'junk-orphans', label: 'Pixels Órfãos Soltos', category: 'Equipamentos', rgba: makeOrphanSpeckledSprite() },
  ];
}

// ============================================================================
// 2. SYNTHETIC CORRUPTOR FOR REAL SPRITES (training negatives for discriminator)
// ============================================================================

export type CorruptionKind =
  | 'noise'
  | 'blur_requantize'
  | 'gradient_overlay'
  | 'shuffled_pixels'
  | 'mixel_2x'
  | 'pillow_blob';

export function corruptSpriteForNegativeSample(
  sourceRgba: Uint8ClampedArray,
  kind: CorruptionKind,
  seed: number,
): Uint8ClampedArray {
  const rng = createSeededRng(seed);
  const out = new Uint8ClampedArray(sourceRgba);

  if (kind === 'noise') {
    for (let i = 0; i < out.length; i += 4) {
      if (out[i + 3] > 0) {
        out[i] = Math.floor(rng() * 256);
        out[i + 1] = Math.floor(rng() * 256);
        out[i + 2] = Math.floor(rng() * 256);
      }
    }
    return out;
  }

  if (kind === 'blur_requantize') {
    const blurred = createBlankRgba();
    for (let y = 1; y < SPRITE_SIZE - 1; y++) {
      for (let x = 1; x < SPRITE_SIZE - 1; x++) {
        let sr = 0;
        let sg = 0;
        let sb = 0;
        let sa = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const p = ((y + dy) * SPRITE_SIZE + (x + dx)) * 4;
            sr += sourceRgba[p];
            sg += sourceRgba[p + 1];
            sb += sourceRgba[p + 2];
            sa += sourceRgba[p + 3];
          }
        }
        const aAvg = Math.round(sa / 9);
        if (aAvg > 35) {
          setPixel(blurred, x, y, sr / 9, sg / 9, sb / 9, aAvg > 210 ? 255 : aAvg);
        }
      }
    }
    return blurred;
  }

  if (kind === 'gradient_overlay') {
    for (let y = 0; y < SPRITE_SIZE; y++) {
      for (let x = 0; x < SPRITE_SIZE; x++) {
        const i = (y * SPRITE_SIZE + x) * 4;
        if (out[i + 3] > 0) {
          const t = (x + y) / 62;
          out[i] = Math.min(255, Math.round(out[i] * 0.35 + (40 + t * 200) * 0.65));
          out[i + 1] = Math.min(255, Math.round(out[i + 1] * 0.35 + (50 + t * 180) * 0.65));
          out[i + 2] = Math.min(255, Math.round(out[i + 2] * 0.35 + (70 + t * 170) * 0.65));
        }
      }
    }
    return out;
  }

  if (kind === 'shuffled_pixels') {
    const opaqueIndices: number[] = [];
    for (let p = 0; p < SPRITE_SIZE * SPRITE_SIZE; p++) {
      if (out[p * 4 + 3] > 0) opaqueIndices.push(p);
    }
    const colors = opaqueIndices.map((p) => [out[p * 4], out[p * 4 + 1], out[p * 4 + 2]] as const);
    for (let i = colors.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = colors[i];
      colors[i] = colors[j];
      colors[j] = tmp;
    }
    for (let k = 0; k < opaqueIndices.length; k++) {
      const p = opaqueIndices[k];
      out[p * 4] = colors[k][0];
      out[p * 4 + 1] = colors[k][1];
      out[p * 4 + 2] = colors[k][2];
    }
    return out;
  }

  if (kind === 'mixel_2x') {
    const down = createBlankRgba();
    for (let y = 0; y < SPRITE_SIZE; y += 2) {
      for (let x = 0; x < SPRITE_SIZE; x += 2) {
        const [r, g, b, a] = getPixel(sourceRgba, x, y);
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            setPixel(down, x + dx, y + dy, r, g, b, a);
          }
        }
      }
    }
    return down;
  }

  return makePillowShadedSprite();
}

// ============================================================================
// 3. CLEAN CLASSIC PIXEL-ART FIXTURES (for unit testing held-out real > 70)
// ============================================================================

const CLASSIC_RAMPS: Array<Array<[number, number, number]>> = [
  // Steel blue-gray ramp (cool shadows -> bright neutral highlights)
  [
    [22, 26, 40],
    [54, 64, 92],
    [96, 112, 146],
    [154, 170, 198],
    [216, 228, 242],
  ],
  // Classic gold-bronze ramp (warm reddish-brown shadow -> yellow highlight)
  [
    [34, 22, 16],
    [92, 56, 26],
    [156, 108, 38],
    [214, 172, 62],
    [248, 232, 138],
  ],
  // Emerald / leather green ramp (deep teal shadow -> warm lime highlight)
  [
    [18, 34, 30],
    [34, 78, 62],
    [58, 126, 84],
    [108, 176, 104],
    [176, 224, 142],
  ],
  // Crimson ruby / potion ramp (deep plum shadow -> bright coral highlight)
  [
    [38, 16, 28],
    [98, 28, 46],
    [164, 48, 56],
    [218, 92, 84],
    [248, 174, 154],
  ],
  // Oak wood ramp (deep umber shadow -> warm ochre highlight)
  [
    [32, 20, 16],
    [78, 48, 30],
    [126, 82, 46],
    [172, 124, 74],
    [212, 174, 118],
  ],
];

/**
 * Generates clean, hand-crafted 32x32 pixel-art sprites with hue-shifted dark outlines,
 * stepped 5-color material ramps, NW directional lighting, clean clusters, and proper margins.
 * Used strictly in automated unit tests to test calibration & scoring without requiring network I/O.
 */
export function makeCleanPixelArtFixture(category: ItemCategory, variantIndex: number): Uint8ClampedArray {
  const rgba = createBlankRgba();
  const ramp = CLASSIC_RAMPS[variantIndex % CLASSIC_RAMPS.length];
  const accent = CLASSIC_RAMPS[(variantIndex + 1) % CLASSIC_RAMPS.length];
  const offset = (variantIndex % 3) - 1;

  if (category === 'Equipamentos') {
    // Classic compact armor / shield silhouette centered around (15,15) with crisp NW lighting
    const halfW = 6 + (variantIndex % 3);
    const halfH = 7 + (variantIndex % 3);
    for (let y = 15 - halfH + offset; y <= 15 + halfH + offset; y++) {
      for (let x = 15 - halfW; x <= 15 + halfW; x++) {
        const dx = x - 15;
        const dy = y - (15 + offset);
        // Taper bottom slightly like a classic armor/shield
        const maxW = dy > 2 ? halfW - Math.floor((dy - 2) * 0.45) : halfW;
        if (Math.abs(dx) > maxW) continue;
        // Cut neck notch on top
        if (dy === -halfH && Math.abs(dx) <= 1) continue;

        const nw = -dx * 0.7 - dy * 0.8;
        let idx = 2;
        if (nw >= 4.5) idx = 4;
        else if (nw >= 1.5) idx = 3;
        else if (nw <= -4.0) idx = 1;

        // Trim accent band
        const useAccent = dy === -2 || (Math.abs(dx) === 0 && dy >= -1 && dy <= 2);
        const col = useAccent ? accent[Math.min(4, idx + 1)] : ramp[idx];
        setPixel(rgba, x, y, col[0], col[1], col[2], 255);
      }
    }
    addCrispOutline(rgba, ramp[0]);
    return rgba;
  }

  if (category === 'Consumíveis') {
    // Classic compact flask / potion bottle with neck, cork, liquid, and NW specular pixel cluster
    for (let y = 8 + offset; y <= 23 + offset; y++) {
      const dy = y - (16 + offset);
      const halfW = dy < -4 ? 2 : dy < -2 ? 3 : dy <= 5 ? 5 + (variantIndex % 2) : 4;
      for (let x = 15 - halfW; x <= 15 + halfW; x++) {
        const dx = x - 15;
        let col = ramp[2];
        if (dy < -4) {
          col = dx <= 0 ? accent[3] : accent[1]; // cork stopper
        } else if (dy < -2) {
          col = dx <= 0 ? CLASSIC_RAMPS[0][3] : CLASSIC_RAMPS[0][1]; // glass neck
        } else {
          const nw = -dx - dy * 0.6;
          if (dx <= -2 && dy <= 0) col = ramp[4]; // crisp NW glass highlight
          else if (nw >= 2) col = ramp[3];
          else if (nw <= -3) col = ramp[1];
          else col = ramp[2];
        }
        setPixel(rgba, x, y, col[0], col[1], col[2], 255);
      }
    }
    addCrispOutline(rgba, ramp[0]);
    return rgba;
  }

  if (category === 'Containers') {
    // Classic 3/4 oblique chest/box showing bright top lid, medium front face, dark right side
    const left = 7 + (variantIndex % 2);
    const right = 23 + (variantIndex % 2);
    const top = 9 + offset;
    const bottom = 23 + offset;
    for (let y = top; y <= bottom; y++) {
      for (let x = left; x <= right; x++) {
        let col = ramp[2];
        if (y <= top + 4) {
          // Bright top lid face
          col = x < left + 8 ? ramp[4] : ramp[3];
        } else if (x >= right - 4) {
          // Dark SE side plane
          col = ramp[1];
        } else {
          // Medium NW front face with metal clasp
          const isClasp = x >= left + 5 && x <= left + 7 && y >= top + 5 && y <= top + 8;
          const isSeam = y === top + 5;
          col = isSeam ? ramp[0] : isClasp ? accent[3] : x < left + 6 ? ramp[3] : ramp[2];
        }
        setPixel(rgba, x, y, col[0], col[1], col[2], 255);
      }
    }
    addCrispOutline(rgba, ramp[0]);
    return rgba;
  }

  // Decoração: classic chalice / crown / lantern with open elliptical rim and base
  for (let y = 7 + offset; y <= 24 + offset; y++) {
    const dy = y - (15 + offset);
    let halfW = 5;
    if (dy <= -4) halfW = 6;
    else if (dy <= 1) halfW = 5 - Math.floor((dy + 4) * 0.5);
    else if (dy <= 5) halfW = 2;
    else halfW = 5;

    for (let x = 15 - halfW; x <= 15 + halfW; x++) {
      const dx = x - 15;
      const nw = -dx * 0.9 - dy * 0.5;
      let col = ramp[2];
      if (dy <= -5 && Math.abs(dx) <= 3) col = ramp[1]; // inner hollow rim
      else if (nw >= 3.5) col = ramp[4];
      else if (nw >= 1) col = ramp[3];
      else if (nw <= -3) col = ramp[1];
      if (dy === 1 && Math.abs(dx) <= 1) col = accent[3];
      setPixel(rgba, x, y, col[0], col[1], col[2], 255);
    }
  }
  addCrispOutline(rgba, ramp[0]);
  return rgba;
}

/**
 * Generates a full 32x32 ground tile fixture (100% coverage) to test tile rule exemption.
 */
export function makeCleanGroundTileFixture(seed = 55): Uint8ClampedArray {
  const rng = createSeededRng(seed);
  const rgba = createBlankRgba();
  const ramp = CLASSIC_RAMPS[4]; // earth/stone tones
  for (let y = 0; y < SPRITE_SIZE; y++) {
    for (let x = 0; x < SPRITE_SIZE; x++) {
      // Structured 4x4 pebble/cobble clusters with NW lighting
      const bx = Math.floor(x / 4);
      const by = Math.floor(y / 4);
      const lx = x % 4;
      const ly = y % 4;
      let idx = 2;
      if (lx === 0 || ly === 0) idx = 3;
      else if (lx === 3 && ly === 3) idx = 1;
      else if (((bx * 7 + by * 13) % 5) === 0 && rng() > 0.5) idx = 3;
      else if (((bx * 5 + by * 11) % 4) === 0) idx = 1;
      const c = ramp[idx];
      setPixel(rgba, x, y, c[0], c[1], c[2], 255);
    }
  }
  return rgba;
}
