import { SPRITE_SIZE } from './tibiaParser';

function hash2d(x: number, y: number, seed: number): number {
  let h = (x * 374761393 + y * 668265263 + seed * 1442695041) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 0xffffffff;
}

let cachedDirtTile: Uint8ClampedArray | null = null;
let cachedStoneTile: Uint8ClampedArray | null = null;

/**
 * Generates a crisp, seamless 32x32 classic RPG dirt ground tile ("Terra")
 * with earthy tones, tiny pebbles, and subtle grass flecks.
 */
export function getDirtBackdropTile(): Uint8ClampedArray {
  if (cachedDirtTile) return cachedDirtTile;
  const data = new Uint8ClampedArray(SPRITE_SIZE * SPRITE_SIZE * 4);

  const dirtPalette: [number, number, number][] = [
    [106, 76, 48],  // base earth
    [118, 86, 55],  // warm light soil
    [94, 66, 41],   // shaded soil
    [84, 58, 36],   // deep furrow
    [126, 95, 64],  // sandy highlight
  ];

  for (let y = 0; y < SPRITE_SIZE; y++) {
    for (let x = 0; x < SPRITE_SIZE; x++) {
      const idx = (y * SPRITE_SIZE + x) * 4;
      const n1 = hash2d(x, y, 17);
      const n2 = hash2d(x >> 1, y >> 1, 41);
      const combined = n1 * 0.65 + n2 * 0.35;

      let col: [number, number, number];
      if (combined < 0.16) col = dirtPalette[3];
      else if (combined < 0.38) col = dirtPalette[2];
      else if (combined < 0.72) col = dirtPalette[0];
      else if (combined < 0.90) col = dirtPalette[1];
      else col = dirtPalette[4];

      // Occasional tiny pebbles or subtle grass blades
      const detail = hash2d(x, y, 99);
      if (detail > 0.965) {
        col = [138, 124, 108]; // small pale pebble
      } else if (detail < 0.025) {
        col = [74, 96, 48]; // muted grass fleck
      } else if (detail < 0.045) {
        col = [68, 46, 28]; // tiny pebble shadow
      }

      data[idx] = col[0];
      data[idx + 1] = col[1];
      data[idx + 2] = col[2];
      data[idx + 3] = 255;
    }
  }

  cachedDirtTile = data;
  return data;
}

/**
 * Generates a crisp, seamless 32x32 classic RPG stone flagstone floor tile ("Piso")
 * with mortar seams, NW bevel highlights, and subtle stone grain.
 */
export function getStoneBackdropTile(): Uint8ClampedArray {
  if (cachedStoneTile) return cachedStoneTile;
  const data = new Uint8ClampedArray(SPRITE_SIZE * SPRITE_SIZE * 4);

  for (let y = 0; y < SPRITE_SIZE; y++) {
    for (let x = 0; x < SPRITE_SIZE; x++) {
      const idx = (y * SPRITE_SIZE + x) * 4;
      // 2x2 brick/flagstone pattern (16x16 blocks with staggered vertical joints)
      const rowBlock = Math.floor(y / 16);
      const localY = y % 16;
      const offsetX = rowBlock === 1 ? 8 : 0;
      const localX = (x + offsetX) % 16;

      const isMortar = localY === 0 || localX === 0;
      const isNwHighlight = !isMortar && (localY === 1 || localX === 1);
      const isSeShadow = !isMortar && (localY === 15 || localX === 15);
      const grain = hash2d(x, y, 73);

      let r = 92;
      let g = 98;
      let b = 108;

      if (isMortar) {
        r = 56;
        g = 60;
        b = 68;
      } else if (isNwHighlight) {
        r = 114;
        g = 121;
        b = 132;
      } else if (isSeShadow) {
        r = 74;
        g = 79;
        b = 88;
      } else {
        const delta = Math.round((grain - 0.5) * 18);
        r += delta;
        g += delta;
        b += delta;
      }

      data[idx] = Math.max(0, Math.min(255, r));
      data[idx + 1] = Math.max(0, Math.min(255, g));
      data[idx + 2] = Math.max(0, Math.min(255, b));
      data[idx + 3] = 255;
    }
  }

  cachedStoneTile = data;
  return data;
}

/**
 * Alpha-composites a 32x32 foreground sprite over a 32x32 ground tile pixel-for-pixel.
 */
export function compositeSpriteOverTile(
  spriteRgba: Uint8ClampedArray,
  tileRgba: Uint8ClampedArray,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(tileRgba);
  for (let i = 0; i < spriteRgba.length; i += 4) {
    const a = spriteRgba[i + 3] / 255;
    if (a <= 0) continue;
    if (a >= 0.995) {
      out[i] = spriteRgba[i];
      out[i + 1] = spriteRgba[i + 1];
      out[i + 2] = spriteRgba[i + 2];
      out[i + 3] = 255;
    } else {
      out[i] = Math.round(spriteRgba[i] * a + out[i] * (1 - a));
      out[i + 1] = Math.round(spriteRgba[i + 1] * a + out[i + 1] * (1 - a));
      out[i + 2] = Math.round(spriteRgba[i + 2] * a + out[i + 2] * (1 - a));
      out[i + 3] = 255;
    }
  }
  return out;
}
