import { SPRITE_SIZE } from './tibiaParser';

/**
 * Converts a 32x32 RGBA pixel array to a PNG data URL in browser environments.
 * Returns null in non-DOM environments (e.g. Node unit tests).
 */
export function rgbaToPngDataUrl(
  rgba: Uint8ClampedArray,
  width = SPRITE_SIZE,
  height = SPRITE_SIZE,
): string | null {
  if (typeof document === 'undefined') return null;
  if (rgba.length !== width * height * 4) return null;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.imageSmoothingEnabled = false;
  ctx.putImageData(new ImageData(new Uint8ClampedArray(rgba), width, height), 0, 0);
  return canvas.toDataURL('image/png');
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function rgbToHex(r: number, g: number, b: number): string {
  return (
    '#' +
    [r, g, b]
      .map((v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0'))
      .join('')
      .toUpperCase()
  );
}

/**
 * Computes a lowercase hex SHA-256 digest of an ArrayBuffer using Web Crypto
 * (supported natively in modern browsers and Node >= 19).
 */
export async function sha256Hex(buffer: ArrayBuffer): Promise<string> {
  const cryptoApi = globalThis.crypto?.subtle;
  if (cryptoApi) {
    const hashBuf = await cryptoApi.digest('SHA-256', buffer);
    const bytes = new Uint8Array(hashBuf);
    let hex = '';
    for (let i = 0; i < bytes.length; i++) {
      hex += bytes[i].toString(16).padStart(2, '0');
    }
    return hex;
  }
  // Fallback FNV-1a 64-bit style deterministic digest if SubtleCrypto is unavailable
  const bytes = new Uint8Array(buffer);
  let h1 = 0x811c9dc5;
  let h2 = 0x9e3779b9;
  for (let i = 0; i < bytes.length; i++) {
    h1 = Math.imul(h1 ^ bytes[i], 0x01000193) >>> 0;
    h2 = Math.imul(h2 ^ bytes[i], 0x85ebca6b) >>> 0;
  }
  return `${h1.toString(16).padStart(8, '0')}${h2.toString(16).padStart(8, '0')}`;
}
