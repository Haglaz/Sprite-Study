import { useEffect, useMemo, useRef } from 'react';
import type { BackdropMode, ZoomLevel } from '../types/studio';
import {
  compositeSpriteOverTile,
  getDirtBackdropTile,
  getStoneBackdropTile,
} from '../utils/backdropTiles';
import { rgbaToPngDataUrl } from '../utils/pixelImage';
import { SPRITE_SIZE } from '../utils/tibiaParser';

let cachedDirtDataUrl: string | null = null;
let cachedStoneDataUrl: string | null = null;

function getTileDataUrl(mode: 'dirt' | 'stone'): string | null {
  if (mode === 'dirt') {
    if (!cachedDirtDataUrl) {
      cachedDirtDataUrl = rgbaToPngDataUrl(getDirtBackdropTile());
    }
    return cachedDirtDataUrl;
  }
  if (!cachedStoneDataUrl) {
    cachedStoneDataUrl = rgbaToPngDataUrl(getStoneBackdropTile());
  }
  return cachedStoneDataUrl;
}

interface PixelSpriteCanvasProps {
  rgbaPixels?: Uint8ClampedArray | null;
  base64Png?: string | null;
  zoom: ZoomLevel;
  backdrop: BackdropMode;
  alt: string;
  testId?: string;
}

export function PixelSpriteCanvas({
  rgbaPixels,
  base64Png,
  zoom,
  backdrop,
  alt,
  testId,
}: PixelSpriteCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const displaySize = SPRITE_SIZE * zoom;

  const groundTileRgba = useMemo(() => {
    if (backdrop === 'dirt') return getDirtBackdropTile();
    if (backdrop === 'stone') return getStoneBackdropTile();
    if (backdrop === 'magenta') {
      const mag = new Uint8ClampedArray(SPRITE_SIZE * SPRITE_SIZE * 4);
      for (let i = 0; i < mag.length; i += 4) {
        mag[i] = 255;
        mag[i + 1] = 0;
        mag[i + 2] = 255;
        mag[i + 3] = 255;
      }
      return mag;
    }
    return null;
  }, [backdrop]);

  const groundTileBgUrl = useMemo(() => {
    if (backdrop === 'dirt' || backdrop === 'stone') {
      return getTileDataUrl(backdrop);
    }
    return null;
  }, [backdrop]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, SPRITE_SIZE, SPRITE_SIZE);

    if (rgbaPixels && rgbaPixels.length === SPRITE_SIZE * SPRITE_SIZE * 4) {
      const pixelsToDraw = groundTileRgba
        ? compositeSpriteOverTile(rgbaPixels, groundTileRgba)
        : rgbaPixels;
      const imageData = new ImageData(new Uint8ClampedArray(pixelsToDraw), SPRITE_SIZE, SPRITE_SIZE);
      ctx.putImageData(imageData, 0, 0);
      return;
    }

    if (groundTileRgba) {
      ctx.putImageData(
        new ImageData(new Uint8ClampedArray(groundTileRgba), SPRITE_SIZE, SPRITE_SIZE),
        0,
        0,
      );
    }

    if (base64Png) {
      let cancelled = false;
      const img = new Image();
      img.onload = () => {
        if (cancelled) return;
        ctx.clearRect(0, 0, SPRITE_SIZE, SPRITE_SIZE);
        if (groundTileRgba) {
          ctx.putImageData(
            new ImageData(new Uint8ClampedArray(groundTileRgba), SPRITE_SIZE, SPRITE_SIZE),
            0,
            0,
          );
        }
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(img, 0, 0, SPRITE_SIZE, SPRITE_SIZE);
      };
      img.src = base64Png.startsWith('data:')
        ? base64Png
        : `data:image/png;base64,${base64Png}`;
      return () => {
        cancelled = true;
      };
    }
  }, [rgbaPixels, base64Png, groundTileRgba]);

  const backdropClass =
    backdrop === 'magenta'
      ? 'bg-[#FF00FF]'
      : backdrop === 'transparent'
        ? 'checkerboard-dark'
        : backdrop === 'dark'
          ? 'bg-[#090B0E]'
          : backdrop === 'light'
            ? 'bg-[#E5E7EB]'
            : 'bg-[#3C2C1E]';

  return (
    <div
      className={`flex items-center justify-center rounded-lg border border-[#262B35] overflow-hidden ${backdropClass} ${
        groundTileBgUrl ? 'pixelated' : ''
      }`}
      style={{
        width: 288,
        height: 288,
        ...(groundTileBgUrl
          ? {
              backgroundImage: `url(${groundTileBgUrl})`,
              backgroundSize: `${displaySize}px ${displaySize}px`,
              backgroundPosition: 'center',
              backgroundRepeat: 'repeat',
            }
          : {}),
      }}
      data-testid={testId}
      data-backdrop={backdrop}
    >
      <canvas
        ref={canvasRef}
        width={SPRITE_SIZE}
        height={SPRITE_SIZE}
        aria-label={alt}
        role="img"
        className="pixelated block shrink-0"
        style={{
          width: displaySize,
          height: displaySize,
        }}
      />
    </div>
  );
}
