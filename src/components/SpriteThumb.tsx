import { useEffect, useRef } from 'react';
import { SPRITE_SIZE } from '../utils/tibiaParser';

export type ThumbBackdrop = 'transparent' | 'magenta' | 'dark';

interface SpriteThumbProps {
  rgba: Uint8ClampedArray;
  scale?: number;
  backdrop?: ThumbBackdrop;
  alt: string;
  className?: string;
}

export function SpriteThumb({
  rgba,
  scale = 3,
  backdrop = 'transparent',
  alt,
  className = '',
}: SpriteThumbProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const displayPx = SPRITE_SIZE * scale;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, SPRITE_SIZE, SPRITE_SIZE);
    if (rgba.length === SPRITE_SIZE * SPRITE_SIZE * 4) {
      ctx.putImageData(new ImageData(new Uint8ClampedArray(rgba), SPRITE_SIZE, SPRITE_SIZE), 0, 0);
    }
  }, [rgba]);

  const bgClass =
    backdrop === 'magenta'
      ? 'bg-[#FF00FF]'
      : backdrop === 'dark'
        ? 'bg-[#090B0E]'
        : 'checkerboard-dark';

  return (
    <div
      className={`inline-flex items-center justify-center rounded border border-[#262B35] overflow-hidden shrink-0 ${bgClass} ${className}`}
      style={{ width: displayPx, height: displayPx }}
    >
      <canvas
        ref={canvasRef}
        width={SPRITE_SIZE}
        height={SPRITE_SIZE}
        role="img"
        aria-label={alt}
        className="pixelated block"
        style={{ width: displayPx, height: displayPx }}
      />
    </div>
  );
}
