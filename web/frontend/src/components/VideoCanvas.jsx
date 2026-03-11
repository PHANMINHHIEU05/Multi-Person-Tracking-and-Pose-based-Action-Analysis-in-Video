import { useEffect, useRef } from "react";

/**
 * High-performance video canvas.
 *
 * Instead of re-rendering on every React state change, this component reads
 * the latest ImageBitmap directly from a ref (bitmapRef) and paints it inside
 * a requestAnimationFrame loop.  This keeps frame delivery completely off the
 * React render cycle and avoids GC pressure from base64 strings.
 *
 * Props:
 *   bitmapRef  — React ref whose .current is an ImageBitmap (from useWebSocket)
 *   isRunning  — boolean, controls whether the rAF loop is active
 */
export default function VideoCanvas({ bitmapRef, isRunning }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);
  const lastBitmapRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const drawPlaceholder = () => {
      canvas.width = 854;
      canvas.height = 480;
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#475569";
      ctx.font = "16px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No video — upload and Start", canvas.width / 2, canvas.height / 2);
    };

    if (!isRunning) {
      drawPlaceholder();
      return;
    }

    const loop = () => {
      const bmp = bitmapRef.current;
      if (bmp && bmp !== lastBitmapRef.current) {
        // Resize canvas to match video aspect ratio (once per size change)
        if (canvas.width !== bmp.width || canvas.height !== bmp.height) {
          canvas.width = bmp.width;
          canvas.height = bmp.height;
        }
        ctx.drawImage(bmp, 0, 0);
        lastBitmapRef.current = bmp;
      }
      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [isRunning, bitmapRef]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full rounded-lg border border-slate-700 bg-slate-900"
    />
  );
}
