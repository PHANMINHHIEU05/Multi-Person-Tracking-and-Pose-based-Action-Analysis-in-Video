import { useEffect, useRef } from "react";

/**
 * PERF: Worker-based canvas renderer (no React state updates per frame).
 */
export default function VideoCanvas({ isRunning, setFrameHandler }) {
  const canvasRef = useRef(null);
  const workerRef = useRef(null);
  const ctxRef = useRef(null);
  const runningRef = useRef(false);

  useEffect(() => {
    runningRef.current = isRunning;
  }, [isRunning]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: false }); // PERF: faster compositing
    ctxRef.current = ctx;

    const drawPlaceholder = () => {
      canvas.width = 854;
      canvas.height = 480;
      ctx.fillStyle = "#0f172a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#475569";
      ctx.font = "16px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(
        "No video — upload and Start",
        canvas.width / 2,
        canvas.height / 2,
      );
    };

    drawPlaceholder();

    // PERF: Create worker once; JPEG decode happens off main thread.
    const worker = new Worker("/frameWorker.js");
    workerRef.current = worker;

    worker.onmessage = (evt) => {
      if (evt.data?.type !== "bitmap") return;
      const bitmap = evt.data.bitmap;
      if (!bitmap || !ctxRef.current) return;
      const c = canvasRef.current;
      if (!c) {
        bitmap.close();
        return;
      }
      if (c.width !== bitmap.width || c.height !== bitmap.height) {
        c.width = bitmap.width;
        c.height = bitmap.height;
      }
      ctxRef.current.drawImage(bitmap, 0, 0);
      bitmap.close(); // PERF: release bitmap memory immediately
    };

    // PERF: Expose non-state frame handler to websocket hook.
    if (typeof setFrameHandler === "function") {
      setFrameHandler((blob) => {
        if (!runningRef.current || !workerRef.current) return;
        workerRef.current.postMessage({ type: "frame", blob });
      });
    }

    return () => {
      if (typeof setFrameHandler === "function") setFrameHandler(null);
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, [setFrameHandler]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full rounded-lg border border-slate-700 bg-slate-900"
    />
  );
}
