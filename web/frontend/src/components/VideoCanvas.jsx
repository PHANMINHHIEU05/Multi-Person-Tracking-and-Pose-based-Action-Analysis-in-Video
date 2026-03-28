import { useEffect, useRef } from "react"; // REFACTOR:

export default function VideoCanvas({
  videoUrl,
  frameMetaBuffer,
  videoFps = 30,
}) {
  // REFACTOR:
  const videoRef = useRef(null); // REFACTOR:
  const canvasRef = useRef(null); // REFACTOR:
  const rafRef = useRef(null); // REFACTOR:
  const lastGoodMetaRef = useRef(null); // # FIX: hold last valid metadata to avoid flicker on brief websocket gaps

  useEffect(() => {
    // REFACTOR:
    const video = videoRef.current; // REFACTOR:
    const canvas = canvasRef.current; // REFACTOR:
    if (!video || !canvas) return; // REFACTOR:

    const ctx = canvas.getContext("2d", { alpha: true }); // REFACTOR:
    if (!ctx) return; // REFACTOR:

    function syncCanvasSize() {
      // REFACTOR:
      canvas.width = video.videoWidth || 854; // REFACTOR:
      canvas.height = video.videoHeight || 480; // REFACTOR:
    }

    video.addEventListener("loadedmetadata", syncCanvasSize); // REFACTOR:
    syncCanvasSize(); // REFACTOR:

    function drawOverlay() {
      // REFACTOR:
      rafRef.current = requestAnimationFrame(drawOverlay); // REFACTOR:

      if (!video.videoWidth) return; // REFACTOR:

      const currentTime = video.currentTime || 0; // # FIX: synchronize overlay using video playback time
      const MAX_SYNC_DRIFT_SEC = 0.2; // # FIX: strict sync threshold for selecting a "fresh" metadata frame
      const HOLD_LAST_META_SEC = 0.5; // # FIX: grace window to keep last good bbox instead of clearing instantly
      const buf = frameMetaBuffer.current; // # FIX:
      if (!buf.length) {
        const fallback = lastGoodMetaRef.current; // # FIX:
        const fallbackTs = Number(
          fallback?.timestamp ?? fallback?.sourceTimeSec ?? -999,
        ); // # FIX:
        if (
          fallback &&
          Math.abs(fallbackTs - currentTime) <= HOLD_LAST_META_SEC
        ) {
          // # FIX: keep drawing recent stable bbox when metadata queue momentarily stalls
        } else {
          ctx.clearRect(0, 0, canvas.width, canvas.height); // # FIX:
          return; // # FIX:
        }
      }

      const nearestMeta = buf.length
        ? buf.reduce((prev, cur) => {
            // # FIX: pick closest metadata snapshot by timestamp
            const curTs = Number(cur.timestamp ?? cur.sourceTimeSec ?? 0); // # FIX:
            const prevTs = Number(prev.timestamp ?? prev.sourceTimeSec ?? 0); // # FIX:
            return Math.abs(curTs - currentTime) <
              Math.abs(prevTs - currentTime) // # FIX:
              ? cur // # FIX:
              : prev; // # FIX:
          })
        : null; // # FIX:

      let meta = nearestMeta; // # FIX:
      let metaTs = Number(meta?.timestamp ?? meta?.sourceTimeSec ?? -999); // # FIX:
      if (!meta || Math.abs(metaTs - currentTime) > MAX_SYNC_DRIFT_SEC) {
        const fallback = lastGoodMetaRef.current; // # FIX:
        const fallbackTs = Number(
          fallback?.timestamp ?? fallback?.sourceTimeSec ?? -999,
        ); // # FIX:
        if (
          fallback &&
          Math.abs(fallbackTs - currentTime) <= HOLD_LAST_META_SEC
        ) {
          meta = fallback; // # FIX: reuse last good overlay snapshot during brief mismatch spikes
          metaTs = fallbackTs; // # FIX:
        }
      }

      if (!meta || Math.abs(metaTs - currentTime) > HOLD_LAST_META_SEC) {
        // # FIX: clear only when both fresh and fallback metadata are too stale
        ctx.clearRect(0, 0, canvas.width, canvas.height); // # FIX:
        return; // # FIX:
      }

      lastGoodMetaRef.current = meta; // # FIX: update sticky fallback state with latest drawable metadata

      ctx.clearRect(0, 0, canvas.width, canvas.height); // REFACTOR:

      const scaleX = canvas.width / (meta.inferenceWidth || canvas.width); // REFACTOR:
      const scaleY = canvas.height / (meta.inferenceHeight || canvas.height); // REFACTOR:

      meta.tracks?.forEach(({ id, bbox, action, conf }) => {
        // REFACTOR:
        if (!bbox || bbox.length < 4) return; // REFACTOR:
        const [x1, y1, x2, y2] = bbox; // REFACTOR:
        const sx1 = x1 * scaleX; // REFACTOR:
        const sy1 = y1 * scaleY; // REFACTOR:
        const sw = (x2 - x1) * scaleX; // REFACTOR:
        const sh = (y2 - y1) * scaleY; // REFACTOR:

        const isFall = String(action || "")
          .toLowerCase()
          .includes("fall"); // REFACTOR:
        const color = isFall ? "#E24B4A" : "#1D9E75"; // REFACTOR:

        ctx.strokeStyle = color; // REFACTOR:
        ctx.lineWidth = 2; // REFACTOR:
        ctx.strokeRect(sx1, sy1, sw, sh); // REFACTOR:

        const label = `ID:${id} ${action} ${Math.round((conf ?? 0) * 100)}%`; // REFACTOR:
        ctx.font = "bold 13px sans-serif"; // # FIX: match requested label style
        const tw = ctx.measureText(label).width; // REFACTOR:
        ctx.fillStyle = color; // REFACTOR:
        ctx.fillRect(sx1, sy1 - 22, tw + 10, 22); // # FIX: larger label background for readability

        ctx.fillStyle = "#ffffff"; // REFACTOR:
        ctx.fillText(label, sx1 + 5, sy1 - 6); // # FIX: align text baseline with updated label box
      });
    }

    rafRef.current = requestAnimationFrame(drawOverlay); // REFACTOR:

    return () => {
      // REFACTOR:
      if (rafRef.current) cancelAnimationFrame(rafRef.current); // REFACTOR:
      video.removeEventListener("loadedmetadata", syncCanvasSize); // REFACTOR:
    };
  }, [videoUrl, frameMetaBuffer, videoFps]); // REFACTOR:

  return (
    // REFACTOR:
    <div style={{ position: "relative", width: "100%" }}>
      {" "}
      {/* REFACTOR: */}
      <video // REFACTOR:
        ref={videoRef} // REFACTOR:
        src={videoUrl || ""} // REFACTOR:
        controls // REFACTOR:
        autoPlay={false} // REFACTOR:
        style={{ width: "100%", display: "block" }} // REFACTOR:
      />
      <canvas // REFACTOR:
        ref={canvasRef} // REFACTOR:
        style={{
          // REFACTOR:
          position: "absolute", // REFACTOR:
          top: 0, // REFACTOR:
          left: 0, // REFACTOR:
          width: "100%", // REFACTOR:
          height: "100%", // REFACTOR:
          pointerEvents: "none", // REFACTOR:
        }}
      />
    </div>
  );
}
