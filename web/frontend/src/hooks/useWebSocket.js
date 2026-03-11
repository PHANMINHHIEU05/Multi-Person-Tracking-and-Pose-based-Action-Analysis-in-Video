import { useCallback, useEffect, useRef, useState } from "react";

const WS_BASE = import.meta.env.VITE_WS_BASE ?? "ws://localhost:8000";

/**
 * Connect to /ws/{run_id} and stream video frames.
 *
 * Protocol (binary WS):
 *   Server sends two messages per frame:
 *     1. Binary  → raw JPEG bytes
 *     2. Text    → JSON metadata {type:"frame", frame_idx, fps, tracks, action_counts, fall_alert}
 *   Status messages are text-only JSON {type:"status"|"ping"}.
 *
 * Returns:
 *   frameBitmap: ImageBitmap | null  (ready for canvas drawImage — zero-copy GPU)
 *   tracks, counts, fallAlert, status, fps, frameIdx, connect, disconnect
 */
export function useWebSocket() {
  const wsRef = useRef(null);
  // Use a ref for the latest bitmap so React doesn't re-render per frame.
  // The VideoCanvas reads it via requestAnimationFrame.
  const bitmapRef = useRef(null);
  const [tracks, setTracks] = useState([]);
  const [counts, setCounts] = useState({});
  const [fallAlert, setFallAlert] = useState(false);
  const [status, setStatus] = useState("idle");
  const [fps, setFps] = useState(0);
  const [frameIdx, setFrameIdx] = useState(0);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus("idle");
  }, []);

  const connect = useCallback((run_id) => {
    if (wsRef.current) wsRef.current.close();

    bitmapRef.current = null;
    setTracks([]);
    setCounts({});
    setFallAlert(false);
    setFrameIdx(0);
    setFps(0);
    setStatus("running");

    const ws = new WebSocket(`${WS_BASE}/ws/${run_id}`);
    ws.binaryType = "blob";
    wsRef.current = ws;

    // We expect: binary (JPEG) → text (meta) pairs.
    // Keep the latest Blob so we can decode it when the meta arrives.
    let pendingBlob = null;

    ws.onmessage = async (evt) => {
      // Binary message = JPEG image
      if (evt.data instanceof Blob) {
        pendingBlob = evt.data;
        return;
      }

      // Text message
      let msg;
      try {
        msg = JSON.parse(evt.data);
      } catch {
        return;
      }

      if (msg.type === "frame" && pendingBlob) {
        // Decode JPEG → ImageBitmap (hardware-decoded, very fast)
        try {
          const bitmap = await createImageBitmap(pendingBlob);
          // Release previous bitmap
          if (bitmapRef.current) bitmapRef.current.close();
          bitmapRef.current = bitmap;
        } catch {
          /* ignore decode errors */
        }
        pendingBlob = null;

        setTracks(msg.tracks ?? []);
        setCounts(msg.action_counts ?? {});
        setFallAlert(msg.fall_alert ?? false);
        setFps(msg.fps ?? 0);
        setFrameIdx(msg.frame_idx ?? 0);
      } else if (msg.type === "status") {
        setStatus(msg.status);
        wsRef.current = null;
      }
      // "ping" — ignore
    };

    ws.onerror = () => {
      setStatus("error");
      wsRef.current = null;
    };

    ws.onclose = () => {
      if (wsRef.current) {
        setStatus((prev) => (prev === "running" ? "error" : prev));
        wsRef.current = null;
      }
    };
  }, []);

  useEffect(() => () => disconnect(), [disconnect]);

  return {
    bitmapRef,
    tracks,
    counts,
    fallAlert,
    status,
    fps,
    frameIdx,
    connect,
    disconnect,
  };
}
