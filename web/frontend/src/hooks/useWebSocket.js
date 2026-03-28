import { useCallback, useEffect, useRef, useState } from "react"; // REFACTOR:

const WS_BASE = import.meta.env.VITE_WS_BASE ?? "ws://localhost:8000"; // REFACTOR:
const META_BUFFER_LIMIT = 300; // # FIX: cap metadata buffer at 300 entries per requirement

export function useWebSocket() {
  // REFACTOR:
  const wsRef = useRef(null); // REFACTOR:
  const frameMetaBuffer = useRef([]); // REFACTOR: rolling metadata buffer for video-time sync
  const [status, setStatus] = useState("idle"); // REFACTOR:
  const [dashboardData, setDashboardData] = useState({
    // REFACTOR:
    tracks: [],
    action_counts: {},
    fall_alert: false,
    fps: 0,
    frame_idx: 0,
  });

  const handleStatus = useCallback((nextStatus) => {
    // REFACTOR:
    setStatus(nextStatus ?? "idle"); // REFACTOR:
    if (
      nextStatus === "done" ||
      nextStatus === "stopped" ||
      nextStatus === "error"
    ) {
      // REFACTOR:
      wsRef.current = null; // REFACTOR:
    }
  }, []);

  const disconnect = useCallback(() => {
    // REFACTOR:
    if (wsRef.current) {
      // REFACTOR:
      wsRef.current.close(); // REFACTOR:
      wsRef.current = null; // REFACTOR:
    }
    frameMetaBuffer.current = []; // REFACTOR:
    setStatus("idle"); // REFACTOR:
  }, []);

  const connect = useCallback(
    (runId) => {
      // REFACTOR:
      if (wsRef.current) wsRef.current.close(); // REFACTOR:

      frameMetaBuffer.current = []; // REFACTOR:
      setDashboardData({
        // REFACTOR:
        tracks: [],
        action_counts: {},
        fall_alert: false,
        fps: 0,
        frame_idx: 0,
      });
      setStatus("running"); // REFACTOR:

      const ws = new WebSocket(`${WS_BASE}/ws/${runId}`); // REFACTOR:
      wsRef.current = ws; // REFACTOR:

      ws.onmessage = (event) => {
        // REFACTOR:
        // REFACTOR: all messages are JSON text now
        try {
          const data = JSON.parse(event.data); // REFACTOR:

          if (data.type === "ping") return; // REFACTOR: ignore keepalive packets

          if (data.status) {
            // REFACTOR: done / stopped / error
            handleStatus(data.status); // REFACTOR:
            return;
          }

          if (data.type && data.type !== "frame") return; // REFACTOR: ignore unknown non-frame payloads

          frameMetaBuffer.current.push(data); // REFACTOR: buffer metadata by frame_idx
          if (frameMetaBuffer.current.length > 1) {
            const n = frameMetaBuffer.current.length; // REFACTOR:
            if (
              (frameMetaBuffer.current[n - 1].frame_idx ?? 0) <
              (frameMetaBuffer.current[n - 2].frame_idx ?? 0)
            ) {
              frameMetaBuffer.current.sort(
                (a, b) => (a.frame_idx ?? 0) - (b.frame_idx ?? 0),
              );
            }
          }
          if (frameMetaBuffer.current.length > META_BUFFER_LIMIT) {
            frameMetaBuffer.current.splice(
              0,
              frameMetaBuffer.current.length - META_BUFFER_LIMIT,
            ); // # FIX: evict oldest entries first when buffer exceeds 300
          }

          if ((data.frame_idx ?? 0) % 6 === 0) {
            // REFACTOR: sparse dashboard updates
            setDashboardData(data); // REFACTOR:
          }
        } catch (e) {
          console.error("WS parse error", e); // REFACTOR:
        }
      };

      ws.onerror = () => {
        // REFACTOR:
        handleStatus("error"); // REFACTOR:
      };

      ws.onclose = () => {
        // REFACTOR:
        if (wsRef.current) handleStatus("error"); // REFACTOR:
      };
    },
    [handleStatus],
  );

  useEffect(() => () => disconnect(), [disconnect]); // REFACTOR:

  return { frameMetaBuffer, dashboardData, status, connect, disconnect }; // REFACTOR:
}
