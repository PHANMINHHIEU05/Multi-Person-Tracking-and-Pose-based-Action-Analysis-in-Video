import { useCallback, useEffect, useRef, useState } from "react";

const WS_BASE = import.meta.env.VITE_WS_BASE ?? "ws://localhost:8000";

/**
 * PERF: WebSocket hook optimized for low-overhead realtime playback.
 */
export function useWebSocket(handleFrame) {
  const wsRef = useRef(null); // PERF: socket ref avoids rerenders
  const metaRef = useRef({}); // PERF: metadata cache without state churn
  const [status, setStatus] = useState("idle");
  const [dashboardData, setDashboardData] = useState({
    tracks: [],
    counts: {},
    fallAlert: false,
    fps: 0,
    frameIdx: 0,
  });

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    metaRef.current = {};
    setStatus("idle");
  }, []);

  const connect = useCallback(
    (run_id) => {
      if (wsRef.current) wsRef.current.close();

      metaRef.current = {};
      setDashboardData({
        tracks: [],
        counts: {},
        fallAlert: false,
        fps: 0,
        frameIdx: 0,
      });
      setStatus("running");

      const ws = new WebSocket(`${WS_BASE}/ws/${run_id}`);
      ws.binaryType = "blob";
      wsRef.current = ws;

      ws.onmessage = (evt) => {
        // PERF: Pass binary frame directly to VideoCanvas worker handler.
        if (evt.data instanceof Blob) {
          if (typeof handleFrame === "function") handleFrame(evt.data);
          return;
        }

        let msg;
        try {
          msg = JSON.parse(evt.data);
        } catch {
          return;
        }

        if (msg.type === "frame") {
          // PERF: Keep latest metadata in ref; do not trigger rerender every frame.
          metaRef.current = msg;
          // PERF: Update dashboard state only every 6th frame.
          if ((msg.frame_idx ?? 0) % 6 === 0) {
            setDashboardData({
              tracks: msg.tracks ?? [],
              counts: msg.action_counts ?? {},
              fallAlert: msg.fall_alert ?? false,
              fps: msg.fps ?? 0,
              frameIdx: msg.frame_idx ?? 0,
            });
          }
        } else if (msg.type === "status") {
          setStatus(msg.status);
          wsRef.current = null;
        }
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
    },
    [handleFrame],
  );

  useEffect(() => () => disconnect(), [disconnect]);

  return {
    tracks: dashboardData.tracks,
    counts: dashboardData.counts,
    fallAlert: dashboardData.fallAlert,
    status,
    fps: dashboardData.fps,
    frameIdx: dashboardData.frameIdx,
    metaRef,
    connect,
    disconnect,
  };
}
