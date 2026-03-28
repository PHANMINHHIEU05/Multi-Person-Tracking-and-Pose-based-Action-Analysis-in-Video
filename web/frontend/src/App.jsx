import { useCallback, useState } from "react"; // REFACTOR: useCallback to stabilize callbacks
import { useWebSocket } from "./hooks/useWebSocket";
import VideoCanvas from "./components/VideoCanvas";
import ControlPanel from "./components/ControlPanel";
import ActionChart from "./components/ActionChart";
import TrackTable from "./components/TrackTable";
import FallAlert from "./components/FallAlert";
import HistoryList from "./components/HistoryList";

function App() {
  const { frameMetaBuffer, dashboardData, status, connect, disconnect } =
    useWebSocket(); // REFACTOR:
  const [runId, setRunId] = useState(null); // REFACTOR:
  const [videoUrl, setVideoUrl] = useState(""); // REFACTOR:

  const tracks = dashboardData.tracks ?? []; // REFACTOR:
  const counts = dashboardData.action_counts ?? {}; // REFACTOR:
  const fallAlert = dashboardData.fall_alert ?? false; // REFACTOR:
  const fps = dashboardData.fps ?? 0; // REFACTOR:
  const frameIdx = dashboardData.frame_idx ?? 0; // REFACTOR:

  const handleStart = useCallback(
    (id) => {
      // REFACTOR: memoize callback to prevent ControlPanel re-creation on every App render
      setRunId(id); // REFACTOR:
      connect(id); // REFACTOR:
    },
    [connect],
  ); // REFACTOR:

  const handleStop = useCallback(() => {
    // REFACTOR: memoize callback
    disconnect(); // REFACTOR:
  }, [disconnect]); // REFACTOR:

  const handleVideoReady = useCallback((fileId) => {
    // REFACTOR: memoize callback to prevent ControlPanel dependency churn
    if (!fileId) return; // REFACTOR:
    setVideoUrl(`/api/video/${fileId}`); // REFACTOR:
  }, []); // REFACTOR: no deps, only sets state

  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col">
      {/* Header */}
      <header className="flex items-center gap-3 px-6 py-3 bg-slate-800/80 border-b border-slate-700 backdrop-blur">
        <div className="w-7 h-7 rounded-md bg-blue-600 flex items-center justify-center text-xs font-bold">
          PT
        </div>
        <h1 className="text-base font-semibold tracking-tight">
          Multi-Person Tracking &amp; Action Analysis
        </h1>
        {status === "running" && (
          <span className="ml-auto text-xs font-mono text-green-400 bg-green-900/40 px-2 py-0.5 rounded">
            {fps.toFixed(1)} fps &middot; frame {frameIdx}
          </span>
        )}
      </header>

      {/* Main 3-column layout */}
      <main className="flex flex-1 gap-0 overflow-hidden">
        {/* Left — Control Panel */}
        <aside className="w-72 shrink-0 overflow-y-auto p-4 bg-slate-800/40 border-r border-slate-700/60">
          <ControlPanel
            status={status}
            onStart={handleStart}
            onStop={handleStop}
            onVideoReady={handleVideoReady}
          />
        </aside>

        {/* Center — Video */}
        <section className="flex-1 flex flex-col gap-3 p-4 min-w-0">
          <FallAlert active={fallAlert} />
          <VideoCanvas
            videoUrl={videoUrl}
            frameMetaBuffer={frameMetaBuffer}
            videoFps={30}
          />{" "}
          {/* REFACTOR: browser-native playback + canvas overlay */}
        </section>

        {/* Right — Analytics */}
        <aside className="w-72 shrink-0 overflow-y-auto p-4 bg-slate-800/40 border-l border-slate-700/60 flex flex-col gap-5">
          <ActionChart counts={counts} />
          <hr className="border-slate-700" />
          <TrackTable tracks={tracks} />
          <hr className="border-slate-700" />
          <HistoryList key={status === "done" ? runId : undefined} />
        </aside>
      </main>
    </div>
  );
}

export default App;
