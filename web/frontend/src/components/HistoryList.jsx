import { useEffect, useState } from "react";
import { getHistory, videoDownloadUrl, csvDownloadUrl } from "../api/client";

const STATUS_COLOR = {
  done: "text-green-400",
  running: "text-blue-400",
  stopped: "text-amber-400",
  error: "text-red-400",
};

/** Run history list with download links. */
export default function HistoryList() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const data = await getHistory();
      setHistory(data);
    } catch {
      // silently ignore
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-300">History</h3>
        <button
          onClick={load}
          className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
        >
          {loading ? "Loading…" : "Refresh"}
        </button>
      </div>

      {history.length === 0 && !loading && (
        <p className="text-slate-500 text-xs">No past runs</p>
      )}

      <div className="flex flex-col gap-1 max-h-52 overflow-y-auto">
        {history.map((run) => (
          <div
            key={run.id}
            className="flex items-center gap-2 bg-slate-800/60 rounded-lg px-3 py-2 text-xs"
          >
            <div className="flex-1 min-w-0">
              <p className="text-slate-200 truncate font-medium">
                {run.video_name}
              </p>
              <p className="text-slate-500">
                {run.started_at?.slice(0, 19).replace("T", " ")}
              </p>
            </div>
            <span
              className={`font-mono shrink-0 ${STATUS_COLOR[run.status] ?? "text-slate-400"}`}
            >
              {run.status}
            </span>
            {run.status === "done" && (
              <div className="flex gap-1 shrink-0">
                <a
                  href={videoDownloadUrl(run.id)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-1.5 py-0.5 rounded bg-slate-700 hover:bg-blue-700 text-slate-300 transition-colors"
                  title="Download video"
                >
                  MP4
                </a>
                <a
                  href={csvDownloadUrl(run.id)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-1.5 py-0.5 rounded bg-slate-700 hover:bg-green-700 text-slate-300 transition-colors"
                  title="Download CSV"
                >
                  CSV
                </a>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
