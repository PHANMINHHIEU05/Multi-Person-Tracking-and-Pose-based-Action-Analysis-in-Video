import { useCallback, useRef, useState } from "react";
import { uploadVideo, startRun, stopRun } from "../api/client";

/**
 * Left-panel: drag-drop upload → configure → start/stop.
 * Props:
 *   status: "idle" | "running" | "done" | "stopped" | "error"
 *   onStart(run_id): called when pipeline starts
 *   onStop(): called when stop clicked
 */
export default function ControlPanel({
  status,
  onStart,
  onStop,
  onVideoReady,
}) {
  // REFACTOR:
  const [file, setFile] = useState(null);
  const [fileId, setFileId] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [conf, setConf] = useState(0.4);
  const [imgsz, setImgsz] = useState(480);
  const [modelPath, setModelPath] = useState(
    "runs/train_horizontal/final_safe_system.pth",
  );
  const [errorMsg, setErrorMsg] = useState("");
  const [runId, setRunId] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [autoStart, setAutoStart] = useState(false); // FIX: auto-start toggle
  const dropRef = useRef(null);

  const handleFile = useCallback(
    async (f) => {
      if (!f) return;
      setFile(f);
      setFileId(null);
      setUploadProgress(0);
      setErrorMsg("");
      setUploading(true);
      try {
        const result = await uploadVideo(f, setUploadProgress);
        setFileId(result.file_id);
        onVideoReady?.(result.file_id); // REFACTOR: provide browser-playable source URL seed to parent
        // FIX: auto-start if enabled
        if (autoStart) {
          setTimeout(async () => {
            try {
              const runResult = await startRun({
                file_id: result.file_id,
                model_path: modelPath,
                conf,
                imgsz,
              });
              setRunId(runResult.run_id);
              onStart?.(runResult.run_id);
            } catch (e) {
              setErrorMsg(
                "Auto-start failed: " +
                  (e?.response?.data?.detail ?? e.message),
              );
            }
          }, 500);
        }
      } catch (e) {
        setErrorMsg(
          "Upload failed: " + (e?.response?.data?.detail ?? e.message),
        );
      } finally {
        setUploading(false);
      }
    },
    [onVideoReady, autoStart, modelPath, conf, imgsz, onStart],
  ); // REFACTOR:

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      const f = e.dataTransfer.files?.[0];
      if (f) handleFile(f);
    },
    [handleFile],
  );

  const handleInputChange = useCallback(
    (e) => {
      const f = e.target.files?.[0];
      if (f) handleFile(f);
    },
    [handleFile],
  );

  const handleStart = async () => {
    if (!fileId) return;
    setErrorMsg("");
    try {
      const result = await startRun({
        file_id: fileId,
        model_path: modelPath,
        conf,
        imgsz,
      });
      setRunId(result.run_id);
      onStart?.(result.run_id);
    } catch (e) {
      setErrorMsg("Start failed: " + (e?.response?.data?.detail ?? e.message));
    }
  };

  const handleStop = async () => {
    if (runId) {
      try {
        await stopRun(runId);
      } catch {}
    }
    onStop?.();
  };

  const isRunning = status === "running";

  return (
    <div className="flex flex-col gap-4 h-full text-sm text-slate-200">
      <h2 className="text-lg font-semibold text-white">Control Panel</h2>

      {/* Drop zone */}
      <div
        ref={dropRef}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => document.getElementById("file-input").click()}
        className="flex flex-col items-center justify-center gap-2 border-2 border-dashed border-slate-600 rounded-lg p-6 cursor-pointer hover:border-blue-500 transition-colors"
      >
        <svg
          className="w-8 h-8 text-slate-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
          />
        </svg>
        {file ? (
          <span className="text-blue-400 font-medium truncate max-w-full px-2">
            {file.name}
          </span>
        ) : (
          <span className="text-slate-400">Drag & drop video or click</span>
        )}
        <input
          id="file-input"
          type="file"
          accept="video/*"
          className="hidden"
          onChange={handleInputChange}
        />
      </div>

      {/* Upload progress */}
      {uploading && (
        <div className="w-full bg-slate-700 rounded-full h-2">
          <div
            className="bg-blue-500 h-2 rounded-full transition-all"
            style={{ width: `${uploadProgress}%` }}
          />
        </div>
      )}
      {fileId && !uploading && (
        <p className="text-green-400 text-xs">✓ Uploaded</p>
      )}

      {/* Config */}
      <div className="flex flex-col gap-3 bg-slate-800 rounded-lg p-3">
        <label className="flex flex-col gap-1">
          <span className="text-slate-400">Model path</span>
          <input
            type="text"
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
            className="bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none focus:border-blue-500"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-slate-400">
            Confidence threshold:{" "}
            <span className="text-white font-mono">{conf.toFixed(2)}</span>
          </span>
          <input
            type="range"
            min="0.1"
            max="0.95"
            step="0.01"
            value={conf}
            onChange={(e) => setConf(parseFloat(e.target.value))}
            className="accent-blue-500"
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-slate-400">Image size</span>
          <select
            value={imgsz}
            onChange={(e) => setImgsz(parseInt(e.target.value))}
            className="bg-slate-900 border border-slate-600 rounded px-2 py-1 text-xs text-slate-200 focus:outline-none focus:border-blue-500"
          >
            {[320, 480, 640, 960, 1280].map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-2 cursor-pointer">
          {" "}
          {/* FIX: auto-start checkbox */}
          <input
            type="checkbox"
            checked={autoStart}
            onChange={(e) => setAutoStart(e.target.checked)}
            className="w-4 h-4 accent-blue-500 cursor-pointer"
          />
          <span className="text-slate-400 select-none">
            Auto-start after upload
          </span>
        </label>
      </div>

      {/* Error */}
      {errorMsg && (
        <p className="text-red-400 text-xs bg-red-900/30 rounded p-2 border border-red-800">
          {errorMsg}
        </p>
      )}

      {/* Start / Stop */}
      {!isRunning ? (
        <button
          onClick={handleStart}
          disabled={!fileId || uploading}
          className="mt-auto w-full py-2.5 rounded-lg font-semibold bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-500 disabled:cursor-not-allowed transition-colors"
        >
          ▶ Start
        </button>
      ) : (
        <button
          onClick={handleStop}
          className="mt-auto w-full py-2.5 rounded-lg font-semibold bg-red-600 hover:bg-red-500 transition-colors"
        >
          ■ Stop
        </button>
      )}

      {/* Status badge */}
      {status !== "idle" && (
        <div
          className={`text-center text-xs py-1 rounded font-mono ${
            status === "running"
              ? "text-green-400 bg-green-900/30"
              : status === "done"
                ? "text-blue-400 bg-blue-900/30"
                : status === "error"
                  ? "text-red-400 bg-red-900/30"
                  : "text-slate-400 bg-slate-800"
          }`}
        >
          {status.toUpperCase()}
        </div>
      )}
    </div>
  );
}
