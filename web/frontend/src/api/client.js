import axios from "axios";

const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

const api = axios.create({ baseURL: BASE });

/** Upload a video file. Returns { file_id, filename, path } */
export async function uploadVideo(file, onProgress) {
  const form = new FormData();
  form.append("file", file);
  const res = await api.post("/api/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => {
      if (onProgress && e.total)
        onProgress(Math.round((e.loaded / e.total) * 100));
    },
  });
  return res.data;
}

/** Start a pipeline run. Returns { run_id, ws_url } */
export async function startRun({ file_id, model_path, conf, imgsz }) {
  const res = await api.post("/api/start", {
    file_id,
    model_path,
    conf,
    imgsz,
  });
  return res.data;
}

/** Stop a running pipeline. */
export async function stopRun(run_id) {
  await api.post(`/api/stop/${run_id}`);
}

/** Get run metadata + status. */
export async function getResult(run_id) {
  const res = await api.get(`/api/results/${run_id}`);
  return res.data;
}

/** Get run history list. */
export async function getHistory() {
  const res = await api.get("/api/history");
  return res.data;
}

/** Build download URL for result video. */
export function videoDownloadUrl(run_id) {
  return `${BASE}/api/results/${run_id}/video`;
}

/** Build download URL for result CSV. */
export function csvDownloadUrl(run_id) {
  return `${BASE}/api/results/${run_id}/csv`;
}
