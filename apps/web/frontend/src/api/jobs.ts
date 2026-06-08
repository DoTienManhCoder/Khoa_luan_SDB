import { API_BASE, AutoShotJob } from "./client";

export type UploadSettings = {
  sensitivity: "low" | "medium" | "high";
  minSceneDurationSec: number;
  backend: "auto" | "phase2" | "baseline";
  temperature: number | null;
  sigma: number | null;
  threshold: number | null;
};

export async function createJobFromUpload(file: File, settings: UploadSettings): Promise<AutoShotJob> {
  const form = new FormData();
  form.append("file", file);
  form.append("sensitivity", settings.sensitivity);
  form.append("min_scene_duration_sec", String(settings.minSceneDurationSec));
  form.append("backend", settings.backend);
  if (settings.temperature !== null) form.append("temperature", String(settings.temperature));
  if (settings.sigma !== null) form.append("sigma", String(settings.sigma));
  if (settings.threshold !== null) form.append("threshold", String(settings.threshold));

  const response = await fetch(`${API_BASE}/api/jobs/from-upload`, {
    method: "POST",
    body: form
  });

  if (!response.ok) {
    const error = await readError(response);
    throw new Error(error);
  }

  return response.json();
}

export async function getJob(jobId: string): Promise<AutoShotJob> {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}`);
  if (!response.ok) {
    const error = await readError(response);
    throw new Error(error);
  }
  return response.json();
}

export async function deleteJob(jobId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}`, { method: "DELETE" });
  if (!response.ok) {
    const error = await readError(response);
    throw new Error(error);
  }
}

export function exportUrl(jobId: string, kind: "json" | "csv" | "txt"): string {
  return `${API_BASE}/api/jobs/${jobId}/exports/${kind}`;
}

async function readError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return body.detail || response.statusText;
  } catch {
    return response.statusText;
  }
}
