import { SlidersHorizontal, Sparkles } from "lucide-react";
import { FormEvent, useState } from "react";

import UploadDropzone from "../components/UploadDropzone";
import ErrorState from "../components/ErrorState";
import AdvancedSettings, { Backend } from "../components/AdvancedSettings";
import { createJobFromUpload } from "../api/jobs";

type Props = {
  onJobCreated: (jobId: string) => void;
};

export default function UploadPage({ onJobCreated }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [sensitivity, setSensitivity] = useState<"low" | "medium" | "high">("medium");
  const [minSceneDurationSec, setMinSceneDurationSec] = useState(0.5);
  const [backend, setBackend] = useState<Backend>("auto");
  const [temperature, setTemperature] = useState<number | null>(null);
  const [sigma, setSigma] = useState<number | null>(null);
  const [threshold, setThreshold] = useState<number | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!file) {
      setError("Choose a video file first.");
      return;
    }

    setError(null);
    setIsSubmitting(true);
    try {
      const job = await createJobFromUpload(file, {
        sensitivity,
        minSceneDurationSec,
        backend,
        temperature,
        sigma,
        threshold,
      });
      onJobCreated(job.id);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Upload failed.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="page upload-page">
      <section className="workspace">
        <form className="upload-panel" onSubmit={handleSubmit}>
          <div className="section-title">
            <span>New analysis</span>
            <strong>Upload video</strong>
          </div>

          <UploadDropzone file={file} disabled={isSubmitting} onFileChange={setFile} />

          <div className="settings-row">
            <label>
              <span>
                <SlidersHorizontal size={16} />
                Sensitivity
              </span>
              <select
                value={sensitivity}
                disabled={isSubmitting}
                onChange={(event) => setSensitivity(event.target.value as "low" | "medium" | "high")}
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </label>
            <label>
              <span>Minimum scene</span>
              <input
                type="number"
                min="0.1"
                max="10"
                step="0.1"
                value={minSceneDurationSec}
                disabled={isSubmitting}
                onChange={(event) => setMinSceneDurationSec(Number(event.target.value))}
              />
            </label>
          </div>

          <AdvancedSettings
            backend={backend}
            temperature={temperature}
            sigma={sigma}
            threshold={threshold}
            disabled={isSubmitting}
            onChange={(next) => {
              setBackend(next.backend);
              setTemperature(next.temperature);
              setSigma(next.sigma);
              setThreshold(next.threshold);
            }}
          />

          {error && <ErrorState message={error} />}

          <button className="primary-action" type="submit" disabled={isSubmitting || !file}>
            <Sparkles size={18} />
            {isSubmitting ? "Uploading" : "Analyze"}
          </button>
        </form>

        <aside className="side-panel">
          <div>
            <span className="eyebrow">MVP</span>
            <h1>Scene boundary detection for uploaded videos</h1>
            <p>
              FastAPI analyzes each upload, stores temporary job results in MongoDB, and publishes generated scene
              assets through Cloudinary when credentials are configured.
            </p>
          </div>
          <div className="limit-list">
            <span>MP4, MOV, WEBM, MKV</span>
            <span>Temporary result links</span>
            <span>JSON, CSV, TXT exports</span>
          </div>
        </aside>
      </section>
    </main>
  );
}
