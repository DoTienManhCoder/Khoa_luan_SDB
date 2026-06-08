import { ChevronDown, ChevronRight, RotateCcw } from "lucide-react";
import { useState } from "react";

export type Backend = "auto" | "phase2" | "baseline";

type Props = {
  backend: Backend;
  temperature: number | null;
  sigma: number | null;
  threshold: number | null;
  disabled?: boolean;
  onChange: (next: { backend: Backend; temperature: number | null; sigma: number | null; threshold: number | null }) => void;
};

export default function AdvancedSettings({ backend, temperature, sigma, threshold, disabled, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const showHyper = backend !== "baseline";

  const reset = () => {
    onChange({ backend: "auto", temperature: null, sigma: null, threshold: null });
  };

  return (
    <div className="advanced-settings">
      <button type="button" className="advanced-toggle" onClick={() => setOpen((value) => !value)} disabled={disabled}>
        {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        Advanced settings
      </button>

      {open && (
        <div className="advanced-body">
          <label className="advanced-field">
            <span>Backend</span>
            <select
              value={backend}
              disabled={disabled}
              onChange={(event) => onChange({ backend: event.target.value as Backend, temperature, sigma, threshold })}
            >
              <option value="auto">Auto (server default)</option>
              <option value="phase2">AutoShotV2</option>
              <option value="baseline">OpenCV baseline</option>
            </select>
          </label>

          {showHyper && (
            <>
              <label className="advanced-field">
                <span>Temperature (0.05 – 5.0)</span>
                <input
                  type="number"
                  min="0.05"
                  max="5"
                  step="0.01"
                  placeholder="checkpoint default"
                  value={temperature ?? ""}
                  disabled={disabled}
                  onChange={(event) => onChange({
                    backend,
                    temperature: event.target.value === "" ? null : Number(event.target.value),
                    sigma,
                    threshold,
                  })}
                />
              </label>

              <label className="advanced-field">
                <span>Sigma (0 – 10)</span>
                <input
                  type="number"
                  min="0"
                  max="10"
                  step="0.1"
                  placeholder="checkpoint default"
                  value={sigma ?? ""}
                  disabled={disabled}
                  onChange={(event) => onChange({
                    backend,
                    temperature,
                    sigma: event.target.value === "" ? null : Number(event.target.value),
                    threshold,
                  })}
                />
              </label>

              <label className="advanced-field">
                <span>Threshold (0 – 1)</span>
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  placeholder="checkpoint default"
                  value={threshold ?? ""}
                  disabled={disabled}
                  onChange={(event) => onChange({
                    backend,
                    temperature,
                    sigma,
                    threshold: event.target.value === "" ? null : Number(event.target.value),
                  })}
                />
              </label>
            </>
          )}

          <button type="button" className="ghost-button advanced-reset" onClick={reset} disabled={disabled}>
            <RotateCcw size={14} />
            Reset
          </button>
        </div>
      )}
    </div>
  );
}
