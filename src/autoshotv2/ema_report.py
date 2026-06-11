"""Regenerate the EMA-vs-no-EMA comparison table from the bundled result JSONs.

This is the JSON-tier reproduce path for `reports/ema_study/`: it reads the six
result JSONs that ship with the bundle (AutoShot / BBC / ClipShots, each in an
EMA `alpha999` and a `noema` variant) and prints a best-threshold F1/P/R table
plus the isolated EMA contribution (D-EMA - D-noEMA).

Usage:
    python -m autoshotv2.ema_report                       # print the table
    python -m autoshotv2.ema_report --out reports/ema_study/ema_vs_noema.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_DIR / "reports" / "ema_study"

# dataset -> (filename stem, container key, {A,B,D method keys})
DATASETS: dict[str, dict[str, Any]] = {
    "AutoShot test (n=200)": {
        "stem": "results_autoshot",
        "container": "models",
        "keys": {"A": "autoshot_phase1_raw", "B": "autoshot_phase1_gaussian", "D": "autoshot_ema"},
    },
    "BBC shot-level (n=11)": {
        "stem": "results_bbc",
        "container": "shot_level",
        "keys": {"A": "A_phase1_raw", "B": "B_phase1_gauss", "D": "D_ema"},
    },
    "ClipShots test (n=500)": {
        "stem": "results_clipshots",
        "container": "models",
        "keys": {"A": "A_phase1_raw", "B": "B_phase1_gauss", "D": "D_ema"},
    },
}


def metric_block(container: dict, key: str):
    """Pull best-threshold metrics from either nested ('best_threshold') or flat ('f1_best') layout."""
    if not container:
        return None
    sub = container.get(key)
    if not sub:
        return None
    if "best_threshold" in sub:
        b = sub["best_threshold"]
        return {"f1": b["f1"], "p": b["precision"], "r": b["recall"], "thr": b["threshold"]}
    if "f1_best" in sub:
        return {"f1": sub["f1_best"], "p": sub["p_best"], "r": sub["r_best"], "thr": sub["thr_best"]}
    return None


def load_container(path: Path, container_key: str) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get(container_key, {})


def fmt(m, field: str) -> str:
    return "—" if m is None else f"{m[field]:.4f}"


def build_report(src_dir: Path) -> str:
    lines = [
        "# EMA vs no-EMA — best-threshold results (regenerated from JSON)",
        "",
        "A = Phase 1 raw · B = Phase 1 + Gaussian · D-EMA = full-model fine-tune + EMA (α=0.999) · "
        "D-noEMA = same fine-tune, EMA off (control). Numbers are JSON-tier (read from "
        "`reports/ema_study/results_*.json`).",
        "",
        "| Dataset | Method | F1 | Precision | Recall |",
        "|---|---|---:|---:|---:|",
    ]
    deltas = []
    for name, spec in DATASETS.items():
        ema = load_container(src_dir / f"{spec['stem']}_alpha999.json", spec["container"])
        noema = load_container(src_dir / f"{spec['stem']}_noema.json", spec["container"])
        a = metric_block(ema, spec["keys"]["A"])
        b = metric_block(ema, spec["keys"]["B"])
        d_ema = metric_block(ema, spec["keys"]["D"])
        d_noema = metric_block(noema, spec["keys"]["D"])
        lines.append(f"| **{name}** | A — Phase 1 raw | {fmt(a, 'f1')} | {fmt(a, 'p')} | {fmt(a, 'r')} |")
        lines.append(f"| | B — Phase 1 + Gauss | {fmt(b, 'f1')} | {fmt(b, 'p')} | {fmt(b, 'r')} |")
        lines.append(f"| | D-EMA (α=0.999) | {fmt(d_ema, 'f1')} | {fmt(d_ema, 'p')} | {fmt(d_ema, 'r')} |")
        lines.append(f"| | D-noEMA (control) | {fmt(d_noema, 'f1')} | {fmt(d_noema, 'p')} | {fmt(d_noema, 'r')} |")
        if d_ema and d_noema:
            deltas.append((name, d_ema["f1"], d_noema["f1"], d_ema["f1"] - d_noema["f1"]))

    lines += ["", "## Isolated EMA contribution (D-EMA − D-noEMA)", "",
              "| Dataset | D-EMA F1 | D-noEMA F1 | ΔF1 |", "|---|---:|---:|---:|"]
    for name, fe, fn, delta in deltas:
        lines.append(f"| {name} | {fe:.4f} | {fn:.4f} | {delta:+.4f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src-dir", default=str(DEFAULT_DIR), help="Folder holding the result JSONs.")
    parser.add_argument("--out", default=None, help="Optional path to write the markdown table.")
    args = parser.parse_args()

    # The table contains non-ASCII (α, Δ, −); avoid UnicodeEncodeError on cp1252 consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        pass  # best-effort: stdout may be wrapped/closed (pytest capture, pipes)

    report = build_report(Path(args.src_dir))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"wrote -> {out_path}")
    print(report)


if __name__ == "__main__":
    main()
