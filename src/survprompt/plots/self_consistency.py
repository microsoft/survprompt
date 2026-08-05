"""
Self-consistency plots for model predictions.

Shows how aggregating predictions across an increasing number of independent
runs (n = 1, 2, 3, ... up to the maximum number of available runs) affects each
headline metric. For a given n we average the predicted survival curves over a
set of run-combinations of that size (so the curve is not sensitive to run
ordering), then score the aggregated prediction with the same machinery used by
the main figures (``compute_all_metrics_with_bootstrap``): C-Index, weighted
Pseudo-observation cMAE, and integrated Brier score. A fourth panel shows the
per-patient prediction variability across the n runs.
"""

import os
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from typing import Dict

from survprompt.configs.exp_config import ExperimentConfig
from survprompt.plots.color_utils import GLOBAL_FONT_SIZE
from survprompt.plots.plot_utils import (
    process_data_for_km, resolve_survprompt_headline_label, save_source_data,
)
from survprompt.plots.plot_all_metrics import compute_all_metrics_with_bootstrap
from survprompt.defaults import DEFAULT_DATA_NAME

# Set global font properties
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams["font.size"] = GLOBAL_FONT_SIZE

# Load environment and set BASE_DIR
BASE_DIR = os.getenv("BASE_DIR")
if BASE_DIR is None:
    load_dotenv()
    BASE_DIR = os.getenv("BASE_DIR")
if BASE_DIR is None:
    raise ValueError("BASE_DIR environment variable not set")

PLOT_DIR = os.path.join(BASE_DIR, "plots", "self_consistency")
os.makedirs(PLOT_DIR, exist_ok=True)

# Cap the number of run-combinations evaluated per n (every combination is used
# when there are few; this bounds cost when many runs are available).
MAX_COMBOS_PER_N = 15
# Survival-curve grid (years). GPT SURV_PROB curves are emitted on 0..10 by 0.5.
CURVE_GRID = np.arange(0.0, 10.5, 0.5)
# Time points the Brier/aggregate metrics are scored on (matches plot_all_metrics).
TIME_POINTS = np.arange(0.5, 10.1, 0.5)

# (metric key in compute_all_metrics_with_bootstrap, panel title, y-label, lower-is-better)
METRICS = [
    ("c_index", "C-Index vs. aggregated runs", "C-Index (↑ better)", False),
    ("cmae", "cMAE vs. aggregated runs", "cMAE (years, ↓ better)", True),
    ("brier_score", "Integrated Brier vs. aggregated runs", "Integrated Brier (↓ better)", True),
]


def _run_prob_matrix(run_time_series: pd.Series):
    """Interpolate each patient's survival curve onto ``CURVE_GRID``. Returns the
    sample-id index and an (N, len(grid)) array (NaN row for uncovered patients)."""
    idx = run_time_series.index
    mat = np.full((len(idx), len(CURVE_GRID)), np.nan)
    for i, v in enumerate(run_time_series.to_numpy()):
        if isinstance(v, (list, tuple)) and len(v) == 2:
            t = np.asarray(v[0], dtype=float)
            p = np.asarray(v[1], dtype=float)
            if t.size and p.size:
                mat[i] = np.interp(CURVE_GRID, t, p, left=1.0, right=p[-1])
    return idx, mat


def _load_runs(datasets, model_pattern):
    """Return (gt_entry, train_entry, sample_idx, prob_array, event_series).

    ``prob_array`` has shape (R, N, len(grid)) for the R matched runs."""
    gt = [x for x in datasets if x[2] == "Ground Truth"][0]
    train = [x for x in datasets if "Training Times/Events" in x[2]][0]
    idx, mats, event_series = None, [], None
    for time_data, event_data, label in datasets:
        if label in ("Ground Truth", "Training Times/Events"):
            continue
        base = label.rsplit(" pred", 1)[0] if " pred" in label else label
        if model_pattern not in base:
            continue
        run_idx, mat = _run_prob_matrix(time_data)
        if idx is None:
            idx = run_idx
            event_series = event_data
        mats.append(mat)
    if not mats:
        return gt, train, None, None, None
    return gt, train, idx, np.stack(mats, axis=0), event_series


def _aggregated_series(idx, agg_probs):
    """Build a SURV_PROB-style Series whose value per patient is [grid, probs]."""
    vals = {}
    grid_list = CURVE_GRID.tolist()
    for i, sid in enumerate(idx):
        row = agg_probs[i]
        vals[sid] = [grid_list, row.tolist()] if not np.isnan(row).all() else np.nan
    return pd.Series(vals)


def _t50_matrix(probs):
    """Per-run, per-patient median survival time t50 (years) from curves
    ``probs`` of shape (R, N, G). t50 is where S(t) crosses 0.5; if the curve
    never reaches 0.5, its area is used (matching the metric code's fallback)."""
    R, N, _ = probs.shape
    t50 = np.full((R, N), np.nan)
    for r in range(R):
        for i in range(N):
            p = probs[r, i]
            if np.isnan(p).all():
                continue
            t50[r, i] = (np.interp(0.5, p[::-1], CURVE_GRID[::-1])
                         if p[-1] <= 0.5 else np.trapz(p, CURVE_GRID))
    return t50


def compute_consistency_curve(datasets, model_pattern):
    """For each n = 1..R, average curves over run-combinations of size n and score
    each aggregate. Returns ({n: {metric: {mean, lo, hi}, 'avg_std': v}}, R)."""
    gt, train, idx, probs, event_series = _load_runs(datasets, model_pattern)
    if probs is None:
        return {}, 0
    n_runs = probs.shape[0]
    t50 = _t50_matrix(probs)  # (R, N), computed once
    metric_keys = [m[0] for m in METRICS]
    curve: Dict[int, Dict] = {}
    for n in range(1, n_runs + 1):
        combos = list(itertools.combinations(range(n_runs), n))[:MAX_COMBOS_PER_N]
        per_metric = {k: [] for k in metric_keys}
        stds = []
        for combo in combos:
            cidx = list(combo)
            with np.errstate(all="ignore"):
                agg = np.nanmean(probs[cidx], axis=0)  # (N, G)
            synth = _aggregated_series(idx, agg)
            ds = [gt, train, (synth, event_series, "SURV_PROB: AGG")]
            m = compute_all_metrics_with_bootstrap(ds, "SURV_PROB: AGG", TIME_POINTS, n_boot=0)
            for k in metric_keys:
                per_metric[k].append(m[k]["value"])
            if n > 1:
                with np.errstate(all="ignore"):
                    stds.append(float(np.nanmean(np.nanstd(t50[cidx], axis=0, ddof=1))))
        entry = {}
        for k in metric_keys:
            arr = np.array([v for v in per_metric[k] if not np.isnan(v)])
            entry[k] = {
                "mean": float(np.mean(arr)) if arr.size else np.nan,
                "lo": float(np.min(arr)) if arr.size else np.nan,
                "hi": float(np.max(arr)) if arr.size else np.nan,
            }
        entry["avg_std"] = float(np.mean(stds)) if stds else 0.0
        curve[n] = entry
    return curve, n_runs


def plot_self_consistency_curves(curves, cancer_types, output_filename="self_consistency_curves.pdf"):
    """Plot each metric (and prediction variability) vs. number of aggregated runs."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    cancer_colors = {ca: palette[i % 10] for i, ca in enumerate(cancer_types)}

    max_n = 1
    panels = list(METRICS) + [("avg_std", "Prediction variability vs. aggregated runs",
                               "Avg. per-patient std dev (years)", True)]
    for ax, (key, title, ylabel, _lower) in zip(axes, panels):
        for ca in cancer_types:
            curve = curves.get(ca)
            if not curve:
                continue
            ns = sorted(curve.keys())
            max_n = max(max_n, max(ns))
            color = cancer_colors[ca]
            if key == "avg_std":
                means = [curve[n]["avg_std"] for n in ns]
                ax.plot(ns, means, marker="o", color=color, label=ca.upper())
            else:
                means = [curve[n][key]["mean"] for n in ns]
                los = [curve[n][key]["lo"] for n in ns]
                his = [curve[n][key]["hi"] for n in ns]
                ax.plot(ns, means, marker="o", color=color, label=ca.upper())
                ax.fill_between(ns, los, his, color=color, alpha=0.15)
        ax.set_title(title, fontsize=GLOBAL_FONT_SIZE)
        ax.set_xlabel("Number of aggregated runs (n)")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes:
        ax.set_xticks(list(range(1, max_n + 1)))
    axes[0].legend(fontsize=9, title="Cancer", ncol=2)

    plt.tight_layout()
    output_path = os.path.join(PLOT_DIR, output_filename)
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    print(f"Saved: {output_path}")
    plt.close(fig)

    # Companion source-data table.
    rows = []
    for ca in cancer_types:
        for n, e in sorted(curves.get(ca, {}).items()):
            row = {"cancer": ca.upper(), "n_runs": n, "avg_pred_std_years": e["avg_std"]}
            for key, *_ in METRICS:
                row[f"{key}_mean"] = e[key]["mean"]
                row[f"{key}_min"] = e[key]["lo"]
                row[f"{key}_max"] = e[key]["hi"]
            rows.append(row)
    save_source_data(output_path, pd.DataFrame(rows))


def main():
    """Generate self-consistency analysis plots."""
    dataset = DEFAULT_DATA_NAME
    cancer_types = ["nsclc", "brca", "crc", "panc", "prostate"]
    prompting_task = "SURV_PROB"
    model_pattern = None

    print("=" * 60)
    print("SELF-CONSISTENCY ANALYSIS")
    print("=" * 60)

    curves = {}
    for ca in cancer_types:
        print(f"\nProcessing {ca}...")
        cfg = ExperimentConfig(base_dir=BASE_DIR, data_name=dataset, cancer_of_interest=ca)
        datasets = process_data_for_km(
            cfg, [prompting_task],
            get_train_times=True,
            system_prompt_path="system",
            load_multiple_runs=True,
        )
        if model_pattern is None:
            model_pattern = resolve_survprompt_headline_label(
                label.rsplit(" pred", 1)[0] if " pred" in label else label
                for _, _, label in datasets
            )
        curve, n_runs = compute_consistency_curve(datasets, model_pattern)
        if n_runs == 0:
            print(f"  {model_pattern}: No runs found")
            continue
        curves[ca] = curve
        c1 = curve[1]["c_index"]["mean"]
        cN = curve[n_runs]["c_index"]["mean"]
        print(f"  {model_pattern}: {n_runs} runs (n=1..{n_runs}); "
              f"C-Index {c1:.3f} -> {cN:.3f}, "
              f"cMAE {curve[1]['cmae']['mean']:.2f} -> {curve[n_runs]['cmae']['mean']:.2f}, "
              f"Brier {curve[1]['brier_score']['mean']:.3f} -> {curve[n_runs]['brier_score']['mean']:.3f}")

    plot_self_consistency_curves(curves, cancer_types)

    print("\n" + "=" * 60)
    print("SELF-CONSISTENCY ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
