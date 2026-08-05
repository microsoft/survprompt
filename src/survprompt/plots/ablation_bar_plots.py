"""Plot feature ablation metrics for Survprompt and RSF outputs."""
from __future__ import annotations

import argparse
import ast
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

from survprompt.configs.exp_config import ExperimentConfig  # noqa: E402
from survprompt.evaluation.metrics import (  # noqa: E402
    _calculate_estimate_for_risk,
    cmae_components,
    concordance_index_censored,
    interpolate_time_at_threshold,
)
from survprompt.experiments.interpretability.feature_sets import ABLATION_FEATURE_LABELS  # noqa: E402
from survprompt.plots.plot_utils import process_data_for_km  # noqa: E402
from survprompt.defaults import DEFAULT_DATA_NAME, DEFAULT_PRED_ENGINE

DEFAULT_DATASET = DEFAULT_DATA_NAME
DEFAULT_CANCER = "nsclc"
DEFAULT_SYSTEM_PROMPT = "system"
N_BOOT = 500
SEED = 0
RSF_COLOR = "#bdbdbd"
SURV_COLOR = "#984ea3"
LABEL_RSF = "RSF"
LABEL_SURV = "Survprompt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ablation cMAE and c-index bars for Survprompt vs RSF.")
    parser.add_argument("--base-dir", default=os.getenv("BASE_DIR"), help="Repository/output base directory. Defaults to BASE_DIR.")
    parser.add_argument("--input-dir", default=os.getenv("INPUT_DIR"), help="Input data directory. Defaults to INPUT_DIR.")
    parser.add_argument("--data-name", default=DEFAULT_DATASET, help="Dataset name.")
    parser.add_argument("--cancer", default=DEFAULT_CANCER, help="Cancer cohort to plot.")
    parser.add_argument("--fold", type=int, default=0, help="Fold to plot.")
    parser.add_argument("--prediction-number", type=int, default=0, help="Survprompt prediction number to plot.")
    parser.add_argument("--model", default=DEFAULT_PRED_ENGINE, help="Model to plot.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt output directory name.")
    parser.add_argument("--surv-dir", default=None, help="Directory containing Survprompt ablation outputs by feature.")
    parser.add_argument("--rsf-dir", default=None, help="Directory containing RSF ablation outputs by feature.")
    parser.add_argument("--out-dir", default=None, help="Directory for output PDFs and source data.")
    args = parser.parse_args()
    if args.base_dir is None:
        raise ValueError("BASE_DIR must be set or passed with --base-dir")
    if args.input_dir is None:
        raise ValueError("INPUT_DIR must be set or passed with --input-dir")
    if args.surv_dir is None:
        args.surv_dir = os.path.join(args.base_dir, "results", "predictions", args.data_name, f"ablation_{args.model}_race")
    if args.rsf_dir is None:
        args.rsf_dir = os.path.join(args.base_dir, "results", "predictions", args.data_name, "RSF_ablation")
    if args.out_dir is None:
        args.out_dir = os.path.join(args.base_dir, "plots", "ablation", args.data_name, args.cancer)
    return args



def _train_times(base_dir: str, input_dir: str, data_name: str, cancer: str, system_prompt: str):
    cfg = ExperimentConfig(
        cancer_of_interest=cancer,
        input_dir=input_dir,
        base_dir=base_dir,
        data_name=data_name,
        features=["demographics", "treatment", "stage", "met", "path", "lab", "genomics"],
        sample_size=0,
    )
    datasets = process_data_for_km(cfg, ["TTE_OS"], system_prompt_path=system_prompt, get_train_times=True)
    train_dataset = [x for x in datasets if x[2] == "Training Times/Events"][0]
    return train_dataset[0].astype(float).values, train_dataset[1].astype(bool).values


def _find_survprompt_file(surv_dir: str, feature_name: str) -> str | None:
    pattern = os.path.join(surv_dir, feature_name, "preds.csv")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def _find_rsf_file(rsf_dir: str, feature_name: str, cancer: str, fold: int) -> str | None:
    path = os.path.join(rsf_dir, feature_name, f"results_{cancer}_fullfts_pred_fold{fold}.csv")
    if os.path.exists(path):
        return path
    return None


def _surv_pred_times_days(df: pd.DataFrame):
    """Per-patient predicted t50 (days) for a SURV_PROB df; drops NaN rows."""
    times_days, keep = [], []
    for idx, row in df.iterrows():
        try:
            times = ast.literal_eval(row["pred_time"]) if isinstance(row["pred_time"], str) else row["pred_time"]
            probs = ast.literal_eval(row["pred_prob"]) if isinstance(row["pred_prob"], str) else row["pred_prob"]
        except Exception:
            continue
        if times is None or probs is None:
            continue
        times = np.asarray(times, dtype=float)
        probs = np.asarray(probs, dtype=float)
        if np.any(np.isnan(times)) or np.any(np.isnan(probs)):
            continue
        crossing_time = interpolate_time_at_threshold(times, probs, 0.5)
        if crossing_time is None:
            crossing_time = float(np.trapz(probs, times))
        times_days.append(crossing_time * 365.25)
        keep.append(idx)
    return df.loc[keep].reset_index(drop=True), np.asarray(times_days)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    if weights.sum() == 0:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def _cmae_point_and_ci(abs_err_days: np.ndarray, weights: np.ndarray):
    """Weighted Pseudo-observation cMAE (years) + 95% bootstrap CI."""
    point = _weighted_mean(abs_err_days, weights) / 365.25
    rng = np.random.default_rng(SEED)
    n_samples = len(abs_err_days)
    bootstrap = []
    for _ in range(N_BOOT):
        indices = rng.integers(0, n_samples, n_samples)
        bootstrap.append(_weighted_mean(abs_err_days[indices], weights[indices]) / 365.25)
    return point, float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))


def _cindex_point_and_ci(df: pd.DataFrame, pred_col: str):
    """C-index + 95% bootstrap CI. Risk is precomputed once (the SURV_PROB risk
    estimate parses pred_time/pred_prob, so doing it per-bootstrap is far too
    slow); the bootstrap then just resamples the (risk, time, event) arrays."""
    if pred_col == "pred_prob":
        risk = np.asarray(_calculate_estimate_for_risk(df, year=None), dtype=float)
    else:
        risk = -df[pred_col].to_numpy(dtype=float)
    time = df["stop_nonlt"].to_numpy(dtype=float)
    event = df["dead_nonlt"].to_numpy().astype(bool)
    valid = ~np.isnan(risk)
    risk, time, event = risk[valid], time[valid], event[valid]

    point = float(concordance_index_censored(event, time, risk)[0])
    rng = np.random.default_rng(SEED)
    n_samples = len(risk)
    bootstrap = []
    for _ in range(N_BOOT):
        indices = rng.integers(0, n_samples, n_samples)
        try:
            bootstrap.append(concordance_index_censored(event[indices], time[indices], risk[indices])[0])
        except Exception:
            pass
    bootstrap = np.asarray(bootstrap, dtype=float)
    bootstrap = bootstrap[~np.isnan(bootstrap)]
    return point, float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))


def collect(args: argparse.Namespace) -> pd.DataFrame:
    train_time, train_event = _train_times(args.base_dir, args.input_dir, args.data_name, args.cancer, args.system_prompt)
    rows = []
    for feature_name, label in ABLATION_FEATURE_LABELS.items():
        surv_path = _find_survprompt_file(args.surv_dir, feature_name)
        rsf_path = _find_rsf_file(args.rsf_dir, feature_name, args.cancer, args.fold)
        if surv_path is None:
            print(f"  [skip] missing Survprompt predictions for {feature_name}")
            continue
        if rsf_path is None:
            print(f"  [skip] missing RSF predictions for {feature_name}")
            continue

        surv_df = pd.read_csv(surv_path)
        reference = surv_df[["sample_ids", "stop_nonlt", "dead_nonlt"]]

        # ---- Survprompt ----
        surv_valid, surv_days = _surv_pred_times_days(surv_df)
        surv_err, surv_weights = cmae_components(
            predicted_times=surv_days,
            event_times=surv_valid["stop_nonlt"].values,
            event_indicators=surv_valid["dead_nonlt"].values,
            train_event_times=train_time,
            train_event_indicators=train_event,
        )
        surv_cmae, surv_cmae_lo, surv_cmae_hi = _cmae_point_and_ci(surv_err, surv_weights)
        surv_ci, surv_ci_lo, surv_ci_hi = _cindex_point_and_ci(
            surv_valid[["pred_time", "pred_prob", "stop_nonlt", "dead_nonlt"]], "pred_prob"
        )

        # ---- RSF baseline (per feature subset) ----
        rsf_df = pd.read_csv(rsf_path).merge(reference, on="sample_ids", how="left")
        rsf_err, rsf_weights = cmae_components(
            predicted_times=rsf_df["pred_num_days"].values,
            event_times=rsf_df["stop_nonlt"].values,
            event_indicators=rsf_df["dead_nonlt"].values,
            train_event_times=train_time,
            train_event_indicators=train_event,
        )
        rsf_cmae, rsf_cmae_lo, rsf_cmae_hi = _cmae_point_and_ci(rsf_err, rsf_weights)
        rsf_ci, rsf_ci_lo, rsf_ci_hi = _cindex_point_and_ci(
            rsf_df[["pred_num_days", "stop_nonlt", "dead_nonlt"]], "pred_num_days"
        )

        print(f"  {label:18}  Survprompt cMAE={surv_cmae:.2f} c-idx={surv_ci:.3f} | RSF cMAE={rsf_cmae:.2f} c-idx={rsf_ci:.3f}")
        rows.append({
            "feature": feature_name,
            "label": label,
            "surv_cmae": surv_cmae,
            "surv_cmae_lo": surv_cmae_lo,
            "surv_cmae_hi": surv_cmae_hi,
            "surv_cindex": surv_ci,
            "surv_cindex_lo": surv_ci_lo,
            "surv_cindex_hi": surv_ci_hi,
            "rsf_cmae": rsf_cmae,
            "rsf_cmae_lo": rsf_cmae_lo,
            "rsf_cmae_hi": rsf_cmae_hi,
            "rsf_cindex": rsf_ci,
            "rsf_cindex_lo": rsf_ci_lo,
            "rsf_cindex_hi": rsf_ci_hi,
        })
    return pd.DataFrame(rows)


def _barh_plot(df: pd.DataFrame, metric: str, xlabel: str, xmax: float, out_path: str, title: str) -> None:
    """Grouped horizontal bars (RSF vs Survprompt) with 95% CI error bars."""
    labels = df["label"].tolist()
    y_positions = np.arange(len(labels))[::-1]
    bar_height = 0.4

    fig, ax = plt.subplots(figsize=(12, 8))
    for offset, color, model, key in [
        (bar_height / 2, RSF_COLOR, LABEL_RSF, "rsf"),
        (-bar_height / 2, SURV_COLOR, LABEL_SURV, "surv"),
    ]:
        values = df[f"{key}_{metric}"].values
        low = df[f"{key}_{metric}_lo"].values
        high = df[f"{key}_{metric}_hi"].values
        xerr = np.vstack([values - low, high - values])
        ax.barh(y_positions + offset, values, height=bar_height, color=color, label=model,
                xerr=xerr, error_kw=dict(ecolor="black", elinewidth=1.1, capsize=3))
        # Place the value label just right of the error bar's upper whisker so it
        # never overlaps the bar or the CI cap.
        for y_pos, value, high_value in zip(y_positions + offset, values, high):
            ax.annotate(f"{value:.2f}", (high_value, y_pos), ha="left", va="center", fontsize=11,
                        xytext=(8, 0), textcoords="offset points", color="black")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend order: RSF then Survprompt
    handles, legend_labels = ax.get_legend_handles_labels()
    order = [legend_labels.index(LABEL_RSF), legend_labels.index(LABEL_SURV)]
    # Outside the axes (upper right) so it never clips the value labels
    ax.legend([handles[i] for i in order], [legend_labels[i] for i in order], title="Model",
              loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    print(f"Saved {out_path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = collect(args)
    if df.empty:
        print("No data collected; run the ablation experiments first.")
        return

    source_dir = os.path.join(args.out_dir, "source_data")
    os.makedirs(source_dir, exist_ok=True)
    df.to_csv(os.path.join(source_dir, "ablation_surv_prob_comparison.csv"), index=False)

    title = f"{args.data_name.upper()} {args.cancer.upper()} (SURV_PROB)"
    _barh_plot(df, "cmae", "cMAE (years)", 8.0,
               os.path.join(args.out_dir, "ablation_surv_prob_comparison_cmae.pdf"), title)
    _barh_plot(df, "cindex", "c-index", 1.0,
               os.path.join(args.out_dir, "ablation_surv_prob_comparison_cindex.pdf"), title)


if __name__ == "__main__":
    main()
