"""RMST-based error figures for SURV_PROB and TTE_OS models.

Three figures, styled to match ``all_metrics_*`` (Set1 model colors, bootstrap
CIs, ``mini`` hatching):

1. ``all_metrics_all_models_RMST_SURV_PROB.pdf`` -- a 4-row bar chart, one row per
   horizon tau in {1, 3, 5, 10} yr. Each row mirrors the existing all-models
   metric panels: x-axis = cancer, grouped bars = model, value = RMST-MAE(tau).

2. ``all_metrics_all_models_RMST_TTE_OS.pdf`` -- the same all-models RMST-MAE
    view for scalar TTE_OS predictions.

3. ``rmst_survprompt_vs_rsf_timecourse.pdf`` -- a 1x5 line figure (one panel per
   cancer) with tau on the x-axis and two series (RSF baseline vs Survprompt),
   showing how the RMST error grows with the horizon.

RMST error = | predicted RMST(tau) - ground-truth RMST(tau) |, averaged over
patients, where the ground truth uses KM imputation for patients censored before
tau (see ``survprompt.evaluation.metrics``).
"""
import os
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from sksurv.metrics import brier_score as sksurv_brier_score

from survprompt.configs.exp_config import ExperimentConfig
from survprompt.evaluation.metrics import (
    DEFAULT_TAUS,
    _curve_for_entry,
    compute_km_curve,
    compute_one_calibration_stats,
    compute_rmst_mae_stats,
    load_rsf_survival_curves,
)
from survprompt.plots.color_utils import (
    GLOBAL_FONT_SIZE, get_line_style, canonical_model_sort_key)
from survprompt.plots.plot_all_metrics import plot_metric_barplot
from survprompt.plots.plot_utils import (
    is_all_models_zero_shot_label,
    format_model_display_label, get_model_color, parse_label,
    process_data_for_km, resolve_survprompt_headline_label, save_source_data,
)
import pandas as pd
from survprompt.defaults import DEFAULT_DATA_NAME

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams["font.size"] = GLOBAL_FONT_SIZE

BASE_DIR = os.getenv("BASE_DIR")
if BASE_DIR is None:
    load_dotenv()
    BASE_DIR = os.getenv("BASE_DIR")

BOOTSTRAP_SAMPLES = 1000
CONFIDENCE_ALPHA = 0.05
TAUS = DEFAULT_TAUS

RSF_DISPLAY = "Random Survival Forest"
SURVPROMPT_DISPLAY = "Survprompt"


def _km_from_datasets(datasets):
    """Training-cohort KM curve in years, used to impute ground-truth RMST."""
    train = [x for x in datasets if x[2] == "Training Times/Events"]
    if not train:
        raise ValueError("Training Times/Events not found; call with get_train_times=True")
    train_time_yr = train[0][0].astype(float).values / 365.25
    train_event = train[0][1].astype(int).values
    return compute_km_curve(train_time_yr, train_event)


def compute_brier_at_taus_stats(
    datasets,
    model_label: str,
    taus: List[float] = TAUS,
    rsf_curves=None,
    n_boot: int = BOOTSTRAP_SAMPLES,
    alpha: float = CONFIDENCE_ALPHA,
    seed: int = 42,
) -> Dict[float, Dict[str, float]]:
    """Time-dependent Brier score BS(tau) at each horizon, with bootstrap CIs.

    Uses sksurv's ``brier_score`` (IPCW) with the predicted survival probability
    S(tau) as the estimate. RSF is scored against its true survival functions
    when ``rsf_curves`` is supplied (its CSV is scalar-only). A tau beyond the
    sample's follow-up is left as NaN for that horizon.
    """
    nan_stat = {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    empty = {tau: dict(nan_stat) for tau in taus}

    gt = [(x[0], x[1]) for x in datasets if x[2] == "Ground Truth"]
    train = [x for x in datasets if x[2] == "Training Times/Events"]
    model = [x for x in datasets if x[2] == model_label]
    if not gt or not train or not model:
        return empty

    gt_time = gt[0][0].astype(float) / 365.25
    gt_event = gt[0][1].astype(float)
    train_time = train[0][0].astype(float).values / 365.25
    train_event = train[0][1].astype(bool).values
    train_struct = np.array(
        [(e, t) for e, t in zip(train_event, train_time)],
        dtype=[("status", "bool"), ("time", "<f8")],
    )
    pred = model[0][0]

    common_idx = gt_time.index.intersection(pred.index)
    if len(common_idx) == 0:
        return empty

    taus_arr = np.asarray(taus, dtype=float)
    surv_rows, obs_t, obs_e = [], [], []
    for sid in common_idx:
        t_obs, ev = gt_time.loc[sid], gt_event.loc[sid]
        if np.isnan(t_obs) or np.isnan(ev):
            continue
        curve = _curve_for_entry(model_label, pred.loc[sid], rsf_curves, sid)
        if curve is None:
            continue
        times_yr, probs = curve
        surv_rows.append(np.interp(taus_arr, times_yr, probs, left=1.0, right=probs[-1]))
        obs_t.append(float(t_obs))
        obs_e.append(bool(int(ev)))
    if not surv_rows:
        return empty

    S = np.asarray(surv_rows)
    test_struct = np.array(
        [(e, t) for e, t in zip(obs_e, obs_t)],
        dtype=[("status", "bool"), ("time", "<f8")],
    )
    n = len(obs_t)

    def _brier_vec(idx) -> np.ndarray:
        sub, sub_S = test_struct[idx], S[idx]
        out = np.full(len(taus), np.nan)
        valid = taus_arr < sub["time"].max()  # sksurv requires tau within follow-up
        if valid.any():
            try:
                _, scores = sksurv_brier_score(train_struct, sub, sub_S[:, valid], taus_arr[valid])
                out[valid] = scores
            except Exception:
                pass
        return out

    point = _brier_vec(np.arange(n))
    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, len(taus)), np.nan)
    for b in range(n_boot):
        boot[b] = _brier_vec(rng.integers(0, n, size=n))

    out: Dict[float, Dict[str, float]] = {}
    for j, tau in enumerate(taus):
        col = boot[:, j]
        col = col[~np.isnan(col)]
        val = float(point[j]) if not np.isnan(point[j]) else np.nan
        if col.size == 0:
            out[tau] = {"value": val, "ci_lower": np.nan, "ci_upper": np.nan}
        else:
            out[tau] = {
                "value": val,
                "ci_lower": float(np.percentile(col, 100 * (alpha / 2))),
                "ci_upper": float(np.percentile(col, 100 * (1 - alpha / 2))),
            }
    return out


def _model_labels_for_task(datasets, prompting_task: str) -> List[str]:
    """Models for the all-models figure, mirroring plot_all_metrics filters."""
    labels = []
    seen = set()
    for _, _, label in datasets:
        if label in ("Ground Truth", "Training Times/Events"):
            continue
        if is_all_models_zero_shot_label(label, prompting_task) and label not in seen:
            seen.add(label)
            labels.append(label)
    # Consistent left-to-right model order across every task/figure.
    labels.sort(key=canonical_model_sort_key)
    return labels


def _model_color(internal_label: str, simplify: str = "Survprompt_appendix") -> str:
    """Same color rule the bar panels use (Set1 palette / grey RSF baseline).

    ``simplify='Survprompt_final'`` is used by the 2-model "baseline vs
    Survprompt" figures so the Survprompt line is drawn purple; the all-models
    figures use the default and keep family colors (e.g. GPT-5 stays blue).
    """
    try:
        m, method, _size, task = parse_label(internal_label)
        return get_model_color(m, method, task, simplify=simplify)
    except Exception:
        return "gray"


def _compute_per_cancer(
    cancer: str,
    model_labels: List[str],
    dataset: str,
    prompting_task: str,
    race_inclusion_path: str,
    system_prompt_path: str,
    metric: str = "rmst",
) -> Dict[str, Dict[float, Dict[str, float]]]:
    """Return {model_label: {tau: stats}} for one cancer.

    ``metric`` selects the per-tau quantity: ``"rmst"`` -> RMST-MAE(tau);
    ``"brier"`` -> time-dependent Brier score BS(tau); ``"onecal"`` ->
    one-calibration p-value at tau.
    """
    cfg = ExperimentConfig(base_dir=BASE_DIR, data_name=dataset, cancer_of_interest=cancer)
    datasets = process_data_for_km(
        cfg, [prompting_task], get_train_times=True,
        race_inclusion_path=race_inclusion_path, system_prompt_path=system_prompt_path,
    )

    rsf_curves = None
    if any("Baseline: RSF" in m for m in model_labels):
        try:
            rsf_curves = load_rsf_survival_curves(cancer, BASE_DIR, data_name=dataset)
        except Exception as exc:  # pragma: no cover - degrade gracefully
            print(f"  [warn] could not load RSF curves for {cancer}: {exc}")

    result = {}
    if metric == "brier":
        for label in model_labels:
            result[label] = compute_brier_at_taus_stats(
                datasets, label, taus=TAUS, rsf_curves=rsf_curves,
                n_boot=BOOTSTRAP_SAMPLES, alpha=CONFIDENCE_ALPHA,
            )
    elif metric == "onecal":
        for label in model_labels:
            result[label] = compute_one_calibration_stats(
                datasets, label, taus=TAUS, rsf_curves=rsf_curves,
            )
    else:
        km_t, km_s = _km_from_datasets(datasets)
        for label in model_labels:
            result[label] = compute_rmst_mae_stats(
                datasets, label, km_t, km_s, taus=TAUS, rsf_curves=rsf_curves,
                n_boot=BOOTSTRAP_SAMPLES, alpha=CONFIDENCE_ALPHA,
            )
    return result


def generate_all_models_rmst_figure(
    plot_dir: str,
    dataset: str,
    cancer_of_interest: List[str],
    prompting_task: str = "SURV_PROB",
    race_inclusion_path: str = "incl_race",
    system_prompt_path: str = "system",
):
    """4-row (one per tau) all-models RMST-MAE bar figure."""
    # Determine the model set from the first cancer.
    cfg0 = ExperimentConfig(base_dir=BASE_DIR, data_name=dataset, cancer_of_interest=cancer_of_interest[0])
    sample_datasets = process_data_for_km(
        cfg0, [prompting_task], get_train_times=True,
        race_inclusion_path=race_inclusion_path, system_prompt_path=system_prompt_path,
    )
    model_labels = _model_labels_for_task(sample_datasets, prompting_task)
    print(f"RMST all-models figure ({prompting_task}) -- models: {model_labels}")

    # stats_by_tau[tau][cancer][model_label] = {value, ci_lower, ci_upper}
    stats_by_tau = {tau: {} for tau in TAUS}
    for ca in cancer_of_interest:
        print(f"Processing RMST for {ca}...")
        per_model = _compute_per_cancer(
            ca, model_labels, dataset, prompting_task, race_inclusion_path, system_prompt_path
        )
        for tau in TAUS:
            stats_by_tau[tau][ca] = {label: per_model[label][tau] for label in model_labels}

    fig, axes = plt.subplots(len(TAUS), 1, figsize=(10, 5 * len(TAUS)))
    for i, tau in enumerate(TAUS):
        plot_metric_barplot(
            stats_by_tau[tau], cancer_of_interest, model_labels,
            metric_name=f"RMST-MAE tau={int(tau)}",
            ylabel=f"RMST-MAE @ {int(tau)}yr (years)",
            ax=axes[i], show_legend=(i == 0), simplify_legend_labels=True,
        )
        axes[i].set_title(f"τ = {int(tau)} years", fontsize=GLOBAL_FONT_SIZE)

    plt.tight_layout()
    out_path = os.path.join(plot_dir, f"all_metrics_all_models_RMST_{prompting_task}.pdf")
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    print(f"Saved: {out_path}")
    plt.close(fig)

    # Companion source-data CSV (RMST-MAE value + 95% CI per cancer x model x tau).
    rows = []
    for tau in TAUS:
        for ca in cancer_of_interest:
            for label in model_labels:
                s = stats_by_tau[tau][ca][label]
                rows.append({
                    "cancer": ca, "model": label, "metric": "RMST-MAE (years)",
                    "tau_years": int(tau), "value": s["value"],
                    "ci_lower": s["ci_lower"], "ci_upper": s["ci_upper"],
                })
    save_source_data(out_path, pd.DataFrame(rows))


def _legend_label(label: str, legend_names: Dict[str, str], simplify: bool) -> str:
    """Series label: explicit override, else 'RSF', else simplified model name."""
    if legend_names and label in legend_names:
        return legend_names[label]
    if "Baseline: RSF" in label:
        return "RSF"
    if simplify:
        try:
            m, _method, size, _task = parse_label(label)
            return format_model_display_label(m, size)
        except Exception:
            return label
    return label


def _timecourse_figure(
    plot_dir: str,
    out_name: str,
    dataset: str,
    cancer_of_interest: List[str],
    model_labels: List[str],
    metric: str,
    ylabel: str,
    suptitle: str,
    race_inclusion_path: str,
    system_prompt_path: str,
    prompting_task: str = "SURV_PROB",
    legend_names: Dict[str, str] = None,
):
    """Generic 1xN line figure: metric vs horizon tau, one line per model, per cancer.

    Confidence intervals are drawn as light shaded bands for every series (2-model
    and all-models figures alike).
    """
    many = len(model_labels) > 2
    # 2-model "baseline vs Survprompt" figures draw Survprompt purple; all-models
    # figures keep family colors.
    color_simplify = "Survprompt_appendix" if many else "Survprompt_final"
    colors = {lab: _model_color(lab, color_simplify) for lab in model_labels}
    styles = {lab: get_line_style(parse_label(lab)[2]) for lab in model_labels}
    names = {lab: _legend_label(lab, legend_names, simplify=True) for lab in model_labels}

    n = len(cancer_of_interest)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.4), sharey=False)
    if n == 1:
        axes = [axes]
    x = np.arange(len(TAUS))
    metric_label = {"rmst": "RMST-MAE (years)", "brier": "Brier score",
                    "onecal": "One-calibration p-value"}.get(metric, metric)
    source_rows = []

    for j, ca in enumerate(cancer_of_interest):
        ax = axes[j]
        per_model = _compute_per_cancer(
            ca, model_labels, dataset, prompting_task,
            race_inclusion_path, system_prompt_path, metric=metric,
        )
        for lab in model_labels:
            vals = np.array([per_model[lab][tau]["value"] for tau in TAUS], dtype=float)
            lo = np.array([per_model[lab][tau]["ci_lower"] for tau in TAUS], dtype=float)
            hi = np.array([per_model[lab][tau]["ci_upper"] for tau in TAUS], dtype=float)
            ax.plot(x, vals, marker="o", linewidth=1.8, markersize=5 if not many else 4,
                    color=colors[lab], linestyle=styles[lab], label=names[lab])
            ax.fill_between(x, lo, hi, color=colors[lab], alpha=0.15, linewidth=0)
            for ti, tau in enumerate(TAUS):
                source_rows.append({
                    "cancer": ca, "model": lab, "metric": metric_label,
                    "tau_years": int(tau), "value": vals[ti],
                    "ci_lower": lo[ti], "ci_upper": hi[ti],
                })
        ax.set_title(ca.upper())
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(t)}" for t in TAUS])
        ax.set_xlabel("Horizon τ (years)")
        if j == 0:
            ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.3)

    handles, labs = axes[0].get_legend_handles_labels()
    ncol = min(len(model_labels), 5)
    fig.legend(handles, labs, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, -0.10 if many else -0.06))
    fig.suptitle(suptitle, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(plot_dir, out_name)
    plt.savefig(out_path, bbox_inches="tight", format="pdf")
    print(f"Saved: {out_path}")
    plt.close(fig)
    save_source_data(out_path, pd.DataFrame(source_rows))


def _all_surv_prob_labels(dataset, cancer, race_inclusion_path, system_prompt_path) -> List[str]:
    cfg = ExperimentConfig(base_dir=BASE_DIR, data_name=dataset, cancer_of_interest=cancer)
    sample = process_data_for_km(
        cfg, ["SURV_PROB"], get_train_times=True,
        race_inclusion_path=race_inclusion_path, system_prompt_path=system_prompt_path,
    )
    return _model_labels_for_task(sample, "SURV_PROB")


def main():
    dataset = DEFAULT_DATA_NAME
    cancer_of_interest = ["nsclc", "brca", "crc", "panc", "prostate"]
    race_inclusion_path = "incl_race"
    system_prompt_path = "system"

    plot_dir = os.path.join(BASE_DIR, "plots", "all_metrics", dataset, race_inclusion_path)
    os.makedirs(plot_dir, exist_ok=True)

    print("=" * 60)
    print("GENERATING RMST ALL-MODELS BAR FIGURES (4 rows by tau)")
    print("=" * 60)
    for task in ("SURV_PROB", "TTE_OS"):
        generate_all_models_rmst_figure(
            plot_dir, dataset, cancer_of_interest, task, race_inclusion_path, system_prompt_path
        )

    all_models = _all_surv_prob_labels(dataset, cancer_of_interest[0], race_inclusion_path, system_prompt_path)
    survprompt_label = resolve_survprompt_headline_label(all_models)
    two = ["Baseline: RSF", survprompt_label]
    two_names = {"Baseline: RSF": "RSF", survprompt_label: "Survprompt"}
    print(f"All-models timecourse model set: {all_models}")

    # 1x5 timecourses: {RMST-MAE, time-dependent Brier} x {RSF-vs-Survprompt, all models}.
    jobs = [
        ("rmst_survprompt_vs_rsf_timecourse.pdf", two, "rmst", "RMST-MAE (years)",
         "RMST-MAE vs horizon: Survprompt vs RSF baseline", two_names),
        ("rmst_all_models_timecourse.pdf", all_models, "rmst", "RMST-MAE (years)",
         "RMST-MAE vs horizon: all SURV_PROB models", None),
        ("brier_survprompt_vs_rsf_timecourse.pdf", two, "brier", "Brier score",
         "Time-dependent Brier vs horizon: Survprompt vs RSF baseline", two_names),
        ("brier_all_models_timecourse.pdf", all_models, "brier", "Brier score",
         "Time-dependent Brier vs horizon: all SURV_PROB models", None),
        ("onecal_survprompt_vs_rsf_timecourse.pdf", two, "onecal", "One-calibration p-value (↑ better)",
         "One-calibration vs horizon: Survprompt vs RSF baseline", two_names),
        ("onecal_all_models_timecourse.pdf", all_models, "onecal", "One-calibration p-value (↑ better)",
         "One-calibration vs horizon: all SURV_PROB models", None),
    ]
    for out_name, labels, metric, ylabel, suptitle, legend_names in jobs:
        print("\n" + "=" * 60)
        print(f"GENERATING {out_name}")
        print("=" * 60)
        _timecourse_figure(
            plot_dir, out_name, dataset, cancer_of_interest, labels, metric,
            ylabel, suptitle, race_inclusion_path, system_prompt_path,
            legend_names=legend_names,
        )


if __name__ == "__main__":
    main()
