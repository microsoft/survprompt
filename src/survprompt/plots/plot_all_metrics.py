"""
Generate 4-subplot figure with bar charts and bootstrap confidence intervals for
C-Index, integrated Brier score, mean absolute error (MAE), and censored MAE across
all cancers of interest.
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from typing import List, Tuple, Optional, Dict, Callable
import ast
try:  # Optional dependency in some environments
    from sksurv.metrics import brier_score as sksurv_brier_score  # type: ignore[import]
except ImportError:  # pragma: no cover - handled at runtime
    sksurv_brier_score = None

from survprompt.configs.exp_config import ExperimentConfig
from survprompt.defaults import DEFAULT_DATA_NAME, SURVPROMPT_DEFAULT_MODEL_LABEL
from survprompt.plots.color_utils import GLOBAL_FONT_SIZE, canonical_model_sort_key
from survprompt.plots.plot_utils import (
    is_all_models_zero_shot_label,
    format_model_display_label, process_data_for_km, parse_label,
    get_model_color, resolve_survprompt_headline_label, save_source_data,
    metric_stats_to_long
)
from survprompt.evaluation.metrics import (
    calculate_error, interpolate_time_at_threshold,
    concordance_index_censored, compute_km_curve,
    compute_rmst_mae_stats, load_rsf_survival_curves, cmae_components,
    compute_d_calibration_stats, pseudo_obs_surrogate_times,
    censoring_confidence_weights, weighted_concordance
)

# Set global font properties
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams["font.size"] = GLOBAL_FONT_SIZE

# Load environment and set BASE_DIR
BASE_DIR = os.getenv("BASE_DIR")
if BASE_DIR is None:
    load_dotenv()
    BASE_DIR = os.getenv("BASE_DIR")

BOOTSTRAP_SAMPLES = 1000
CONFIDENCE_ALPHA = 0.05

RMST_TAU_REP = 5.0

# Parallelism for the per-(cancer, model) bootstrap (the expensive part). Each
# task is independent, so we fan them out across processes.
N_WORKERS = min(48, max(1, (os.cpu_count() or 1) - 2))


def _minimal_datasets(selected_datasets, model_label):
    """Trim a cancer's datasets to just what one model's metrics need: ground
    truth, training times/events, and that model -- so each parallel task pickles
    only a small payload."""
    keep = {"Ground Truth", "Training Times/Events", model_label}
    return [x for x in selected_datasets if x[2] in keep]


def _bar_metrics_for_model(selected_datasets, model_label, time_points,
                           n_boot, alpha, rsf_curves):
    """All bar-figure stats for one (cancer, model). Module-level so it can be
    dispatched to worker processes."""
    metrics = compute_all_metrics_with_bootstrap(
        selected_datasets, model_label, time_points, n_boot=n_boot,
        alpha=alpha, rsf_curves=rsf_curves,
    )
    metrics = dict(metrics)
    metrics["d_cal"] = compute_d_calibration_stats(
        selected_datasets, model_label, rsf_curves=rsf_curves)
    return metrics


def _comparison_metrics_for_model(selected_datasets, model_label, time_points,
                                  n_boot, alpha, rsf_curves, km_t, km_s, rmst_tau):
    """Bar metrics + RMST-MAE@tau for one (cancer, model) of the comparison."""
    metrics = _bar_metrics_for_model(
        selected_datasets, model_label, time_points, n_boot, alpha, rsf_curves)
    nan_stat = {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    try:
        metrics["rmst"] = compute_rmst_mae_stats(
            selected_datasets, model_label, km_t, km_s, taus=[rmst_tau],
            rsf_curves=rsf_curves, n_boot=n_boot, alpha=alpha)[rmst_tau]
    except Exception:
        metrics["rmst"] = dict(nan_stat)
    return metrics

def _bootstrap_ci(
    values: np.ndarray,
    agg_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """Return point estimate and bootstrap percentile CI for 1D data."""
    clean = np.asarray(values, dtype=float)
    clean = clean[~np.isnan(clean)]
    if clean.size == 0:
        return np.nan, np.nan, np.nan
    point = agg_fn(clean)
    if clean.size == 1 or n_boot <= 0:
        return point, point, point
    rng = np.random.default_rng(random_state)
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(clean, size=clean.size, replace=True)
        boot_stats.append(agg_fn(sample))
    lower = np.percentile(boot_stats, 100 * (alpha / 2))
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return point, lower, upper


def _bootstrap_indices(
    metric_fn: Callable[[np.ndarray], float],
    n_observations: int,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """Bootstrap a metric that can be recomputed on index subsets."""
    if n_observations == 0:
        return np.nan, np.nan, np.nan
    full_indices = np.arange(n_observations)
    point = metric_fn(full_indices)
    if n_observations == 1 or n_boot <= 0:
        return point, point, point
    rng = np.random.default_rng(random_state)
    stats = []
    for _ in range(n_boot):
        sample_idx = rng.choice(full_indices, size=n_observations, replace=True)
        stats.append(metric_fn(sample_idx))
    lower = np.percentile(stats, 100 * (alpha / 2))
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return point, lower, upper


def _summarize_error_array(
    errors: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
    take_abs: bool = True
) -> Dict[str, float]:
    errs = np.asarray(errors, dtype=float)
    errs = errs[~np.isnan(errs)]
    if errs.size == 0:
        return {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    if take_abs:
        errs = np.abs(errs)
    point, lower, upper = _bootstrap_ci(errs, n_boot=n_boot, alpha=alpha)
    return {"value": point, "ci_lower": lower, "ci_upper": upper}


def compute_all_metrics_with_bootstrap(
    datasets: List[Tuple[pd.Series, pd.Series, str]],
    model_name: str,
    time_points: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rsf_curves: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute all four metrics (C-Index, Brier, MAE, cMAE) with a single bootstrap loop.
    Returns dict with keys: 'c_index', 'brier_score', 'mae', 'cmae'

    The RSF/Cox baselines store only a scalar predicted time, so their Brier
    matrix would otherwise be a crude step function. Pass ``rsf_curves``
    (``{sample_id: (times_years, probs)}`` from
    ``survprompt.evaluation.metrics.load_rsf_survival_curves``) to score the RSF
    baseline against its true survival function instead; the time-based metrics
    (c-index/MAE/cMAE) still use the scalar t50.
    """
    if sksurv_brier_score is None:
        raise ImportError("sksurv.metrics is required to compute Brier scores")
    # Extract ground truth and training data
    gt_time, gt_event = [(x[0], x[1]) for x in datasets if x[2] == 'Ground Truth'][0]
    gt_time = gt_time.astype(float) / 365.25
    gt_event = gt_event.astype(bool)
    
    train_time, train_event = None, None
    for time, event, name in datasets:
        if "Training Times/Events" in name:
            train_time = time.astype(float) / 365.25
            train_event = event.astype(bool)
            break
    
    # Find model data
    model_data = None
    for time, event, name in datasets:
        if name == model_name:
            model_data = (time, event, name)
            break
    
    if model_data is None:
        return {
            'c_index': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan},
            'pseudo_c_index': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan},
            'pseudo_c_index_weighted': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan},
            'brier_score': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan},
            'mae': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan},
            'cmae': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
        }
    
    time_data, event_data, _ = model_data
    
    # Prepare aligned data
    common_idx = gt_time.index.intersection(time_data.index)
    if len(common_idx) == 0:
        return {
            'c_index': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan},
            'pseudo_c_index': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan},
            'pseudo_c_index_weighted': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan},
            'brier_score': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan},
            'mae': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan},
            'cmae': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
        }
    
    aligned_gt_time = gt_time.loc[common_idx]
    aligned_gt_event = gt_event.loc[common_idx]
    aligned_pred_data = time_data.loc[common_idx]
    
    # Convert predictions based on type
    if 'SURV_PROB' in model_name:
        pred_times = aligned_pred_data.apply(
            lambda x: interpolate_time_at_threshold(x[0], x[1], 0.5)
            if isinstance(x, (list, tuple)) else np.nan
        ).astype(float)
        
        # Build survival probability matrix for Brier score
        def extract_survival_probs_at_times(surv_data, target_times):
            if isinstance(surv_data, (list, tuple)) and len(surv_data) == 2:
                times, probs = surv_data
                if isinstance(times, str):
                    times = ast.literal_eval(times)
                if isinstance(probs, str):
                    probs = ast.literal_eval(probs)
                # SURV_PROB curve times are already in years (same convention the
                # t50/MAE path relies on); do NOT divide by 365.25.
                times = np.array(times, dtype=float)
                probs = np.array(probs, dtype=float)
                if len(times) > 0 and len(probs) > 0:
                    return np.interp(target_times, times, probs)
            return np.full(len(target_times), np.nan)

        pred_surv_matrix = np.array([
            extract_survival_probs_at_times(entry, time_points)
            for entry in aligned_pred_data
        ])
    else:
        # Time-to-event predictions
        pred_times = aligned_pred_data.astype(float) / 365.25
        pred_surv_matrix = np.zeros((len(pred_times), len(time_points)))
        for i, pred_time in enumerate(pred_times):
            pred_surv_matrix[i] = (time_points <= pred_time).astype(float)

    if rsf_curves is not None and 'Baseline: RSF' in model_name:
        pred_surv_matrix = np.array([
            np.interp(time_points, rsf_curves[sid][0], rsf_curves[sid][1],
                      left=1.0, right=rsf_curves[sid][1][-1])
            if sid in rsf_curves else np.full(len(time_points), np.nan)
            for sid in common_idx
        ])

    # Filter valid samples
    valid_mask = ~(pred_times.isna() | aligned_gt_time.isna() | aligned_gt_event.isna() |
                   np.isnan(pred_surv_matrix).any(axis=1))
    
    if valid_mask.sum() == 0:
        return {
            'c_index': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan},
            'pseudo_c_index': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan},
            'pseudo_c_index_weighted': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan},
            'brier_score': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan},
            'mae': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan},
            'cmae': {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
        }
    
    final_gt_time = aligned_gt_time[valid_mask].values
    final_gt_event = aligned_gt_event[valid_mask].values
    final_pred_times = pred_times[valid_mask].values
    final_pred_surv_matrix = pred_surv_matrix[valid_mask]
    
    # Prepare structured arrays for Brier score
    if train_time is not None and train_event is not None:
        train_structured = np.array([
            (ev, tm) for ev, tm in zip(train_event.values, train_time.values)
        ], dtype=[("status", "bool"), ("time", "<f8")])
    else:
        train_structured = None
    
    test_structured = np.array([
        (ev, tm) for ev, tm in zip(final_gt_event, final_gt_time)
    ], dtype=[("status", "bool"), ("time", "<f8")])
    
    surv_estimate = final_pred_surv_matrix

    # Compute errors for MAE and cMAE
    mae_errors = np.abs(final_pred_times - final_gt_time)
    
    if train_time is not None and train_event is not None:
        # Pseudo-observation surrogate (decensored times). Reused for both the
        # weighted cMAE and the pseudo-observation concordance below.
        pseudo_surrogate = pseudo_obs_surrogate_times(
            final_gt_time, final_gt_event,
            train_time.values, train_event.values,
        )
        cmae_errors = np.abs(final_pred_times - pseudo_surrogate)
        cmae_weights = censoring_confidence_weights(
            final_gt_time, final_gt_event,
            train_time.values, train_event.values,
        )
    else:
        cmae_errors = mae_errors.copy()
        cmae_weights = np.ones_like(mae_errors)
        pseudo_surrogate = None
    
    # Single bootstrap loop for all metrics
    n_samples = len(final_gt_time)
    rng = np.random.default_rng(42)
    
    c_index_boots = []
    pseudo_c_boots = []          # unweighted pseudo-observation concordance
    pseudo_c_w_boots = []        # confidence-weighted pseudo-observation concordance
    brier_boots = []
    mae_boots = []
    cmae_boots = []

    # Decensored surrogate times -> all pairs comparable (every patient "complete").
    all_events = np.ones(n_samples, dtype=bool)

    for _ in range(n_boot):
        indices = rng.choice(n_samples, size=n_samples, replace=True)

        # C-Index
        try:
            c_res = concordance_index_censored(
                event_indicator=final_gt_event[indices],
                event_time=final_gt_time[indices],
                estimate=-final_pred_times[indices]
            )
            c_index_boots.append(c_res[0])
        except Exception:
            c_index_boots.append(np.nan)

        # Pseudo-observation C-Index (unweighted): ordinary concordance on the
        # decensored times.
        if pseudo_surrogate is not None:
            try:
                pseudo_c_boots.append(concordance_index_censored(
                    all_events[indices], pseudo_surrogate[indices],
                    -final_pred_times[indices])[0])
            except Exception:
                pseudo_c_boots.append(np.nan)
        else:
            pseudo_c_boots.append(np.nan)

        # Brier Score
        if train_structured is not None:
            try:
                sub_structured = test_structured[indices]
                sub_surv = surv_estimate[indices]

                max_sample_time = sub_structured["time"].max()
                valid_time_mask = time_points < max_sample_time

                if valid_time_mask.sum() > 0:
                    sample_time_points = time_points[valid_time_mask]
                    sample_surv = sub_surv[:, valid_time_mask]

                    times_out, scores = sksurv_brier_score(
                        survival_train=train_structured,
                        survival_test=sub_structured,
                        estimate=sample_surv,
                        times=sample_time_points
                    )
                    
                    mask = ~np.isnan(scores)
                    if mask.sum() > 0:
                        valid_times = times_out[mask]
                        valid_scores = scores[mask]
                        if valid_scores.size == 1:
                            brier_boots.append(float(valid_scores[0]))
                        else:
                            integrated = np.trapz(valid_scores, valid_times) / (valid_times[-1] - valid_times[0])
                            brier_boots.append(float(integrated))
                    else:
                        brier_boots.append(np.nan)
                else:
                    brier_boots.append(np.nan)
            except Exception:
                brier_boots.append(np.nan)
        else:
            brier_boots.append(np.nan)
        
        # MAE
        mae_boots.append(np.mean(mae_errors[indices]))

        # cMAE (confidence-weighted mean)
        w = cmae_weights[indices]
        cmae_boots.append(
            np.average(cmae_errors[indices], weights=w) if w.sum() > 0
            else np.mean(cmae_errors[indices])
        )
    
    # Compute point estimates
    c_index_point = concordance_index_censored(
        event_indicator=final_gt_event,
        event_time=final_gt_time,
        estimate=-final_pred_times
    )[0]

    # Pseudo-observation concordance point estimates: unweighted (ordinary
    # concordance on decensored times) and confidence-weighted. The weighted one
    # is bootstrapped in its own loop via multinomial resample counts folded into
    # the pair weights (count_i * w_i) -- identical to resampling indices, but it
    # collapses duplicate draws so each replicate runs on fewer distinct nodes
    # (the O(n log n) weighted concordance is the bottleneck here).
    if pseudo_surrogate is not None:
        pseudo_risk = -final_pred_times
        try:
            pseudo_c_point = concordance_index_censored(
                all_events, pseudo_surrogate, pseudo_risk)[0]
        except Exception:
            pseudo_c_point = np.nan
        pseudo_c_w_point = weighted_concordance(pseudo_surrogate, pseudo_risk, cmae_weights)
        rng_pc = np.random.default_rng(123)
        for _ in range(n_boot):
            counts = np.bincount(rng_pc.integers(0, n_samples, n_samples), minlength=n_samples)
            present = counts > 0
            try:
                pseudo_c_w_boots.append(weighted_concordance(
                    pseudo_surrogate[present], pseudo_risk[present],
                    cmae_weights[present] * counts[present],
                ))
            except Exception:
                pseudo_c_w_boots.append(np.nan)
    else:
        pseudo_c_point = np.nan
        pseudo_c_w_point = np.nan

    mae_point = np.mean(mae_errors)
    cmae_point = (
        np.average(cmae_errors, weights=cmae_weights) if cmae_weights.sum() > 0
        else np.mean(cmae_errors)
    )
    
    # Brier point estimate
    if train_structured is not None:
        try:
            max_test_time = test_structured["time"].max()
            valid_time_mask = time_points < max_test_time
            if valid_time_mask.sum() > 0:
                sample_time_points = time_points[valid_time_mask]
                sample_surv = surv_estimate[:, valid_time_mask]

                times_out, scores = sksurv_brier_score(
                    survival_train=train_structured,
                    survival_test=test_structured,
                    estimate=sample_surv,
                    times=sample_time_points
                )
                
                mask = ~np.isnan(scores)
                if mask.sum() > 0:
                    valid_times = times_out[mask]
                    valid_scores = scores[mask]
                    if valid_scores.size == 1:
                        brier_point = float(valid_scores[0])
                    else:
                        brier_point = float(np.trapz(valid_scores, valid_times) / (valid_times[-1] - valid_times[0]))
                else:
                    brier_point = np.nan
            else:
                brier_point = np.nan
        except Exception:
            brier_point = np.nan
    else:
        brier_point = np.nan
    
    # Compute CIs
    def get_ci(boots, point, alpha):
        boots_clean = np.array([b for b in boots if not np.isnan(b)])
        if len(boots_clean) == 0:
            return point, point, point
        lower = np.percentile(boots_clean, 100 * (alpha / 2))
        upper = np.percentile(boots_clean, 100 * (1 - alpha / 2))
        return point, lower, upper
    
    c_val, c_low, c_up = get_ci(c_index_boots, c_index_point, alpha)
    pc_val, pc_low, pc_up = get_ci(pseudo_c_boots, pseudo_c_point, alpha)
    pcw_val, pcw_low, pcw_up = get_ci(pseudo_c_w_boots, pseudo_c_w_point, alpha)
    b_val, b_low, b_up = get_ci(brier_boots, brier_point, alpha)
    mae_val, mae_low, mae_up = get_ci(mae_boots, mae_point, alpha)
    cmae_val, cmae_low, cmae_up = get_ci(cmae_boots, cmae_point, alpha)

    return {
        'c_index': {"value": c_val, "ci_lower": c_low, "ci_upper": c_up},
        'pseudo_c_index': {"value": pc_val, "ci_lower": pc_low, "ci_upper": pc_up},
        'pseudo_c_index_weighted': {"value": pcw_val, "ci_lower": pcw_low, "ci_upper": pcw_up},
        'brier_score': {"value": b_val, "ci_lower": b_low, "ci_upper": b_up},
        'mae': {"value": mae_val, "ci_lower": mae_low, "ci_upper": mae_up},
        'cmae': {"value": cmae_val, "ci_lower": cmae_low, "ci_upper": cmae_up}
    }

def calculate_c_index_for_models(
    datasets: List[Tuple[pd.Series, pd.Series, str]],
    n_boot: int = 500,
    alpha: float = 0.05
) -> Dict[str, Dict[str, float]]:
    """
    Calculate c-index with bootstrap confidence intervals for all models.
    """
    c_indices: Dict[str, Dict[str, float]] = {}

    gt_time, gt_event = [(x[0], x[1]) for x in datasets if x[2] == 'Ground Truth'][0]
    gt_time = gt_time.astype(float) / 365.25
    gt_event = gt_event.astype(bool)

    for time, event, name in datasets:
        if name in ['Ground Truth', 'Training Times/Events']:
            continue

        try:
            if 'SURV_PROB' in name:
                pred_times = time.apply(
                    lambda x: interpolate_time_at_threshold(x[0], x[1], 0.5)
                ).astype(float)
            else:
                pred_times = time.astype(float) / 365.25

            common_idx = gt_time.index.intersection(pred_times.index)
            if len(common_idx) == 0:
                c_indices[name] = {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
                continue

            aligned_gt_time = gt_time.loc[common_idx]
            aligned_gt_event = gt_event.loc[common_idx]
            aligned_pred_times = pred_times.loc[common_idx]

            valid_mask = ~(aligned_pred_times.isna() |
                           aligned_gt_time.isna() |
                           aligned_gt_event.isna())
            final_gt_time = aligned_gt_time[valid_mask].values
            final_gt_event = aligned_gt_event[valid_mask].values
            final_pred_times = aligned_pred_times[valid_mask].values

            if final_pred_times.size == 0:
                c_indices[name] = {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
                continue

            def metric_fn(indices: np.ndarray) -> float:
                if indices.size == 0:
                    return np.nan
                res = concordance_index_censored(
                    event_indicator=final_gt_event[indices],
                    event_time=final_gt_time[indices],
                    estimate=-final_pred_times[indices]
                )
                return res[0]

            point, lower, upper = _bootstrap_indices(
                metric_fn,
                n_observations=final_pred_times.size,
                n_boot=n_boot,
                alpha=alpha
            )

            c_indices[name] = {
                "value": point,
                "ci_lower": lower,
                "ci_upper": upper
            }

        except Exception as e:
            print(f"Error calculating c-index for {name}: {e}")
            c_indices[name] = {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}

    return c_indices


def calculate_brier_scores_for_models(
    datasets: List[Tuple[pd.Series, pd.Series, str]],
    time_points: Optional[np.ndarray] = None,
    n_boot: int = 500,
    alpha: float = 0.05
) -> Dict[str, Dict[str, float]]:
    """Calculate integrated Brier score with bootstrap confidence intervals."""
    if sksurv_brier_score is None:
        raise ImportError("sksurv.metrics is required to compute Brier scores")
    if time_points is None:
        time_points = np.arange(0.5, 10.1, 0.5)

    brier_scores: Dict[str, Dict[str, float]] = {}

    gt_time, gt_event = [(x[0], x[1]) for x in datasets if x[2] == 'Ground Truth'][0]
    gt_time = gt_time.astype(float) / 365.25
    gt_event = gt_event.astype(bool)

    train_time, train_event = None, None
    for time, event, name in datasets:
        if "Training Times/Events" in name:
            train_time = time.astype(float) / 365.25
            train_event = event.astype(bool)
            break

    if train_time is None or train_event is None:
        print("Warning: No training data found for Brier score calculation")
        return brier_scores

    train_structured = np.array([
        (ev, tm) for ev, tm in zip(train_event.values, train_time.values)
    ], dtype=[("status", "bool"), ("time", "<f8")])

    for time, event, name in datasets:
        if name in ['Ground Truth', 'Training Times/Events']:
            continue

        try:
            common_idx = gt_time.index.intersection(time.index)
            if len(common_idx) == 0:
                brier_scores[name] = {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
                continue

            aligned_gt_time = gt_time.loc[common_idx]
            aligned_gt_event = gt_event.loc[common_idx]
            aligned_pred_data = time.loc[common_idx]

            if 'SURV_PROB' in name:
                def extract_survival_probs_at_times(surv_data, target_times):
                    if isinstance(surv_data, (list, tuple)) and len(surv_data) == 2:
                        times, probs = surv_data
                        if isinstance(times, str):
                            times = ast.literal_eval(times)
                        if isinstance(probs, str):
                            probs = ast.literal_eval(probs)
                        # SURV_PROB curve times are already in years; no /365.25.
                        times = np.array(times, dtype=float)
                        probs = np.array(probs, dtype=float)
                        if len(times) > 0 and len(probs) > 0:
                            return np.interp(target_times, times, probs)
                    return np.full(len(target_times), np.nan)

                pred_surv_matrix = np.array([
                    extract_survival_probs_at_times(entry, time_points)
                    for entry in aligned_pred_data
                ])

            else:
                pred_times = aligned_pred_data.astype(float) / 365.25
                pred_surv_matrix = np.zeros((len(pred_times), len(time_points)))
                for i, pred_time in enumerate(pred_times):
                    pred_surv_matrix[i] = (time_points <= pred_time).astype(float)

            valid_mask = ~(np.isnan(pred_surv_matrix).any(axis=1) |
                          aligned_gt_time.isna() |
                          aligned_gt_event.isna())
            if valid_mask.sum() == 0:
                brier_scores[name] = {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
                continue

            final_gt_time = aligned_gt_time[valid_mask].values
            final_gt_event = aligned_gt_event[valid_mask].values
            final_pred_matrix = pred_surv_matrix[valid_mask]

            test_structured = np.array([
                (ev, tm) for ev, tm in zip(final_gt_event, final_gt_time)
            ], dtype=[("status", "bool"), ("time", "<f8")])

            # sksurv.brier_score expects the SURVIVAL probability S(t), not risk.
            surv_estimate = final_pred_matrix

            def compute_integrated(indices: np.ndarray) -> float:
                if indices.size == 0:
                    return np.nan
                sub_structured = test_structured[indices]
                sub_surv = surv_estimate[indices]

                # Filter time points based on the maximum time in THIS bootstrap sample
                max_sample_time = sub_structured["time"].max()
                valid_time_mask = time_points < max_sample_time
                if valid_time_mask.sum() == 0:
                    return np.nan

                sample_time_points = time_points[valid_time_mask]
                sample_surv = sub_surv[:, valid_time_mask]

                try:
                    times_out, scores = sksurv_brier_score(
                        survival_train=train_structured,
                        survival_test=sub_structured,
                        estimate=sample_surv,
                        times=sample_time_points
                    )
                except Exception:
                    return np.nan
                    
                mask = ~np.isnan(scores)
                if mask.sum() == 0:
                    return np.nan
                valid_times = times_out[mask]
                valid_scores = scores[mask]
                if valid_scores.size == 1:
                    return float(valid_scores[0])
                return float(np.trapz(valid_scores, valid_times) / (valid_times[-1] - valid_times[0]))

            point, lower, upper = _bootstrap_indices(
                compute_integrated,
                n_observations=surv_estimate.shape[0],
                n_boot=n_boot,
                alpha=alpha
            )

            brier_scores[name] = {
                "value": point,
                "ci_lower": lower,
                "ci_upper": upper
            }

        except Exception as e:
            print(f"Error calculating Brier score for {name}: {e}")
            brier_scores[name] = {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}

    return brier_scores


def calculate_mae_for_models(datasets: List[Tuple[pd.Series, pd.Series, str]]) -> Dict[str, np.ndarray]:
    """
    Calculate MAE for all models.
    
    Args:
        datasets: List of (time, event, label) tuples
        
    Returns:
        Dictionary mapping model name to array of errors
    """
    mae_errors = {}
    
    # Get ground truth
    gt_time, gt_event = [(x[0], x[1]) for x in datasets if x[2] == 'Ground Truth'][0]
    gt_time = gt_time.astype(float) / 365.25  # Convert to years
    gt_event = gt_event.astype(bool)
    gt_df = pd.DataFrame({'time': gt_time, 'event': gt_event})
    
    for time, event, name in datasets:
        if name in ['Ground Truth', 'Training Times/Events']:
            continue
            
        try:
            if 'SURV_PROB' in name:
                pred_times = time.apply(lambda x: interpolate_time_at_threshold(x[0], x[1], 0.5)).astype(float)
            else:
                pred_times = time.astype(float) / 365.25
            
            event_series = event.astype(bool)
            df = pd.DataFrame({'time': pred_times, 'event': event_series})
            aligned = gt_df.join(df, lsuffix='_gt', rsuffix='_pred', how='inner')
            
            # Calculate error
            errors = calculate_error(aligned['time_pred'], aligned['time_gt'], aligned['event_gt'])
            mae_errors[name] = errors
            
        except Exception as e:
            print(f"Error calculating MAE for {name}: {e}")
            mae_errors[name] = np.array([np.nan])
            
    return mae_errors


def calculate_cmae_for_models(datasets: List[Tuple[pd.Series, pd.Series, str]]) -> Dict[str, np.ndarray]:
    """
    Calculate censored MAE for all models.
    
    Args:
        datasets: List of (time, event, label) tuples
        
    Returns:
        Dictionary mapping model name to array of censored errors
    """
    cmae_errors = {}
    
    # Get ground truth and training data
    gt_time, gt_event = [(x[0], x[1]) for x in datasets if x[2] == 'Ground Truth'][0]
    gt_time = gt_time.astype(float) / 365.25  # Convert to years
    gt_event = gt_event.astype(bool)
    gt_df = pd.DataFrame({'time': gt_time, 'event': gt_event})
    
    train_time, train_event = None, None
    for time, event, name in datasets:
        if "Training Times/Events" in name:
            train_time, train_event = time.astype(float) / 365.25, event.astype(bool)
            break
    
    if train_time is None or train_event is None:
        print("Warning: No training data found for cMAE calculation")
        return cmae_errors
    
    for time, event, name in datasets:
        if name in ['Ground Truth', 'Training Times/Events']:
            continue
            
        try:
            if 'SURV_PROB' in name:
                pred_times = time.apply(lambda x: interpolate_time_at_threshold(x[0], x[1], 0.5)).astype(float)
            else:
                pred_times = time.astype(float) / 365.25
            
            event_series = event.astype(bool)
            df = pd.DataFrame({'time': pred_times, 'event': event_series})
            aligned = gt_df.join(df, lsuffix='_gt', rsuffix='_pred', how='inner')
            
            # Calculate censored error
            censored_errors = calculate_error(
                aligned['time_pred'], aligned['time_gt'], aligned['event_gt'],
                train_event_times=train_time, train_event_indicators=train_event
            )
            cmae_errors[name] = censored_errors
            
        except Exception as e:
            print(f"Error calculating cMAE for {name}: {e}")
            cmae_errors[name] = np.array([np.nan])
            
    return cmae_errors


def plot_metric_barplot(
    metric_stats: Dict[str, Dict[str, Dict[str, float]]],
    cancer_of_interest: List[str],
    model_names: List[str],
    metric_name: str,
    ylabel: str,
    ax,
    show_legend: bool = False,
    simplify_legend_labels: bool = False,
    show_xlabel: bool = False,
    use_fixed_y_axis: bool = False,
):
    """Bar chart with bootstrap confidence intervals for each metric."""
    cancer_labels = [ca.upper() for ca in cancer_of_interest]

    color_map: Dict[str, str] = {}
    legend_labels: Dict[str, str] = {}
    hatch_map: Dict[str, str] = {}
    
    for model in model_names:
        if model == 'Random Survival Forest':
            model_label = 'Baseline: RSF'
        elif model == 'Survprompt':
            model_label = SURVPROMPT_DEFAULT_MODEL_LABEL
        else:
            model_label = model
        try:
            m, method, size, task = parse_label(model_label)
            # Headline "Survprompt" series is purple; all-models bars keep family
            # colors (Set1 palette) to match km_plots.
            simplify_param = 'Survprompt_final' if model == 'Survprompt' else 'Survprompt_appendix'
            color_map[model] = get_model_color(m, method, task, simplify=simplify_param)

            # Hatch compact model variants only; reasoning effort should not
            # reuse the model-size encoding.
            hatch_map[model] = '///' if size == 'mini' else None

            if simplify_legend_labels:
                legend_labels[model] = format_model_display_label(m, size)
            else:
                legend_labels[model] = model
        except Exception:
            color_map[model] = 'gray'
            legend_labels[model] = model
            hatch_map[model] = None

    x = np.arange(len(cancer_labels))
    
    # Adjust bar width based on number of models
    # For many models, make bars narrower to avoid overlap
    if len(model_names) <= 2:
        width = 0.35
    elif len(model_names) <= 4:
        width = 0.20
    elif len(model_names) <= 6:
        width = 0.13
    else:
        width = 0.8 / len(model_names)

    for i, model in enumerate(model_names):
        values = []
        err_low = []
        err_high = []
        for ca in cancer_of_interest:
            stats = metric_stats.get(ca, {}).get(model, {})
            value = stats.get("value", np.nan)
            lower = stats.get("ci_lower", np.nan)
            upper = stats.get("ci_upper", np.nan)
            values.append(value)
            if np.isnan(value) or np.isnan(lower) or np.isnan(upper):
                err_low.append(0.0)
                err_high.append(0.0)
            else:
                err_low.append(max(0.0, value - lower))
                err_high.append(max(0.0, upper - value))

        offset = (i - len(model_names) / 2 + 0.5) * width
        
        # Set hatch color to white for hatched bars
        if hatch_map.get(model, None):
            mpl.rcParams['hatch.color'] = 'white'
        
        bar = ax.bar(
            x + offset,
            values,
            width,
            label=legend_labels.get(model, model),
            color=color_map.get(model, 'gray'),
            alpha=0.85,
            hatch=hatch_map.get(model, None)
        )
        
        yerr = np.array([err_low, err_high])
        ax.errorbar(
            x + offset,
            values,
            yerr=yerr,
            fmt='none',
            ecolor='black',
            elinewidth=1.2,
            capsize=4,
            capthick=1
        )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Cancer" if show_xlabel else "")
    ax.set_xticks(x)
    ax.set_xticklabels(cancer_labels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if use_fixed_y_axis:
        if metric_name == "C-Index":
            ax.set_ylim(0.5, 1.0)
        elif metric_name == "Brier Score":
            ax.set_ylim(0.00, 0.39)
            ax.set_yticks(np.arange(0.00, 0.40, 0.05))
            ax.set_yticklabels([f"{tick:.2f}" for tick in np.arange(0.00, 0.40, 0.05)])
        elif metric_name == "cMAE":
            ax.set_ylim(0, 13)
            ax.set_yticks([0, 2, 4, 6, 8, 10, 12])

    if show_legend:
        # Use 3 columns for legend when there are many models
        ncol = 3 if len(model_names) > 4 else 1
        ax.legend(fontsize=8, loc='upper right', ncol=ncol)
    elif ax.get_legend():
        ax.get_legend().remove()


def plot_all_models_metrics(
    plot_dir: str,
    dataset: str,
    cancer_of_interest: List[str],
    prompting_task: str,
    race_inclusion_path: str = 'incl_race',
    system_prompt_path: str = 'system'
):
    """
    Generate 4-subplot figure showing all metrics for all models (like supplementary figures).
    
    Args:
        plot_dir: Directory to save plots
        dataset: Dataset name
        cancer_of_interest: List of cancer types
        prompting_task: Either 'SURV_PROB' or 'TTE_OS'
        race_inclusion_path: Path to race inclusion criteria
        system_prompt_path: Path to system prompt
    """
    # Initialize metric dictionaries
    c_index_stats = {}
    pseudo_c_stats = {}
    pseudo_c_w_stats = {}
    brier_score_stats = {}
    mae_stats = {}
    cmae_stats = {}
    dcal_stats = {}

    # Get all model names from first cancer type to determine what models to plot
    ca_first = cancer_of_interest[0]
    cfg = ExperimentConfig(
            base_dir=BASE_DIR,
            data_name=dataset,
            cancer_of_interest=ca_first,
        )
    ca_datasets_sample = process_data_for_km(
        cfg, [prompting_task], 
        get_train_times=True, 
        race_inclusion_path=race_inclusion_path,
        system_prompt_path=system_prompt_path
    )
    
    # Filter models like in km_curves supplementary figures (all 0-shot models, no Temp)
    all_model_labels = []
    for _, _, label in ca_datasets_sample:
        if is_all_models_zero_shot_label(label, prompting_task):
                all_model_labels.append(label)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_model_labels = []
    for label in all_model_labels:
        if label not in seen:
            seen.add(label)
            unique_model_labels.append(label)

    # Plot models in a consistent left-to-right order across every task/figure.
    unique_model_labels.sort(key=canonical_model_sort_key)

    print(f"Processing all models for {prompting_task}...")
    print(f"Models: {unique_model_labels}")
    
    time_points = np.arange(0.5, 10.1, 0.5)

    # Load each cancer's data once, then fan the per-(cancer, model) bootstrap
    # out across worker processes (each task gets a minimal, cheap-to-pickle slice
    # of the data; RSF curves are only sent to the RSF task that needs them).
    tasks = []
    for ca in cancer_of_interest:
        print(f"Loading {ca}...")
        cfg = ExperimentConfig(base_dir=BASE_DIR, data_name=dataset, cancer_of_interest=ca)
        ca_datasets = process_data_for_km(
            cfg, [prompting_task], get_train_times=True,
            race_inclusion_path=race_inclusion_path, system_prompt_path=system_prompt_path,
        )
        selected_datasets = [
            x for x in ca_datasets
            if x[2] in ['Ground Truth', 'Training Times/Events'] or x[2] in unique_model_labels
        ]
        rsf_curves = None
        if any('Baseline: RSF' in m for m in unique_model_labels):
            try:
                rsf_curves = load_rsf_survival_curves(ca, BASE_DIR, data_name=dataset)
            except Exception as exc:
                print(f"  [warn] could not load RSF curves for {ca}: {exc}")
        for model_label in unique_model_labels:
            rc = rsf_curves if 'Baseline: RSF' in model_label else None
            tasks.append((ca, model_label, _minimal_datasets(selected_datasets, model_label), rc))

    print(f"Computing {len(tasks)} (cancer, model) metric sets on {N_WORKERS} workers...")
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=N_WORKERS)(
        delayed(_bar_metrics_for_model)(ds, model_label, time_points,
                                        BOOTSTRAP_SAMPLES, CONFIDENCE_ALPHA, rc)
        for (_ca, model_label, ds, rc) in tasks
    )

    for ca in cancer_of_interest:
        for d in (c_index_stats, pseudo_c_stats, pseudo_c_w_stats, brier_score_stats,
                  mae_stats, cmae_stats, dcal_stats):
            d[ca] = {}
    for (ca, model_label, _ds, _rc), m in zip(tasks, results):
        c_index_stats[ca][model_label] = m['c_index']
        pseudo_c_stats[ca][model_label] = m['pseudo_c_index']
        pseudo_c_w_stats[ca][model_label] = m['pseudo_c_index_weighted']
        brier_score_stats[ca][model_label] = m['brier_score']
        mae_stats[ca][model_label] = m['mae']
        cmae_stats[ca][model_label] = m['cmae']
        dcal_stats[ca][model_label] = m['d_cal']

    # Main all-models figure: C-Index, Integrated Brier, cMAE.
    main_metrics = [
        (c_index_stats, "C-Index", "c-index"),
        (brier_score_stats, "Brier Score", "IBS"),
        (cmae_stats, "cMAE", "cMAE (years)"),
    ]
    fig, axes = plt.subplots(len(main_metrics), 1, figsize=(10, 5 * len(main_metrics)))
    for i, (stats, name, ylabel) in enumerate(main_metrics):
        plot_metric_barplot(stats, cancer_of_interest, unique_model_labels,
                            name, ylabel, axes[i], show_legend=(i == 0),
                            simplify_legend_labels=True)
    plt.tight_layout()
    output_path = os.path.join(plot_dir, f"all_metrics_all_models_{prompting_task}.pdf")
    plt.savefig(output_path, bbox_inches='tight', format="pdf")
    print(f"Saved: {output_path}")
    plt.close(fig)
    save_source_data(output_path, metric_stats_to_long(
        {
            "C-Index": c_index_stats,
            "Integrated Brier Score": brier_score_stats,
            "cMAE (years)": cmae_stats,
        },
        cancer_of_interest, unique_model_labels,
    ))

    # Supplementary all-models figure: MAE, PseudoObs C-Index (unweighted +
    # weighted), D-calibration.
    supp_metrics = [
        (mae_stats, "MAE", "Mean Absolute Error (years)"),
        (pseudo_c_stats, "PseudoObs C-Index", "PseudoObs C-Index"),
        (pseudo_c_w_stats, "PseudoObs C-Index", "PseudoObs C-Index (weighted)"),
        (dcal_stats, "D-Calibration", "D-Calibration p-value (↑ better)"),
    ]
    fig, axes = plt.subplots(len(supp_metrics), 1, figsize=(10, 5 * len(supp_metrics)))
    for i, (stats, name, ylabel) in enumerate(supp_metrics):
        plot_metric_barplot(stats, cancer_of_interest, unique_model_labels,
                            name, ylabel, axes[i], show_legend=(i == 0),
                            simplify_legend_labels=True)
    plt.tight_layout()
    supp_path = os.path.join(plot_dir, f"all_metrics_all_models_{prompting_task}_supplementary.pdf")
    plt.savefig(supp_path, bbox_inches='tight', format="pdf")
    print(f"Saved: {supp_path}")
    plt.close(fig)
    save_source_data(supp_path, metric_stats_to_long(
        {
            "MAE (years)": mae_stats,
            "PseudoObs C-Index": pseudo_c_stats,
            "PseudoObs C-Index (weighted)": pseudo_c_w_stats,
            "D-Calibration p-value": dcal_stats,
        },
        cancer_of_interest, unique_model_labels,
    ))


def main():
    """
    Main function to generate the 4-subplot figure with all metrics.
    """
    # Data selection
    dataset = DEFAULT_DATA_NAME
    cancer_of_interest = ['nsclc', 'brca', 'crc', 'panc', 'prostate']
    prompting_tasks = ['TTE_OS', 'SURV_PROB']
    race_inclusion_path = 'incl_race'
    system_prompt_path = 'system'

    # Define PLOT_DIR
    PLOT_DIR = os.path.join(BASE_DIR, "plots", "all_metrics", dataset, race_inclusion_path)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # ========== ORIGINAL FIGURE: Baseline + Survprompt only ==========
    print("=" * 60)
    print("GENERATING ORIGINAL FIGURE: Baseline + Survprompt")
    print("=" * 60)
    
    # Model names for plotting
    model_names = [
        'Random Survival Forest',
        'Survprompt'
    ]
    
    time_points = np.arange(0.5, 10.1, 0.5)

    # Initialize metric dictionaries
    c_index_stats = {ca: {} for ca in cancer_of_interest}
    pseudo_c_stats = {ca: {} for ca in cancer_of_interest}
    pseudo_c_w_stats = {ca: {} for ca in cancer_of_interest}
    brier_score_stats = {ca: {} for ca in cancer_of_interest}
    mae_stats = {ca: {} for ca in cancer_of_interest}
    cmae_stats = {ca: {} for ca in cancer_of_interest}
    rmst_stats = {ca: {} for ca in cancer_of_interest}
    dcal_stats = {ca: {} for ca in cancer_of_interest}

    # Build (cancer, model) tasks (10 total) and fan out across workers.
    tasks = []
    survprompt_label = None
    for ca in cancer_of_interest:
        print(f"Loading {ca}...")
        cfg = ExperimentConfig(base_dir=BASE_DIR, data_name=dataset, cancer_of_interest=ca)
        ca_datasets = process_data_for_km(cfg, prompting_tasks, get_train_times=True,
                                          race_inclusion_path=race_inclusion_path,
                                          system_prompt_path=system_prompt_path)
        if survprompt_label is None:
            survprompt_label = resolve_survprompt_headline_label(x[2] for x in ca_datasets)
        selected_datasets = [
            x for x in ca_datasets if x[2] in [
                'Ground Truth', 'Baseline: RSF',
                survprompt_label, "Training Times/Events"]
        ]
        try:
            rsf_curves = load_rsf_survival_curves(ca, BASE_DIR, data_name=dataset)
        except Exception as exc:
            print(f"  [warn] could not load RSF curves for {ca}: {exc}")
            rsf_curves = None
        train_ds = [x for x in selected_datasets if x[2] == "Training Times/Events"][0]
        km_t, km_s = compute_km_curve(train_ds[0].astype(float).values / 365.25,
                                      train_ds[1].astype(int).values)
        for label in ('Baseline: RSF', survprompt_label):
            rc = rsf_curves if 'Baseline: RSF' in label else None
            tasks.append((ca, label, _minimal_datasets(selected_datasets, label), rc, km_t, km_s))

    label_to_display = {
        'Baseline: RSF': 'Random Survival Forest',
        survprompt_label: 'Survprompt',
    }

    print(f"Computing {len(tasks)} (cancer, model) metric sets on {N_WORKERS} workers...")
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=N_WORKERS)(
        delayed(_comparison_metrics_for_model)(ds, label, time_points,
                                               BOOTSTRAP_SAMPLES, CONFIDENCE_ALPHA, rc,
                                               km_t, km_s, RMST_TAU_REP)
        for (_ca, label, ds, rc, km_t, km_s) in tasks
    )
    for (ca, label, _ds, _rc, _kt, _ks), m in zip(tasks, results):
        disp = label_to_display[label]
        c_index_stats[ca][disp] = m['c_index']
        pseudo_c_stats[ca][disp] = m['pseudo_c_index']
        pseudo_c_w_stats[ca][disp] = m['pseudo_c_index_weighted']
        brier_score_stats[ca][disp] = m['brier_score']
        mae_stats[ca][disp] = m['mae']
        cmae_stats[ca][disp] = m['cmae']
        rmst_stats[ca][disp] = m['rmst']
        dcal_stats[ca][disp] = m['d_cal']

    # Main comparison figure: C-Index, Integrated Brier, cMAE.
    main_metrics = [
        (c_index_stats, "C-Index", "c-index"),
        (brier_score_stats, "Brier Score", "IBS"),
        (cmae_stats, "cMAE", "cMAE (years)"),
    ]
    fig, axes = plt.subplots(len(main_metrics), 1, figsize=(6, 5 * len(main_metrics)))
    for i, (stats, name, ylabel) in enumerate(main_metrics):
        plot_metric_barplot(stats, cancer_of_interest, model_names,
                            name, ylabel, axes[i], show_legend=(i == 0),
                            use_fixed_y_axis=True)
    plt.tight_layout()
    comparison_pdf = os.path.join(PLOT_DIR, "all_metrics_comparison.pdf")
    plt.savefig(comparison_pdf, bbox_inches='tight', format="pdf")
    print(f"Plot saved to {comparison_pdf}")
    plt.close(fig)
    save_source_data(comparison_pdf, metric_stats_to_long(
        {
            "C-Index": c_index_stats,
            "Integrated Brier Score": brier_score_stats,
            "cMAE (years)": cmae_stats,
        },
        cancer_of_interest, model_names,
    ))

    # Supplementary figure: MAE, PseudoObs C-Index (unweighted + weighted), D-cal.
    supp_metrics = [
        (mae_stats, "MAE", "Mean Absolute Error (years)"),
        (pseudo_c_stats, "PseudoObs C-Index", "PseudoObs C-Index"),
        (pseudo_c_w_stats, "PseudoObs C-Index", "PseudoObs C-Index (weighted)"),
        (dcal_stats, "D-Calibration", "D-Calibration p-value (↑ better)"),
    ]
    fig, axes = plt.subplots(len(supp_metrics), 1, figsize=(6, 5 * len(supp_metrics)))
    for i, (stats, name, ylabel) in enumerate(supp_metrics):
        plot_metric_barplot(stats, cancer_of_interest, model_names,
                            name, ylabel, axes[i], show_legend=(i == 0))
    plt.tight_layout()
    supp_pdf = os.path.join(PLOT_DIR, "all_metrics_comparison_supplementary.pdf")
    plt.savefig(supp_pdf, bbox_inches='tight', format="pdf")
    print(f"Plot saved to {supp_pdf}")
    plt.close(fig)
    save_source_data(supp_pdf, metric_stats_to_long(
        {
            "MAE (years)": mae_stats,
            "PseudoObs C-Index": pseudo_c_stats,
            "PseudoObs C-Index (weighted)": pseudo_c_w_stats,
            "D-Calibration p-value": dcal_stats,
        },
        cancer_of_interest, model_names,
    ))
    
    # ========== NEW FIGURES: All models for each prompting task ==========
    print("\n" + "=" * 60)
    print("GENERATING SUPPLEMENTARY FIGURES: All Models")
    print("=" * 60)
    
    # Generate figure for SURV_PROB (all models)
    plot_all_models_metrics(
        plot_dir=PLOT_DIR,
        dataset=dataset,
        cancer_of_interest=cancer_of_interest,
        prompting_task='SURV_PROB',
        race_inclusion_path=race_inclusion_path,
        system_prompt_path=system_prompt_path
    )
    
    # Generate figure for TTE_OS (all models)
    plot_all_models_metrics(
        plot_dir=PLOT_DIR,
        dataset=dataset,
        cancer_of_interest=cancer_of_interest,
        prompting_task='TTE_OS',
        race_inclusion_path=race_inclusion_path,
        system_prompt_path=system_prompt_path
    )
    
    print("\n" + "=" * 60)
    print("ALL FIGURES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
