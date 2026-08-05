from typing import List, Dict, Tuple, Optional, Any
import os
import warnings
import pandas as pd
import numpy as np
import ast
from sksurv.metrics import concordance_index_censored
from scipy.stats import chi2, chisquare
from survprompt.evaluation.synthcity_utils import nonparametric_distance
from survprompt.defaults import DEFAULT_DATA_NAME
try:
    from scipy.integrate import trapz
except ImportError:
    try:
        from numpy import trapz  # NumPy < 2.0
    except ImportError:
        from numpy import trapezoid as trapz  # NumPy >= 2.0

DEFAULT_MAX_TRUNCATION_TIME_YEARS = 30.0

def interpolate_time_at_threshold(times, surv_probs, threshold=0.5):
    """
    Given times and survival probabilities, find the time when survival probability crosses the threshold.
    Uses linear interpolation between the two nearest points.
    """
    times = np.asarray(times)
    surv_probs = np.asarray(surv_probs)
    if len(times) != len(surv_probs):
        return None
    
    # Find indices where surv_prob crosses the threshold
    above = surv_probs >= threshold
    below = surv_probs < threshold
    crossing_indices = np.where(above[:-1] & below[1:])[0]
    if len(crossing_indices) == 0:
        if len(surv_probs) > 0 and surv_probs[0] < threshold:
            return times[0]
        elif len(surv_probs) > 0 and surv_probs[-1] >= threshold:
            return times[-1]
        return None  # No crossing found
    idx = crossing_indices[0]
    # Linear interpolation
    t1, t2 = times[idx], times[idx+1]
    s1, s2 = surv_probs[idx], surv_probs[idx+1]
    if s1 == s2:
        return t1
    t_cross = t1 + (threshold - s1) * (t2 - t1) / (s2 - s1)
    return t_cross

def get_metrics (save_df: pd.DataFrame,
                 pred_ids_1: List[str],
                 pred_ids_2: List[str]) -> Dict[str, Tuple[float, float]]:
    metrics_dict = {}
    subset_metrics_dict = {}
    
    # C-index
    metrics_dict['c_index'] = calculate_c_index(save_df)

    # print sub-cohort c-indexes as well
    subset_metrics_dict['c_index'] = assess_on_subcohorts(save_df, {'subcohort_1': pred_ids_1, 'subcohort_2': pred_ids_2})

    return metrics_dict, subset_metrics_dict

def assess_on_subcohorts(save_df: pd.DataFrame, subcohorts: List[str]):
    """Assess c-index on subcohorts"""
    cindex_subcohorts = {}
    pred_cols =  [col for col in save_df.columns if col.startswith('pred_')]
    for subcohort_name, subcohort_ids in subcohorts.items():
        cindex_subcohorts[subcohort_name] = {}
        for pred_col in pred_cols:
            df = save_df.copy()
            df.loc[:, 'incohort'] = df.sample_ids.apply(lambda x: x in subcohort_ids.values)
            df = df[df.incohort]
            if len(df) == 0:
                print(f"Warning: no samples in subcohort {subcohort_name}")
                cindex_subcohorts[subcohort_name][pred_col] = None
            else:
                n_na = df[df[pred_col].isna()].shape[0]
                df = df[~df[pred_col].isna()]
                if n_na > 0:
                    print(f"Warning: skipping {n_na} NaN predictions for c-index at {pred_col}")
                print(f"Predicting for subset of n={df.shape[0]} patients at {pred_col}")
                c_idx_res = concordance_index_censored(event_indicator=df['dead_nonlt'].astype(bool), 
                                                       event_time=df['stop_nonlt'], 
                                                       estimate=-df[pred_col])
                print(f"C-index at {pred_col}: {c_idx_res[0]:.5f}")
                cindex_subcohorts[subcohort_name][pred_col] = c_idx_res[0]
    return cindex_subcohorts

def get_comprehensive_metrics(
    predictions: Dict[str, Any],
    ground_truth: Dict[str, Any],
    save_df: Optional[pd.DataFrame] = None,
    pred_ids_1: Optional[List[str]] = None,
    pred_ids_2: Optional[List[str]] = None,
    include_brier: bool = False,
    include_distance: bool = False
) -> Dict[str, Any]:
    """
    Enhanced version of get_metrics with additional survival analysis metrics.
    """
    # Get basic metrics using existing function if save_df is provided
    if save_df is not None and pred_ids_1 is not None and pred_ids_2 is not None:
        metrics_dict, subset_metrics_dict = get_metrics(save_df, pred_ids_1, pred_ids_2)
    else:
        metrics_dict = {}
        subset_metrics_dict = {}
    
    # Add comprehensive metrics
    comprehensive_metrics = {
        'basic_metrics': metrics_dict,
        'subset_metrics': subset_metrics_dict
    }
    
    if include_brier and 'brier_data' in predictions:
        brier_data = predictions['brier_data']
        comprehensive_metrics['brier_score'] = calculate_brier_score(**brier_data)
    
    if include_distance and 'distance_data' in predictions:
        distance_data = predictions['distance_data']
        comprehensive_metrics['nonparametric_distance'] = nonparametric_distance(**distance_data)
    
    return comprehensive_metrics

###############
# C-index
###############

def calculate_c_index(df: pd.DataFrame, year=None):
    """Calculate c-index for the entire cohort from saved output df"""
    cindex = {}

    if 'pred_num_days' in df.columns:
        # TTE predictions - for TTE_OS, our predictions are in the pred_num_days column (predicted number of days until event) 
        pred_type = 'TTE'
        pred_col = 'pred_num_days'
    elif 'pred_prob' in df.columns:
        # Risk predictions - for SURV_PROB, our predictions are in the pred_prob column (survival probability for each specified time point)
        pred_type = 'risk'
        pred_col = 'pred_prob'
    else:
        raise ValueError("No valid prediction column found in the DataFrame. Expected 'pred_num_days' or 'pred_prob'.")

    # Skip na
    num_na = df[df[pred_col].isna()].shape[0]
    if num_na > 0:
        print(f"Warning: skipping {num_na} NaN predictions for c-index at {pred_col}")
    df = df[~df[pred_col].isna()]

    # Get estimates of risk
    if pred_type == 'TTE':
        risk_estimate = -df[pred_col]
    elif pred_type == 'risk':
        risk_estimate = _calculate_estimate_for_risk(df, year=year)
    
    c_idx_res = concordance_index_censored(event_indicator=df['dead_nonlt'].astype(bool),
                                            event_time=df['stop_nonlt'],
                                            estimate=risk_estimate)
    cindex[pred_col] = c_idx_res[0]
    return cindex

def calculate_c_index_ipcw(
    T_train: np.ndarray,
    Y_train: np.ndarray,
    Prediction: np.ndarray,
    T_test: np.ndarray,
    Y_test: np.ndarray,
    Time: float,
) -> float:
    """Helper for evaluating the C-INDEX metric."""
    from sksurv.metrics import concordance_index_ipcw as sksurv_concordance_index_ipcw

    T_train = pd.Series(T_train)
    Y_train = pd.Series(Y_train)
    T_test = pd.Series(T_test)
    Y_test = pd.Series(Y_test)
    Prediction = np.asarray(Prediction).squeeze()

    Y_train_structured = [
        (Y_train.iloc[i], T_train.iloc[i]) for i in range(len(Y_train))
    ]
    Y_train_structured = np.array(
        Y_train_structured, dtype=[("status", "bool"), ("time", "<f8")]
    )

    Y_test_structured = [(Y_test.iloc[i], T_test.iloc[i]) for i in range(len(Y_test))]
    Y_test_structured = np.array(
        Y_test_structured, dtype=[("status", "bool"), ("time", "<f8")]
    )

    # concordance_index_ipcw expects risk scores
    return sksurv_concordance_index_ipcw(Y_train_structured, Y_test_structured, Prediction, tau=Time)[0]


def _calculate_estimate_for_risk(df, year):
    """
    Calculate the estimate for risk predictions restricted_mean survival time for entire duration, or get risk at a specific year.
    """
    df_time = df['pred_time'].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)
    df_col = df['pred_prob'].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)

    if year is None:
        # Calculate risk from RMST (inverse RMST) for the entire duration - summarise the entire curve in one number (area under the curve)
        # Use each row's own time_grid to handle varying prediction lengths
        risk_from_rmst = pd.Series(
            [-trapz(prob, np.asarray(time, dtype=float)) for prob, time in zip(df_col, df_time)],
            index=df_col.index
        )
        return risk_from_rmst
    else:
        # Risk at a specific year - use the first row's time_grid to find the index
        time_grid = np.asarray(df_time.iloc[0], dtype=float)
        idx = np.abs(time_grid - year).argmin()
        risk_at_year = df_col.apply(lambda x: 1 - x[idx] if isinstance(x, list) and idx < len(x) else np.nan)
        return risk_at_year
    
###############
# Optimism
###############

def calculate_optimism(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate optimism metrics comparing predictions against a ground truth dataset.
    
    Args:
        df: DataFrame containing
            - pred_num_days or pred_prob: predicted time to event (TTE) in days or survival probabilities
            - pred_event: predicted event indicators (if provided)
            - stop_nonlt: ground truth event times
            - dead_nonlt: ground truth event indicators

    Returns:
        Dictionary containing optimism metrics
    """
    optimism = {}

    T_base = df['stop_nonlt'].astype(int)
    E_base = df['dead_nonlt'].astype(int)
    E_pred = df['pred_event'] if 'pred_event' in df.columns else E_base # Assume the event predictions are the same as ground truth if not provided

    if 'pred_num_days' in df.columns:
        # TTE predictions - for TTE_OS, our predictions are in the pred_num_days column
        T_pred = df['pred_num_days'].astype(int)
        auc_opt, auc_abs_opt, sightedness = nonparametric_distance(
            (T_base, E_base),
            (T_pred, E_pred)
        )
        
    elif 'pred_prob' in df.columns:
        # Risk predictions - survival probabilities at specified time points
        T_pred = df['pred_prob']

        all_time_points = []
        all_survival_probs = []
        for time_points, probs in T_pred.values:
            # ax.plot(time_points, probs, label=label)
            if len(time_points) > 0:
                all_time_points.append(time_points)
                all_survival_probs.append(probs)

        time_grid = np.linspace(0, max([max(times) for times in all_time_points]), len(T_base))
        interpolated_survivals = []
        for times, probs in zip(all_time_points, all_survival_probs):
            # linear interpolation
            interpolated_survivals.append(np.interp(time_grid, times, probs))

        # mean predicted survival curve and bounds
        mean_predicted_survival = np.mean(interpolated_survivals, axis=0)
        
        auc_opt, auc_abs_opt, sightedness = nonparametric_distance(
            (T_base, E_base),
            (dict(zip(time_grid*365, mean_predicted_survival)), E_pred),  # Convert time to days
            is_syn_type='prob'
        )

    else:
        raise ValueError("No valid prediction column found in the DataFrame. Expected 'pred_num_days' or 'pred_prob'.")
    
    optimism['auc_opt'] = auc_opt
    optimism['auc_abs_opt'] = auc_abs_opt
    optimism['sightedness'] = sightedness
    
    return optimism

###############
# Brier Score
###############

def calculate_brier_score(
    T_train: np.ndarray,
    Y_train: np.ndarray,
    Prediction: np.ndarray,
    T_test: np.ndarray,
    Y_test: np.ndarray,
    Time: float,
) -> float:
    """Helper for evaluating the Brier score."""
    from sksurv.metrics import brier_score as sksurv_brier_score

    T_train = pd.Series(T_train)
    Y_train = pd.Series(Y_train)
    T_test = pd.Series(T_test)
    Y_test = pd.Series(Y_test)

    Y_train_structured = [
        (Y_train.iloc[i], T_train.iloc[i]) for i in range(len(Y_train))
    ]
    Y_train_structured = np.array(
        Y_train_structured, dtype=[("status", "bool"), ("time", "<f8")]
    )

    Y_test_structured = [(Y_test.iloc[i], T_test.iloc[i]) for i in range(len(Y_test))]
    Y_test_structured = np.array(
        Y_test_structured, dtype=[("status", "bool"), ("time", "<f8")]
    )

    # brier_score expects survival scores
    return sksurv_brier_score(Y_train_structured, Y_test_structured, 1 - Prediction, times=Time)[0]

###############
# MAE
###############

def compute_km_curve(event_times: np.ndarray, event_indicators: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the Kaplan-Meier survival curve."""
    unique_times, event_counts = np.unique(event_times[event_indicators == 1], return_counts=True)
    population_at_risk = np.array([np.sum(event_times >= t) for t in unique_times])
    survival_probs = np.cumprod(1 - event_counts / population_at_risk)
    return unique_times, survival_probs

def restricted_mean_survival_time(times: np.ndarray, survival_probs: np.ndarray) -> float:
    """Computes the restricted mean survival time (RMST)."""
    time_diffs = np.diff(np.insert(times, 0, 0))
    avg_probs = (np.insert(survival_probs, 0, 1)[:-1] + np.insert(survival_probs, 0, 1)[1:]) / 2
    return np.sum(time_diffs * avg_probs)

def calculate_error(
        predicted_times: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        train_event_times: Optional[np.ndarray] = None,
        train_event_indicators: Optional[np.ndarray] = None,
        truncation_time_years: Optional[float] = None,
) -> np.ndarray:
    """
    Calculates the signed error (predicted - observed) for survival predictions.

    With training data provided, each censored test patient's observed time is
    replaced by the Pseudo-observation surrogate (the jackknife pseudo-value of
    the Kaplan-Meier mean), matching the reference cMAE implementation in
    ``SurvivalEVAL`` (``method="Pseudo_obs"``); see ``pseudo_obs_surrogate_times``
    below. Note this returns *unweighted* signed errors; the canonical cMAE
    additionally applies confidence weights
    (``1 - S_KM(c)``) -- callers that need the weighted metric should use
    ``censored_mae`` / ``cmae_components`` instead.
    """
    if train_event_times is not None and train_event_indicators is not None:
        estimated_event_times = pseudo_obs_surrogate_times(
            event_times, event_indicators, train_event_times, train_event_indicators,
            truncation_time_years=truncation_time_years,
        )
        errors = np.asarray(predicted_times, dtype=float) - estimated_event_times
    else:
        errors = predicted_times - event_times
    return errors

def get_mae(
        errors: np.ndarray,
) -> float:
    """
    Computes the Mean Absolute Error (MAE) for survival predictions.
    """
    return np.mean(np.abs(errors))

def calculate_error_metrics(
    predicted_times: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: Optional[np.ndarray] = None,
    train_event_indicators: Optional[np.ndarray] = None,
    metric_type: str = "mae",
    truncation_time_years: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate various error metrics for survival predictions.
    Supports MAE, cMAE (conditional MAE), MSE, and RMSE.
    """
    results = {}

    if metric_type.lower() == "mae":
        errors = calculate_error(predicted_times, event_times, event_indicators)
        results['mae'] = get_mae(errors)
    elif metric_type.lower() == "cmae":
        errors = calculate_error(
            predicted_times, event_times, event_indicators,
            train_event_times, train_event_indicators,
            truncation_time_years=truncation_time_years,
        )
        results['cmae'] = get_mae(errors)
    elif metric_type.lower() == "mse":
        mse = np.mean((predicted_times - event_times) ** 2)
        results['mse'] = mse
        results['rmse'] = np.sqrt(mse)
    return results


# ===========================================================================
# Censored MAE (cMAE) via the Pseudo-observation surrogate
#
# Implementation of the censored-MAE used by Qi et al.,
# "Effective Ways to Build and Evaluate Individual Survival Distributions"
# and its repo CensoredMAE (https://github.com/shi-ang/CensoredMAE). A censored
# test patient is decensored with the jackknife *pseudo-observation* of the
# Kaplan-Meier mean survival time (Andersen, Klein & Rosthoj, Biometrics 2003):
# the patient's contribution to the cohort mean, (n+1)*mu_with - n*mu_train,
# where mu_with adds that subject (at risk until its censoring time) to the
# training KM. A confidence weight 1 - S_KM(c) down-weights patients censored
# early, where the pseudo-value is least certain (Haider et al., JMLR 2020).
# ===========================================================================

def _km_product_limit(times, events):
    """Kaplan-Meier product-limit estimator over all distinct observed times.

    Returns ``(distinct_times, at_risk, n_events, survival)`` where ``at_risk``
    and ``n_events`` are the number at risk and the number of events at each
    distinct time, and ``survival`` is the right-continuous S(t).
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    distinct = np.unique(times)
    at_risk = np.array([(times >= u).sum() for u in distinct], dtype=float)
    n_events = np.array([events[times == u].sum() for u in distinct], dtype=float)
    survival = np.cumprod(1.0 - n_events / at_risk)
    return distinct, at_risk, n_events, survival


def _km_curve_mean(times, survival):
    """Mean of a KM curve = area under S(t). When S does not reach 0 the tail is
    closed off with a straight line to S=0 (so the mean is finite); area by the
    trapezoidal rule over (0, 1) -> (times, survival) -> (t_zero, 0)."""
    t = np.concatenate([[0.0], np.asarray(times, dtype=float)])
    s = np.concatenate([[1.0], np.asarray(survival, dtype=float)])
    if s[-1] > 0:
        t_zero = t[-1] / (1.0 - s[-1])  # where the (0,1)->(t_last,S_last) line hits 0
        t = np.concatenate([t, [t_zero]])
        s = np.concatenate([s, [0.0]])
    return float(trapz(s, t))


def _km_survival_at(distinct, survival, query):
    """S(query) for the step KM; once past the last distinct time it is
    extrapolated along the (first point)->(last point) line, floored at 0."""
    query = np.asarray(query, dtype=float)
    idx = np.searchsorted(distinct, query, side="right") - 1
    out = np.where(idx >= 0, survival[np.clip(idx, 0, survival.size - 1)], 1.0)
    beyond = query > distinct[-1]
    if np.any(beyond):
        span = distinct[-1] - distinct[0]
        slope = 0.0 if span == 0 else (survival[-1] - survival[0]) / span
        out = np.where(beyond, np.maximum(survival[-1] + slope * (query - distinct[-1]), 0.0), out)
    return out


def pseudo_obs_surrogate_times(
    event_times,
    event_indicators,
    train_event_times,
    train_event_indicators,
    truncation_time_years: Optional[float] = None,
) -> np.ndarray:
    """Per-sample surrogate times: the observed time for events, the KM-mean
    pseudo-observation for censored patients (see section header).

    If ``truncation_time_years`` is ``None``, censored pseudo-values are clipped at
    ``min(max training follow-up, 30 years)`` to avoid unsupported tail
    extrapolation. A warning is emitted when the inferred training follow-up
    exceeds 30, since that may indicate day-based inputs or a caller that should
    override the default cap explicitly.
    """
    event_times = np.asarray(event_times, dtype=float)
    event_indicators = np.asarray(event_indicators).astype(bool)
    train_event_times = np.asarray(train_event_times, dtype=float)
    train_event_indicators = np.asarray(train_event_indicators).astype(bool)
    n_train = train_event_times.size
    max_train_followup = float(np.max(train_event_times))
    if truncation_time_years is None:
        if max_train_followup > DEFAULT_MAX_TRUNCATION_TIME_YEARS:
            warnings.warn(
                "pseudo_obs_surrogate_times inferred `truncation_time_years` from training times, "
                f"but max_train_followup={max_train_followup:.3f} exceeds the default cap of "
                f"{DEFAULT_MAX_TRUNCATION_TIME_YEARS:.1f}. If these inputs are in days, convert them "
                "to years before calling this helper; otherwise pass `truncation_time_years` explicitly "
                "or adjust the default cap.",
                stacklevel=2,
            )
        truncation_time_years = min(max_train_followup, DEFAULT_MAX_TRUNCATION_TIME_YEARS)

    distinct, at_risk, n_events, survival = _km_product_limit(
        train_event_times, train_event_indicators
    )

    # The mean only needs the times where S drops (the events) plus the final
    # follow-up time, which anchors the straight-line tail.
    keep = np.flatnonzero(n_events > 0)
    if keep.size == 0:
        raise ValueError("Pseudo-observation cMAE needs at least one training event.")
    if keep[-1] != distinct.size - 1:
        keep = np.append(keep, distinct.size - 1)
    ev_times, ev_at_risk, ev_n_events, ev_surv = (
        distinct[keep], at_risk[keep], n_events[keep], survival[keep],
    )
    mu_train = _km_curve_mean(ev_times, ev_surv)

    # mu_with(c): mean of the training KM after adding one censored subject at
    # time c. That subject is at risk for every event time <= c, so the at-risk
    # count rises by one over that prefix; recompute the product limit and mean.
    # Surrogates depend only on the censoring time, so compute once per value.
    surrogate = event_times.copy()
    for c in np.unique(event_times[~event_indicators]):
        prefix = int(np.searchsorted(ev_times, c, side="right"))
        at_risk_aug = ev_at_risk.copy()
        at_risk_aug[:prefix] += 1.0
        surv_aug = np.cumprod(1.0 - ev_n_events / at_risk_aug)
        if prefix == ev_times.size:
            # c extends follow-up past the last event: carry S flat out to c.
            mu_with = _km_curve_mean(np.append(ev_times, c), np.append(surv_aug, surv_aug[-1]))
        else:
            mu_with = _km_curve_mean(ev_times, surv_aug)
        surrogate[event_times == c] = (n_train + 1) * mu_with - n_train * mu_train

    surrogate[event_indicators] = event_times[event_indicators]
    censored = ~event_indicators
    if truncation_time_years is not None and np.any(censored):
        surrogate[censored] = np.minimum(surrogate[censored], truncation_time_years)
    return surrogate


def censoring_confidence_weights(
    event_times,
    event_indicators,
    train_event_times,
    train_event_indicators,
) -> np.ndarray:
    """Per-sample weights: 1 for events, ``1 - S_KM(c)`` for a patient censored
    at time c (later censoring -> more certain surrogate -> larger weight)."""
    event_times = np.asarray(event_times, dtype=float)
    event_indicators = np.asarray(event_indicators).astype(bool)
    distinct, _, _, survival = _km_product_limit(train_event_times, train_event_indicators)
    weights = np.ones(event_times.size)
    censored = ~event_indicators
    if censored.any():
        weights[censored] = 1.0 - _km_survival_at(distinct, survival, event_times[censored])
    return weights


def cmae_components(
    predicted_times,
    event_times,
    event_indicators,
    train_event_times,
    train_event_indicators,
    truncation_time_years: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(abs_errors, weights)`` for weighted Pseudo-observation cMAE.

    The weighted mean of ``abs_errors`` is the cMAE; returning the components
    (rather than the scalar) lets callers bootstrap a weighted mean.
    """
    predicted = np.asarray(predicted_times, dtype=float)
    surrogate = pseudo_obs_surrogate_times(
        event_times, event_indicators, train_event_times, train_event_indicators,
        truncation_time_years=truncation_time_years,
    )
    weights = censoring_confidence_weights(
        event_times, event_indicators, train_event_times, train_event_indicators
    )
    return np.abs(predicted - surrogate), weights


def ccae_components(
    predicted_times,
    event_times,
    event_indicators,
    train_event_times,
    train_event_indicators,
    truncation_time_years: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(signed_errors, weights)`` for weighted censored average error.

    This matches ``cmae_components`` except it preserves the sign of
    ``predicted - surrogate`` instead of taking the absolute value.
    """
    predicted = np.asarray(predicted_times, dtype=float)
    surrogate = pseudo_obs_surrogate_times(
        event_times, event_indicators, train_event_times, train_event_indicators,
        truncation_time_years=truncation_time_years,
    )
    weights = censoring_confidence_weights(
        event_times, event_indicators, train_event_times, train_event_indicators
    )
    return predicted - surrogate, weights


def censored_mae(
    predicted_times,
    event_times,
    event_indicators,
    train_event_times,
    train_event_indicators,
    weighted: bool = True,
    truncation_time_years: Optional[float] = None,
) -> float:
    """Weighted (default) Pseudo-observation cMAE."""
    abs_errors, weights = cmae_components(
        predicted_times, event_times, event_indicators,
        train_event_times, train_event_indicators,
        truncation_time_years=truncation_time_years,
    )
    if not weighted or weights.sum() == 0:
        return float(np.mean(abs_errors))
    return float(np.average(abs_errors, weights=weights))


def weighted_concordance(time, risk, weight) -> float:
    """Confidence-weighted concordance on complete (uncensored) data.

    ``C_w = sum_{t_i<t_j} w_i w_j [1(r_i>r_j) + 0.5*1(r_i=r_j)] / sum_{t_i<t_j} w_i w_j``,
    with ``risk`` higher for shorter predicted survival. Each pair is weighted by
    the product of its endpoints' confidence weights, so a pair touching a
    low-confidence surrogate contributes little (cf. Uno's IPCW concordance,
    which also weights pairs). Computed in O(n log n) with a Fenwick tree over
    risk ranks, so it bootstraps cheaply.
    """
    time = np.asarray(time, dtype=float)
    risk = np.asarray(risk, dtype=float)
    weight = np.asarray(weight, dtype=float)
    n = time.size
    if n < 2:
        return np.nan
    ranks = np.unique(risk, return_inverse=True)[1] + 1  # 1-indexed risk ranks
    K = int(ranks.max())
    order = np.argsort(time, kind="mergesort")
    ts, rk, ws = time[order], ranks[order], weight[order]

    tree = np.zeros(K + 1)
    num = den = total = 0.0
    i = 0
    while i < n:
        # Patients sharing a time are not comparable to each other, so score the
        # whole time-group against the tree (strictly-earlier times) before adding
        # it to the tree.
        j = i
        while j < n and ts[j] == ts[i]:
            j += 1
        for k in range(i, j):
            rj = rk[k]
            s_le = 0.0
            idx = rj
            while idx > 0:
                s_le += tree[idx]
                idx -= idx & (-idx)
            s_lt = 0.0
            idx = rj - 1
            while idx > 0:
                s_lt += tree[idx]
                idx -= idx & (-idx)
            # concordant: earlier-failer has higher risk (rank > rk[k]).
            num += ws[k] * (total - s_le + 0.5 * (s_le - s_lt))
            den += ws[k] * total
        for k in range(i, j):
            idx = rk[k]
            while idx <= K:
                tree[idx] += ws[k]
                idx += idx & (-idx)
            total += ws[k]
        i = j
    return float(num / den) if den > 0 else np.nan


def pseudo_obs_concordance(
    predicted_times,
    event_times,
    event_indicators,
    train_event_times,
    train_event_indicators,
    weighted: bool = False,
    truncation_time_years: Optional[float] = None,
) -> float:
    """Concordance index after decensoring with pseudo-observations.

    Standard (Harrell) concordance only scores *comparable* pairs -- pairs where
    the shorter time is an observed event. Following the idea of Kumar et al.
    (Sci. Rep. 2022) of imputing censored times and then scoring an ordinary
    concordance, we replace every censored time with its pseudo-observation
    surrogate (see ``pseudo_obs_surrogate_times``) so that all patients are
    "complete" and every pair is comparable, then compute the C-statistic of the
    predicted risk against the completed times.

    With ``weighted=True`` each pair is weighted by ``w_i * w_j`` where
    ``w = 1 - S_KM(c)`` is the same surrogate-confidence weight used by the
    weighted cMAE, so pairs touching an uncertain (early-censored) surrogate
    count less.

    ``predicted_times`` are predicted survival times; risk is taken as their
    negative (longer predicted survival -> lower risk), matching the rest of the
    codebase.
    """
    predicted_times = np.asarray(predicted_times, dtype=float)
    surrogate = pseudo_obs_surrogate_times(
        event_times, event_indicators, train_event_times, train_event_indicators,
        truncation_time_years=truncation_time_years,
    )
    if weighted:
        w = censoring_confidence_weights(
            event_times, event_indicators, train_event_times, train_event_indicators
        )
        return weighted_concordance(surrogate, -predicted_times, w)
    all_events = np.ones(surrogate.size, dtype=bool)
    return float(concordance_index_censored(all_events, surrogate, -predicted_times)[0])


# ===========================================================================
# Restricted Mean Survival Time (RMST) error metric
#
# RMST(tau) = integral_0^tau S(t) dt. Used as an *error* metric by comparing a
# model's predicted RMST(tau) against a ground-truth RMST(tau) with KM imputation
# for patients censored before tau. Unit conventions: SURV_PROB pred curves and
# all RMST math are in YEARS; RSF sksurv survival functions and ground-truth /
# training observed times are in DAYS (converted on the way in).
# ===========================================================================

# Shared default horizons (years).
DEFAULT_TAUS: List[float] = [1.0, 3.0, 5.0, 10.0]
_RMST_GRID_POINTS = 200


def predicted_rmst_at_tau(times_yr, probs, tau: float) -> float:
    """Predicted RMST(tau) = integral_0^tau S(t) dt from a (times, probs) curve.

    ``times_yr`` and ``probs`` describe a survival curve with times in years.
    S(t) is held at 1.0 before the first point and at the last value beyond the
    last point.
    """
    t = np.asarray(times_yr, dtype=float)
    s = np.asarray(probs, dtype=float)
    if t.size == 0 or s.size == 0 or t.size != s.size:
        return np.nan
    grid = np.linspace(0.0, tau, _RMST_GRID_POINTS)
    s_on_grid = np.interp(grid, t, s, left=1.0, right=s[-1])
    return float(trapz(s_on_grid, grid))


def predicted_tte_rmst_at_tau(pred_time_days, tau: float) -> float:
    """RMST(tau) for a scalar TTE prediction represented as a step curve."""
    try:
        pred_time_yr = float(pred_time_days) / 365.25
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(pred_time_yr):
        return np.nan
    return float(np.clip(pred_time_yr, 0.0, tau))


def true_rmst_at_tau(t_obs_yr: float, event: int, km_t, km_s, tau: float) -> float:
    """Ground-truth RMST(tau) = E[min(T, tau) | observed data] for one patient.

    - If the patient is observed past tau, the truth is exactly tau.
    - If the event was observed before tau, the truth is the observed time.
    - If censored before tau, impute the residual area under the KM curve
      (built from the training cohort) conditional on survival to the censoring
      time, matching ``calculate_error``'s conditional-expectation approach.
    """
    if t_obs_yr >= tau:
        return float(tau)
    if event == 1:
        return float(t_obs_yr)
    insert_idx = np.searchsorted(km_t, t_obs_yr, side="right")
    if insert_idx == 0:
        s_at_c = 1.0
    elif insert_idx - 1 >= len(km_s):
        s_at_c = km_s[-1]
    else:
        s_at_c = km_s[insert_idx - 1]
    if s_at_c <= 0:
        return float(t_obs_yr)
    grid = np.linspace(t_obs_yr, tau, _RMST_GRID_POINTS)
    s_on_grid = np.interp(grid, km_t, km_s, left=1.0, right=km_s[-1])
    cond_area = trapz(s_on_grid, grid) / s_at_c
    return float(t_obs_yr + cond_area)


def load_rsf_survival_curves(
    cancer: str,
    base_dir: str,
    model_suffix: str = "fullfts",
    fold: int = 0,
    data_name: str = DEFAULT_DATA_NAME,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load the saved fold-0 RSF model and return per-patient survival curves.

    Returns ``{sample_id: (times_years, probs)}`` for the held-out test fold.
    Mirrors ``analysis/figures/reasoning_effort_comparison.py``'s ``_rsf_metrics``:
    it uses the model's own ``feature_names_in_`` so the extra ``has_{cancer}``
    indicator column added by ``get_tabular_data`` is dropped, and race columns
    are included. ``data_name`` selects which dataset's tabular features and
    saved RSF joblib to load.
    """
    import joblib  # Local import: keep module import lightweight.
    from survprompt.configs.exp_config import ExperimentConfig
    from survprompt.predictor_utils import get_tabular_data

    cfg = ExperimentConfig(
        cancer_of_interest=cancer,
        input_dir=os.getenv("INPUT_DIR"),
        base_dir=base_dir,
        data_name=data_name,
        features=["demographics", "treatment", "stage", "met", "path", "lab", "genomics"],
        num_folds=5,
        fold=fold,
        sample_size=-1,
    )
    _, test_features, sample_ids, _, _, _ = get_tabular_data(cfg)

    model_path = os.path.join(
        base_dir, "results", "models", data_name,
        f"rsf_baseline_{cancer}_{model_suffix}_model_fold{fold}.joblib",
    )
    rsf = joblib.load(model_path)
    feat_cols = list(rsf.feature_names_in_)
    X_te = pd.DataFrame(test_features)[feat_cols].fillna(0).astype(float).values
    sfs = rsf.predict_survival_function(X_te)

    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sid, sf in zip(sample_ids, sfs):
        curves[sid] = (np.asarray(sf.x, dtype=float) / 365.25, np.asarray(sf.y, dtype=float))
    return curves


def _curve_for_entry(
    label: str,
    entry,
    rsf_curves: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    sample_id,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Resolve a (times_yr, probs) curve for one patient/model, or None.

    RSF predictions are scalar in the prediction CSVs, so they are sourced from
    ``rsf_curves`` (the loaded joblib survival functions). SURV_PROB entries are
    ``[times_yr, probs]`` lists already in years.
    """
    if rsf_curves is not None and "Baseline: RSF" in label:
        return rsf_curves.get(sample_id)
    if "SURV_PROB" in label and isinstance(entry, (list, tuple)) and len(entry) == 2:
        times, probs = entry
        times = np.asarray(times, dtype=float)
        probs = np.asarray(probs, dtype=float)
        if times.size == 0 or probs.size == 0 or times.size != probs.size:
            return None
        if np.any(np.isnan(times)) or np.any(np.isnan(probs)):
            return None
        return times, probs
    return None


def _rmst_bootstrap_ci(values: np.ndarray, n_boot: int, alpha: float, seed: int) -> Tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[~np.isnan(clean)]
    if clean.size == 0:
        return np.nan, np.nan, np.nan
    point = float(np.mean(clean))
    if clean.size == 1 or n_boot <= 0:
        return point, point, point
    rng = np.random.default_rng(seed)
    n = clean.size
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        boot[b] = np.mean(clean[rng.integers(0, n, size=n)])
    lower = float(np.percentile(boot, 100 * (alpha / 2)))
    upper = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return point, lower, upper


def compute_rmst_mae_stats(
    datasets: List[Tuple[pd.Series, pd.Series, str]],
    model_label: str,
    km_t: np.ndarray,
    km_s: np.ndarray,
    taus: List[float] = DEFAULT_TAUS,
    rsf_curves: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[float, Dict[str, float]]:
    """RMST-MAE(tau) with bootstrap CIs for a single model.

    ``datasets`` follows ``process_data_for_km``'s output: a list of
    ``(time_series, event_series, label)`` tuples including ``"Ground Truth"``.
    ``km_t``/``km_s`` is the training-cohort KM curve in years used to impute
    the ground-truth RMST for censored patients.

    Returns ``{tau: {"value", "ci_lower", "ci_upper"}}``.
    """
    nan_stat = {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    empty = {tau: dict(nan_stat) for tau in taus}

    gt = [(x[0], x[1]) for x in datasets if x[2] == "Ground Truth"]
    if not gt:
        return empty
    gt_time = gt[0][0].astype(float) / 365.25  # years
    gt_event = gt[0][1].astype(float)

    model = [x for x in datasets if x[2] == model_label]
    if not model:
        return empty
    pred_series = model[0][0]

    common_idx = gt_time.index.intersection(pred_series.index)
    if len(common_idx) == 0:
        return empty

    # Accumulate per-patient absolute RMST errors per tau (shared row order so a
    # single bootstrap index draw applies to every tau consistently).
    abs_err = {tau: [] for tau in taus}
    for sid in common_idx:
        t_obs_yr = gt_time.loc[sid]
        ev = gt_event.loc[sid]
        if np.isnan(t_obs_yr) or np.isnan(ev):
            continue
        pred_entry = pred_series.loc[sid]
        if "TTE_OS" in model_label:
            curve = None
        else:
            curve = _curve_for_entry(model_label, pred_entry, rsf_curves, sid)
            if curve is None:
                continue
        for tau in taus:
            if "TTE_OS" in model_label:
                p_rmst = predicted_tte_rmst_at_tau(pred_entry, tau)
            else:
                times_yr, probs = curve
                p_rmst = predicted_rmst_at_tau(times_yr, probs, tau)
            if np.isnan(p_rmst):
                abs_err[tau].append(np.nan)
                continue
            t_rmst = true_rmst_at_tau(float(t_obs_yr), int(ev), km_t, km_s, tau)
            abs_err[tau].append(abs(p_rmst - t_rmst))

    out: Dict[float, Dict[str, float]] = {}
    for tau in taus:
        value, lo, hi = _rmst_bootstrap_ci(np.asarray(abs_err[tau], dtype=float), n_boot, alpha, seed)
        out[tau] = {"value": value, "ci_lower": lo, "ci_upper": hi}
    return out


# ===========================================================================
# Calibration: D-calibration and one-calibration
#
# Implementations of two survival-calibration tests, both reporting a
# chi-square p-value where higher = better calibrated.
#
# - D-calibration (Haider et al., "Effective Ways to Build and Evaluate
#   Individual Survival Distributions", JMLR 2020): a model is D-calibrated if
#   the predicted survival probabilities at the patients' own event times are
#   Uniform(0,1). Each uncensored S_i(t_i) drops in one bin; a patient censored
#   at c with S_i(c)=p is spread uniformly (in survival units) over [0, p].
# - One-calibration (Hosmer-Lemeshow with the D'Agostino-Nam censoring
#   correction): bin patients by predicted event probability at a horizon, and
#   compare the bin's mean predicted probability with the Kaplan-Meier observed
#   event probability at that horizon.
# ===========================================================================

def d_calibration(pred_probs, event_indicators, num_bins: int = 10):
    """D-calibration chi-square test.

    ``pred_probs`` are the predicted survival probabilities at each patient's own
    event/censoring time. Returns ``(statistic, p_value, histogram)``.
    """
    p = np.clip(np.asarray(pred_probs, dtype=float), 0.0, 1.0)
    censored = ~np.asarray(event_indicators).astype(bool)
    hist = np.zeros(num_bins)

    # Uncensored: the event-time survival probability lands entirely in one bin.
    for prob in p[~censored]:
        hist[min(int(prob * num_bins), num_bins - 1)] += 1.0

    # Censored at survival p: the true event has survival in [0, p]; spread the
    # unit of mass uniformly (in survival units) over the bins covering [0, p].
    for prob in p[censored]:
        if prob <= 0:
            hist[0] += 1.0
            continue
        b = min(int(prob * num_bins), num_bins - 1)
        hist[b] += (prob - b / num_bins) / prob          # partial bin containing p
        if b > 0:
            hist[:b] += (1.0 / num_bins) / prob          # full lower bins

    statistic, p_value = chisquare(hist)
    return float(statistic), float(p_value), hist


def one_calibration(preds, event_time, event_indicator, target_time,
                    num_bins: int = 10):
    """One-calibration (Hosmer-Lemeshow, D'Agostino-Nam) at ``target_time``.

    ``preds`` are predicted *event* probabilities at the horizon (i.e.
    ``1 - S(target_time)``). Returns ``(p_value, statistic, observed, expected)``.
    """
    preds = np.asarray(preds, dtype=float)
    event_time = np.asarray(event_time, dtype=float)
    event_indicator = np.asarray(event_indicator)

    order = np.argsort(-preds)  # high predicted risk first
    bins_p = np.array_split(preds[order], num_bins)
    bins_t = np.array_split(event_time[order], num_bins)
    bins_e = np.array_split(event_indicator[order], num_bins)

    statistic = 0.0
    observed, expected = [], []
    for bp, bt, be in zip(bins_p, bins_t, bins_e):
        n = bt.size
        if n == 0:
            continue
        mean_pred = float(bp.mean())
        distinct, _, _, surv = _km_product_limit(bt, be)
        observed_prob = 1.0 - float(_km_survival_at(distinct, surv, target_time))
        statistic += n * (observed_prob - mean_pred) ** 2 / (mean_pred * (1.0 - mean_pred))
        observed.append(observed_prob)
        expected.append(mean_pred)

    k = len(observed)
    dof = k - 1  # D'Agostino-Nam degrees of freedom (k <= 15)
    if dof <= 0:
        raise ValueError("Too few non-empty bins to compute one-calibration.")
    p_value = float(1.0 - chi2.cdf(statistic, dof))
    return p_value, float(statistic), observed, expected


def _patient_survival_at(model_label, entry, rsf_curves, sample_id, query_yr):
    """Predicted S(query_yr) for one patient: from the survival curve for
    SURV_PROB / RSF models, or a step function for scalar TTE_OS predictions."""
    if "TTE_OS" in model_label:
        try:
            pred_yr = float(entry) / 365.25
        except (TypeError, ValueError):
            return np.nan
        if np.isnan(pred_yr):
            return np.nan
        return 1.0 if query_yr < pred_yr else 0.0
    curve = _curve_for_entry(model_label, entry, rsf_curves, sample_id)
    if curve is None:
        return np.nan
    times_yr, probs = curve
    return float(np.interp(query_yr, times_yr, probs, left=1.0, right=probs[-1]))


def _calibration_inputs(datasets, model_label):
    """Shared setup: ground-truth times (years) + events, and the model's
    prediction series aligned to the common patients. Returns
    ``(gt_time, gt_event, pred_series, common_idx)`` or None if unavailable."""
    gt = [(x[0], x[1]) for x in datasets if x[2] == "Ground Truth"]
    model = [x for x in datasets if x[2] == model_label]
    if not gt or not model:
        return None
    gt_time = gt[0][0].astype(float) / 365.25
    gt_event = gt[0][1].astype(float)
    pred_series = model[0][0]
    common_idx = gt_time.index.intersection(pred_series.index)
    if len(common_idx) == 0:
        return None
    return gt_time, gt_event, pred_series, common_idx


def compute_d_calibration_stats(datasets, model_label, rsf_curves=None, num_bins: int = 10):
    """D-calibration p-value for one model (point estimate; higher = better).

    Returns ``{"value", "ci_lower", "ci_upper"}`` with the CI equal to the value
    (a single cohort-level test statistic is not bootstrapped)."""
    nan_stat = {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    setup = _calibration_inputs(datasets, model_label)
    if setup is None:
        return dict(nan_stat)
    gt_time, gt_event, pred_series, common_idx = setup

    surv_at_own, events = [], []
    for sid in common_idx:
        t_obs, ev = gt_time.loc[sid], gt_event.loc[sid]
        if np.isnan(t_obs) or np.isnan(ev):
            continue
        s = _patient_survival_at(model_label, pred_series.loc[sid], rsf_curves, sid, float(t_obs))
        if np.isnan(s):
            continue
        surv_at_own.append(s)
        events.append(ev)
    if len(surv_at_own) < num_bins:
        return dict(nan_stat)
    try:
        _, p_value, _ = d_calibration(np.asarray(surv_at_own), np.asarray(events), num_bins)
    except Exception:
        p_value = np.nan
    return {"value": p_value, "ci_lower": p_value, "ci_upper": p_value}


def compute_one_calibration_stats(datasets, model_label, taus=DEFAULT_TAUS,
                                  rsf_curves=None, num_bins: int = 10):
    """One-calibration p-value per horizon for one model (point estimate; higher
    = better). Returns ``{tau: {"value", "ci_lower", "ci_upper"}}``."""
    nan_stat = {"value": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    setup = _calibration_inputs(datasets, model_label)
    if setup is None:
        return {tau: dict(nan_stat) for tau in taus}
    gt_time, gt_event, pred_series, common_idx = setup

    out = {}
    for tau in taus:
        preds, et, ei = [], [], []
        for sid in common_idx:
            t_obs, ev = gt_time.loc[sid], gt_event.loc[sid]
            if np.isnan(t_obs) or np.isnan(ev):
                continue
            s = _patient_survival_at(model_label, pred_series.loc[sid], rsf_curves, sid, float(tau))
            if np.isnan(s):
                continue
            preds.append(1.0 - s)
            et.append(t_obs)
            ei.append(ev)
        if len(preds) < num_bins:
            out[tau] = dict(nan_stat)
            continue
        try:
            p_value = one_calibration(np.asarray(preds), np.asarray(et), np.asarray(ei), float(tau), num_bins)[0]
        except Exception:
            p_value = np.nan
        out[tau] = {"value": p_value, "ci_lower": p_value, "ci_upper": p_value}
    return out
