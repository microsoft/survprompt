import argparse
import atexit
import os
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import numpy as np
from dotenv import load_dotenv
from typing import List, Tuple, Optional

from survprompt.configs.exp_config import ExperimentConfig
from survprompt.defaults import SURVPROMPT_DEFAULT_MODEL_LABEL
from survprompt.plots.color_utils import GLOBAL_FONT_SIZE
from survprompt.plots.plot_utils import (
    is_all_models_zero_shot_label,
    format_model_display_label, get_line_style, get_model_color,
    is_survprompt_headline_label, parse_label, process_data_for_km,
    resolve_survprompt_headline_label
)
from survprompt.evaluation.metrics import ccae_components, calculate_error, interpolate_time_at_threshold
from survprompt.defaults import DEFAULT_DATA_NAME

SURVPROMPT_MODEL_LABEL = SURVPROMPT_DEFAULT_MODEL_LABEL

_BOOTSTRAP_EXECUTOR: Optional[ProcessPoolExecutor] = None
_BOOTSTRAP_MAX_WORKERS = min(12, os.cpu_count() or 1)


def _shutdown_bootstrap_executor() -> None:
    global _BOOTSTRAP_EXECUTOR
    if _BOOTSTRAP_EXECUTOR is not None:
        _BOOTSTRAP_EXECUTOR.shutdown(wait=True, cancel_futures=True)
        _BOOTSTRAP_EXECUTOR = None


atexit.register(_shutdown_bootstrap_executor)


def _get_bootstrap_executor() -> ProcessPoolExecutor:
    global _BOOTSTRAP_EXECUTOR
    if _BOOTSTRAP_EXECUTOR is None:
        _BOOTSTRAP_EXECUTOR = ProcessPoolExecutor(max_workers=_BOOTSTRAP_MAX_WORKERS)
    return _BOOTSTRAP_EXECUTOR


def _smooth_curve_array(values: np.ndarray, smoothing_window: int) -> np.ndarray:
    return pd.Series(values).rolling(window=smoothing_window, center=True).mean().to_numpy(dtype=float)


def _curve_from_binned_samples(
    errors: np.ndarray,
    weights: np.ndarray,
    bin_codes: np.ndarray,
    n_bins: int,
    weighted_censored_error: bool,
    density_scaled_error: bool,
    smoothing_window: int,
) -> np.ndarray:
    valid = np.isfinite(errors) & np.isfinite(weights) & (bin_codes >= 0)
    if weighted_censored_error:
        valid &= weights > 0
    if not valid.any():
        return np.full(n_bins, np.nan, dtype=float)

    err = errors[valid]
    wt = weights[valid]
    codes = bin_codes[valid].astype(int, copy=False)

    if weighted_censored_error:
        counts = np.bincount(codes, weights=wt, minlength=n_bins).astype(float)
        err_sum = np.bincount(codes, weights=wt * err, minlength=n_bins).astype(float)
    else:
        counts = np.bincount(codes, minlength=n_bins).astype(float)
        err_sum = np.bincount(codes, weights=err, minlength=n_bins).astype(float)

    mean = np.full(n_bins, np.nan, dtype=float)
    nonzero = counts > 0
    mean[nonzero] = err_sum[nonzero] / counts[nonzero]

    if density_scaled_error:
        max_count = counts.max()
        if not np.isfinite(max_count) or max_count <= 0:
            mean[:] = np.nan
        else:
            mean = mean * np.sqrt(counts / max_count)

    return _smooth_curve_array(mean, smoothing_window)


def _bootstrap_curve_worker(task: Tuple[np.ndarray, np.ndarray, np.ndarray, int, bool, bool, int, int]) -> np.ndarray:
    errors, weights, bin_codes, n_bins, weighted_censored_error, density_scaled_error, smoothing_window, seed = task
    rng = np.random.default_rng(seed)
    n = len(errors)
    sample_idx = rng.integers(0, n, size=n)
    return _curve_from_binned_samples(
        errors[sample_idx],
        weights[sample_idx],
        bin_codes[sample_idx],
        n_bins,
        weighted_censored_error,
        density_scaled_error,
        smoothing_window,
    )

# Set global font properties
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["font.size"] = GLOBAL_FONT_SIZE

# Load environment and set BASE_DIR
BASE_DIR = os.getenv("BASE_DIR")
if BASE_DIR is None:
    load_dotenv()
    BASE_DIR = os.getenv("BASE_DIR")
 

def _nice_count_ceiling(value: float) -> int:
    # Round a positive count up to a clean axis endpoint (e.g. 37 -> 50, 63 -> 75).
    if not np.isfinite(value) or value <= 0:
        return 1
    exponent = np.floor(np.log10(value))
    scale = 10 ** exponent
    fraction = value / scale
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5:
        nice_fraction = 5
    elif fraction <= 7.5:
        nice_fraction = 7.5
    else:
        nice_fraction = 10
    return int(nice_fraction * scale)


def _ground_truth_histogram_counts(
    datasets: List[Tuple[pd.Series, pd.Series, str]],
    bin_width: float,
):
    T_base, E_base, _ = datasets[0]
    T_base = T_base.astype(int) / 365.25
    E_base = E_base.astype(int).astype(bool)

    time_min, time_max = T_base.min(), T_base.max()
    bins = np.arange(time_min, time_max + bin_width, bin_width)
    if bins.size < 2:
        bins = np.array([time_min, time_min + bin_width], dtype=float)
    bin_lefts = bins[:-1]

    event_counts, _ = np.histogram(T_base[E_base], bins=bins)
    censor_counts, _ = np.histogram(T_base[~E_base], bins=bins)
    return bin_lefts, event_counts.astype(float), censor_counts.astype(float), bins


def plot_support_histogram(
    ax,
    datasets: List[Tuple[pd.Series, pd.Series, str]],
    bin_width: float = 0.25,
    title: Optional[str] = None,
    legend: bool = False,
    xaxis_label: Optional[str] = None,
    yaxis_label: Optional[str] = None,
) -> None:
    bin_lefts, event_counts, censor_counts, bins = _ground_truth_histogram_counts(datasets, bin_width)
    ax.bar(
        bin_lefts,
        event_counts,
        width=bin_width,
        align='edge',
        color='#5b8db8',
        alpha=0.55,
        linewidth=0,
        label='Events',
    )
    ax.bar(
        bin_lefts,
        censor_counts,
        width=bin_width,
        align='edge',
        color='#c47c5a',
        alpha=0.45,
        linewidth=0,
        label='Censored',
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(float(bins[0]), float(bins[-1]))
    ax.set_ylim(0, _nice_count_ceiling(max(float(event_counts.max(initial=0)), float(censor_counts.max(initial=0)))))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(25))
    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.tick_params(axis='y', which='minor', labelleft=False)
    ax.tick_params(axis='x', which='major', labelsize=14)

    if title:
        ax.text(-0.15, 0.5, title, transform=ax.transAxes,
                rotation=0, verticalalignment='center',
                horizontalalignment='center', fontsize=18, fontweight='bold')
    if legend:
        ax.legend(loc='upper right', ncol=2)
    if xaxis_label:
        ax.set_xlabel(xaxis_label, fontsize=16)
    if yaxis_label:
        ax.set_ylabel(yaxis_label, fontsize=16)


def plot_support_hist_panel(
    n_rows,
    n_cols,
    cancer_of_interest,
    datasets,
    datasets_filter,
    bin_width: float = 0.25,
    title_fmt=lambda ca: ca.upper(),
    yaxis_superlabel: str = 'Number of Patients',
):
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(12 * n_cols, 4.5 * n_rows), sharex='col'
    )
    axes = np.array(axes).reshape(n_rows, n_cols)
    total_plots = n_rows * n_cols
    n_cancers = len(cancer_of_interest)

    for idx in range(total_plots):
        if idx >= n_cancers:
            axes.flat[idx].axis('off')
            continue
        ca = cancer_of_interest[idx]
        ax = axes.flat[idx]
        dset = datasets_filter(datasets, ca)
        plot_support_histogram(
            ax,
            dset,
            bin_width=bin_width,
            title=title_fmt(ca),
            legend=(idx == 0),
            xaxis_label='Time (years)' if idx == n_cancers - 1 else None,
            yaxis_label=None,
        )

    fig.supylabel(yaxis_superlabel)
    plt.tight_layout()
    return fig


def plot_errors(
    ax,
    datasets: List[Tuple[pd.Series, pd.Series, str]],
    bin_width: float = 0.1,
    smoothing_window: int = 3,
    xaxis_label: str = "Time (years)",
    yaxis_label: str = "Average Signed Error",
    title: str = None,
    legend_pos: str = 'upper right',
    xtick_spacing: Optional[float] = None,
    density_scaled_error: bool = False,
    censored_error: bool = False,
    plot_censored_separately: bool = False,
    show_confidence_interval: bool = False,
    simplify_legend_labels: str = None,
    abs_error: bool = False,
    show_bin_density: bool = False,
    show_support_axis: bool = False,
    weighted_censored_error: bool = False,
    ci_bootstrap_samples: int = 300,
    min_bin_patients: Optional[int] = None,
    y_limits: Optional[Tuple[float, float]] = None
) -> None:
    """
    Plot binned and smoothed average errors over time for each model.
    Optional density scaling shrinks sparse-bin amplitudes, but is off for the
    main signed-error figures. When ``weighted_censored_error`` is enabled,
    censored errors use the pseudo-observation surrogate together with
    censoring-confidence weights. Confidence intervals are bootstrapped for the
    plotted, smoothed curve itself. Optionally, bins with too few patients can
    be masked out before plotting. A subtle support ribbon can be shown behind
    the curves to reveal event/censoring density by time bin.
    An optional secondary axis can expose the mirrored support scale.
    """
    T_base, E_base, base_label = datasets[0]
    T_base = T_base.astype(int) / 365.25
    E_base = E_base.astype(int)

    time_min, time_max = T_base.min(), T_base.max()
    bins = np.arange(time_min, time_max + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    all_bins = pd.Index(bin_centers, dtype='float')
    base_time_bins = pd.cut(T_base, bins=bins, labels=bin_centers, include_lowest=True)
    base_bin_counts = base_time_bins.value_counts(sort=False).reindex(all_bins, fill_value=0).astype(float)
    event_bin_counts = (
        base_time_bins[E_base.astype(bool)]
        .value_counts(sort=False)
        .reindex(all_bins, fill_value=0)
        .astype(float)
    )
    censor_bin_counts = (
        base_time_bins[~E_base.astype(bool)]
        .value_counts(sort=False)
        .reindex(all_bins, fill_value=0)
        .astype(float)
    )

    if y_limits is not None:
        effective_y_limits = y_limits
    elif abs_error:
        effective_y_limits = (0, 5)
    else:
        effective_y_limits = (-22, 22)

    def _smooth_series(series: pd.Series) -> pd.Series:
        return pd.Series(_smooth_curve_array(series.to_numpy(dtype=float), smoothing_window), index=series.index)

    def _bin_patient_counts(frame: pd.DataFrame) -> pd.Series:
        valid = np.isfinite(frame['error']) & np.isfinite(frame['weight'])
        if weighted_censored_error:
            valid &= frame['weight'] > 0
        counts = frame.loc[valid].groupby('time_bin', observed=False).size().reindex(all_bins, fill_value=0)
        return counts.astype(float)

    def _bin_mean_and_mass(frame: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        if weighted_censored_error:
            def _weighted_bin_stats(group: pd.DataFrame) -> pd.Series:
                valid = (
                    np.isfinite(group['error'])
                    & np.isfinite(group['weight'])
                    & (group['weight'] > 0)
                )
                if not valid.any():
                    return pd.Series({'mean': np.nan, 'mass': 0.0})
                err = group.loc[valid, 'error'].to_numpy(dtype=float)
                wt = group.loc[valid, 'weight'].to_numpy(dtype=float)
                return pd.Series({
                    'mean': float(np.average(err, weights=wt)),
                    'mass': float(wt.sum()),
                })

            grouped_stats = frame.groupby('time_bin', observed=False).apply(_weighted_bin_stats, include_groups=False)
            mean_errors = grouped_stats['mean'].reindex(all_bins)
            counts = grouped_stats['mass'].reindex(all_bins)
        else:
            grouped = frame.groupby('time_bin', observed=False)['error']
            mean_errors = grouped.mean().reindex(all_bins)
            counts = grouped.count().reindex(all_bins)
        return mean_errors, counts

    def _scale_curve(mean_errors: pd.Series, counts: pd.Series) -> pd.Series:
        if not density_scaled_error:
            return mean_errors
        max_count = counts.max()
        if not np.isfinite(max_count) or max_count <= 0:
            return mean_errors * np.nan
        scaling = np.sqrt(counts / max_count)
        return mean_errors * scaling

    def _bootstrap_curve_interval(frame: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        valid_rows = frame[np.isfinite(frame['error']) & np.isfinite(frame['weight'])].copy()
        if weighted_censored_error:
            valid_rows = valid_rows[valid_rows['weight'] > 0]
        if valid_rows.empty or ci_bootstrap_samples <= 1:
            empty = pd.Series(np.nan, index=all_bins, dtype=float)
            return empty, empty

        errors = valid_rows['error'].to_numpy(dtype=float)
        weights = valid_rows['weight'].to_numpy(dtype=float)
        bin_codes = valid_rows['time_bin'].cat.codes.to_numpy(dtype=int)
        n_bins = len(all_bins)
        seeds = np.random.default_rng(0).integers(0, np.iinfo(np.int32).max, size=ci_bootstrap_samples, dtype=np.int32)
        tasks = [
            (errors, weights, bin_codes, n_bins, weighted_censored_error, density_scaled_error, smoothing_window, int(seed))
            for seed in seeds
        ]

        executor = _get_bootstrap_executor()
        boot_curves = list(executor.map(_bootstrap_curve_worker, tasks, chunksize=max(1, ci_bootstrap_samples // _BOOTSTRAP_MAX_WORKERS)))
        boot_arr = np.asarray(boot_curves, dtype=float)
        lower = pd.Series(np.nan, index=all_bins, dtype=float)
        upper = pd.Series(np.nan, index=all_bins, dtype=float)
        all_nan = np.all(np.isnan(boot_arr), axis=0)
        valid_cols = ~all_nan
        if np.any(valid_cols):
            lower.iloc[valid_cols] = np.nanpercentile(boot_arr[:, valid_cols], 2.5, axis=0)
            upper.iloc[valid_cols] = np.nanpercentile(boot_arr[:, valid_cols], 97.5, axis=0)
        return lower, upper

    support_height = None
    support_count_scale = None
    if show_bin_density and base_bin_counts.max() > 0:
        support_x = base_bin_counts.index.to_numpy(dtype=float)
        support_height = 0.5 * min(abs(effective_y_limits[0]), abs(effective_y_limits[1]))

        if abs_error:
            normalized_counts = base_bin_counts / base_bin_counts.max()
            ax.fill_between(
                support_x,
                0,
                support_height * normalized_counts.to_numpy(dtype=float),
                step='mid',
                color='#9ecae1',
                alpha=0.28,
                linewidth=0,
                zorder=0,
            )
        else:
            count_scale = max(event_bin_counts.max(), censor_bin_counts.max())
            support_count_scale = count_scale
            if count_scale > 0:
                event_heights = support_height * (event_bin_counts / count_scale).to_numpy(dtype=float)
                censor_heights = support_height * (censor_bin_counts / count_scale).to_numpy(dtype=float)
                ax.fill_between(
                    support_x,
                    0,
                    event_heights,
                    step='mid',
                    color='#5b8db8',
                    alpha=0.18,
                    linewidth=0,
                    zorder=0,
                )
                ax.fill_between(
                    support_x,
                    0,
                    -censor_heights,
                    step='mid',
                    color='#c47c5a',
                    alpha=0.16,
                    linewidth=0,
                    zorder=0,
                )

    if show_support_axis and support_height is not None and support_height > 0:
        if abs_error:
            support_count_scale = base_bin_counts.max()

        if support_count_scale is not None and support_count_scale > 0:
            support_axis_limit = _nice_count_ceiling(float(support_count_scale))
            support_axis = ax.secondary_yaxis(
                'right',
                functions=(
                    lambda y, y_scale=support_height, count_scale=support_count_scale: y * count_scale / y_scale,
                    lambda y, y_scale=support_height, count_scale=support_count_scale: y * y_scale / count_scale,
                ),
            )
            support_axis.spines['right'].set_visible(True)
            support_axis.spines['right'].set_color('#6b7280')
            support_axis.spines['right'].set_linewidth(1.0)
            support_axis.tick_params(axis='y', labelsize=11, colors='#6b7280', length=4, width=1.0)
            if abs_error:
                ticks = [support_axis_limit]
                if support_axis_limit > 25:
                    ticks = [25, support_axis_limit]
                support_axis.set_yticks(ticks)
                support_axis.set_yticklabels([str(int(tick)) for tick in ticks])
            else:
                ticks = [-support_axis_limit, support_axis_limit]
                if support_axis_limit > 25:
                    ticks = [-support_axis_limit, -25, 25, support_axis_limit]
                support_axis.set_yticks(ticks)
                support_axis.set_yticklabels([str(int(tick)) for tick in ticks])

    for (T, E, label) in datasets[1:]:
        if "Training Times/Events" in label:
            train_time, train_event = T.astype(float)/365.25, E.astype(int)
            continue
        elif 'SURV_PROB' in label:
            # Entries for patients a model didn't cover are NaN floats (not
            # [times, probs]); guard the subscript -> NaN error, dropped in binning.
            T = T.apply(
                lambda x: interpolate_time_at_threshold(x[0], x[1], 0.5)
                if isinstance(x, (list, tuple)) else np.nan
            ).astype(float)
        else:
            # astype(float) (not int) keeps NaN for uncovered patients.
            T = T.astype(float) / 365.25

        if weighted_censored_error:
            errors, sample_weights = ccae_components(
                predicted_times=T,
                event_times=T_base,
                event_indicators=E_base,
                train_event_times=train_time,
                train_event_indicators=train_event,
            )
        elif censored_error:
            errors = calculate_error(
                predicted_times=T,
                event_times=T_base,
                event_indicators=E_base,
                train_event_times=train_time,
                train_event_indicators=train_event
            )
            sample_weights = np.ones_like(np.asarray(errors, dtype=float))
        else:
            errors = calculate_error(
                predicted_times=T,
                event_times=T_base,
                event_indicators=E_base
            )
            sample_weights = np.ones_like(np.asarray(errors, dtype=float))

        if abs_error:
            errors = np.abs(errors)

        errors = np.asarray(errors, dtype=float)
        sample_weights = np.asarray(sample_weights, dtype=float)

        model, method, size, task = parse_label(label)
        color = get_model_color(model, method, task, simplify=simplify_legend_labels)
        if "Baseline" in label:
            color = "black"
        if "Baseline: RSF" in label:
            label = "Random Survival Forest"
        elif is_survprompt_headline_label(label) and simplify_legend_labels != 'Survprompt_appendix':
            label = "Survprompt"
        elif simplify_legend_labels == 'Survprompt_appendix':
            # Keep compact-model qualifiers (e.g. ' mini'); drop default full-size
            # and reasoning-effort text from display labels.
            label = format_model_display_label(model, size)
        ls = get_line_style(size)

        df = pd.DataFrame({
            'time': T_base.to_numpy(dtype=float),
            'error': errors,
            'weight': sample_weights,
            'event': E_base.to_numpy(dtype=float),
        }, index=T_base.index)
        df['time_bin'] = pd.cut(df['time'], bins=bins, labels=bin_centers, include_lowest=True)

        mean_errors, counts = _bin_mean_and_mass(df)
        patient_counts = _bin_patient_counts(df)
        y_values = _scale_curve(mean_errors, counts)
        smoothed = _smooth_series(y_values)
        ci_lower, ci_upper = _bootstrap_curve_interval(df)
        curve_valid = ~(smoothed.isna() | ci_lower.isna() | ci_upper.isna())
        if min_bin_patients is not None:
            curve_valid &= patient_counts >= float(min_bin_patients)
        plot_values = smoothed.where(curve_valid)

        ax.plot(plot_values.index, plot_values.values, label=label, alpha=1.0, color=color, linestyle=ls, linewidth=3)

        if show_confidence_interval:
            x_values = smoothed.index.to_numpy(dtype=float)
            lower_values = ci_lower.to_numpy(dtype=float)
            upper_values = ci_upper.to_numpy(dtype=float)
            valid_mask = curve_valid.to_numpy(dtype=bool)
            ax.fill_between(
                x_values,
                lower_values,
                upper_values,
                where=valid_mask,
                interpolate=False,
                color=color, alpha=0.10, linewidth=0
            )
            
        if yaxis_label:
            ax.set_ylabel(yaxis_label, fontsize=16)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('data', 0))
    
    if title:
        ax.text(-0.15, 0.5, title, transform=ax.transAxes, 
                rotation=0, verticalalignment='center', 
                horizontalalignment='center', fontsize=18, fontweight='bold')
    if legend_pos:
        ax.legend(loc=legend_pos, ncol=1 if simplify_legend_labels == 'Survprompt_final' else 3)

    if xtick_spacing:
        ax.xaxis.set_major_locator(plt.MultipleLocator(2 * xtick_spacing))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(xtick_spacing))
        ax.tick_params(axis='x', which='major', labelsize=12)
        ax.tick_params(axis='x', which='minor', labelbottom=False)
    else:
        ax.set_xlim(0, 10)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))   
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5)) 
        ax.tick_params(axis='x', which='major', labelbottom=False)
        ax.tick_params(axis='x', which='minor', labelbottom=False)
        
    if xaxis_label:
        ax.set_xlabel(xaxis_label, fontsize=16)
        ax.tick_params(axis='x', which='major', labelsize=14, labelbottom=True)
        
    ax.set_ylim(*effective_y_limits)

    ax.yaxis.set_major_locator(plt.MultipleLocator(1 if abs_error else 4))  
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1 if abs_error else 2))  
    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.tick_params(axis='y', which='minor', labelleft=False)

    ax.figure.canvas.draw_idle()

def plot_error_panel_finalfig(
    n_rows,
    n_cols,
    cancer_of_interest,
    datasets,
    datasets_filter,
    plot_kwargs=None,
    title_fmt=lambda ca: ca.upper(),
    yaxis_superlabel: str = 'Average Censored Error (years)',
    support_axis_superlabel: Optional[str] = None,
):
    """
    Panel plotting for main figure error visualizations.
    """
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(12* n_cols, 6 * n_rows), sharey='row'
    )
    axes = np.array(axes)
    axes = axes.reshape(n_rows, n_cols)
    total_plots = n_rows * n_cols
    n_cancers = len(cancer_of_interest)
    
    for idx in range(total_plots):
        if idx >= n_cancers:
            axes.flat[idx].axis('off')
            continue
        ca = cancer_of_interest[idx]
        dset = datasets_filter(datasets, ca)
        ax = axes.flat[idx]
        kwargs = plot_kwargs.copy() if plot_kwargs else {}
        
        if ca == 'nsclc':
            kwargs.setdefault('legend_pos', 'upper right')
            kwargs.setdefault('yaxis_label', None)
            kwargs.setdefault('xaxis_label', None)
        elif ca in ['crc']:
            kwargs.setdefault('legend_pos', None)
            kwargs.setdefault('yaxis_label', None)
            kwargs.setdefault('xaxis_label', None)
        elif ca in ['brca', 'panc']:
            kwargs.setdefault('legend_pos', None)
            kwargs.setdefault('yaxis_label', None)
            kwargs.setdefault('xaxis_label', None)
        elif ca in ['prostate']:
            kwargs.setdefault('legend_pos', None)
            kwargs.setdefault('yaxis_label', None)
            kwargs.setdefault('xaxis_label', "Time (years)")

        if kwargs.get('abs_error'):
            kwargs.setdefault('y_limits', (0, 5))
        else:
            kwargs.setdefault('y_limits', (-25, 25))

        kwargs.setdefault('title', title_fmt(ca))
        plot_errors(ax, dset, **kwargs)
        
    fig.supylabel(yaxis_superlabel)
    if support_axis_superlabel:
        fig.text(0.992, 0.5, support_axis_superlabel, rotation=-90,
                 va='center', ha='center', fontsize=16, color='#6b7280')
    plt.tight_layout(rect=(0, 0, 0.985, 1))
    return fig

def plot_error_distribution(
    plot_dir,
    errors,
    cancer_of_interest,
    model_names,
    filename,
    ylabel="Error",
    abs=False
):
    """
    Plot grouped boxplots of error distributions across cancers for selected models.
    """
    data = []
    for ca in cancer_of_interest:
        err_dict = errors[ca]
        for model in model_names:
            if model in err_dict:
                vals = np.array(err_dict[model], dtype=np.float64)
                if abs:
                    vals = np.abs(vals)
                for val in vals:
                    data.append({
                        "Cancer": ca.upper(),
                        "Model": model,
                        "Error": val
                    })
    df = pd.DataFrame(data)

    color_map = {}
    for model in model_names:
        if model == 'Random Survival Forest':
            model_label = 'Baseline: RSF'
        elif model == 'Survprompt':
            model_label = SURVPROMPT_MODEL_LABEL
        else:
            model_label = model
        m, method, size, task = parse_label(model_label)
        color_map[model] = get_model_color(m, method, task)

    plt.figure(figsize=(1.2 * len(cancer_of_interest), 6))
    ax = sns.boxplot(
        data=df,
        x="Cancer",
        y="Error",
        hue="Model",
        showfliers=False,
        palette=color_map
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Cancer")
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight', format="pdf")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-plot-hist', '--plot-hist', action='store_true',
                        help='Generate a separate event/censor histogram figure.')
    args = parser.parse_args()

    # Data selection
    dataset = DEFAULT_DATA_NAME
    cancer_of_interest = ['nsclc', 'brca', 'crc', 'panc', 'prostate']
    prompting_tasks = ['TTE_OS', 'SURV_PROB']
    race_inclusion_path = 'incl_race'
    system_prompt_path = 'system'
 
    # Define PLOT_DIR
    PLOT_DIR = os.path.join(BASE_DIR, "plots", "errors", dataset, race_inclusion_path)
    os.makedirs(PLOT_DIR, exist_ok=True)
 
    datasets = {}
    censored_errors = {}
 
    for ca in cancer_of_interest:
        cfg = ExperimentConfig(
            base_dir=BASE_DIR,
            data_name=dataset,
            cancer_of_interest=ca)
        ca_datasets = process_data_for_km(cfg, prompting_tasks, race_inclusion_path=race_inclusion_path, system_prompt_path=system_prompt_path, get_train_times=True)
        datasets[ca] = ca_datasets

        survprompt_label = resolve_survprompt_headline_label(x[2] for x in ca_datasets)
        selected = [
            x for x in ca_datasets if x[2] in [
                'Ground Truth',
                'Baseline: RSF',
                survprompt_label,
                "Training Times/Events"
            ]
        ]
        gt_time, gt_event = [
            (x[0], x[1]) for x in selected if x[2] == 'Ground Truth'
        ][0]
        gt_time = gt_time.astype(float) / 365.25
        gt_event = gt_event.astype(float)
        gt_df = pd.DataFrame({'time': gt_time, 'event': gt_event})

        censored_errors[ca] = {}
        train_time, train_event = None, None
        for time, event, name in selected[1:]:
            if "Training Times/Events" in name:
                train_time, train_event = time.astype(float) / 365.25, event.astype(bool)
                continue
            if 'SURV_PROB' in name:
                time = time.apply(lambda x: interpolate_time_at_threshold(x[0], x[1], 0.5)).astype(float)
            else:
                time = time.astype(float) / 365.25
            
            event = event.astype(bool)
            df = pd.DataFrame({'time': time, 'event': event})
            aligned = gt_df.join(df, lsuffix='_gt', rsuffix='_pred', how='inner')
            censored_error = calculate_error(
                aligned['time_pred'], aligned['time_gt'],
                aligned['event_gt'],
                train_event_times=train_time,
                train_event_indicators=train_event
            )
            censored_errors[ca][name] = censored_error

    def filter_row2_cens(dsets, ca):
        survprompt_label = resolve_survprompt_headline_label(x[2] for x in dsets[ca])
        return [x for x in dsets[ca] if x[2] in [
            'Ground Truth', 'Training Times/Events', 'Baseline: RSF', survprompt_label
        ]]

    # Main figure: Average Signed Censored Prediction Error
    fig = plot_error_panel_finalfig(
        5,
        1,
        cancer_of_interest,
        datasets,
        filter_row2_cens,
        plot_kwargs={
            'bin_width': 1/4,
            'smoothing_window': 3,
            'density_scaled_error': False,
            'censored_error': True,
            'simplify_legend_labels': 'Survprompt_final',
            'abs_error': False,
            'show_bin_density': False,
            'show_support_axis': False,
            'weighted_censored_error': True,
            'min_bin_patients': 0,
            'show_confidence_interval': True
        },
        yaxis_superlabel='Average Censored Error (years)',
        support_axis_superlabel=None
    )
    fig.savefig(os.path.join(PLOT_DIR, f"average_weighted_censored_prediction_error_over_time_all_cancers_mainfig_{dataset}.pdf"), 
                bbox_inches='tight', format="pdf")
    plt.close(fig)

    # Supp Figure 3: Average Signed Censored Prediction Error (all models) - SURV_PROB
    fig = plot_error_panel_finalfig(
        5,
        1,
        cancer_of_interest,
        datasets,
        lambda dsets, ca: [
            x for x in dsets[ca]
            if x[2] == 'Ground Truth' or x[2] == 'Training Times/Events' or x[2] == 'Baseline: RSF' or is_all_models_zero_shot_label(x[2], 'SURV_PROB')
        ],
        plot_kwargs={
            'bin_width': 1/4,
            'smoothing_window': 3,
            'density_scaled_error': False,
            'censored_error': True,
            'simplify_legend_labels': 'Survprompt_appendix',
            'abs_error': False,
            'show_bin_density': False,
            'show_support_axis': False,
            'weighted_censored_error': True,
            'min_bin_patients': 0,
            'show_confidence_interval': True
        },
        yaxis_superlabel='Average Censored Error (years)',
        support_axis_superlabel=None
    )
    fig.savefig(os.path.join(PLOT_DIR, f"average_weighted_censored_prediction_error_over_time_all_cancers_all_models_suppfig2_{dataset}.pdf"), 
                bbox_inches='tight', format="pdf")
    plt.close(fig)

    # Supp Figure 3: Average Signed Censored Prediction Error (all models) - TTE_OS
    fig = plot_error_panel_finalfig(
        5,
        1,
        cancer_of_interest,
        datasets,
        lambda dsets, ca: [
            x for x in dsets[ca]
            if x[2] == 'Ground Truth' or x[2] == 'Training Times/Events' or x[2] == 'Baseline: RSF' or is_all_models_zero_shot_label(x[2], 'TTE_OS')
        ],
        plot_kwargs={
            'bin_width': 1/4,
            'smoothing_window': 3,
            'density_scaled_error': False,
            'censored_error': True,
            'simplify_legend_labels': 'Survprompt_appendix',
            'abs_error': False,
            'show_bin_density': False,
            'show_support_axis': False,
            'weighted_censored_error': True,
            'min_bin_patients': 0,
            'show_confidence_interval': True
        },
        yaxis_superlabel='Average Censored Error (years)',
        support_axis_superlabel=None
    )
    fig.savefig(os.path.join(PLOT_DIR, f"average_weighted_censored_prediction_error_over_time_all_cancers_all_models_suppfig3_{dataset}.pdf"), 
                bbox_inches='tight', format="pdf")
    plt.close(fig)

    if args.plot_hist:
        fig = plot_support_hist_panel(
            5,
            1,
            cancer_of_interest,
            datasets,
            lambda dsets, ca: [x for x in dsets[ca] if x[2] == 'Ground Truth'],
            bin_width=1/4,
            yaxis_superlabel='Number of Patients',
        )
        fig.savefig(
            os.path.join(PLOT_DIR, f"event_censor_histograms_all_cancers_{dataset}.pdf"),
            bbox_inches='tight', format='pdf'
        )
        plt.close(fig)

    # Main figure: Error distributions boxplot
    model_names_survprompt = ['Random Survival Forest', 'Survprompt']
    censored_errors_survprompt = {
        k: {
            'Random Survival Forest': v['Baseline: RSF'],
            'Survprompt': v[resolve_survprompt_headline_label(v.keys())],
        }
        for k, v in censored_errors.items()
    }
    
    plot_error_distribution(
        plot_dir=PLOT_DIR,
        errors=censored_errors_survprompt,
        cancer_of_interest=cancer_of_interest,
        model_names=model_names_survprompt,
        ylabel='Absolute Censored Error [years]',
        abs=True,
        filename=f"absolute_censored_error_distributions_all_cancers_mainfig_{dataset}.pdf"
    )