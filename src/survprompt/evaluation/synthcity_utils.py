"""
Adapted from synthcity codebase
https://github.com/vanderschaarlab/synthcity
"""

# Standard and typing imports
from typing import Tuple, Optional, Any
import numpy as np
import pandas as pd

# Third party imports
from lifelines import KaplanMeierFitter
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.validation import check_is_fitted
from sksurv.nonparametric import kaplan_meier_estimator

try:
    from scipy.integrate import trapz
except ImportError:
    try:
        from numpy import trapz  # NumPy < 2.0
    except ImportError:
        from numpy import trapezoid as trapz  # NumPy >= 2.0

# ========================
# Classes
# ========================

class SurvivalFunctionEstimator(BaseEstimator):
    """Kaplan–Meier estimate of the survival function."""

    def __init__(self) -> None:
        pass

    def fit(self, y: np.ndarray) -> "SurvivalFunctionEstimator":
        """Estimate survival distribution from training data.

        Parameters
        ----------
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        self
        """
        event, time = check_y_survival(y, allow_all_censored=True)

        unique_time, prob = kaplan_meier_estimator(event, time)
        self.unique_time_ = np.r_[-np.infty, unique_time]
        self.prob_ = np.r_[1.0, prob]

        return self

    def predict_proba(self, time: np.ndarray) -> np.ndarray:
        """Return probability of an event after given time point.

        :math:`\\hat{S}(t) = P(T > t)`

        Parameters
        ----------
        time : array, shape = (n_samples,)
            Time to estimate probability at.

        Returns
        -------
        prob : array, shape = (n_samples,)
            Probability of an event.
        """
        check_is_fitted(self, "unique_time_")
        time = check_array(time, ensure_2d=False)

        # K-M is undefined if estimate at last time point is non-zero
        extends = time > self.unique_time_[-1]
        if self.prob_[-1] > 0 and extends.any():
            raise ValueError(
                "time must be smaller than largest "
                "observed time point: {}".format(self.unique_time_[-1])
            )

        # beyond last time point is zero probability
        Shat = np.empty(time.shape, dtype=float)
        Shat[extends] = 0.0

        valid = ~extends
        time = time[valid]
        idx = np.searchsorted(self.unique_time_, time)
        # for non-exact matches, we need to shift the index to left
        eps = np.finfo(self.unique_time_.dtype).eps
        exact = np.absolute(self.unique_time_[idx] - time) < eps
        idx[~exact] -= 1
        Shat[valid] = self.prob_[idx]

        return Shat
    

class CensoringDistributionEstimator(SurvivalFunctionEstimator):
    """Kaplan–Meier estimator for the censoring distribution."""

    def fit(self, y: np.ndarray) -> "CensoringDistributionEstimator":
        """Estimate censoring distribution from training data.

        Parameters
        ----------
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        self
        """
        event, time = check_y_survival(y)
        if event.all():
            self.unique_time_ = np.unique(time)
            self.prob_ = np.ones(self.unique_time_.shape[0])
        else:
            unique_time, prob = kaplan_meier_estimator(event, time, reverse=True)
            self.unique_time_ = np.r_[-np.infty, unique_time]
            self.prob_ = np.r_[1.0, prob]

        return self

    def predict_ipcw(self, y: np.ndarray) -> np.ndarray:
        """Return inverse probability of censoring weights at given time points.

        :math:`\\omega_i = \\delta_i / \\hat{G}(y_i)`

        Parameters
        ----------
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns
        -------
        ipcw : array, shape = (n_samples,)
            Inverse probability of censoring weights.
        """
        event, time = check_y_survival(y)
        Ghat = self.predict_proba(time[event])

        if (Ghat == 0.0).any():
            raise ValueError(
                "censoring survival function is zero at one or more time points"
            )

        weights = np.zeros(time.shape[0])
        weights[event] = 1.0 / Ghat

        return weights

# ========================
# Helper Functions
# ========================

def _get_conditional_probs_from_survival(surv):
    """
    Retrieved from: xgbse.non_parametric import _get_conditional_probs_from_survival

    Return conditional failure probabilities (for each time interval) from survival curve.
    P(T < t+1 | T > t): probability of failure up to time t+1 conditional on individual
    survival up to time t.

    Args:
        surv (pd.DataFrame): dataframe of survival estimates, as .predict() methods return

    Returns:
        pd.DataFrame: conditional failurer probability of event
            specifically at time bucket
    """

    conditional_preds = 1 - (surv / surv.shift(1, axis=1).fillna(1))
    conditional_preds = conditional_preds.fillna(0)

    return conditional_preds


def km_survival_function(
    T: np.ndarray, E: np.ndarray
) -> Tuple[KaplanMeierFitter, np.ndarray, np.ndarray, np.ndarray]:
    kmf = KaplanMeierFitter().fit(T, E)
    surv_fn = kmf.survival_function_.T.reset_index(drop=True)
    if len(surv_fn.columns) < 2:
        raise RuntimeError("invalid survival functin for extrapolation")

    hazards = _get_conditional_probs_from_survival(surv_fn)
    constant_hazard = hazards.values[:, -1:].mean(axis=1)[0]

    return kmf, surv_fn, hazards, constant_hazard


def generate_score(metric: np.ndarray) -> Tuple[float, float]:
    percentile_val = 1.96
    return (np.mean(metric), percentile_val * np.std(metric) / np.sqrt(len(metric)))


def print_score(score: Tuple[float, float]) -> str:
    return str(round(score[0], 4)) + " +/- " + str(round(score[1], 4))


def _check_estimate_1d(estimate: np.ndarray, test_time: np.ndarray) -> np.ndarray:
    estimate = check_array(estimate, ensure_2d=False)
    if estimate.ndim != 1:
        raise ValueError(
            "Expected 1D array, got {:d}D array instead:\narray={}.\n".format(
                estimate.ndim, estimate
            )
        )
    check_consistent_length(test_time, estimate)
    return estimate


def _check_inputs(
    event_indicator: np.ndarray, event_time: np.ndarray, estimate: np.ndarray
) -> np.ndarray:
    check_consistent_length(event_indicator, event_time, estimate)
    event_indicator = check_array(event_indicator, ensure_2d=False)
    event_time = check_array(event_time, ensure_2d=False)
    estimate = _check_estimate_1d(estimate, event_time)

    if not np.issubdtype(event_indicator.dtype, np.bool_):
        raise ValueError(
            "only boolean arrays are supported as class labels for survival analysis, got {0}".format(
                event_indicator.dtype
            )
        )

    if len(event_time) < 2:
        raise ValueError("Need a minimum of two samples")

    if not event_indicator.any():
        raise ValueError("All samples are censored")

    return event_indicator, event_time, estimate


def _check_times(test_time: np.ndarray, times: np.ndarray) -> np.ndarray:
    times = check_array(np.atleast_1d(times), ensure_2d=False, dtype=test_time.dtype)
    times = np.unique(times)

    if times.max() >= test_time.max() or times.min() < test_time.min():
        raise ValueError(
            "all times must be within follow-up time of test data: [{}; {}[".format(
                test_time.min(), test_time.max()
            )
        )

    return times


def _check_estimate_2d(
    estimate: np.ndarray, test_time: np.ndarray, time_points: np.ndarray
) -> np.ndarray:
    estimate = check_array(estimate, ensure_2d=False, allow_nd=False)
    time_points = _check_times(test_time, time_points)
    check_consistent_length(test_time, estimate)

    if estimate.ndim == 2 and estimate.shape[1] != time_points.shape[0]:
        raise ValueError(
            "expected estimate with {} columns, but got {}".format(
                time_points.shape[0], estimate.shape[1]
            )
        )

    return estimate, time_points


def _get_comparable(
    event_indicator: np.ndarray, event_time: np.ndarray, order: np.ndarray
) -> np.ndarray:
    n_samples = len(event_time)
    tied_time = 0
    comparable = {}
    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        start = i + 1
        end = start
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time
        for j in range(i, end):
            if event_indicator[order[j]]:
                mask = np.zeros(n_samples, dtype=bool)
                mask[end:] = True
                # an event is comparable to censored samples at same time point
                mask[i:end] = censored_at_same_time
                comparable[j] = mask
                tied_time += censored_at_same_time.sum()
        i = end

    return comparable, tied_time


def _estimate_concordance_index(
    event_indicator: np.ndarray,
    event_time: np.ndarray,
    estimate: np.ndarray,
    weights: np.ndarray,
    tied_tol: float = 1e-8,
) -> float:
    order = np.argsort(event_time)

    comparable, tied_time = _get_comparable(event_indicator, event_time, order)

    if len(comparable) == 0:
        raise RuntimeError(
            "Data has no comparable pairs, cannot estimate concordance index."
        )

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        w_i = weights[order[ind]]

        est = estimate[order[mask]]

        ties = np.absolute(est - est_i) <= tied_tol
        n_ties = ties.sum()
        # an event should have a higher score
        con = est < est_i
        n_con = con[~ties].sum()

        numerator += w_i * n_con + 0.5 * w_i * n_ties
        denominator += w_i * mask.sum()

        tied_risk += n_ties
        concordant += n_con
        discordant += est.size - n_con - n_ties

    cindex = numerator / denominator
    return cindex


def concordance_index_censored(
    event_indicator: np.ndarray,
    event_time: np.ndarray,
    estimate: np.ndarray,
    tied_tol: float = 1e-8,
) -> float:
    """Concordance index for right-censored data

    The concordance index is defined as the proportion of all comparable pairs
    in which the predictions and outcomes are concordant.

    Two samples are comparable if (i) both of them experienced an event (at different times),
    or (ii) the one with a shorter observed survival time experienced an event, in which case
    the event-free subject "outlived" the other. A pair is not comparable if they experienced
    events at the same time.

    Concordance intuitively means that two samples were ordered correctly by the model.
    More specifically, two samples are concordant, if the one with a higher estimated
    risk score has a shorter actual survival time.
    When predicted risks are identical for a pair, 0.5 rather than 1 is added to the count
    of concordant pairs.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb>`
    and [1]_ for further description.

    Parameters
    ----------
    event_indicator : array-like, shape = (n_samples,)
        Boolean array denotes whether an event occurred

    event_time : array-like, shape = (n_samples,)
        Array containing the time of an event or time of censoring

    estimate : array-like, shape = (n_samples,)
        Estimated risk of experiencing an event

    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.

    Returns
    -------
    cindex : float
        Concordance index
    """
    event_indicator, event_time, estimate = _check_inputs(
        event_indicator, event_time, estimate
    )

    w = np.ones_like(estimate)

    return _estimate_concordance_index(
        event_indicator, event_time, estimate, w, tied_tol
    )


def concordance_index_ipcw(
    survival_train: np.ndarray,
    survival_test: np.ndarray,
    estimate: np.ndarray,
    tau: Optional[float] = None,
    tied_tol: float = 1e-8,
) -> float:
    """Concordance index for right-censored data based on inverse probability of censoring weights.

    This is an alternative to the estimator in :func:`concordance_index_censored`
    that does not depend on the distribution of censoring times in the test data.
    Therefore, the estimate is unbiased and consistent for a population concordance
    measure that is free of censoring.

    It is based on inverse probability of censoring weights, thus requires
    access to survival times from the training data to estimate the censoring
    distribution. Note that this requires that survival times `survival_test`
    lie within the range of survival times `survival_train`. This can be
    achieved by specifying the truncation time `tau`.
    The resulting `cindex` tells how well the given prediction model works in
    predicting events that occur in the time range from 0 to `tau`.

    The estimator uses the Kaplan-Meier estimator to estimate the
    censoring survivor function. Therefore, it is restricted to
    situations where the random censoring assumption holds and
    censoring is independent of the features.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb>`
    and [1]_ for further description.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    estimate : array-like, shape = (n_samples,)
        Estimated risk of experiencing an event of test data.

    tau : float, optional
        Truncation time. The survival function for the underlying
        censoring time distribution :math:`D` needs to be positive
        at `tau`, i.e., `tau` should be chosen such that the
        probability of being censored after time `tau` is non-zero:
        :math:`P(D > \\tau) > 0`. If `None`, no truncation is performed.

    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.

    Returns
    -------
    cindex : float
        Concordance index

    """
    test_event, test_time = check_y_survival(survival_test)

    if tau is not None:
        mask = test_time < tau
        survival_test = survival_test[mask]

    estimate = _check_estimate_1d(estimate, test_time)

    cens = CensoringDistributionEstimator()
    cens.fit(survival_train)
    ipcw_test = cens.predict_ipcw(survival_test)
    if tau is None:
        ipcw = ipcw_test
    else:
        ipcw = np.empty(estimate.shape[0], dtype=ipcw_test.dtype)
        ipcw[mask] = ipcw_test
        ipcw[~mask] = 0

    w = np.square(ipcw)

    return _estimate_concordance_index(test_event, test_time, estimate, w, tied_tol)


def brier_score(
    survival_train: np.ndarray,
    survival_test: np.ndarray,
    estimate: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Estimate the time-dependent Brier score for right censored data.

    The time-dependent Brier score is the mean squared error at time point :math:`t`:

    .. math::

        \\mathrm{BS}^c(t) = \\frac{1}{n} \\sum_{i=1}^n I(y_i \\leq t \\land \\delta_i = 1)
        \\frac{(0 - \\hat{\\pi}(t | \\mathbf{x}_i))^2}{\\hat{G}(y_i)} + I(y_i > t)
        \\frac{(1 - \\hat{\\pi}(t | \\mathbf{x}_i))^2}{\\hat{G}(t)} ,

    where :math:`\\hat{\\pi}(t | \\mathbf{x})` is the predicted probability of
    remaining event-free up to time point :math:`t` for a feature vector :math:`\\mathbf{x}`,
    and :math:`1/\\hat{G}(t)` is a inverse probability of censoring weight, estimated by
    the Kaplan-Meier estimator.

    See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb#Time-dependent-Brier-Score>`
    and [1]_ for details.

    Parameters
    ----------
    survival_train : structured array, shape = (n_train_samples,)
        Survival times for training data to estimate the censoring
        distribution from.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    survival_test : structured array, shape = (n_samples,)
        Survival times of test data.
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    estimate : array-like, shape = (n_samples, n_times)
        Estimated risk of experiencing an event for test data at `times`.
        The i-th column must contain the estimated probability of
        remaining event-free up to the i-th time point.

    times : array-like, shape = (n_times,)
        The time points for which to estimate the Brier score.
        Values must be within the range of follow-up times of
        the test data `survival_test`.

    Returns
    -------
    brier_scores : array , shape = (n_times,)
        Values of the brier score.

    Examples
    --------
    >>> from sksurv.datasets import load_gbsg2
    >>> from sksurv.linear_model import CoxPHSurvivalAnalysis
    >>> from sksurv.metrics import brier_score
    >>> from sksurv.preprocessing import OneHotEncoder

    Load and prepare data.

    >>> X, y = load_gbsg2()
    >>> X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)
    >>> Xt = OneHotEncoder().fit_transform(X)

    Fit a Cox model.

    >>> est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)

    Retrieve individual survival functions and get probability
    of remaining event free up to 5 years (=1825 days).

    >>> survs = est.predict_survival_function(Xt)
    >>> preds = [fn(1825) for fn in survs]

    Compute the Brier score at 5 years.

    >>> times, score = brier_score(y, y, preds, 1825)
    >>> print(score)
    [0.20881843]

    See also
    --------
    integrated_brier_score
        Computes the average Brier score over all time points.

    References
    ----------
    .. [1] E. Graf, C. Schmoor, W. Sauerbrei, and M. Schumacher,
           "Assessment and comparison of prognostic classification schemes for survival data,"
           Statistics in Medicine, vol. 18, no. 17-18, pp. 2529–2545, 1999.
    """
    test_event, test_time = check_y_survival(survival_test)
    train_event, train_time = check_y_survival(survival_train)
    estimate, times = _check_estimate_2d(estimate, test_time, times)
    if estimate.ndim == 1 and times.shape[0] == 1:
        estimate = estimate.reshape(-1, 1)

    # fit IPCW estimator
    cens = CensoringDistributionEstimator().fit(survival_train)
    # calculate inverse probability of censoring weight at current time point t.
    prob_cens_t = cens.predict_proba(times)
    prob_cens_t[prob_cens_t == 0] = np.inf
    # calculate inverse probability of censoring weights at observed time point
    prob_cens_y = cens.predict_proba(test_time)
    prob_cens_y[prob_cens_y == 0] = np.inf

    # Calculating the brier scores at each time point
    brier_scores = np.empty(times.shape[0], dtype=float)
    for i, t in enumerate(times):
        est = estimate[:, i]
        is_case = (test_time <= t) & test_event
        is_control = test_time > t

        brier_scores[i] = np.mean(
            np.square(est) * is_case.astype(int) / prob_cens_y
            + np.square(1.0 - est) * is_control.astype(int) / prob_cens_t[i]
        )

    return brier_scores

def check_y_survival(
    y_or_event: np.ndarray, *args: Any, allow_all_censored: bool = False
) -> tuple:
    """Check that array correctly represents an outcome for survival analysis.

    Parameters
    ----------
    y_or_event : structured array with two fields, or boolean array
        If a structured array, it must contain the binary event indicator
        as first field, and time of event or time of censoring as
        second field. Otherwise, it is assumed that a boolean array
        representing the event indicator is passed.

    *args : list of array-likes
        Any number of array-like objects representing time information.
        Elements that are `None` are passed along in the return value.

    allow_all_censored : bool, optional, default: False
        Whether to allow all events to be censored.

    Returns
    -------
    event : array, shape=[n_samples,], dtype=bool
        Binary event indicator.

    time : array, shape=[n_samples,], dtype=float
        Time of event or censoring.
    """
    y = y_or_event

    if (
        not isinstance(y, np.ndarray)
        or y.dtype.fields is None
        or len(y.dtype.fields) != 2
    ):
        raise ValueError(
            "y must be a structured array with the first field"
            " being a binary class event indicator and the second field"
            " the time of the event/censoring"
        )

    event_field, time_field = y.dtype.names
    y_event = y[event_field]
    time_args = (y[time_field],)

    event = check_array(y_event, ensure_2d=False)
    if not np.issubdtype(event.dtype, np.bool_):
        raise ValueError(
            f"elements of event indicator must be boolean, but found {event.dtype}"
        )

    if not (allow_all_censored or np.any(event)):
        raise ValueError("all samples are censored")

    return_val = [event]
    for i, yt in enumerate(time_args):
        if yt is None:
            return_val.append(yt)
            continue

        yt = check_array(yt, ensure_2d=False)
        if not np.issubdtype(yt.dtype, np.number):
            raise ValueError(
                f"time must be numeric, but found {yt.dtype} for argument {i + 2}"
            )

        return_val.append(yt)

    return tuple(return_val)

def dataframe_hash(df: pd.DataFrame) -> str:
    """Dataframe hashing, used for caching/backups"""
    cols = sorted(list(df.columns))
    return str(abs(pd.util.hash_pandas_object(df[cols].fillna(0)).sum()))

def nonparametric_distance(
    real: Tuple[np.ndarray, np.ndarray],
    syn: Tuple[np.ndarray, np.ndarray],
    n_points: int = 1000,
    is_syn_type: str = 'time',
) -> Tuple[float, float, float]:
    """Calculate nonparametric distance between survival distributions."""
    if len(syn) == 0 or len(real) == 0:
        raise ValueError("Empty evaluation sets")

    real_T, real_E = real
    real_kmf, real_surv, real_hazards, real_constant_hazard = km_survival_function(real_T, real_E)

    syn_T, syn_E = syn
    if is_syn_type == 'prob':  # Synthetic data are probabilities
        time_points = np.array(list(syn_T.keys()))
        Tmax = time_points.max()
    else:  # Synthetic data are times
        syn_kmf, syn_surv, syn_hazards, syn_constant_hazard = km_survival_function(syn_T, syn_E)
        Tmin = max(0, min(real_T.min(), syn_T.min()))
        Tmax = max(real_T.max(), syn_T.max())
        time_points = np.linspace(Tmin, Tmax, n_points)

    abs_opt = []
    opt = []
    for t in time_points:
        real_local_pred = real_kmf.predict(t)
        if is_syn_type == 'prob':
            syn_local_pred = syn_T[t]
        else:  # Synthetic data are times
            syn_local_pred = syn_kmf.predict(t)

        if np.isnan(syn_local_pred):
            raise RuntimeError("syn_local_pred contains NaNs")
        if np.isnan(real_local_pred):
            raise RuntimeError("real_local_pred contains NaNs")

        abs_opt.append(abs(syn_local_pred - real_local_pred))
        opt.append(syn_local_pred - real_local_pred)

    auc_abs_opt = trapz(abs_opt, time_points) / Tmax
    auc_opt = trapz(opt, time_points) / Tmax
    if is_syn_type == 'prob':
        # Not useful metric here because we provide the max predicted time in the prompt
        sightedness = (real_T.max() - Tmax) / Tmax
    else:  # Synthetic data are times
        sightedness = (real_T.max() - syn_T.max()) / Tmax

    return auc_opt, auc_abs_opt, sightedness