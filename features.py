"""Feature extraction (frozen definitions).

Every formula below was recovered from the frozen training dataset
(``prime_gap_features_FINAL_THESIS.csv``) and reproduces all 115,228 rows to
float round-off (see PGI_CRYPTO_CORE_VALIDATION.md).  Notation: ``g`` = the k
gaps of a window, ``m`` = mean(g), ``p`` = p_mid, ``ln`` = natural log.

ML features (24, frozen order, see ``config.ML_FEATURE_NAMES``)
---------------------------------------------------------------
mean_gap                 m
std_dev                  population std (ddof = 0)
skewness                 biased Fisher-Pearson  mean(d^3) / mean(d^2)^1.5, d = g - m
excess_kurtosis          biased  mean(d^4) / mean(d^2)^2 - 3
autocorr_lag{1..5}       sum_{t<k-h} d_t d_{t+h} / sum_t d_t^2   (biased sample ACF)
twin_prime_density       count(g == 2) / k
gap_entropy              Shannon entropy, log2, of the empirical gap-value distribution
interquartile_range      Q3 - Q1, index-based:  sorted[ceil(.75(k-1))] - sorted[ceil(.25(k-1))]
                         (== sorted[75] - sorted[25] for k = 100; numpy method="higher")
max_gap_ratio            max(g) / m
min_gap, max_gap         min(g), max(g)
gap_range                max(g) - min(g)
cv_coeff_variation       std_dev / m
prop_even_gt_2logp       count(g even AND g > 2 ln p) / k
median_gap               sorted[k // 2]  (upper median; numpy method="higher")
poisson_lambda_fit       m   (exact duplicate of mean_gap, kept: the frozen model uses it)
gap_freq_mode            most frequent gap value; ties -> smallest value
large_gap_fraction       count(g > 2 m) / k
log_mean_gap             ln(m)
CR_log2_legacy_...       m / ln(p)^2   (= CR / ln p; a deterministic proxy of CR, see note)

NOTE (unresolved, reported not fixed): the last feature is part of the frozen
model's input although it is a deterministic function of CR and p_mid.  The
library reproduces the frozen contract exactly; it does not decide this issue.

Analytical statistics (reporting only, never ML inputs)
-------------------------------------------------------
CR                         m / ln p
CR_null_pvalue_gamma_k_k   2 * min(F, 1 - F), F = Gamma(shape k, rate k) CDF at CR
ks_statistic_exponential   KS D of (g / ln p) against Exp(1), fixed parameter (no fitting)
ks_pvalue_exponential      asymptotic Kolmogorov survival function at sqrt(k) * D
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy import special

from . import config
from .exceptions import FeatureExtractionError
from .result_types import AnalyticalStatistics, FeatureMatrix
from .windows import WindowSet

_CHUNK = 8192


def ln_p_mid(p_mid) -> np.ndarray:
    """Natural log of p_mid as float64 (handles Python ints beyond int64)."""
    arr = np.asarray(p_mid)
    if arr.dtype == object:
        vals = [int(v) for v in arr]
        out = np.array([math.log(v) if v >= 2 else float("nan") for v in vals], dtype=np.float64)
    else:
        if arr.dtype.kind not in "iuf":
            raise FeatureExtractionError("p_mid must be numeric")
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(arr >= 2, np.log(arr.astype(np.float64)), np.nan)
    if np.isnan(out).any():
        raise FeatureExtractionError("every p_mid must be >= 2")
    return out


def _value_run_stats(sorted_int: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row Shannon entropy (log2) and mode (smallest value among ties)."""
    n, k = sorted_int.shape
    new_run = np.ones((n, k), dtype=np.bool_)
    new_run[:, 1:] = sorted_int[:, 1:] != sorted_int[:, :-1]
    starts = np.flatnonzero(new_run.ravel())
    row = starts // k
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = n * k
    counts = ends - starts
    values = sorted_int.ravel()[starts]
    p = counts / float(k)
    entropy = -np.bincount(row, weights=p * np.log2(p), minlength=n)
    max_count = np.zeros(n, dtype=np.int64)
    np.maximum.at(max_count, row, counts)
    hit = np.flatnonzero(counts == max_count[row])
    _, first = np.unique(row[hit], return_index=True)   # runs ascend in value within a row
    return entropy, values[hit[first]].astype(np.float64)


def _features_chunk(gaps_int: np.ndarray, ln_p: np.ndarray) -> np.ndarray:
    n, k = gaps_int.shape
    g = gaps_int.astype(np.float64)
    out = np.empty((n, config.N_ML_FEATURES), dtype=np.float64)
    mean = g.mean(axis=1)
    d = g - mean[:, None]
    d2 = d * d
    m2 = d2.mean(axis=1)
    std = np.sqrt(m2)
    srt = np.sort(gaps_int, axis=1)
    entropy, mode = _value_run_stats(srt)
    with np.errstate(divide="ignore", invalid="ignore"):
        valid = m2 > 0
        skew = np.where(valid, (d2 * d).mean(axis=1) / m2 ** 1.5, np.nan)
        kurt = np.where(valid, (d2 * d2).mean(axis=1) / m2 ** 2 - 3.0, np.nan)
        denom = d2.sum(axis=1)
        acf = [np.where(valid, (d[:, : k - h] * d[:, h:]).sum(axis=1) / denom, np.nan) for h in range(1, 6)]
    q1 = srt[:, (k + 2) // 4].astype(np.float64)
    q3 = srt[:, (3 * k) // 4].astype(np.float64)
    gmin = srt[:, 0].astype(np.float64)
    gmax = srt[:, -1].astype(np.float64)
    cols = {
        "mean_gap": mean,
        "std_dev": std,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "autocorr_lag1": acf[0], "autocorr_lag2": acf[1], "autocorr_lag3": acf[2],
        "autocorr_lag4": acf[3], "autocorr_lag5": acf[4],
        "twin_prime_density": (gaps_int == 2).mean(axis=1),
        "gap_entropy": entropy,
        "interquartile_range": q3 - q1,
        "max_gap_ratio": gmax / mean,
        "min_gap": gmin,
        "max_gap": gmax,
        "gap_range": gmax - gmin,
        "cv_coeff_variation": std / mean,
        "prop_even_gt_2logp": (((gaps_int % 2) == 0) & (g > 2.0 * ln_p[:, None])).mean(axis=1),
        "median_gap": srt[:, k // 2].astype(np.float64),
        "poisson_lambda_fit": mean,
        "gap_freq_mode": mode,
        "large_gap_fraction": (g > 2.0 * mean[:, None]).mean(axis=1),
        "log_mean_gap": np.log(mean),
        "CR_log2_legacy_DO_NOT_USE_FOR_CLASSIFICATION": mean / ln_p ** 2,
    }
    for j, name in enumerate(config.ML_FEATURE_NAMES):
        out[:, j] = cols[name]
    return out


def _check_inputs(gap_matrix: np.ndarray, p_mid) -> Tuple[np.ndarray, np.ndarray]:
    g = np.asarray(gap_matrix)
    if g.ndim != 2 or g.shape[0] == 0:
        raise FeatureExtractionError("gap matrix must be 2-D with at least one window")
    if g.shape[1] < 6:
        raise FeatureExtractionError("windows need at least 6 gaps (autocorrelation up to lag 5)")
    if g.dtype.kind not in "iu":
        raise FeatureExtractionError("gap matrix must have an integer dtype (no silent float coercion)")
    if (g < 1).any():
        raise FeatureExtractionError("prime gaps must be >= 1")
    lnp = ln_p_mid(p_mid)
    if lnp.shape[0] != g.shape[0]:
        raise FeatureExtractionError("p_mid must have one entry per window")
    return g.astype(np.int64, copy=False), lnp


def compute_features(gap_matrix: np.ndarray, p_mid) -> FeatureMatrix:
    """ML feature matrix for arbitrary (n, k) integer gap windows and their p_mid.

    The frozen classifier requires k = 100; other k are accepted here so the
    formulas can be unit-tested on small hand-checkable windows.
    """
    g, lnp = _check_inputs(gap_matrix, p_mid)
    parts = [_features_chunk(g[i:i + _CHUNK], lnp[i:i + _CHUNK]) for i in range(0, g.shape[0], _CHUNK)]
    values = np.concatenate(parts, axis=0)
    values.flags.writeable = False
    return FeatureMatrix(values=values)


def compute_analytical_statistics(gap_matrix: np.ndarray, p_mid) -> AnalyticalStatistics:
    """CR, CR null p-value and fixed-parameter Exp(1) KS statistics (reporting only)."""
    g, lnp = _check_inputs(gap_matrix, p_mid)
    n, k = g.shape
    cr = np.empty(n)
    cr_p = np.empty(n)
    ks_d = np.empty(n)
    ks_p = np.empty(n)
    idx = np.arange(1, k + 1, dtype=np.float64)
    for s in range(0, n, _CHUNK):
        sl = slice(s, min(s + _CHUNK, n))
        gc = g[sl].astype(np.float64)
        cr_c = gc.mean(axis=1) / lnp[sl]
        cdf = special.gammainc(k, k * cr_c)
        sf = special.gammaincc(k, k * cr_c)
        cr[sl] = cr_c
        cr_p[sl] = 2.0 * np.minimum(cdf, sf)
        x = np.sort(gc / lnp[sl][:, None], axis=1)
        f = 1.0 - np.exp(-x)
        d_plus = (idx / k - f).max(axis=1)
        d_minus = (f - (idx - 1.0) / k).max(axis=1)
        ks_d[sl] = np.maximum(d_plus, d_minus)
    ks_p[:] = special.kolmogorov(math.sqrt(k) * ks_d)
    for a in (cr, cr_p, ks_d, ks_p):
        a.flags.writeable = False
    return AnalyticalStatistics(CR=cr, CR_null_pvalue_gamma_k_k=cr_p,
                                ks_statistic_exponential=ks_d, ks_pvalue_exponential=ks_p)


def extract_ml_features(windows: WindowSet) -> FeatureMatrix:
    """ML feature matrix (24 columns, frozen order) for a :class:`WindowSet`."""
    return compute_features(windows.gap_matrix, windows.p_mid)


def extract_analytical_statistics(windows: WindowSet) -> AnalyticalStatistics:
    """Analytical (non-ML) statistics for a :class:`WindowSet`."""
    return compute_analytical_statistics(windows.gap_matrix, windows.p_mid)
