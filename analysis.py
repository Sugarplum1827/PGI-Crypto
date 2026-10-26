"""Window analysis and frozen-model prediction for a single centered window.

``analyze_window``  statistics only (features, CR, KS, range status)      -> WindowAnalysis
``predict_window``  frozen two-stage XGBoost on that window's features   -> WindowClassification
``labels.apply_label_rules``  the analytical label definitions            -> LabelRuleResult

The three are deliberately separate: the XGBoost output is never presented as the label rule.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import config
from .classification import FrozenClassifier
from .exceptions import FeatureMismatchError
from .features import compute_analytical_statistics, compute_features
from .labels import MaximalGapCatalogue, exact_ma_interval
from .result_types import WindowAnalysis, WindowClassification
from .windows import CenteredWindow


def analyze_window(window: CenteredWindow, *, window_index: Optional[int] = None,
                   catalogue: Optional[MaximalGapCatalogue] = None, ks_alpha: float = config.KS_ALPHA) -> WindowAnalysis:
    """Features + analytical statistics of one centered window (no ML)."""
    gaps = window.gap_matrix()
    center = window.center
    feats = compute_features(gaps, [center])
    stats = compute_analytical_statistics(gaps, [center])
    cr = float(stats.CR[0])
    ks_p = float(stats.ks_pvalue_exponential[0])
    lo, hi = exact_ma_interval()
    band_lo, band_hi = config.CR_CC_BAND
    outside = center > config.VALIDATED_MAX_P_MID
    warnings = []
    if outside:
        warnings.append("p_mid is outside the newly trained model's supported range (%s): any ML output for this window is an "
                        "unvalidated extrapolation." % config.VALIDATED_RANGE_TEXT)
    if window.neighbour_primality == "probable":
        warnings.append("Neighbour primes of this window are strong probable primes (bases 2 and 3), not proofs.")
    info = None
    if catalogue is not None and window_index is not None and not outside:
        info = {"records_nearby": catalogue.records_near_window(int(window_index)),
                "is_mgv_by_catalogue_rule": int(window_index) in catalogue.mgv_window_ids(int(window_index) + 6)}
    return WindowAnalysis(
        window=window, window_size=window.window_size, p_mid=center, p_start=window.p_start, p_end=window.p_end,
        features=feats, feature_values=feats.row_as_dict(0), cr=cr, cr_null_pvalue=float(stats.CR_null_pvalue_gamma_k_k[0]),
        ks_statistic=float(stats.ks_statistic_exponential[0]), ks_pvalue=ks_p, ks_pass=ks_p > ks_alpha, ks_alpha=ks_alpha,
        cr_in_null_interval=bool(lo <= cr <= hi), cr_in_cc_band=bool(band_lo <= cr <= band_hi),
        outside_training_range=bool(outside), validated_range=config.VALIDATED_RANGE_TEXT, maximal_gap_info=info,
        neighbour_primality=window.neighbour_primality, warnings=tuple(warnings))


def predict_window(analysis: WindowAnalysis, classifier,
                   stage1_threshold: float = config.STAGE1_THRESHOLD_BASELINE) -> WindowClassification:
    """Classify one window using the fresh four-class model or legacy baseline."""
    if analysis.window_size != config.WINDOW_GAPS:
        raise FeatureMismatchError("the classifier needs window_size=%d (got %d)" % (config.WINDOW_GAPS, analysis.window_size))
    if hasattr(classifier, "predict_windows"):
        prediction = classifier.predict_windows(
            analysis.window.gap_matrix(), [analysis.p_mid], [analysis.p_start], [analysis.p_end])
        probs = {name: float(value) for name, value in zip(prediction.classes, prediction.probabilities[0])}
        p_event = sum(probs[name] for name in ("CC", "MA", "MGV"))
        if p_event > 0:
            p2 = {name: probs[name] / p_event for name in ("CC", "MA", "MGV")}
        else:
            p2 = {name: 1.0 / 3.0 for name in ("CC", "MA", "MGV")}
        stage2_class = max(p2, key=p2.get)
        event = p_event >= float(stage1_threshold)
        info = classifier.info
        return WindowClassification(
            stage1_class="EVENT" if event else "NONE", stage1_p_event=p_event,
            stage1_threshold=float(stage1_threshold), stage2_class=stage2_class,
            stage2_probabilities=p2, p_cc_given_event=p2["CC"], p_cc=p_event * p2["CC"],
            final_class=stage2_class if event else "NONE",
            model_version="%s / %s" % (info.methodology_version, info.feature_schema_version),
            backend=info.backend, validated_range=config.VALIDATED_RANGE_TEXT,
            outside_training_range=bool(analysis.outside_training_range or prediction.outside_training_range[0]),
            model_info=info)
    r = classifier.classify(analysis.features, stage1_threshold=stage1_threshold, stage2="all")
    p_event = float(r.stage1_event_probability[0])
    p2 = {c: float(v) for c, v in zip(config.STAGE2_CLASSES, r.stage2_probabilities[0])}
    stage2_class = max(p2, key=p2.get)
    event = bool(r.stage1_event[0])
    return WindowClassification(
        stage1_class="EVENT" if event else "NONE", stage1_p_event=p_event, stage1_threshold=float(stage1_threshold),
        stage2_class=stage2_class, stage2_probabilities=p2, p_cc_given_event=p2["CC"], p_cc=p_event * p2["CC"],
        final_class=str(r.final_class[0]), model_version="%s / %s" % (config.METHODOLOGY_VERSION, config.FEATURE_SCHEMA_VERSION),
        backend=r.model_info.backend, validated_range=config.VALIDATED_RANGE_TEXT,
        outside_training_range=analysis.outside_training_range, model_info=r.model_info)
