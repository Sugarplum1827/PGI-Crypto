"""PGI-Crypto Process 1: CC eligibility of a candidate prime's centered window.

Finalized rule (G3), thresholds frozen and NOT tunable here::

    eligible  <=>  final class is CC          (MA, MGV, NONE are rejected)
              AND  P_CC >= 0.80               P_CC = P(EVENT) * P(CC | EVENT)
              AND  0.85 <= CR <= 1.15
              AND  fixed-parameter Exp(1) KS pass   (asymptotic p > 0.05)
              AND  CR inside the Gamma null interval (i.e. not MA by the analytical rule)

Every component is reported in :class:`EligibilityResult`; nothing is hidden.

Policies
--------
POLICY_STRICT enforces the rebuilt model's 10^12 range boundary and ML thresholds.
POLICY_EXPERIMENTAL reports the model output but does not enforce it; only the
analytical CR and KS conditions gate candidates. RSA-2048 factor primes are far
outside the rebuilt model's support, so Experimental results are not validation
of classifier behavior or cryptographic key quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from . import config
from .exceptions import CandidateRejectedError
from .labels import exact_ma_interval
from .result_types import EligibilityResult, WindowAnalysis, WindowClassification

ML_CHECKS = ("ml_final_class_is_cc", "p_cc_at_least_threshold")
RULE_CHECKS = ("cr_in_cc_band", "ks_pass", "not_ma_by_rule")


@dataclass(frozen=True)
class EligibilityPolicy:
    name: str
    enforce_ml_gate: bool
    p_cc_threshold: float = config.P_CC_THRESHOLD
    cr_band: tuple = config.CR_CC_BAND
    require_validated_range: bool = False       # if True, windows with p_mid > 1e12 are ineligible outright


POLICY_STRICT = EligibilityPolicy(name="strict_spec", enforce_ml_gate=True, require_validated_range=True)
POLICY_EXPERIMENTAL = EligibilityPolicy(name="experimental_rule_gate", enforce_ml_gate=False)


def compute_p_cc(p_event: float, p_cc_given_event: float) -> float:
    """``P_CC = P(EVENT) * P(CC | EVENT)`` (law of total probability for the cascade)."""
    return float(p_event) * float(p_cc_given_event)


def check_eligibility(analysis: WindowAnalysis, classification: WindowClassification,
                      policy: EligibilityPolicy = POLICY_STRICT, *, raise_on_reject: bool = False) -> EligibilityResult:
    lo, hi = exact_ma_interval()
    cr = analysis.cr
    band_lo, band_hi = policy.cr_band
    checks: Dict[str, bool] = {
        "ml_final_class_is_cc": classification.final_class == "CC",
        "p_cc_at_least_threshold": classification.p_cc >= policy.p_cc_threshold,
        "cr_in_cc_band": band_lo <= cr <= band_hi,
        "ks_pass": bool(analysis.ks_pass),
        "not_ma_by_rule": lo <= cr <= hi,
    }
    enforced: List[str] = list(RULE_CHECKS) + (list(ML_CHECKS) if policy.enforce_ml_gate else [])
    if policy.require_validated_range:
        checks["inside_validated_range"] = not analysis.outside_training_range
        enforced.append("inside_validated_range")
    failed = [n for n in enforced if not checks[n]]
    labels = {
        "ml_final_class_is_cc": "ml_class_%s_rejected" % classification.final_class,
        "p_cc_at_least_threshold": "p_cc_below_threshold",
        "cr_in_cc_band": "cr_outside_cc_band",
        "ks_pass": "ks_test_failed",
        "not_ma_by_rule": "ma_by_analytical_rule",
        "inside_validated_range": "outside_validated_range",
    }
    order = ["inside_validated_range", "ml_final_class_is_cc", "p_cc_at_least_threshold", "cr_in_cc_band", "ks_pass", "not_ma_by_rule"]
    failed.sort(key=order.index)
    reasons = tuple(labels[n] for n in failed)
    eligible = not failed
    warnings = []
    if analysis.outside_training_range:
        warnings.append(config.KEY_SCALE_WARNING)
    if not policy.enforce_ml_gate:
        ml_failed = [n for n in ML_CHECKS if not checks[n]]
        warnings.append("EXPERIMENTAL policy: ML conditions are reported but NOT enforced%s."
                        % ((" (would have failed: %s)" % ", ".join(ml_failed)) if ml_failed else ""))
    result = EligibilityResult(
        eligible=eligible, reason="eligible" if eligible else reasons[0], reasons=reasons, policy_name=policy.name,
        ml_gate_enforced=policy.enforce_ml_gate, p_event=classification.stage1_p_event,
        p_cc_given_event=classification.p_cc_given_event, p_cc=classification.p_cc, cr=cr, ks_pass=analysis.ks_pass,
        ks_pvalue=analysis.ks_pvalue, final_class=classification.final_class,
        outside_training_range=analysis.outside_training_range, checks=checks, enforced_checks=tuple(enforced),
        classification=classification, warnings=tuple(warnings))
    if raise_on_reject and not eligible:
        raise CandidateRejectedError("candidate rejected: %s" % ", ".join(reasons), eligibility=result)
    return result
