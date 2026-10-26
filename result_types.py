"""Typed result objects.

Design rules
------------
* Analytical statistics (CR, KS, CR null p-value) are a different type from the
  ML feature matrix, so they cannot be passed to the classifier by accident.
* Key material is wrapped in :class:`SecretValue`, whose ``repr``/``str`` never
  reveal the secret.  Nothing in this package prints or logs a secret.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import config
from .windows import CenteredWindow

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureMatrix:
    """The ML feature vector(s): shape (n_windows, 24), frozen column order."""

    values: np.ndarray
    names: Tuple[str, ...] = config.ML_FEATURE_NAMES
    schema_version: str = config.FEATURE_SCHEMA_VERSION

    def __post_init__(self):
        if self.values.ndim != 2 or self.values.shape[1] != len(self.names):
            raise ValueError("feature matrix must have shape (n, %d), got %s" % (len(self.names), self.values.shape))

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    def column(self, name: str) -> np.ndarray:
        return self.values[:, self.names.index(name)]

    def row_as_dict(self, i: int) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.values[i])}


@dataclass(frozen=True)
class AnalyticalStatistics:
    """Reporting-only statistics.  NEVER used as ML predictors (leakage)."""

    CR: np.ndarray                         # mean_gap / ln(p_mid)
    CR_null_pvalue_gamma_k_k: np.ndarray   # two-sided, Gamma(k, rate k) null
    ks_statistic_exponential: np.ndarray   # KS D of gap/ln(p_mid) vs Exp(1), fixed parameter
    ks_pvalue_exponential: np.ndarray      # asymptotic Kolmogorov p-value of D

    @property
    def n_rows(self) -> int:
        return int(self.CR.shape[0])

    def cr_in_null_interval(self) -> np.ndarray:
        """True where CR lies inside the 95 % Gamma null interval [0.8136, 1.2053]."""
        lo, hi = config.CR_NULL_INTERVAL
        return (self.CR >= lo) & (self.CR <= hi)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    methodology_version: str
    feature_schema_version: str
    backend: str
    stage1_file: str
    stage1_sha256: str
    stage2_file: str
    stage2_sha256: str
    stage2_classes: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ClassificationResult:
    """Two-stage frozen XGBoost output for n windows.

    ``stage1_event_probability``  P(EVENT) from Stage 1.
    ``stage1_event``              probability >= ``stage1_threshold``.
    ``stage2_probabilities``      (n, 3) P(class | EVENT) in ``stage2_classes``
                                  order; NaN where Stage 2 was not evaluated.
    ``stage2_prediction``         argmax class where evaluated, else None.
    ``final_class``               NONE if Stage 1 says no event, else Stage 2.
    """

    stage1_event_probability: np.ndarray
    stage1_threshold: float
    stage1_event: np.ndarray
    stage2_probabilities: np.ndarray
    stage2_evaluated: np.ndarray
    stage2_prediction: np.ndarray
    final_class: np.ndarray
    model_info: ModelInfo

    @property
    def n_rows(self) -> int:
        return int(self.stage1_event_probability.shape[0])

    def counts(self) -> Dict[str, int]:
        out = {c: 0 for c in config.FINAL_CLASSES}
        for c in self.final_class:
            out[str(c)] += 1
        return out


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisResult:
    """Result of ``PGICrypto.analyze`` for a prime sequence."""

    n_primes: int
    n_windows: int
    window_spec: Tuple[int, int, int]
    validation_level: str
    p_start: np.ndarray
    p_mid: np.ndarray
    p_end: np.ndarray
    features: FeatureMatrix
    analytics: AnalyticalStatistics
    classification: ClassificationResult
    outside_training_range: np.ndarray   # bool per window: p_mid > config.FROZEN_LIMIT_N (classification unvalidated)

    def to_records(self) -> List[Dict[str, Any]]:
        """One plain dict per window (no secrets are involved)."""
        c = self.classification
        recs = []
        for i in range(self.n_windows):
            recs.append({
                "window_index": i,
                "p_start": int(self.p_start[i]), "p_mid": int(self.p_mid[i]), "p_end": int(self.p_end[i]),
                "stage1_event_probability": float(c.stage1_event_probability[i]),
                "stage2_prediction": None if c.stage2_prediction[i] is None else str(c.stage2_prediction[i]),
                "final_class": str(c.final_class[i]),
                "outside_training_range": bool(self.outside_training_range[i]),
                "CR": float(self.analytics.CR[i]),
                "ks_statistic_exponential": float(self.analytics.ks_statistic_exponential[i]),
                "ks_pvalue_exponential": float(self.analytics.ks_pvalue_exponential[i]),
                "CR_null_pvalue_gamma_k_k": float(self.analytics.CR_null_pvalue_gamma_k_k[i]),
            })
        return recs


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False      # True: not applicable here (passed is True, detail says why)


@dataclass(frozen=True)
class ValidationReport:
    subject: str
    checks: Tuple[ValidationCheck, ...]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    @property
    def failures(self) -> Tuple[ValidationCheck, ...]:
        return tuple(c for c in self.checks if not c.passed)

    @property
    def skipped(self) -> Tuple["ValidationCheck", ...]:
        return tuple(c for c in self.checks if c.skipped)

    def summary(self) -> str:
        extra = (" (%d skipped: not applicable)" % len(self.skipped)) if self.skipped else ""
        return "%s: %d/%d checks passed%s" % (self.subject, sum(c.passed for c in self.checks), len(self.checks), extra)


class ValidationStatus(enum.Enum):
    NOT_VALIDATED = "not_validated"
    PASSED = "passed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


class SecretValue:
    """Opaque holder for secret material.  ``repr``/``str`` are redacted."""

    __slots__ = ("_value",)

    def __init__(self, value: Any):
        object.__setattr__(self, "_value", value)

    def reveal(self) -> Any:
        """Return the secret.  The only way to read it; call it deliberately."""
        return self._value

    def __repr__(self) -> str:
        return "SecretValue(<redacted>)"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return repr(self)

    def __setattr__(self, name, value):
        raise AttributeError("SecretValue is immutable")

    def __eq__(self, other) -> bool:
        return isinstance(other, SecretValue) and self._value == other._value

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SourceWindow:
    """Prime-gap window a key was derived from (provenance, not secret)."""

    primes: Tuple[int, ...]
    start_index: Optional[int] = None

    @property
    def p_mid(self) -> int:
        return self.primes[len(self.primes) // 2]


@dataclass(frozen=True)
class KeyResult:
    """Structured key-generation output.

    ``key`` is a :class:`SecretValue`; everything else is non-secret metadata.
    ``to_public_dict`` never contains the secret.
    """

    key: SecretValue
    key_length_bits: int
    algorithm: str
    source_window: Optional[SourceWindow]
    classification: Optional[Dict[str, Any]]
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    validation_status: ValidationStatus = ValidationStatus.NOT_VALIDATED

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "key": "<redacted>",
            "key_length_bits": self.key_length_bits,
            "algorithm": self.algorithm,
            "source_window": None if self.source_window is None else {
                "start_index": self.source_window.start_index,
                "p_mid": self.source_window.p_mid,
                "n_primes": len(self.source_window.primes),
            },
            "classification": self.classification,
            "generation_metadata": dict(self.generation_metadata),
            "validation_status": self.validation_status.value,
        }


# ===========================================================================
# PGI-Crypto system types (finalized specification)
# ===========================================================================


@dataclass(frozen=True)
class WindowAnalysis:
    """Statistical analysis of ONE centered window (no ML involved).

    ``window`` and the prime values are excluded from ``repr``; for key
    candidates use ``summary(redact_primes=True)``.
    """

    window: CenteredWindow = field(repr=False)
    window_size: int
    p_mid: int = field(repr=False)
    p_start: int = field(repr=False)
    p_end: int = field(repr=False)
    features: FeatureMatrix = field(repr=False)
    feature_values: Dict[str, float] = field(repr=False)
    cr: float
    cr_null_pvalue: float
    ks_statistic: float
    ks_pvalue: float
    ks_pass: bool
    ks_alpha: float
    cr_in_null_interval: bool
    cr_in_cc_band: bool
    outside_training_range: bool
    validated_range: str
    maximal_gap_info: Optional[Dict[str, Any]]
    neighbour_primality: str
    warnings: Tuple[str, ...]

    def summary(self, redact_primes: bool = False) -> Dict[str, Any]:
        red = "REDACTED"
        return {
            "p_mid": red if redact_primes else self.p_mid,
            "p_start": red if redact_primes else self.p_start,
            "p_end": red if redact_primes else self.p_end,
            "p_mid_bits": self.p_mid.bit_length(),
            "window_size": self.window_size,
            "mean_gap": self.feature_values["mean_gap"],
            "cr": self.cr, "cr_null_pvalue": self.cr_null_pvalue,
            "ks_statistic": self.ks_statistic, "ks_pvalue": self.ks_pvalue, "ks_pass": self.ks_pass,
            "cr_in_cc_band": self.cr_in_cc_band, "cr_in_null_interval": self.cr_in_null_interval,
            "outside_training_range": self.outside_training_range, "validated_range": self.validated_range,
            "neighbour_primality": self.neighbour_primality,
        }


@dataclass(frozen=True)
class WindowClassification:
    """Frozen two-stage XGBoost output for ONE window.

    ``stage2_probabilities`` are P(class | EVENT) and are reported for every
    window (including those Stage 1 calls NONE) so that
    ``p_cc = P(EVENT) * P(CC | EVENT)`` is always defined.  ``final_class`` follows the frozen
    cascade: NONE unless Stage 1 says EVENT, otherwise the Stage 2 argmax.
    """

    stage1_class: str                       # "EVENT" | "NONE"
    stage1_p_event: float
    stage1_threshold: float
    stage2_class: str                       # argmax of P(. | EVENT); informational when stage1_class == "NONE"
    stage2_probabilities: Dict[str, float]  # {"CC": .., "MA": .., "MGV": ..}
    p_cc_given_event: float
    p_cc: float                             # P(EVENT) * P(CC | EVENT)
    final_class: str                        # CC | MA | MGV | NONE
    model_version: str
    backend: str
    validated_range: str
    outside_training_range: bool
    model_info: ModelInfo = field(repr=False)


@dataclass(frozen=True)
class EligibilityResult:
    """PGI-Crypto Process 1 outcome for one candidate window (all components visible)."""

    eligible: bool
    reason: str
    reasons: Tuple[str, ...]
    policy_name: str
    ml_gate_enforced: bool
    p_event: float
    p_cc_given_event: float
    p_cc: float
    cr: float
    ks_pass: bool
    ks_pvalue: float
    final_class: str
    outside_training_range: bool
    checks: Dict[str, bool]
    enforced_checks: Tuple[str, ...]
    classification: WindowClassification = field(repr=False)
    warnings: Tuple[str, ...] = ()

    @property
    def final_cc_eligible(self) -> bool:
        return self.eligible


@dataclass(frozen=True)
class PrimeCandidate:
    """One candidate prime examined during rejection sampling (public metadata only).

    ``value`` and ``window`` are secret (a prime's neighbours locate it) and are only kept
    for accepted candidates; both are redacted in ``repr``.
    """

    attempt: int
    bits: int
    role: str                       # "p" | "q" | "pool" | "rejected"
    accepted: bool
    rejection_reason: Optional[str]
    eligibility: Optional[EligibilityResult]
    analysis: Optional[Dict[str, Any]]      # WindowAnalysis.summary(redact_primes=True)
    seconds: float
    value: Optional[SecretValue] = field(default=None, repr=False)
    window: Optional[SecretValue] = field(default=None, repr=False)


@dataclass(frozen=True)
class SeparationResult:
    """Prime-separation constraint ``|p - q| > 2 ** (modulus_bits / 2 - 100)``.

    ``modulus_bits`` (never the ambiguous ``k``) is the explicit parameter.  |p - q| together with N
    reveals p and q ((p+q)^2 = (p-q)^2 + 4N), so the exact distance is a SecretValue; only its bit length is public.
    """

    modulus_bits: int
    threshold_exponent: int
    separation_threshold: int
    actual_prime_distance_bits: int
    actual_prime_distance: SecretValue = field(repr=False)
    separation_pass: bool
    interpretation: str


@dataclass(frozen=True)
class RSAPrivateMaterial:
    """Raw RSA research values.  Every field is secret; ``repr`` is redacted."""

    p: int = field(repr=False)
    q: int = field(repr=False)
    d: int = field(repr=False)
    dp: int = field(repr=False)
    dq: int = field(repr=False)
    qinv: int = field(repr=False)
    n: int
    e: int

    def __repr__(self) -> str:
        return "RSAPrivateMaterial(n_bits=%d, e=%d, p/q/d/CRT=<redacted>)" % (self.n.bit_length(), self.e)

    __str__ = __repr__


class KeyValidationResult(ValidationReport):
    """:class:`ValidationReport` for a generated key (subject ``generated_key``)."""


@dataclass(frozen=True)
class ProgressEvent:
    """Progress notification passed to the optional ``progress`` callback (public data only)."""

    stage: str            # candidate_generated | window_built | classified | candidate_rejected | candidate_accepted | ...
    attempt: int
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KeyGenerationResult:
    """Outcome of ``generate``.  The private key is in ``private_material`` (a SecretValue)."""

    success: bool
    modulus_bits: int
    modulus: int
    public_exponent: int
    p_info: Optional[PrimeCandidate]
    q_info: Optional[PrimeCandidate]
    entropy: Any                                   # entropy.EntropyResult
    separation: Optional[SeparationResult]
    candidate_attempts: int
    candidates: Tuple[PrimeCandidate, ...]
    generation_time_seconds: float
    timings: Dict[str, float]
    validation: Optional[KeyValidationResult]
    policy_name: str
    ml_gate_enforced: bool
    pool_size: int
    selection_effect: bool                         # True only if pool_size > 2 allowed an actual choice
    rng: Dict[str, Any]
    is_secure_mode: bool
    spec_version: str
    warnings: Tuple[str, ...]
    research_limitations: Tuple[str, ...]
    private_material: Optional[SecretValue] = field(default=None, repr=False)
    policy: Any = field(default=None, repr=False)      # the EligibilityPolicy object used (needed to re-validate custom policies)

    def redacted_info(self) -> Dict[str, Any]:
        """Everything a UI may show: no p, q, d, CRT values, no private key."""
        red = "REDACTED"
        return {
            "success": self.success, "modulus_bits": self.modulus_bits, "public_exponent": self.public_exponent,
            "p": red, "q": red, "d": red, "CRT": red, "private_key": red,
            "modulus_N_bits": self.modulus.bit_length(),
            "entropy_statistic": None if self.entropy is None else self.entropy.statistic,
            "separation_pass": None if self.separation is None else self.separation.separation_pass,
            "candidate_attempts": self.candidate_attempts,
            "generation_time_seconds": self.generation_time_seconds,
            "policy": self.policy_name, "ml_gate_enforced": self.ml_gate_enforced,
            "rng": dict(self.rng), "validation": None if self.validation is None else self.validation.summary(),
            "warnings": list(self.warnings),
        }
