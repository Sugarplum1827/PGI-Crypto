"""Archived two-stage compatibility pipeline from the earlier PGI-Crypto system.

``PGICrypto`` loads the historical frozen classifiers. It is retained only to
reproduce the legacy implementation and is not used by the fresh 10^12-range
model comparisons, rebuilt Streamlit panel, or rebuilt Experimental key
benchmark. Use ``PGICryptoResearchSystem`` and ``PGICryptoKeyGenerator`` for
the current research prototype; their predictions do not approve cryptographic
keys.
"""
from __future__ import annotations

import platform
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np

from . import config
from .classification import BACKEND_JSON, STAGE2_GATED, FrozenClassifier
from .analysis import analyze_window, predict_window
from .eligibility import POLICY_STRICT, EligibilityPolicy, check_eligibility
from .exceptions import InvalidPrimeInputError, OutOfTrainingRangeWarning
from .labels import LabelRuleResult, MaximalGapCatalogue, apply_label_rules
from .features import compute_features, extract_analytical_statistics, extract_ml_features
from .key_generation import PGICryptoKeyGenerator, specification_status
from .serialization import serialize_private_key, serialize_public_key
from .primes import LEVEL_PRIMALITY, PrimeSequence, validate_prime_sequence
from .result_types import (AnalysisResult, ClassificationResult, EligibilityResult, FeatureMatrix, KeyGenerationResult, KeyResult,
                    KeyValidationResult, ValidationReport, WindowAnalysis, WindowClassification)
from .validation import validate_analysis, validate_generated_key, validate_generation, validate_key, validate_output
from .windows import (FROZEN_WINDOW_SPEC, CenteredWindow, WindowSpec, build_centered_window, build_windows,
                      centered_window_from_primes)


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.array(a, copy=True)
    a.flags.writeable = False
    return a


class PGICrypto:
    """Legacy frozen-model facade retained for historical reproducibility.

    The recreated 10^12-range public workflow is ``PGICryptoResearchSystem``.
    Do not use this compatibility facade for new thesis results.
    """

    def __init__(self, model_dir=None, backend: str = BACKEND_JSON, verify_hashes: bool = True,
                 validation_level: str = LEVEL_PRIMALITY):
        self.classifier = FrozenClassifier(model_dir=model_dir, backend=backend, verify_hashes=verify_hashes)
        self.validation_level = validation_level
        self.catalogue = MaximalGapCatalogue.default()
        self._keygen = PGICryptoKeyGenerator(self.classifier)
        self._frozen_primes = None

    # -- inputs -------------------------------------------------------------
    def validate_primes(self, primes, level: Optional[str] = None) -> PrimeSequence:
        return validate_prime_sequence(primes, level=level or self.validation_level)

    # -- analysis -----------------------------------------------------------
    def analyze(self, primes, stage1_threshold: float = config.STAGE1_THRESHOLD_BASELINE,
                stage2: str = STAGE2_GATED, validation_level: Optional[str] = None) -> AnalysisResult:
        """Run the full analysis pipeline on a sequence of consecutive primes (>= 101)."""
        seq = primes if isinstance(primes, PrimeSequence) else self.validate_primes(primes, validation_level)
        ws = build_windows(seq, FROZEN_WINDOW_SPEC)
        feats = extract_ml_features(ws)
        analytics = extract_analytical_statistics(ws)
        cls = self.classifier.classify(feats, stage1_threshold=stage1_threshold, stage2=stage2)
        outside = np.array([int(v) > config.FROZEN_LIMIT_N for v in ws.p_mid], dtype=np.bool_)
        outside.flags.writeable = False
        if outside.any():
            warnings.warn(
                "%d of %d windows have p_mid > %d (the range the frozen classifier was trained on); their "
                "classification is unvalidated. See result.outside_training_range." % (int(outside.sum()), len(outside), config.FROZEN_LIMIT_N),
                OutOfTrainingRangeWarning, stacklevel=2)
        return AnalysisResult(
            n_primes=len(seq), n_windows=ws.n_windows,
            window_spec=(ws.spec.k, ws.spec.step, ws.spec.mid_offset), validation_level=seq.level,
            p_start=_readonly(ws.p_start), p_mid=_readonly(ws.p_mid), p_end=_readonly(ws.p_end),
            features=feats, analytics=analytics, classification=cls, outside_training_range=outside,
        )

    def classify_window(self, primes, stage1_threshold: float = config.STAGE1_THRESHOLD_BASELINE,
                        stage2: str = STAGE2_GATED) -> ClassificationResult:
        """Classify exactly one window (101 consecutive primes)."""
        seq = primes if isinstance(primes, PrimeSequence) else self.validate_primes(primes)
        need = FROZEN_WINDOW_SPEC.k + 1
        if len(seq) != need:
            raise InvalidPrimeInputError("wrong_length", "classify_window needs exactly %d primes, got %d (use analyze for longer input)" % (need, len(seq)))
        ws = build_windows(seq)
        return self.classifier.classify(extract_ml_features(ws), stage1_threshold=stage1_threshold, stage2=stage2)

    def classify_features(self, features: Union[FeatureMatrix, np.ndarray], **kwargs) -> ClassificationResult:
        """Classify pre-computed ML feature rows (frozen schema)."""
        return self.classifier.classify(features, **kwargs)

    # -- single-window analysis (finalized specification) ---------------------
    def analyze_prime(self, prime: int, *, window_size: int = config.WINDOW_GAPS) -> WindowAnalysis:
        """Centered-window analysis around ``prime`` (statistics only; no ML)."""
        return analyze_window(build_centered_window(prime, window_size), catalogue=self.catalogue)

    def frozen_primes(self):
        """All primes <= 1e8 (computed once; the frozen dataset's prime sequence)."""
        if self._frozen_primes is None:
            from .primes import primes_up_to
            self._frozen_primes = primes_up_to(config.FROZEN_LIMIT_N)
        return self._frozen_primes

    def analyze_frozen_window(self, window_id: int) -> WindowAnalysis:
        """Window ``window_id`` of the frozen dataset (0 .. 115,227), with catalogue information."""
        P = self.frozen_primes()
        n_windows = (len(P) - 1 - FROZEN_WINDOW_SPEC.k) // FROZEN_WINDOW_SPEC.step + 1
        if not (0 <= window_id < n_windows):
            raise InvalidPrimeInputError("out_of_range", "window id must be in [0, %d]" % (n_windows - 1))
        s0 = window_id * FROZEN_WINDOW_SPEC.step
        win = centered_window_from_primes([int(x) for x in P[s0:s0 + FROZEN_WINDOW_SPEC.k + 1]])
        return analyze_window(win, window_index=window_id, catalogue=self.catalogue)

    def predict_window(self, analysis: WindowAnalysis, stage1_threshold: float = config.STAGE1_THRESHOLD_BASELINE) -> WindowClassification:
        """Frozen two-stage XGBoost for one analysed window (exposes P(EVENT), P(class|EVENT), P_CC)."""
        return predict_window(analysis, self.classifier, stage1_threshold)

    def apply_label_rules(self, analysis: WindowAnalysis, window_index=None) -> LabelRuleResult:
        """Analytical label definitions (NOT the ML prediction)."""
        return apply_label_rules(analysis, window_index=window_index, catalogue=self.catalogue)

    def check_eligibility(self, analysis: WindowAnalysis, classification: WindowClassification = None,
                          policy: EligibilityPolicy = POLICY_STRICT, require: bool = False) -> EligibilityResult:
        """PGI-Crypto Process 1 for one window.  ``require=True`` raises CandidateRejectedError on rejection."""
        cls = classification if classification is not None else self.predict_window(analysis)
        return check_eligibility(analysis, cls, policy, raise_on_reject=require)

    # -- key generation ---------------------------------------------------------
    def generate(self, modulus_bits: int = config.DEFAULT_MODULUS_BITS, **kwargs) -> KeyGenerationResult:
        """PGI-Crypto key generation (research prototype).  See ``key_generation`` for all options.

        Default policy is the finalized specification (strict ML gate); at key scale this normally fails
        with ``KeyGenerationError`` because the frozen classifier is outside its validated range.
        ``policy=POLICY_EXPERIMENTAL`` must be requested explicitly.
        """
        return self._keygen.generate(modulus_bits, **kwargs)

    generate_key = generate          # backward-compatible name

    def serialize_public_key(self, result, format: str = "SubjectPublicKeyInfo") -> bytes:
        return serialize_public_key(result, format)

    def serialize_private_key(self, result, **kwargs) -> bytes:
        """PEM private key; requires ``password=`` (or explicit ``unencrypted=True``); refuses NON-SECURE keys by default."""
        return serialize_private_key(result, **kwargs)

    # -- validation hooks ---------------------------------------------------
    def validate_output(self, obj) -> ValidationReport:
        return validate_output(obj)

    def validate(self, result: KeyGenerationResult, deep: bool = False, rebuild_windows: bool = False) -> KeyValidationResult:
        """Comprehensive validation of a generated key (see ``validation.validate_generated_key``)."""
        return validate_generated_key(result, classifier=self.classifier, deep=deep, rebuild_windows=rebuild_windows)

    def validate_key(self, key: KeyResult, **kwargs) -> ValidationReport:
        return validate_key(key, **kwargs)

    def validate_generation(self, key: KeyResult) -> ValidationReport:
        return validate_generation(key, self.classifier)

    # -- metadata -------------------------------------------------------------
    def info(self) -> Dict[str, Any]:
        return {
            "library_version": config.LIBRARY_VERSION,
            "methodology_version": config.METHODOLOGY_VERSION,
            "feature_schema_version": config.FEATURE_SCHEMA_VERSION,
            "python": platform.python_version(),
            "models": self.classifier.info.as_dict(),
            "key_generation": specification_status(),
        }
