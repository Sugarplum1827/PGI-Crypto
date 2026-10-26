"""Public, range-gated API for the recreated PGI-Crypto research model.

This API classifies a supplied 100-gap window. It does not generate, approve,
or certify cryptographic keys, and it cannot prove that hand-entered gaps are
consecutive prime gaps.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .features import compute_analytical_statistics, compute_features
from .primes import LEVEL_CONSECUTIVE, validate_prime_sequence
from .rebuilt_classifier import DEFAULT_MODEL_PATH, RebuiltClassifier

_MANIFEST_PATH = Path(__file__).resolve().parent / "panel_data" / "model_manifest.json"
with _MANIFEST_PATH.open("r", encoding="utf-8") as _manifest_stream:
    MODEL_MANIFEST = json.load(_manifest_stream)
MODEL_SHA256 = MODEL_MANIFEST["sha256"]
MIN_SUPPORTED_P_MID = int(MODEL_MANIFEST["observed_p_mid_min"])
MAX_SUPPORTED_P_MID = int(MODEL_MANIFEST["observed_p_mid_max"])
WINDOW_GAPS = 100


class PGICryptoResearchSystem:
    """Load the packaged 10^12-range XGBoost model and classify one window.

    The midpoint must fall inside the actual sampled training support. The
    returned class is a research prediction, not a safe/unsafe key decision.
    """

    def __init__(self, model_path: str | Path | None = None, *, verify_model: bool = True):
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"PGI model not found: {self.model_path}")
        self.model_sha256 = self._sha256(self.model_path)
        if verify_model and self.model_sha256 != MODEL_SHA256:
            raise ValueError("model checksum differs from the panel's recorded model artifact")
        self.classifier = RebuiltClassifier(model_path=self.model_path)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _validated_gaps(gaps: Iterable[int]) -> np.ndarray:
        values = list(gaps)
        if len(values) != WINDOW_GAPS:
            raise ValueError(f"provide exactly {WINDOW_GAPS} consecutive-prime gap values")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError("gap values must be positive integers")
            if int(value) < 1 or int(value) > 540:
                raise ValueError("gap values must be between 1 and 540, the maximum observed through 10^12")
        arr = np.asarray(values, dtype=np.int64)
        return arr

    def classify_window(self, gaps: Iterable[int], p_mid: int) -> dict:
        """Calculate features, descriptive statistics, and model probabilities."""
        if isinstance(p_mid, bool) or not isinstance(p_mid, (int, np.integer)):
            raise ValueError("p_mid must be an integer prime value")
        midpoint = int(p_mid)
        if not MIN_SUPPORTED_P_MID <= midpoint <= MAX_SUPPORTED_P_MID:
            raise ValueError(
                "p_mid is outside the model's observed training support "
                f"[{MIN_SUPPORTED_P_MID}, {MAX_SUPPORTED_P_MID}]"
            )

        gap_values = self._validated_gaps(gaps)
        p_start = midpoint - int(gap_values[:50].sum())
        p_end = midpoint + int(gap_values[50:].sum())
        if p_start < 2:
            raise ValueError("the supplied midpoint and preceding gaps imply p_start < 2")
        odd_positions = np.flatnonzero(gap_values % 2)
        if len(odd_positions) and not (p_start == 2 and len(odd_positions) == 1 and odd_positions[0] == 0 and gap_values[0] == 1):
            raise ValueError("prime gaps are even above 2; only the first gap 2-to-3 may equal 1")

        primes = [p_start]
        for gap in gap_values:
            primes.append(primes[-1] + int(gap))
        validated_sequence = validate_prime_sequence(primes, level=LEVEL_CONSECUTIVE)

        matrix = gap_values.reshape(1, WINDOW_GAPS)
        features = compute_features(matrix, [midpoint])
        statistics = compute_analytical_statistics(matrix, [midpoint])
        prediction = self.classifier.predict_windows(matrix, [midpoint], [p_start], [p_end])
        probabilities = {
            label: float(probability)
            for label, probability in zip(prediction.classes, prediction.probabilities[0])
        }
        predicted_label = prediction.predictions[0]
        feature_values = features.row_as_dict(0)
        return {
            "predicted_label": predicted_label,
            "confidence": max(probabilities.values()),
            "probabilities": probabilities,
            "window": {
                "gap_count": WINDOW_GAPS,
                "p_start": p_start,
                "p_mid": midpoint,
                "p_end": p_end,
            },
            "sequence_validation": {
                "status": "consecutive primes verified",
                "deterministic_primality_checks": validated_sequence.deterministic_primality,
            },
            "descriptive_statistics": {
                "cr": float(statistics.CR[0]),
                "cr_null_pvalue": float(statistics.CR_null_pvalue_gamma_k_k[0]),
                "ks_statistic": float(statistics.ks_statistic_exponential[0]),
                "ks_pvalue": float(statistics.ks_pvalue_exponential[0]),
            },
            "selected_features": {
                name: float(feature_values[name])
                for name in ("mean_gap", "std_dev", "max_gap", "twin_prime_density", "gap_entropy", "cv_coeff_variation")
            },
            "model": {
                "track": prediction.model_track,
                "artifact": self.model_path.name,
                "sha256": self.model_sha256,
                "feature_count": len(prediction.feature_names),
            },
            "training_support": {
                "p_mid_min": MIN_SUPPORTED_P_MID,
                "p_mid_max": MAX_SUPPORTED_P_MID,
                "in_support": True,
            },
            "interpretation": (
                "Research classification only. The implied 101-prime window passed consecutive-prime "
                "checks in the supported range; this output does not approve a cryptographic key."
            ),
        }

