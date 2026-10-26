"""Runtime adapter for the newly trained, leakage-clean 10^12-range models.

The archived frozen two-stage classifier is retained for legacy reproduction;
this adapter does not load it. The adapter loads the separate four-class model
fitted by ``pgicrypto.fit_final_models`` and selects its exact 22-feature schema
from the library's audited 24-feature matrix.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .features import extract_ml_features
from .primes import LEVEL_CONSECUTIVE, validate_prime_sequence
from .windows import build_windows

from .config import ML_FEATURE_NAMES as _LIBRARY_FEATURE_NAMES

# Keep the high-scale inference package independent of the optional research
# pipeline (which brings in pandas/pyarrow). This order matches
# pgicrypto.features.MODEL_FEATURE_NAMES exactly.
MODEL_FEATURE_NAMES = tuple(name for name in _LIBRARY_FEATURE_NAMES if name not in {
    "CR_log2_legacy_DO_NOT_USE_FOR_CLASSIFICATION", "poisson_lambda_fit",
})

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGED_MODEL_PATH = Path(__file__).resolve().parent / "models" / "rebuilt_adviser_classical_4class_xgboost.json"
PROJECT_MODEL_PATH = PROJECT_ROOT / "models" / "final_adviser_classical_4class_xgboost.json"
DEFAULT_MODEL_PATH = PACKAGED_MODEL_PATH if PACKAGED_MODEL_PATH.is_file() else PROJECT_MODEL_PATH
CLASS_NAMES = ("CC", "MA", "MGV", "NONE")
VALIDATED_LIMIT = 10**12


@dataclass(frozen=True)
class RebuiltPrediction:
    """Labels and probabilities from one recreated four-class model."""

    classes: tuple[str, ...]
    probabilities: np.ndarray
    predictions: tuple[str, ...]
    p_start: np.ndarray
    p_mid: np.ndarray
    p_end: np.ndarray
    outside_training_range: np.ndarray
    model_path: str
    model_track: str
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class RebuiltModelInfo:
    methodology_version: str
    feature_schema_version: str
    backend: str
    model_file: str
    model_sha256: str
    model_track: str
    classes: tuple[str, ...]
    validated_range: str

    def as_dict(self) -> dict:
        return {
            "methodology_version": self.methodology_version,
            "feature_schema_version": self.feature_schema_version,
            "backend": self.backend,
            "model_file": self.model_file,
            "model_sha256": self.model_sha256,
            "model_track": self.model_track,
            "classes": list(self.classes),
            "validated_range": self.validated_range,
        }


class RebuiltClassifier:
    """Predict four audited/adviser labels using a newly trained XGBoost model.

    This model supports empirical classification through p_mid <= 10^12 using
    a range-spanning systematic training sample. Cryptographic-size prime
    candidates remain outside its domain and are not approved for key use.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH,
                 track: str = "label_adviser_classical_4class"):
        if track not in ("label_adviser_classical_4class", "label_audited_4class"):
            raise ValueError("track must be label_adviser_classical_4class or label_audited_4class")
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"newly trained model not found: {self.model_path}")
        import xgboost as xgb
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(self.model_path))
        self.track = track
        self.model_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        self._column_indices = tuple(
            self._library_feature_names().index(name) for name in MODEL_FEATURE_NAMES
        )
        if int(self.model.n_features_in_) != len(MODEL_FEATURE_NAMES):
            raise ValueError("trained model and 22-feature input contract do not match")
        if tuple(int(v) for v in self.model.classes_) != tuple(range(len(CLASS_NAMES))):
            raise ValueError("trained model class order differs from CC, MA, MGV, NONE")

    @property
    def info(self) -> RebuiltModelInfo:
        return RebuiltModelInfo(
            methodology_version="recreated-range-spanning-four-class-model",
            feature_schema_version="22-feature-v1",
            backend="xgboost-rebuilt",
            model_file=self.model_path.name,
            model_sha256=self.model_sha256,
            model_track=self.track,
            classes=CLASS_NAMES,
            validated_range="p_mid <= 10^12",
        )

    @staticmethod
    def _library_feature_names() -> tuple[str, ...]:
        from .config import ML_FEATURE_NAMES
        return ML_FEATURE_NAMES

    def predict_windows(self, gap_matrix: np.ndarray, p_mid: Iterable[int],
                        p_start: Iterable[int] | None = None,
                        p_end: Iterable[int] | None = None) -> RebuiltPrediction:
        """Classify rows of 100 consecutive gaps and their midpoint primes."""
        gaps = np.asarray(gap_matrix)
        if gaps.ndim != 2 or gaps.shape[1] != 100 or gaps.shape[0] == 0:
            raise ValueError("gap_matrix must have shape (n, 100) with n > 0")
        midpoint = np.asarray(tuple(p_mid), dtype=object)
        if midpoint.shape != (len(gaps),):
            raise ValueError("p_mid must contain one value per gap window")
        p0 = np.asarray(tuple(p_start), dtype=object) if p_start is not None else np.zeros(len(gaps), dtype=object)
        p1 = np.asarray(tuple(p_end), dtype=object) if p_end is not None else np.zeros(len(gaps), dtype=object)
        if p0.shape != (len(gaps),) or p1.shape != (len(gaps),):
            raise ValueError("p_start and p_end must contain one value per gap window")
        full_features = extract_ml_features_from_gaps(gaps, midpoint)
        inputs = full_features[:, self._column_indices].astype(np.float32, copy=False)
        probabilities = self.model.predict_proba(inputs)
        pred_ids = np.argmax(probabilities, axis=1)
        outside = np.fromiter((int(value) > VALIDATED_LIMIT for value in midpoint), dtype=np.bool_, count=len(midpoint))
        for array in (probabilities, midpoint, p0, p1, outside):
            array.flags.writeable = False
        return RebuiltPrediction(
            classes=CLASS_NAMES, probabilities=probabilities,
            predictions=tuple(CLASS_NAMES[int(i)] for i in pred_ids),
            p_start=p0, p_mid=midpoint, p_end=p1, outside_training_range=outside,
            model_path=str(self.model_path), model_track=self.track,
            feature_names=tuple(MODEL_FEATURE_NAMES),
        )

    def predict_primes(self, primes: Iterable[int],
                       validation_level: str = LEVEL_CONSECUTIVE) -> RebuiltPrediction:
        """Validate a consecutive-prime sequence, window it, and classify it."""
        sequence = validate_prime_sequence(primes, level=validation_level)
        windows = build_windows(sequence)
        return self.predict_windows(windows.gap_matrix, windows.p_mid,
                                    windows.p_start, windows.p_end)

    def shap_summary(self, summary_path: str | Path | None = None) -> dict:
        """Read the matching global SHAP summary when it was produced."""
        if summary_path is None:
            suffix = "adviser_classical_4class" if self.track == "label_adviser_classical_4class" else "audited_4class"
            summary_path = PROJECT_ROOT / "results" / "metrics" / "full_1e12" / f"shap_{suffix}_global.json"
        path = Path(summary_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"SHAP summary not found: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if tuple(record.get("features", ())) != tuple(MODEL_FEATURE_NAMES):
            raise ValueError("SHAP summary feature order differs from this model contract")
        return record


def extract_ml_features_from_gaps(gap_matrix: np.ndarray, p_mid: np.ndarray) -> np.ndarray:
    """Use the existing library feature formulas for the rebuilt 22 inputs."""
    from .features import compute_features
    return compute_features(gap_matrix, p_mid).values
