"""Frozen two-stage XGBoost classification (inference only).

This module NEVER trains, tunes or modifies a model.  It loads the two frozen
model files that ship in ``pgi_crypto/models/`` (native XGBoost JSON), checks
their SHA-256 against ``config.MODEL_ARTIFACTS`` and evaluates them.

Pipeline (identical to the validated external experiment)
---------------------------------------------------------
Stage 1  P(EVENT) = model1(features);  EVENT iff P >= threshold (baseline 0.5).
Stage 2  P(CC|MA|MGV | EVENT) = model2(features), evaluated for windows that
         Stage 1 flagged ("gated", the frozen end-to-end definition).
Final    NONE if Stage 1 says no event, otherwise argmax of Stage 2.

Back-ends
---------
``"json"``     (default) pure-NumPy tree evaluator reading the model JSON.
               Needs only NumPy.  Verified against the predictions produced by
               the original xgboost run: max |dP| = 4.1e-7 (Stage 1) and
               6.9e-7 (Stage 2), 100 % identical decisions on the test split.
``"xgboost"``  the ``xgboost`` package (the engine of the original run).
               Optional; not exercised in the environment where this library
               was built (see PGI_CRYPTO_CORE_VALIDATION.md).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from . import config
from .exceptions import (BackendUnavailableError, FeatureExtractionError, FeatureMismatchError, ModelArtifactError,
                         ModelLoadError)
from .result_types import ClassificationResult, FeatureMatrix, ModelInfo

BACKEND_JSON = "json"
BACKEND_XGBOOST = "xgboost"
STAGE2_GATED = "gated"
STAGE2_ALL = "all"
_MAX_TREE_DEPTH = 200


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _JsonBoosterModel:
    """Flattened-tree evaluator for an XGBoost JSON model (numeric splits only)."""

    def __init__(self, path: Path, expected_objective: str, expected_trees: int):
        try:
            with open(path, "r") as fh:
                doc = json.load(fh)
            learner = doc["learner"]
            return self._init_from_doc(path, doc, learner, expected_objective, expected_trees)
        except (ModelLoadError, ModelArtifactError):
            raise
        except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
            raise ModelLoadError("%s: model file could not be parsed (%s)" % (Path(path).name, type(exc).__name__)) from exc

    def _init_from_doc(self, path, doc, learner, expected_objective, expected_trees):
        self.objective = learner["objective"]["name"]
        if self.objective != expected_objective:
            raise ModelArtifactError("%s: objective %r, expected %r" % (path.name, self.objective, expected_objective))
        param = learner["learner_model_param"]
        self.n_features = int(param["num_feature"])
        self.n_class = max(1, int(param.get("num_class", 0) or 0))
        self.base_score = float(str(param["base_score"]).strip("[]"))
        model = learner["gradient_booster"]["model"]
        trees = model["trees"]
        if len(trees) != expected_trees:
            raise ModelArtifactError("%s: %d trees, expected %d" % (path.name, len(trees), expected_trees))
        tree_info = model.get("tree_info") or [i % self.n_class for i in range(len(trees))]
        left, right, feat, cond, dleft, roots = [], [], [], [], [], []
        offset = 0
        for t in trees:
            if any(int(x) != 0 for x in t.get("split_type", [])):
                raise ModelArtifactError("%s: categorical splits are not supported" % path.name)
            n_nodes = len(t["left_children"])
            lc = np.asarray(t["left_children"], dtype=np.int64)
            rc = np.asarray(t["right_children"], dtype=np.int64)
            left.append(np.where(lc < 0, -1, lc + offset))
            right.append(np.where(rc < 0, -1, rc + offset))
            feat.append(np.asarray(t["split_indices"], dtype=np.int64))
            cond.append(np.asarray(t["split_conditions"], dtype=np.float32))
            dleft.append(np.asarray(t["default_left"], dtype=np.bool_))
            roots.append(offset)
            offset += n_nodes
        self.left = np.concatenate(left)
        self.right = np.concatenate(right)
        self.feat = np.concatenate(feat)
        self.cond = np.concatenate(cond)
        self.dleft = np.concatenate(dleft)
        self.roots = np.asarray(roots, dtype=np.int64)
        self.tree_class = np.asarray(tree_info, dtype=np.int64)
        self._class_cols = [np.flatnonzero(self.tree_class == c) for c in range(self.n_class)]

    def margins(self, x32: np.ndarray) -> np.ndarray:
        n = x32.shape[0]
        n_trees = self.roots.shape[0]
        out = np.zeros((n, self.n_class), dtype=np.float64)
        chunk = max(64, 1_000_000 // n_trees)
        for s in range(0, n, chunk):
            xc = x32[s:s + chunk]
            m = xc.shape[0]
            rows = np.arange(m)[:, None]
            node = np.broadcast_to(self.roots, (m, n_trees)).copy()
            for _ in range(_MAX_TREE_DEPTH):
                lc = self.left[node]
                leaf = lc < 0
                if leaf.all():
                    break
                x = xc[rows, self.feat[node]]
                go_left = np.where(np.isnan(x), self.dleft[node], x < self.cond[node])
                node = np.where(leaf, node, np.where(go_left, lc, self.right[node]))
            else:
                raise ModelArtifactError("tree depth exceeded %d; model file is malformed" % _MAX_TREE_DEPTH)
            leaf_val = self.cond[node]
            for c, cols in enumerate(self._class_cols):
                out[s:s + m, c] = leaf_val[:, cols].sum(axis=1, dtype=np.float64)
        return out

    def predict_proba(self, x32: np.ndarray) -> np.ndarray:
        m = self.margins(x32)
        if self.objective == "binary:logistic":
            z = m[:, 0] + np.log(self.base_score / (1.0 - self.base_score))
            return 1.0 / (1.0 + np.exp(-z))
        # multi:softprob: the constant base_score offset is identical for every class and cancels.
        e = np.exp(m - m.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


class _XGBoostBoosterModel:
    """Thin wrapper around ``xgboost.Booster`` (optional back-end)."""

    def __init__(self, path: Path):
        try:
            import xgboost as xgb  # noqa: WPS433 (optional dependency)
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise BackendUnavailableError("backend='xgboost' requires the xgboost package; use backend='json'") from exc
        self._xgb = xgb
        self._booster = xgb.Booster()
        self._booster.load_model(str(path))

    def predict_proba(self, x32: np.ndarray) -> np.ndarray:  # pragma: no cover - needs xgboost
        return np.asarray(self._booster.predict(self._xgb.DMatrix(x32)), dtype=np.float64)


class FrozenClassifier:
    """The frozen two-stage XGBoost baseline.  Inference only."""

    def __init__(self, model_dir: Optional[Union[str, Path]] = None, backend: str = BACKEND_JSON,
                 verify_hashes: bool = True):
        if backend not in (BACKEND_JSON, BACKEND_XGBOOST):
            raise ValueError("backend must be 'json' or 'xgboost'")
        self.backend = backend
        self.model_dir = config.resolve_model_dir(model_dir)
        art = config.MODEL_ARTIFACTS
        self._hashes: Dict[str, str] = {}
        paths: Dict[str, Path] = {}
        for stage in ("stage1", "stage2"):
            p = self.model_dir / art[stage]["file"]
            if not p.is_file():
                raise ModelArtifactError("missing frozen model artifact %s (looked in %s). Required: %s"
                                         % (art[stage]["file"], self.model_dir, ", ".join(a["file"] for a in art.values())))
            digest = _sha256(p)
            if verify_hashes and digest != art[stage]["sha256"]:
                raise ModelArtifactError("%s: SHA-256 %s does not match the frozen artifact %s"
                                         % (p.name, digest, art[stage]["sha256"]))
            self._hashes[stage] = digest
            paths[stage] = p
        if backend == BACKEND_JSON:
            self._m1 = _JsonBoosterModel(paths["stage1"], art["stage1"]["objective"], art["stage1"]["n_trees"])
            self._m2 = _JsonBoosterModel(paths["stage2"], art["stage2"]["objective"], art["stage2"]["n_trees"])
            for m in (self._m1, self._m2):
                if m.n_features != config.N_ML_FEATURES:
                    raise ModelArtifactError("model expects %d features, schema has %d" % (m.n_features, config.N_ML_FEATURES))
            if self._m2.n_class != len(config.STAGE2_CLASSES):
                raise ModelArtifactError("Stage 2 must have %d classes" % len(config.STAGE2_CLASSES))
        else:
            self._m1 = _XGBoostBoosterModel(paths["stage1"])
            self._m2 = _XGBoostBoosterModel(paths["stage2"])

    # -- metadata ---------------------------------------------------------
    @property
    def info(self) -> ModelInfo:
        art = config.MODEL_ARTIFACTS
        return ModelInfo(
            methodology_version=config.METHODOLOGY_VERSION,
            feature_schema_version=config.FEATURE_SCHEMA_VERSION,
            backend=self.backend,
            stage1_file=art["stage1"]["file"], stage1_sha256=self._hashes["stage1"],
            stage2_file=art["stage2"]["file"], stage2_sha256=self._hashes["stage2"],
            stage2_classes=config.STAGE2_CLASSES,
        )

    # -- inference --------------------------------------------------------
    @staticmethod
    def _prepare(features: Union[FeatureMatrix, np.ndarray]) -> np.ndarray:
        if isinstance(features, FeatureMatrix):
            if tuple(features.names) != config.ML_FEATURE_NAMES:
                raise FeatureMismatchError("feature names/order differ from the frozen schema %s" % config.FEATURE_SCHEMA_VERSION)
            values = features.values
        else:
            values = np.asarray(features)
            if values.ndim == 1:
                values = values[None, :]
            if values.ndim != 2 or values.shape[1] != config.N_ML_FEATURES:
                raise FeatureMismatchError("expected shape (n, %d), got %s" % (config.N_ML_FEATURES, values.shape))
            if values.dtype.kind not in "fiu":
                raise FeatureExtractionError("features must be numeric")
        if values.shape[0] == 0:
            raise FeatureExtractionError("no rows to classify")
        values = values.astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            bad = np.argwhere(~np.isfinite(values))[0]
            raise FeatureExtractionError("non-finite feature at row %d, column %r; refusing to pass NaN/inf to the "
                                         "classifier (XGBoost would treat it as 'missing')" % (bad[0], config.ML_FEATURE_NAMES[bad[1]]))
        return values.astype(np.float32)

    def classify(self, features: Union[FeatureMatrix, np.ndarray],
                 stage1_threshold: float = config.STAGE1_THRESHOLD_BASELINE,
                 stage2: str = STAGE2_GATED) -> ClassificationResult:
        """Run the two-stage classifier.

        ``stage2="gated"`` evaluates Stage 2 only where Stage 1 flagged an event
        (frozen end-to-end behaviour); ``"all"`` also reports Stage 2
        probabilities for every window (informational; the final class is
        unchanged).  The default threshold is the frozen baseline 0.5.
        """
        if not (0.0 < float(stage1_threshold) < 1.0):
            raise ValueError("stage1_threshold must be in (0, 1)")
        if stage2 not in (STAGE2_GATED, STAGE2_ALL):
            raise ValueError("stage2 must be 'gated' or 'all'")
        x32 = self._prepare(features)
        n = x32.shape[0]
        p_event = np.asarray(self._m1.predict_proba(x32), dtype=np.float64).reshape(n)
        event = p_event >= stage1_threshold
        evaluate = np.ones(n, dtype=np.bool_) if stage2 == STAGE2_ALL else event.copy()
        p2 = np.full((n, len(config.STAGE2_CLASSES)), np.nan)
        if evaluate.any():
            p2[evaluate] = np.asarray(self._m2.predict_proba(x32[evaluate]), dtype=np.float64)
        pred2 = np.full(n, None, dtype=object)
        final = np.full(n, config.NONE_LABEL, dtype=object)
        if evaluate.any():
            idx = np.argmax(p2[evaluate], axis=1)
            labels = np.array(config.STAGE2_CLASSES, dtype=object)[idx]
            pred2[evaluate] = labels
        gated = event & evaluate
        final[gated] = pred2[gated]
        for a in (p_event, event, p2, evaluate):
            a.flags.writeable = False
        return ClassificationResult(
            stage1_event_probability=p_event, stage1_threshold=float(stage1_threshold), stage1_event=event,
            stage2_probabilities=p2, stage2_evaluated=evaluate, stage2_prediction=pred2,
            final_class=final, model_info=self.info,
        )
