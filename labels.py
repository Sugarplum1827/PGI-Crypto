"""Analytical label rules (the DATASET-LABEL definitions), independent of any ML model.

    CR   = mean_gap / ln(p_mid)                       (single natural log; log^2 is legacy)
    CC   : 0.85 < CR < 1.15  AND  fixed-parameter Exp(1) KS pass
    MA   : CR < lo  OR  CR > hi        [lo, hi] = 95 % interval of Gamma(shape 100, rate 100)
                                       = [0.8136, 1.2053] to 4 d.p. (exact quantiles are used)
    MGV  : catalogue-based: every window that contains a maximal-gap record, plus the 5 windows
           on each side (window = 100 gaps, step 50)
    priority MGV > MA > CC > NONE

KS pass: asymptotic Kolmogorov p-value > ``ks_alpha`` (0.05).  This is the rule under which the
frozen dataset (SHA-256 3a760687...73452) was labelled; the adviser-approved file used the
classical D <= 0.136 cutoff and differs in 283 CC/NONE labels (documented, PGI_CRYPTO_FINAL_SPECIFICATION.md).

These rules are NOT the XGBoost prediction.  The frozen models approximate this labelling
(and do so imperfectly: MGV recovery is 0/12 end-to-end); they are never silently equated.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from . import config

_CATALOGUE_FILE = Path(__file__).resolve().parent / "data" / "maximal_gap_records_1e8.csv"


@lru_cache(maxsize=1)
def exact_ma_interval() -> Tuple[float, float]:
    """Exact 2.5 % / 97.5 % quantiles of Gamma(shape=100, rate=100) (0.8136 / 1.2053 to 4 d.p.)."""
    g = stats.gamma(a=config.CR_NULL_SHAPE, scale=1.0 / config.CR_NULL_SHAPE)
    return float(g.ppf(0.025)), float(g.ppf(0.975))


@dataclass(frozen=True)
class MaximalGapRecord:
    record_number: int
    start_prime: int
    gap: int
    gap_index: int          # 0-based index of the gap in the global gap sequence (= index of start_prime among primes)


class MaximalGapCatalogue:
    """Maximal-gap records (OEIS A002386/A005250/A005669) and the +/- 5-window MGV rule."""

    def __init__(self, records: Sequence[MaximalGapRecord]):
        self.records: Tuple[MaximalGapRecord, ...] = tuple(records)

    @classmethod
    def default(cls) -> "MaximalGapCatalogue":
        recs: List[MaximalGapRecord] = []
        with open(_CATALOGUE_FILE, newline="") as fh:
            for r in csv.DictReader(fh):
                recs.append(MaximalGapRecord(int(r["record_number"]), int(r["start_prime"]), int(r["gap"]),
                                             int(r["oeis_A005669_index_1based"]) - 1))
        return cls(recs)

    @staticmethod
    def containing_windows(gap_index: int, window_size: int = config.WINDOW_GAPS, step: int = config.WINDOW_STEP) -> List[int]:
        """Window ids ``w`` with ``step*w <= gap_index < step*w + window_size``."""
        out = []
        for w in range(max(0, (gap_index - window_size + step) // step), gap_index // step + 1):
            if step * w <= gap_index < step * w + window_size:
                out.append(w)
        return out

    def mgv_window_ids(self, n_windows: int, window_size: int = config.WINDOW_GAPS, step: int = config.WINDOW_STEP,
                       radius: int = 5) -> FrozenSet[int]:
        """Frozen-dataset window ids labelled MGV (reproduces all 185 of the frozen dataset)."""
        ids = set()
        for rec in self.records:
            for c in self.containing_windows(rec.gap_index, window_size, step):
                ids.update(range(max(0, c - radius), min(n_windows - 1, c + radius) + 1))
        return frozenset(ids)

    def records_near_window(self, window_id: int, window_size: int = config.WINDOW_GAPS, step: int = config.WINDOW_STEP,
                            radius: int = 5) -> List[Dict[str, object]]:
        """Records whose +/- radius neighbourhood contains the frozen window ``window_id``."""
        out = []
        for rec in self.records:
            cont = self.containing_windows(rec.gap_index, window_size, step)
            if any(abs(window_id - c) <= radius for c in cont):
                out.append({"record_number": rec.record_number, "start_prime": rec.start_prime, "gap": rec.gap,
                            "contained_in_window": window_id in cont, "containing_windows": cont})
        return out


@dataclass(frozen=True)
class LabelRuleResult:
    """Outcome of the analytical rules for one window (not an ML prediction)."""

    label: str                       # MGV | MA | CC | NONE
    cc_rule: bool
    ma_rule: bool
    mgv_rule: Optional[bool]         # None = not determinable (no window id / outside catalogue range)
    mgv_determinable: bool
    cr: float
    ks_pvalue: float
    ks_pass: bool
    ks_alpha: float
    notes: Tuple[str, ...]


def apply_label_rules_batch(cr: np.ndarray, ks_pvalue: np.ndarray, mgv_mask: Optional[np.ndarray] = None,
                            ks_alpha: float = config.KS_ALPHA) -> np.ndarray:
    """Vectorised labels (object array of 'MGV'/'MA'/'CC'/'NONE'); ``mgv_mask`` None means no MGV rule."""
    cr = np.asarray(cr, dtype=float)
    lo, hi = exact_ma_interval()
    band_lo, band_hi = config.CR_CC_BAND
    cc = (cr > band_lo) & (cr < band_hi) & (np.asarray(ks_pvalue) > ks_alpha)
    ma = (cr < lo) | (cr > hi)
    mg = np.zeros(cr.shape, dtype=bool) if mgv_mask is None else np.asarray(mgv_mask, dtype=bool)
    return np.where(mg, "MGV", np.where(ma, "MA", np.where(cc, "CC", "NONE"))).astype(object)


def apply_label_rules(analysis, *, window_index: Optional[int] = None, catalogue: Optional[MaximalGapCatalogue] = None,
                      ks_alpha: float = config.KS_ALPHA) -> LabelRuleResult:
    """Apply the analytical label rules to a :class:`WindowAnalysis`.

    MGV is only determinable for a frozen-dataset window (``window_index`` = dataset window id,
    step 50 from prime 2) with p_mid <= 1e8: the catalogue covers records up to 1e8 only.  For any
    other window ``mgv_rule`` is ``None`` (not applicable), never silently False.
    """
    notes = []
    lo, hi = exact_ma_interval()
    band_lo, band_hi = config.CR_CC_BAND
    cr = float(analysis.cr)
    ks_p = float(analysis.ks_pvalue)
    ks_pass = ks_p > ks_alpha
    cc = (band_lo < cr < band_hi) and ks_pass
    ma = (cr < lo) or (cr > hi)
    mgv: Optional[bool] = None
    determinable = False
    if window_index is not None and catalogue is not None and not analysis.outside_training_range:
        mgv = int(window_index) in catalogue.mgv_window_ids(window_index + 6)
        determinable = True
    else:
        notes.append("MGV rule not determinable: needs a frozen-dataset window id, the catalogue, and p_mid <= 1e8")
    label = "MGV" if mgv else ("MA" if ma else ("CC" if cc else "NONE"))
    return LabelRuleResult(label=label, cc_rule=cc, ma_rule=ma, mgv_rule=mgv, mgv_determinable=determinable, cr=cr,
                           ks_pvalue=ks_p, ks_pass=ks_pass, ks_alpha=ks_alpha, notes=tuple(notes))
