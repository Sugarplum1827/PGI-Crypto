"""Consecutive prime gaps.

``gaps[i] = primes[i + 1] - primes[i]``.  A sequence of n primes has n - 1
gaps, all >= 1 (the single odd gap is 3 - 2 = 1; every other gap is even).
The input is never modified.
"""
from __future__ import annotations

import numpy as np

from .exceptions import InvalidPrimeInputError
from .primes import PrimeSequence


def compute_gaps(primes) -> np.ndarray:
    """Return the gap array (int64, read-only) of a validated prime sequence.

    ``primes`` may be a :class:`PrimeSequence` or a NumPy array/sequence that
    the caller has already validated; strict monotonicity is re-checked
    because a non-positive gap would silently corrupt every downstream feature.
    """
    values = primes.values if isinstance(primes, PrimeSequence) else np.asarray(primes)
    if values.ndim != 1 or values.shape[0] < 2:
        raise InvalidPrimeInputError("too_few", "at least 2 primes are needed to form a gap")
    gaps = np.diff(values)
    if gaps.dtype == object:
        if any(g < 1 for g in gaps):
            raise InvalidPrimeInputError("unsorted", "non-positive gap found; primes must be strictly increasing")
        if any(g > (1 << 62) for g in gaps):
            raise InvalidPrimeInputError("gap_too_large", "a gap exceeds 2**62")
        gaps = np.array([int(g) for g in gaps], dtype=np.int64)
    else:
        if (gaps < 1).any():
            raise InvalidPrimeInputError("unsorted", "non-positive gap found; primes must be strictly increasing")
        gaps = gaps.astype(np.int64, copy=False)
    gaps = np.array(gaps, copy=True)
    gaps.flags.writeable = False
    return gaps
