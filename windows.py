"""Window construction (frozen thesis definition).

* A window is ``k = 100`` consecutive prime gaps, i.e. 101 consecutive primes.
* Windows start every ``step = 50`` gaps (50 % overlap).
* Window ``w`` covers ``gaps[50*w : 50*w + 100]`` and primes
  ``primes[50*w : 50*w + 101]``.
* ``p_start = primes[50*w]``, ``p_mid = primes[50*w + 50]``,
  ``p_end = primes[50*w + 100]``.
* ``n_windows = (n_gaps - k) // step + 1``.  A trailing partial window is
  dropped, never padded.  Minimum input: 101 primes (one window).

Alignment with the frozen dataset: the dataset's window 0 starts at prime 2.
Windows built from a sequence that starts at another prime are valid windows
of that sequence but are only identical to the dataset's windows if the
sequence starts at ``primes[50*m]`` for some m (counting from prime 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from . import config
from .exceptions import InsufficientPrimesError, InvalidPrimeInputError
from .gaps import compute_gaps
from .primes import PrimeSequence


@dataclass(frozen=True)
class WindowSpec:
    """Window geometry.  The default is the frozen thesis definition."""

    k: int = config.WINDOW_GAPS
    step: int = config.WINDOW_STEP
    mid_offset: int = config.WINDOW_MID_OFFSET

    def __post_init__(self):
        if self.k < 6:
            raise ValueError("k must be >= 6 (autocorrelation up to lag 5 needs 6 values)")
        if self.step < 1:
            raise ValueError("step must be >= 1")
        if not (0 <= self.mid_offset <= self.k):
            raise ValueError("mid_offset must lie within the window")

    @property
    def is_frozen_definition(self) -> bool:
        return (self.k, self.step, self.mid_offset) == (config.WINDOW_GAPS, config.WINDOW_STEP, config.WINDOW_MID_OFFSET)


FROZEN_WINDOW_SPEC = WindowSpec()


@dataclass(frozen=True)
class WindowSet:
    """All windows of a prime sequence.

    ``gap_matrix`` has shape ``(n_windows, k)`` and is a read-only strided
    *view* of the gap array (no copy).  ``start_index`` is the index of each
    window's first prime in the input sequence.
    """

    spec: WindowSpec
    gap_matrix: np.ndarray
    start_index: np.ndarray
    p_start: np.ndarray
    p_mid: np.ndarray
    p_end: np.ndarray

    @property
    def n_windows(self) -> int:
        return int(self.gap_matrix.shape[0])

    def __len__(self) -> int:
        return self.n_windows


def minimum_primes(spec: WindowSpec = FROZEN_WINDOW_SPEC) -> int:
    """Smallest number of primes that yields one window."""
    return spec.k + 1


def build_windows(primes, spec: WindowSpec = FROZEN_WINDOW_SPEC) -> WindowSet:
    """Build the windows of a validated :class:`PrimeSequence`.

    Raises :class:`InsufficientPrimesError` if fewer than ``k + 1`` primes are
    supplied.
    """
    if not isinstance(primes, PrimeSequence):
        raise InvalidPrimeInputError("not_validated", "build_windows expects a PrimeSequence; call validate_prime_sequence first")
    values = primes.values
    need = minimum_primes(spec)
    if len(values) < need:
        raise InsufficientPrimesError("too_few", "need at least %d primes for one window of k=%d gaps, got %d" % (need, spec.k, len(values)))
    gaps = compute_gaps(primes)
    n_windows = (len(gaps) - spec.k) // spec.step + 1
    matrix = np.lib.stride_tricks.sliding_window_view(gaps, spec.k)[:: spec.step]
    assert matrix.shape == (n_windows, spec.k)
    starts = np.arange(n_windows, dtype=np.int64) * spec.step
    return WindowSet(
        spec=spec,
        gap_matrix=matrix,
        start_index=starts,
        p_start=values[starts],
        p_mid=values[starts + spec.mid_offset],
        p_end=values[starts + spec.k],
    )


# ---------------------------------------------------------------------------
# Centered window around a single prime (PGI-Crypto G2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CenteredWindow:
    """``window_size + 1`` consecutive primes with the centre prime in the middle.

    ``window_size`` is the number of *gaps* (frozen definition: 100, hence 101
    primes: 50 below the centre, the centre, 50 above).  The centre prime is the
    frozen dataset's ``p_mid``.  ``window_size`` has nothing to do with the
    modulus size; the two are never abbreviated to the same symbol.

    ``primes`` is excluded from ``repr``: for a key candidate the neighbours of
    the centre locate it within a few hundred integers, so they are secret.
    """

    primes: Tuple[int, ...] = field(repr=False)
    window_size: int = config.WINDOW_GAPS
    neighbour_primality: str = "proven"        # "proven" | "probable" (see primesearch)
    center_primality: str = "verified"         # "verified" | "assumed_by_caller"

    def __post_init__(self):
        if self.window_size < 6 or self.window_size % 2:
            raise ValueError("window_size must be an even integer >= 6")
        if len(self.primes) != self.window_size + 1:
            raise InvalidPrimeInputError("wrong_length", "a centered window of %d gaps needs %d primes, got %d"
                                         % (self.window_size, self.window_size + 1, len(self.primes)))
        if any(b <= a for a, b in zip(self.primes, self.primes[1:])):
            raise InvalidPrimeInputError("unsorted", "window primes must be strictly increasing")

    @property
    def center_index(self) -> int:
        return self.window_size // 2

    @property
    def center(self) -> int:
        return self.primes[self.center_index]

    @property
    def p_start(self) -> int:
        return self.primes[0]

    @property
    def p_end(self) -> int:
        return self.primes[-1]

    def gap_matrix(self) -> np.ndarray:
        """Shape (1, window_size) int64 gaps of the window."""
        g = np.array([b - a for a, b in zip(self.primes, self.primes[1:])], dtype=np.int64)[None, :]
        g.flags.writeable = False
        return g


def build_centered_window(center: int, window_size: int = config.WINDOW_GAPS, *, verify_center: bool = True,
                          center_is_prime: bool = False, sieve_bound: int = None) -> CenteredWindow:
    """Centered window around the prime ``center`` (found with a segmented sieve).

    ``verify_center=True`` tests that ``center`` is prime (deterministic below
    3.3e24, strong probable-prime above) and raises
    :class:`InvalidPrimeInputError` otherwise.  Key generation passes
    ``center_is_prime=True`` after its own, stricter primality test.
    """
    from .primes import is_prime, primality_is_proven
    from .primesearch import DEFAULT_SIEVE_BOUND, neighbor_primes

    if window_size < 6 or window_size % 2:
        raise ValueError("window_size must be an even integer >= 6")
    if isinstance(center, bool) or not isinstance(center, (int, np.integer)):
        raise InvalidPrimeInputError("not_integer", "center must be an integer")
    center = int(center)
    if center_is_prime:
        center_status = "assumed_by_caller"
    elif verify_center:
        if not is_prime(center):
            raise InvalidPrimeInputError("composite", "center %d is not prime" % center if center < 10 ** 30 else "center is not prime")
        center_status = "verified"
    else:
        center_status = "assumed_by_caller"
    half = window_size // 2
    below, above = neighbor_primes(center, half, DEFAULT_SIEVE_BOUND if sieve_bound is None else sieve_bound)
    primes = tuple(below) + (center,) + tuple(above)
    proven = primality_is_proven(primes[-1])
    return CenteredWindow(primes=primes, window_size=window_size,
                          neighbour_primality="proven" if proven else "probable", center_primality=center_status)


def centered_window_from_primes(primes, window_size: int = config.WINDOW_GAPS) -> CenteredWindow:
    """Wrap ``window_size + 1`` already-validated consecutive primes as a centered window."""
    seq = primes.values if isinstance(primes, PrimeSequence) else primes
    from .primes import primality_is_proven
    vals = tuple(int(v) for v in seq)
    return CenteredWindow(primes=vals, window_size=window_size,
                          neighbour_primality="proven" if primality_is_proven(vals[-1]) else "probable",
                          center_primality="verified" if isinstance(primes, PrimeSequence) and primes.level != "structure" else "assumed_by_caller")
