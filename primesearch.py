"""Neighbour-prime search for centered windows around (possibly huge) primes.

A *centered window* (frozen definition, see ``windows.py``) is 101 consecutive
primes: 50 below the centre prime, the centre prime, and 50 above it.  For a
1024-bit centre this spans ~100 * ln(p) ~ 71,000 integers, so primes are found
with a segmented sieve (small-prime sieve, then a strong probable-prime test of
the survivors) instead of testing every integer.

Primality used for the *neighbour* primes
-----------------------------------------
* below 3.317e24: ``primes.is_prime`` (deterministic proof);
* above: strong probable-prime test to bases 2 and 3.  For numbers that already
  survived sieving by every prime below the sieve bound this is the standard
  "probable prime" filter; the chance that a composite passes both bases is
  negligible for numbers of this size, but it is NOT a proof.  The *centre*
  prime of a key candidate is tested far more strictly (``key_generation``).
This is stated in every result as ``neighbour_primality = "probable"``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from .exceptions import InvalidPrimeInputError
from .primes import is_prime, primes_up_to, primality_is_proven

DEFAULT_SIEVE_BOUND = 1 << 20          # 1,048,576 -> 82,025 sieving primes
_TABLES: Dict[int, np.ndarray] = {}


def sieve_primes(bound: int = DEFAULT_SIEVE_BOUND) -> np.ndarray:
    """Cached odd primes below ``bound`` (int64)."""
    if bound not in _TABLES:
        p = primes_up_to(bound)
        _TABLES[bound] = p[p > 2]
    return _TABLES[bound]


def strong_probable_prime(n: int, base: int) -> bool:
    """Strong probable-prime (Miller-Rabin) test of odd ``n`` > 2 to one base."""
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    x = pow(base % n, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(r - 1):
        x = x * x % n
        if x == n - 1:
            return True
    return False


def probable_prime_for_window(n: int) -> bool:
    """Primality filter for neighbour primes (see module docstring)."""
    if primality_is_proven(n):
        return is_prime(n)
    return strong_probable_prime(n, 2) and strong_probable_prime(n, 3)


def _odd_segment_survivors(lo: int, count: int, bound: int) -> List[int]:
    """Odd numbers ``lo, lo+2, ..., lo+2*(count-1)`` (``lo`` odd, >= 3) that survive sieving by all primes < ``bound``.

    Sorted ascending.  Survivors are *candidates*: they still need a primality test.
    Small primes are never sieved out themselves.
    """
    small = sieve_primes(bound)
    mask = np.ones(count, dtype=np.bool_)
    small_case = lo <= bound * 2 + 2 * count
    res = np.array([lo % int(r) for r in small], dtype=np.int64)
    # first index i with lo + 2 i == 0 (mod r):  i = (-lo) * inv2 (mod r),  inv2 = (r + 1) / 2
    i0 = ((-res) % small) * ((small + 1) // 2) % small
    for r, i in zip(small.tolist(), i0.tolist()):
        if i >= count:
            continue
        if small_case and lo + 2 * i == r:      # never sieve out a small prime itself
            i += r
        mask[i::r] = False
    return [lo + 2 * int(i) for i in np.flatnonzero(mask)]


def neighbor_primes(center: int, n_each_side: int = 50, sieve_bound: int = DEFAULT_SIEVE_BOUND) -> Tuple[List[int], List[int]]:
    """``(below, above)``: the ``n_each_side`` primes just below / above ``center``.

    ``below`` is ascending and ends with the largest prime < center; ``above``
    is ascending and starts with the smallest prime > center.  ``center`` itself
    is not tested here and need not be prime.  Survivors of each sieved segment
    are tested in scan order and the scan stops as soon as enough primes are
    found, so no primality test is wasted on overshoot.  Raises
    :class:`InvalidPrimeInputError` if fewer than ``n_each_side`` primes lie
    below ``center`` (``center`` too small for a centered window).
    """
    if isinstance(center, bool) or not isinstance(center, (int, np.integer)):
        raise InvalidPrimeInputError("not_integer", "center must be an integer")
    center = int(center)
    if center < 3:
        raise InvalidPrimeInputError("too_small", "center must be >= 3")
    seg = max(2000, int(n_each_side * math.log(center) * 0.5))   # integers per sieved segment
    seg += seg % 2
    # ---- above (ascending scan) ----
    above: List[int] = []
    lo = center + 1 if (center + 1) % 2 == 1 else center + 2
    while len(above) < n_each_side:
        for n in _odd_segment_survivors(lo, seg // 2, sieve_bound):
            if probable_prime_for_window(n):
                above.append(n)
                if len(above) == n_each_side:
                    break
        lo += seg
    # ---- below (descending scan) ----
    below_desc: List[int] = []
    hi = center - 1 if (center - 1) % 2 == 1 else center - 2     # largest odd < center
    while len(below_desc) < n_each_side and hi >= 3:
        lo_seg = max(3, hi - seg + 2)
        lo_seg += (lo_seg % 2 == 0)
        n_odd = (hi - lo_seg) // 2 + 1
        for n in reversed(_odd_segment_survivors(lo_seg, n_odd, sieve_bound)):
            if probable_prime_for_window(n):
                below_desc.append(n)
                if len(below_desc) == n_each_side:
                    break
        hi = lo_seg - 2
    if len(below_desc) < n_each_side and center > 2:
        below_desc.append(2)                                      # the only even prime
    if len(below_desc) < n_each_side:
        raise InvalidPrimeInputError("too_small", "only %d primes lie below %d; a centered window needs %d" % (len(below_desc), center, n_each_side))
    return sorted(below_desc[:n_each_side]), above[:n_each_side]
