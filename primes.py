"""Prime input handling.

Input contract for every public entry point that takes primes
-------------------------------------------------------------
* A one-dimensional sequence of Python ``int`` / NumPy integer values
  (list, tuple, range, ``numpy.ndarray`` of integer or object dtype).
* Every element is a prime >= 2.
* Strictly increasing order.  The library NEVER sorts or de-duplicates:
  unsorted input and duplicates are errors (``reason`` = ``"unsorted"`` /
  ``"duplicate"``).
* Floats, booleans, strings and ``None`` are rejected, even if integral
  (no silent coercion).
* For the gap/window pipeline the primes must also be *consecutive*
  (no prime missing between neighbours).  That is only proven when
  ``level="consecutive"``; the cheaper levels trust the caller.

Validation levels
-----------------
``"structure"``    type, range, ordering, duplicates, parity.  O(n), vectorised.
``"primality"``    + every element passes Miller-Rabin.  (default)
``"consecutive"``  + no prime exists between adjacent elements (expensive for
                   large inputs; intended for small inputs and tests).

Primality
---------
``is_prime`` is deterministic (a proof) for n < 3 317 044 064 679 887 385 961 981
using fixed Miller-Rabin base sets (Sinclair's 7 bases below 2**64, the first 13
prime bases above).  Beyond that bound it is a fixed-base probabilistic test;
``primality_is_proven(n)`` tells you which regime applies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .exceptions import InvalidPrimeInputError

LEVEL_STRUCTURE = "structure"
LEVEL_PRIMALITY = "primality"
LEVEL_CONSECUTIVE = "consecutive"
_LEVELS = (LEVEL_STRUCTURE, LEVEL_PRIMALITY, LEVEL_CONSECUTIVE)

_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)
_BASES_64 = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)       # n < 2**64
_BASES_13 = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)         # n < 3.317e24
_BASES_EXTRA = _BASES_13 + (43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)  # probabilistic above
_PROVEN_BOUND = 3_317_044_064_679_887_385_961_981
_INT64_MAX = (1 << 63) - 1


def _is_strong_probable_prime(n: int, a: int, d: int, r: int) -> bool:
    a %= n
    if a == 0:
        return True
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(r - 1):
        x = x * x % n
        if x == n - 1:
            return True
    return False


def primality_is_proven(n: int) -> bool:
    """True if ``is_prime(n)`` is a proof (deterministic regime)."""
    return n < _PROVEN_BOUND


def is_prime(n: int) -> bool:
    """Miller-Rabin primality test with fixed bases (see module docstring)."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError("is_prime expects an integer, got %r" % type(n).__name__)
    n = int(n)
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    if n < 97 * 97:
        return True
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    if n < (1 << 64):
        bases = _BASES_64
    elif n < _PROVEN_BOUND:
        bases = _BASES_13
    else:
        bases = _BASES_EXTRA
    return all(_is_strong_probable_prime(n, a, d, r) for a in bases)


def primes_up_to(limit: int) -> np.ndarray:
    """All primes <= ``limit`` as an int64 array (odd-only sieve of Eratosthenes).

    Convenience for examples, tests and reproduction scripts.  Memory is
    about ``limit / 2`` bytes, so ``limit`` is capped at 2 * 10**9.
    """
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)):
        raise TypeError("limit must be an integer")
    limit = int(limit)
    if limit < 2:
        return np.empty(0, dtype=np.int64)
    if limit > 2_000_000_000:
        raise ValueError("primes_up_to is capped at 2e9; supply primes explicitly for larger ranges")
    size = (limit - 1) // 2
    sieve = np.ones(size + 1, dtype=np.bool_)
    sieve[0] = False
    for i in range(1, int(limit ** 0.5) // 2 + 1):
        if sieve[i]:
            p = 2 * i + 1
            sieve[(p * p) // 2::p] = False
    return np.concatenate(([2], 2 * np.nonzero(sieve)[0] + 1)).astype(np.int64)


@dataclass(frozen=True)
class PrimeSequence:
    """A validated, immutable prime sequence.

    ``values`` is a read-only copy of the caller's data (int64 when every value
    fits, otherwise an object array of Python ints).  ``level`` records the
    strongest validation that was actually performed.
    """

    values: np.ndarray
    level: str
    deterministic_primality: bool

    def __len__(self) -> int:
        return int(self.values.shape[0])


def _as_int_list_or_array(values) -> np.ndarray:
    if isinstance(values, (str, bytes)) or values is None:
        raise InvalidPrimeInputError("not_a_sequence", "primes must be a sequence of integers, got %r" % type(values).__name__)
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise InvalidPrimeInputError("not_1d", "primes must be one-dimensional, got ndim=%d" % values.ndim)
        if values.dtype.kind in "iu":
            if values.size and values.dtype.kind == "u" and int(values.max()) > _INT64_MAX:
                arr = np.empty(values.shape[0], dtype=object)
                arr[:] = [int(v) for v in values]
                return arr
            return values.astype(np.int64)   # int8/16/32/uint* -> int64 so later arithmetic cannot overflow
        if values.dtype.kind == "b":
            raise InvalidPrimeInputError("not_integer", "boolean array is not a prime sequence", 0)
        if values.dtype.kind != "O":
            raise InvalidPrimeInputError("not_integer", "primes must have an integer dtype, got %s" % values.dtype, 0)
        items = list(values)
    else:
        try:
            items = list(values)
        except TypeError:
            raise InvalidPrimeInputError("not_a_sequence", "primes must be iterable, got %r" % type(values).__name__)
    for i, v in enumerate(items):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
            raise InvalidPrimeInputError("not_integer", "element %d is %r, expected an integer" % (i, type(v).__name__), i)
    items = [int(v) for v in items]
    if all(-_INT64_MAX <= v <= _INT64_MAX for v in items):
        return np.array(items, dtype=np.int64)
    arr = np.empty(len(items), dtype=object)
    arr[:] = items
    return arr


def validate_prime_sequence(primes: Iterable[int], level: str = LEVEL_PRIMALITY) -> PrimeSequence:
    """Validate ``primes`` and return an immutable :class:`PrimeSequence`.

    Raises :class:`InvalidPrimeInputError` (never repairs the input).
    """
    if level not in _LEVELS:
        raise ValueError("level must be one of %s, got %r" % (_LEVELS, level))
    arr = _as_int_list_or_array(primes)
    n = arr.shape[0]
    if n == 0:
        raise InvalidPrimeInputError("empty", "prime sequence is empty")
    arr = np.array(arr, copy=True)  # never alias or mutate caller data

    if arr.dtype != object:
        if int(arr.min()) < 2:
            i = int(np.argmax(arr < 2))
            raise InvalidPrimeInputError("too_small", "element %d (%d) is < 2" % (i, int(arr[i])), i)
        d = np.diff(arr)
        if n > 1 and (d <= 0).any():
            i = int(np.argmax(d <= 0)) + 1
            reason = "duplicate" if arr[i] == arr[i - 1] else "unsorted"
            raise InvalidPrimeInputError(reason, "element %d (%d) does not strictly increase after element %d (%d): %s"
                                         % (i, int(arr[i]), i - 1, int(arr[i - 1]), reason), i)
        even_big = (arr % 2 == 0) & (arr > 2)
        if even_big.any():
            i = int(np.argmax(even_big))
            raise InvalidPrimeInputError("composite", "element %d (%d) is even and > 2, hence not prime" % (i, int(arr[i])), i)
    else:
        for i in range(n):
            v = arr[i]
            if v < 2:
                raise InvalidPrimeInputError("too_small", "element %d (%d) is < 2" % (i, v), i)
            if i and v <= arr[i - 1]:
                reason = "duplicate" if v == arr[i - 1] else "unsorted"
                raise InvalidPrimeInputError(reason, "element %d does not strictly increase: %s" % (i, reason), i)
            if v > 2 and v % 2 == 0:
                raise InvalidPrimeInputError("composite", "element %d is even and > 2, hence not prime" % i, i)

    achieved = LEVEL_STRUCTURE
    if level in (LEVEL_PRIMALITY, LEVEL_CONSECUTIVE):
        for i in range(n):
            if not is_prime(arr[i]):
                raise InvalidPrimeInputError("composite", "element %d (%d) failed the primality test" % (i, int(arr[i])), i)
        achieved = LEVEL_PRIMALITY
    if level == LEVEL_CONSECUTIVE:
        for i in range(1, n):
            lo, hi = int(arr[i - 1]), int(arr[i])
            start = lo + 1 if lo == 2 else lo + 2   # only odd numbers can be prime above 2
            for c in range(start, hi, 2 if lo != 2 else 1):
                if is_prime(c):
                    raise InvalidPrimeInputError("not_consecutive",
                                                 "prime %d lies between element %d (%d) and element %d (%d)" % (c, i - 1, lo, i, hi), i)
        achieved = LEVEL_CONSECUTIVE

    # Elements are sorted, so arr[-1] is the largest; the proof regime depends on it.
    deterministic = achieved != LEVEL_STRUCTURE and primality_is_proven(int(arr[-1]))
    arr.flags.writeable = False
    return PrimeSequence(values=arr, level=achieved, deterministic_primality=deterministic)
