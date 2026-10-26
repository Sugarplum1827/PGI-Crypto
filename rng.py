"""Randomness sources for key generation (G6).

Two explicitly separated modes:

``"secure"``       OS-backed CSPRNG (``random.SystemRandom`` -> ``os.urandom``).  Default.
                   A seed is rejected: secure mode is never reproducible.
``"reproducible"`` Deterministic, seeded Mersenne Twister.  **NON-SECURE.**  For tests and
                   research only; every result produced from it carries ``is_secure=False``
                   and the private key cannot be serialized unless the caller opts in.

The key-generation code only ever calls the small interface below, so no module
touches ``random``/NumPy generators directly.
"""
from __future__ import annotations

import random
from typing import Optional

from . import config


class RandomSource:
    """Uniform random integers from either the OS CSPRNG or a seeded (NON-SECURE) generator."""

    def __init__(self, mode: str = config.RNG_MODE_SECURE, seed: Optional[int] = None):
        if mode == config.RNG_MODE_SECURE:
            if seed is not None:
                raise ValueError("a seed is not allowed in secure mode; use mode='reproducible' for seeded (NON-SECURE) generation")
            self._rng = random.SystemRandom()
            self.is_secure = True
        elif mode == config.RNG_MODE_REPRODUCIBLE:
            if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("reproducible mode requires an integer seed")
            self._rng = random.Random(seed)
            self.is_secure = False
        else:
            raise ValueError("mode must be 'secure' or 'reproducible', got %r" % (mode,))
        self.mode = mode

    def randbits(self, k: int) -> int:
        if k < 1:
            raise ValueError("k must be >= 1")
        return self._rng.getrandbits(k)

    def randrange(self, lo: int, hi: int) -> int:
        """Uniform integer in ``[lo, hi)``."""
        if hi <= lo:
            raise ValueError("empty range")
        return self._rng.randrange(lo, hi)

    def describe(self) -> dict:
        """Non-secret description (never includes the seed)."""
        return {"mode": self.mode, "is_secure": self.is_secure,
                "source": "OS CSPRNG (os.urandom via random.SystemRandom)" if self.is_secure else "seeded Mersenne Twister (NON-SECURE)"}
