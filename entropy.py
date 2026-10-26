"""Binary bit-frequency Shannon entropy: the PGI-Crypto Process 2 SELECTION STATISTIC (G5).

Definition (exact)
------------------
For a positive integer ``n`` let ``b`` be its binary representation **without leading
zeros** (``len(b) == n.bit_length()``).  With ``f1`` = fraction of ``1`` bits and
``f0 = 1 - f1``::

    H(n) = -( f1 * log2(f1) + f0 * log2(f0) )          (0 * log2(0) := 0)

``0 <= H(n) <= 1`` bit.  A typical random 2048-bit modulus has H close to 1; H is
the entropy of the *bit histogram of one integer*.

What it is NOT
--------------
It is a descriptive statistic used to rank candidate pairs.  It is **not**
cryptographic entropy: it says nothing about the unpredictability of the
generator, nor about the security or factoring resistance of the modulus.  Do not
report it as such.  (The byte-histogram entropy of the old prototype saturates at
toy sizes and is not used.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

DEFINITION = ("Shannon entropy (log2) of the 0/1 frequency of the binary representation of N without leading zeros; "
              "range [0, 1] bit. A candidate-selection statistic, NOT cryptographic entropy.")


@dataclass(frozen=True)
class EntropyResult:
    statistic: float          # H(N) in bits, in [0, 1]
    bit_length: int
    ones: int
    zeros: int
    definition: str = DEFINITION


def bit_frequency_entropy(n: int) -> float:
    """H(n) as defined in the module docstring."""
    return bit_histogram_entropy(n).statistic


def bit_histogram_entropy(n: int) -> EntropyResult:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an int")
    if n < 1:
        raise ValueError("n must be a positive integer")
    length = n.bit_length()
    ones = bin(n).count("1")
    zeros = length - ones
    h = 0.0
    for c in (ones, zeros):
        if c:
            f = c / length
            h -= f * math.log2(f)
    return EntropyResult(statistic=h, bit_length=length, ones=ones, zeros=zeros)


def entropy(p: int, q: int, n: int = None) -> EntropyResult:
    """Entropy statistic of the modulus ``N = p * q`` (``n`` is checked against ``p * q`` if given)."""
    if n is not None and n != p * q:
        raise ValueError("n != p * q")
    return bit_histogram_entropy(p * q)


pair_entropy = entropy      # package-level name (``pgi_crypto.pair_entropy``): ``pgi_crypto.entropy`` stays the submodule
