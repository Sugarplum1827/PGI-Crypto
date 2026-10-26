"""PGI-Crypto key generation (finalized specification, research prototype).

Pipeline per prime (rejection sampling)::

    draw a uniformly random odd integer of the requested size (CSPRNG in secure mode)
      -> reject composites / gcd(e, p-1) != 1                      [candidate generation]
      -> centered 100-gap window around the prime                    [window]
      -> features, newly trained four-class probabilities, P_CC, CR, KS            [analysis + classification]
      -> eligibility under the chosen policy                         [accept or REJECT and redraw]

Candidates are never produced by incremental search from a previous candidate: every attempt is
a fresh uniform draw.  (The 100 neighbour primes of an accepted-or-rejected candidate are found
by a deterministic scan because the window is *defined* as consecutive primes.)

Pair selection (Process 2): ``pool_size`` eligible primes are collected; among all pairs that
satisfy the separation constraint the pair with the highest bit-frequency entropy of N = p*q is
chosen.  With the default ``pool_size=2`` there is exactly one pair, so the entropy value is
REPORTED and no selection takes place (``selection_effect=False``).  The entropy is a statistical
selection criterion, not a security claim.

Separation: ``|p - q| > 2 ** (modulus_bits // 2 - 100)`` (see ``separation_check``).

RSA: e = 65537 (default); phi = (p-1)(q-1); d = e^-1 mod phi; CRT dp, dq, qinv.

Scale limitation (G4): the new range-spanning classifier was trained and evaluated on sampled windows through p_mid <= 10^12. RSA-2048 prime factors are near 2^1024 and remain outside that support. Strict mode blocks before candidate generation; Experimental reports out-of-domain predictions and is not security validation.
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Callable, Dict, List, Optional

from . import config
from .analysis import analyze_window, predict_window
from .rebuilt_classifier import RebuiltClassifier
from .eligibility import POLICY_EXPERIMENTAL, POLICY_STRICT, EligibilityPolicy, check_eligibility
from .entropy import bit_histogram_entropy
from .exceptions import (KeyGenerationError, KeyValidationError, OutsideValidatedRangeError,
                         SpecificationNotResolvedError)
from .primes import is_prime
from .primesearch import strong_probable_prime
from .rng import RandomSource
from .result_types import (KeyGenerationResult, PrimeCandidate, ProgressEvent, RSAPrivateMaterial, SecretValue,
                    SeparationResult)
from .windows import build_centered_window

SPEC_RESOLVED = True

#: Finalized decisions (details: PGI_CRYPTO_FINAL_SPECIFICATION.md).
SPEC_DECISIONS: Dict[str, str] = {
    "G1": "RSA-style modulus N = p*q, default 2048 bits (p, q ~ 1024 bits); modulus_bits configurable; 4096 architecturally allowed, not validated.",
    "G2": "Centered window: 100 gaps (101 primes), candidate prime = p_mid = 51st prime; window_size is distinct from modulus_bits.",
    "G3": "P_CC = P(EVENT)*P(CC|EVENT) >= 0.80 AND 0.85 <= CR <= 1.15 AND KS pass AND final class CC; MA, MGV, NONE rejected.",
    "G4": "New four-class model trained on a range-spanning sample through p_mid <= 1e12; RSA-2048 prime factors remain outside support and are never called validated.",
    "G5": "Binary bit-frequency Shannon entropy of N (statistical selection criterion, not cryptographic entropy).",
    "G6": "Secure mode: OS CSPRNG (default). Reproducible mode: seeded, NON-SECURE, test/research only.",
    "G7": "Rejection sampling of uniformly drawn candidates; MA/MGV/NONE candidates are rejected.",
    "G8": "RSA key with e=65537, d, CRT; separation |p-q| > 2^(modulus_bits/2 - 100) with modulus_bits explicit; validation + PEM serialization via 'cryptography'.",
}
SPEC_GAPS: Dict[str, str] = {}      # backward compatibility: no open specification items remain


def specification_status() -> Dict[str, object]:
    """Machine-readable status of the key-generation specification."""
    return {"resolved": SPEC_RESOLVED, "spec_version": config.KEY_GENERATION_SPEC_VERSION,
            "decisions": dict(SPEC_DECISIONS), "open_items": dict(SPEC_GAPS),
            "document": "PGI_CRYPTO_FINAL_SPECIFICATION.md",
            "scale_limitation": config.KEY_SCALE_WARNING}


POLICIES = {POLICY_STRICT.name: POLICY_STRICT, POLICY_EXPERIMENTAL.name: POLICY_EXPERIMENTAL}

_SMALL_PRIMES = [p for p in range(3, 2000, 2) if all(p % q for q in range(3, int(p ** 0.5) + 1, 2))]
_PRIMORIAL = 1
for _p in _SMALL_PRIMES:
    _PRIMORIAL *= _p


# ---------------------------------------------------------------------------
# Building blocks (public so that the validators and tests use the same code)
# ---------------------------------------------------------------------------

def prime_bits_for(modulus_bits: int) -> int:
    return modulus_bits // 2


def prime_range(modulus_bits: int):
    """``[lo, hi)`` such that any two primes in it multiply to exactly ``modulus_bits`` bits."""
    b = prime_bits_for(modulus_bits)
    lo = math.isqrt(1 << (2 * b - 1)) + 1          # lo^2 >= 2^(2b-1)  =>  p*q >= 2^(2b-1)
    return lo, 1 << b


def separation_check(p: int, q: int, modulus_bits: int) -> SeparationResult:
    """Prime-separation constraint with the explicit parameter ``modulus_bits``.

    Interpretation (documented decision): the thesis writes ``|p - q| > 2^(k/2 - 100)`` without defining
    ``k`` (elsewhere ``k`` is the window size 100, which would make the bound vacuous: 2^-50).  The
    only reading under which the bound is meaningful is the standard RSA key-generation condition,
    with ``k`` = modulus bit length; it is implemented with ``modulus_bits``.  For
    ``modulus_bits < 200`` the exponent is negative and the condition reduces to ``p != q``.
    """
    exponent = modulus_bits // 2 - config.SEPARATION_SLACK_BITS
    threshold = (1 << exponent) if exponent >= 0 else 0
    dist = abs(p - q)
    return SeparationResult(
        modulus_bits=modulus_bits, threshold_exponent=exponent, separation_threshold=threshold,
        actual_prime_distance_bits=dist.bit_length(), actual_prime_distance=SecretValue(dist),
        separation_pass=dist > threshold,
        interpretation="|p - q| > 2^(modulus_bits/2 - %d) with k := modulus_bits (thesis does not define k; see FINAL_SPECIFICATION D-SEP)"
                       % config.SEPARATION_SLACK_BITS)


def build_rsa_material(p: int, q: int, e: int = config.DEFAULT_PUBLIC_EXPONENT) -> RSAPrivateMaterial:
    """N, phi, d = e^-1 mod phi and the CRT parameters.  Raises ``KeyGenerationError`` if e is not invertible."""
    if p == q:
        raise KeyGenerationError("p == q")
    phi = (p - 1) * (q - 1)
    if math.gcd(e, phi) != 1:
        raise KeyGenerationError("gcd(e, phi) != 1")
    d = pow(e, -1, phi)
    return RSAPrivateMaterial(p=p, q=q, d=d, dp=d % (p - 1), dq=d % (q - 1), qinv=pow(q, -1, p), n=p * q, e=e)


def _probably_prime(n: int, rng: RandomSource, extra_rounds: int = 8) -> bool:
    """Candidate-prime test: trial gcd, deterministic/fixed-base Miller-Rabin, plus random-base rounds from ``rng``."""
    if n < 4:
        return n in (2, 3)
    if n % 2 == 0:
        return False
    if n > 2000 and math.gcd(n, _PRIMORIAL) != 1:
        return False
    if not is_prime(n):
        return False
    if n < 1 << 64:
        return True                                  # deterministic proof already
    for _ in range(extra_rounds):
        if not strong_probable_prime(n, rng.randrange(2, n - 1)):
            return False
    return True


def draw_random_prime(rng: RandomSource, modulus_bits: int, e: int = config.DEFAULT_PUBLIC_EXPONENT):
    """Uniformly random prime ``p`` with ``gcd(e, p-1) = 1`` in the prime range (rejection sampling).

    Returns ``(p, integers_examined)``.
    """
    lo, hi = prime_range(modulus_bits)
    examined = 0
    while True:
        n = rng.randrange(lo, hi) | 1
        if n >= hi:
            continue
        examined += 1
        if math.gcd(e, n - 1) != 1:
            continue
        if _probably_prime(n, rng):
            return n, examined


def _check_params(modulus_bits, e, pool_size, max_attempts, window_size):
    if isinstance(modulus_bits, bool) or not isinstance(modulus_bits, int):
        raise ValueError("modulus_bits must be an int")
    if modulus_bits < config.MIN_MODULUS_BITS or modulus_bits % 2:
        raise ValueError("modulus_bits must be even and >= %d" % config.MIN_MODULUS_BITS)
    if prime_bits_for(modulus_bits) < 9:
        raise ValueError("primes need >= 9 bits so that a centered window (50 primes below) exists")
    if e < 3 or e % 2 == 0:
        raise ValueError("public exponent must be an odd integer >= 3")
    if pool_size < 2:
        raise ValueError("pool_size must be >= 2")
    if max_attempts < pool_size:
        raise ValueError("max_attempts must be >= pool_size")
    if window_size != config.WINDOW_GAPS:
        raise ValueError("the rebuilt classifier expects window_size=%d" % config.WINDOW_GAPS)


class PGICryptoKeyGenerator:
    """PGI-Crypto key generator (research prototype).  See the module docstring."""

    def __init__(self, classifier: Optional[RebuiltClassifier] = None):
        self._classifier = classifier

    @property
    def classifier(self) -> RebuiltClassifier:
        if self._classifier is None:
            self._classifier = RebuiltClassifier()
        return self._classifier

    @property
    def is_available(self) -> bool:
        return SPEC_RESOLVED

    def generate(self, modulus_bits: int = config.DEFAULT_MODULUS_BITS, *, public_exponent: int = config.DEFAULT_PUBLIC_EXPONENT,
                 policy: EligibilityPolicy = POLICY_STRICT, pool_size: int = config.DEFAULT_POOL_SIZE,
                 max_attempts: int = config.DEFAULT_MAX_ATTEMPTS, rng_mode: str = config.RNG_MODE_SECURE,
                 seed: Optional[int] = None, stage1_threshold: float = config.STAGE1_THRESHOLD_BASELINE,
                 window_size: int = config.DEFAULT_WINDOW_SIZE, progress: Optional[Callable[[ProgressEvent], None]] = None,
                 validate: bool = True) -> KeyGenerationResult:
        _check_params(modulus_bits, public_exponent, pool_size, max_attempts, window_size)
        lo, hi = prime_range(modulus_bits)
        if policy.require_validated_range and hi > config.VALIDATED_MAX_P_MID:
            raise OutsideValidatedRangeError(
                "policy %r requires the validated range (%s) but %d-bit primes reach %d bits" %
                (policy.name, config.VALIDATED_RANGE_TEXT, prime_bits_for(modulus_bits), hi.bit_length()))
        rng = RandomSource(rng_mode, seed)
        t_start = time.perf_counter()
        timings = {"candidate_generation": 0.0, "window_analysis_classification": 0.0, "pair_selection": 0.0,
                   "rsa_construction": 0.0, "validation": 0.0}
        candidates: List[PrimeCandidate] = []
        pool: List[PrimeCandidate] = []
        state = {"attempts": 0}

        def emit(stage, attempt, message, **data):
            if progress is not None:
                progress(ProgressEvent(stage=stage, attempt=attempt, message=message, data=data))

        def next_eligible() -> Optional[PrimeCandidate]:
            while state["attempts"] < max_attempts:
                state["attempts"] += 1
                n_try = state["attempts"]
                t0 = time.perf_counter()
                prime, examined = draw_random_prime(rng, modulus_bits, public_exponent)
                timings["candidate_generation"] += time.perf_counter() - t0
                emit("candidate_generated", n_try, "candidate prime generated by rejection sampling", bits=prime.bit_length(),
                     integers_examined=examined)
                if any(c.value is not None and c.value.reveal() == prime for c in pool):
                    continue
                t1 = time.perf_counter()
                window = build_centered_window(prime, window_size, center_is_prime=True)
                analysis = analyze_window(window)
                emit("window_built", n_try, "centered %d-gap window built and analysed" % window_size,
                     outside_training_range=analysis.outside_training_range)
                cls = predict_window(analysis, self.classifier, stage1_threshold)
                elig = check_eligibility(analysis, cls, policy)
                timings["window_analysis_classification"] += time.perf_counter() - t1
                summary = analysis.summary(redact_primes=True)
                summary["random_integers_examined"] = examined
                cand = PrimeCandidate(
                    attempt=n_try, bits=prime.bit_length(), role="pool" if elig.eligible else "rejected", accepted=elig.eligible,
                    rejection_reason=None if elig.eligible else elig.reason, eligibility=elig, analysis=summary,
                    seconds=time.perf_counter() - t0, value=SecretValue(prime) if elig.eligible else None,
                    window=SecretValue(window) if elig.eligible else None)
                candidates.append(cand)
                emit("candidate_accepted" if elig.eligible else "candidate_rejected", n_try,
                     "eligible" if elig.eligible else "rejected: " + ", ".join(elig.reasons),
                     p_event=elig.p_event, p_cc_given_event=elig.p_cc_given_event, p_cc=elig.p_cc, cr=elig.cr,
                     ks_pass=elig.ks_pass, final_class=elig.final_class)
                if elig.eligible:
                    return cand
            return None

        def best_pair():
            best = None
            n_valid = 0
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    a, b = pool[i], pool[j]
                    big, small = (a, b) if a.value.reveal() > b.value.reveal() else (b, a)
                    sep = separation_check(big.value.reveal(), small.value.reveal(), modulus_bits)
                    if not sep.separation_pass:
                        continue
                    n_valid += 1
                    ent = bit_histogram_entropy(big.value.reveal() * small.value.reveal())
                    if best is None or ent.statistic > best[2].statistic:
                        best = (big, small, ent, sep)
            return best, n_valid

        selection = None
        n_valid = 0
        while selection is None:
            while len(pool) < pool_size:
                cand = next_eligible()
                if cand is None:
                    return self._fail(modulus_bits, public_exponent, candidates, state["attempts"], policy, pool_size, rng, t_start,
                                      timings, "no eligible prime found after %d attempts (policy %s)" % (state["attempts"], policy.name))
                pool.append(cand)
            t0 = time.perf_counter()
            selection, n_valid = best_pair()
            timings["pair_selection"] += time.perf_counter() - t0
            if selection is None:                 # separation failed for every pair: add one more eligible prime
                cand = next_eligible()
                if cand is None:
                    return self._fail(modulus_bits, public_exponent, candidates, state["attempts"], policy, pool_size, rng, t_start,
                                      timings, "separation constraint not satisfiable within %d attempts" % state["attempts"])
                pool.append(cand)
        big, small, ent, sep = selection
        emit("pair_selected", state["attempts"], "prime pair selected", entropy_statistic=ent.statistic,
             separation_pass=sep.separation_pass, pairs_considered=n_valid)

        t0 = time.perf_counter()
        p_val, q_val = big.value.reveal(), small.value.reveal()
        material = build_rsa_material(p_val, q_val, public_exponent)
        if material.n.bit_length() != modulus_bits:
            raise KeyGenerationError("internal error: modulus has %d bits, requested %d" % (material.n.bit_length(), modulus_bits))
        timings["rsa_construction"] += time.perf_counter() - t0
        roles = {big.attempt: "p", small.attempt: "q"}
        candidates = [dataclasses.replace(c, role=roles[c.attempt]) if c.attempt in roles else c for c in candidates]
        p_info = next(c for c in candidates if c.role == "p")
        q_info = next(c for c in candidates if c.role == "q")

        warnings = self._warnings(policy, rng, pool_size, n_valid, candidates)
        result = KeyGenerationResult(
            success=True, modulus_bits=modulus_bits, modulus=material.n, public_exponent=public_exponent, p_info=p_info,
            q_info=q_info, entropy=ent, separation=sep, candidate_attempts=state["attempts"], candidates=tuple(candidates),
            generation_time_seconds=time.perf_counter() - t_start, timings=dict(timings), validation=None,
            policy_name=policy.name, ml_gate_enforced=policy.enforce_ml_gate, pool_size=pool_size,
            selection_effect=n_valid > 1, rng=rng.describe(), is_secure_mode=rng.is_secure,
            spec_version=config.KEY_GENERATION_SPEC_VERSION, warnings=tuple(warnings),
            research_limitations=config.RESEARCH_LIMITATIONS, private_material=SecretValue(material), policy=policy)
        if validate:
            from .validation import validate_generated_key
            emit("validating", state["attempts"], "validating generated key")
            tv = time.perf_counter()
            validation = validate_generated_key(result, classifier=self.classifier)
            timings["validation"] = time.perf_counter() - tv
            result = dataclasses.replace(result, validation=validation, success=validation.passed, timings=dict(timings),
                                         generation_time_seconds=time.perf_counter() - t_start)
            if not validation.passed:
                err = KeyValidationError("generated key failed validation: %s" % ", ".join(c.name for c in validation.failures),
                                         validation=validation)
                err.result = result
                raise err
        else:
            result = dataclasses.replace(result, warnings=result.warnings + ("Key was NOT validated (validate=False).",))
        emit("done", state["attempts"], "key generation finished")
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _warnings(policy, rng, pool_size, n_valid, candidates):
        w = []
        if any(c.analysis and c.analysis["outside_training_range"] for c in candidates):
            w.append(config.KEY_SCALE_WARNING)
        if not policy.enforce_ml_gate:
            w.append("EXPERIMENTAL eligibility policy: the rebuilt model's class and P_CC conditions were reported but NOT enforced; "
                     "only the analytical CR-band and KS conditions gated the candidates. This does not satisfy the finalized "
                     "specification (G3) and must not be described as ML-validated key generation.")
        if not rng.is_secure:
            w.append("NON-SECURE reproducible RNG mode: this key is for tests/research only and must never be used for real data.")
        if pool_size == 2 or n_valid <= 1:
            w.append("Only one eligible prime pair was available: the entropy statistic is reported but no entropy-based selection took place.")
        w.append(config.ENTROPY_NOTE)
        n_prob = [c for c in candidates if c.analysis and c.analysis.get("neighbour_primality") == "probable"]
        if n_prob:
            w.append("Window neighbour primes are strong probable primes (bases 2 and 3), not proofs.")
        return w

    def _fail(self, modulus_bits, e, candidates, attempts, policy, pool_size, rng, t_start, timings, message):
        reasons: Dict[str, int] = {}
        for c in candidates:
            if not c.accepted:
                reasons[c.rejection_reason or "unknown"] = reasons.get(c.rejection_reason or "unknown", 0) + 1
        best = max((c.eligibility.p_cc for c in candidates if c.eligibility), default=float("nan"))
        partial = KeyGenerationResult(
            success=False, modulus_bits=modulus_bits, modulus=0, public_exponent=e, p_info=None, q_info=None, entropy=None,
            separation=None, candidate_attempts=attempts, candidates=tuple(candidates),
            generation_time_seconds=time.perf_counter() - t_start, timings=dict(timings), validation=None,
            policy_name=policy.name, ml_gate_enforced=policy.enforce_ml_gate, pool_size=pool_size, selection_effect=False,
            rng=rng.describe(), is_secure_mode=rng.is_secure, spec_version=config.KEY_GENERATION_SPEC_VERSION,
            warnings=tuple(self._warnings(policy, rng, pool_size, 0, candidates)) + ("rejection reasons: %s" % reasons,),
            research_limitations=config.RESEARCH_LIMITATIONS, policy=policy)
        raise KeyGenerationError("%s; rejection reasons: %s; highest P_CC seen: %.4f" % (message, reasons, best), result=partial)
