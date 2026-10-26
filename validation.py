"""Validation hooks.

These are *structural and consistency* checks that the KEY VALIDATION phase can
call and extend.  They deliberately make no cryptographic-strength claim: a key
that passes ``validate_key`` is well-formed, not secure.

All functions return a :class:`ValidationReport` (never raise on a failed
check) so callers can aggregate results; programming errors (wrong argument
type) raise ``TypeError``.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

from . import config
from .primes import is_prime, primality_is_proven, validate_prime_sequence
from .result_types import (AnalysisResult, ClassificationResult, FeatureMatrix, KeyGenerationResult, KeyResult,
                    KeyValidationResult, SecretValue, ValidationCheck, ValidationReport)

_TOL = 1e-9


def _c(name: str, ok: bool, detail: str = "") -> ValidationCheck:
    return ValidationCheck(name=name, passed=bool(ok), detail=detail)


def _skip(name: str, why: str) -> ValidationCheck:
    return ValidationCheck(name=name, passed=True, detail="SKIPPED (not applicable): " + why, skipped=True)


def validate_features(features: FeatureMatrix) -> ValidationReport:
    """Schema, ordering, finiteness and internal-consistency checks of an ML feature matrix."""
    if not isinstance(features, FeatureMatrix):
        raise TypeError("expected FeatureMatrix")
    v, names = features.values, features.names
    col = {n: i for i, n in enumerate(names)}
    checks = [
        _c("schema_names_and_order", tuple(names) == config.ML_FEATURE_NAMES, "must equal config.ML_FEATURE_NAMES"),
        _c("schema_version", features.schema_version == config.FEATURE_SCHEMA_VERSION),
        _c("shape", v.ndim == 2 and v.shape[1] == config.N_ML_FEATURES, "got %s" % (v.shape,)),
        _c("dtype_float64", v.dtype == np.float64, str(v.dtype)),
    ]
    finite = bool(np.isfinite(v).all())
    checks.append(_c("all_finite", finite))
    if finite and tuple(names) == config.ML_FEATURE_NAMES:
        f = lambda n: v[:, col[n]]  # noqa: E731
        checks += [
            _c("mean_gap_at_least_1", (f("mean_gap") >= 1.0).all()),
            _c("std_nonnegative", (f("std_dev") >= 0).all()),
            _c("fractions_in_unit_interval", all(((f(n) >= 0) & (f(n) <= 1)).all()
                                                 for n in ("twin_prime_density", "prop_even_gt_2logp", "large_gap_fraction"))),
            _c("min_le_median_le_max", ((f("min_gap") <= f("median_gap")) & (f("median_gap") <= f("max_gap"))).all()),
            _c("gap_range_consistent", np.allclose(f("gap_range"), f("max_gap") - f("min_gap"), atol=_TOL)),
            _c("poisson_lambda_equals_mean", np.array_equal(f("poisson_lambda_fit"), f("mean_gap"))),
            _c("log_mean_gap_consistent", np.allclose(f("log_mean_gap"), np.log(f("mean_gap")), atol=_TOL)),
            _c("entropy_nonnegative", (f("gap_entropy") >= 0).all()),
        ]
    return ValidationReport("features", tuple(checks))


def validate_classification(result: ClassificationResult) -> ValidationReport:
    """Output-schema and internal-consistency checks of a classification result."""
    if not isinstance(result, ClassificationResult):
        raise TypeError("expected ClassificationResult")
    n = result.n_rows
    p1, p2, ev = result.stage1_event_probability, result.stage2_probabilities, result.stage2_evaluated
    checks = [
        _c("shapes", p1.shape == (n,) and p2.shape == (n, len(config.STAGE2_CLASSES)) and result.final_class.shape == (n,)),
        _c("stage1_probability_in_unit_interval", ((p1 >= 0) & (p1 <= 1)).all()),
        _c("stage1_decision_matches_threshold", np.array_equal(result.stage1_event, p1 >= result.stage1_threshold)),
        _c("final_class_labels_valid", all(str(c) in config.FINAL_CLASSES for c in result.final_class)),
    ]
    if ev.any():
        rows = p2[ev]
        checks.append(_c("stage2_rows_sum_to_one", np.allclose(rows.sum(axis=1), 1.0, atol=1e-5)))
        checks.append(_c("stage2_probability_in_unit_interval", ((rows >= 0) & (rows <= 1)).all()))
        expect = np.array(config.STAGE2_CLASSES, dtype=object)[rows.argmax(axis=1)]
        checks.append(_c("stage2_prediction_is_argmax", np.array_equal(result.stage2_prediction[ev], expect)))
    checks.append(_c("stage2_unevaluated_rows_are_nan", bool(np.isnan(p2[~ev]).all())))
    gated = result.stage1_event & ev
    expected_final = np.where(gated, result.stage2_prediction, config.NONE_LABEL)
    checks.append(_c("final_class_follows_cascade", np.array_equal(expected_final.astype(str), result.final_class.astype(str))))
    checks.append(_c("model_info_present", bool(result.model_info.stage1_sha256) and bool(result.model_info.stage2_sha256)))
    frozen = config.MODEL_ARTIFACTS
    checks.append(_c("model_hashes_match_frozen",
                     result.model_info.stage1_sha256 == frozen["stage1"]["sha256"]
                     and result.model_info.stage2_sha256 == frozen["stage2"]["sha256"]))
    return ValidationReport("classification", tuple(checks))


def validate_analysis(result: AnalysisResult) -> ValidationReport:
    """Geometry + sub-result checks of an :class:`AnalysisResult`."""
    if not isinstance(result, AnalysisResult):
        raise TypeError("expected AnalysisResult")
    k, step, _ = result.window_spec
    n = result.n_windows
    checks = [
        _c("window_count_formula", n == (result.n_primes - 1 - k) // step + 1, "n_primes=%d" % result.n_primes),
        _c("array_lengths", len(result.p_start) == len(result.p_mid) == len(result.p_end) == n
           == result.features.n_rows == result.classification.n_rows == result.analytics.n_rows == len(result.outside_training_range)),
        _c("p_start_lt_p_mid_lt_p_end", all(int(a) < int(b) < int(c) for a, b, c in zip(result.p_start, result.p_mid, result.p_end))),
        _c("frozen_window_definition", (k, step) == (config.WINDOW_GAPS, config.WINDOW_STEP)),
    ]
    checks += list(validate_features(result.features).checks)
    checks += list(validate_classification(result.classification).checks)
    return ValidationReport("analysis", tuple(checks))


def validate_modulus_structure(p: int, q: int, n: Optional[int] = None, min_bits: Optional[int] = None,
                               min_prime_distance: Optional[int] = None) -> ValidationReport:
    """Textbook structural checks for a product of two primes.

    Algorithm-independent.  ``min_prime_distance`` (|p - q| lower bound) and
    ``min_bits`` are explicit arguments: the library does not choose them
    (their interpretation is specification gap G8).
    """
    p, q = int(p), int(q)
    checks = [
        _c("p_is_prime", is_prime(p), "deterministic proof" if primality_is_proven(p) else "probabilistic"),
        _c("q_is_prime", is_prime(q), "deterministic proof" if primality_is_proven(q) else "probabilistic"),
        _c("p_ne_q", p != q),
    ]
    if n is not None:
        checks.append(_c("n_equals_p_times_q", int(n) == p * q))
    modulus = p * q if n is None else int(n)
    if min_bits is not None:
        checks.append(_c("modulus_bit_length", modulus.bit_length() >= min_bits, "%d bits" % modulus.bit_length()))
    if min_prime_distance is not None:
        checks.append(_c("prime_distance_bound", abs(p - q) > min_prime_distance))
    return ValidationReport("modulus_structure", tuple(checks))


def validate_key(key: KeyResult, min_prime_distance: Optional[int] = None) -> ValidationReport:
    """Well-formedness of a :class:`KeyResult` (NOT a security assessment)."""
    if not isinstance(key, KeyResult):
        raise TypeError("expected KeyResult")
    public = key.to_public_dict()
    checks = [
        _c("key_is_secret_value", isinstance(key.key, SecretValue)),
        _c("key_length_positive_int", isinstance(key.key_length_bits, int) and key.key_length_bits > 0),
        _c("algorithm_identified", isinstance(key.algorithm, str) and bool(key.algorithm)),
        _c("public_dict_redacts_key", public.get("key") == "<redacted>"),
        _c("repr_does_not_leak", "SecretValue(<redacted>)" in repr(key) or "<redacted>" in repr(key.key)),
    ]
    payload = key.key.reveal() if isinstance(key.key, SecretValue) else None
    if isinstance(payload, tuple) and len(payload) == 2 and all(isinstance(x, int) for x in payload):
        checks += list(validate_modulus_structure(payload[0], payload[1], min_prime_distance=min_prime_distance).checks)
    return ValidationReport("key", tuple(checks))


def validate_generation(key: KeyResult, classifier) -> ValidationReport:
    """Provenance check: re-derive the source window's class and compare with the record.

    ``classifier`` is a :class:`FrozenClassifier`.  Needs ``key.source_window``
    (101 consecutive primes) and ``key.classification['final_class']``.
    """
    from .features import extract_ml_features
    from .windows import build_windows

    if not isinstance(key, KeyResult):
        raise TypeError("expected KeyResult")
    if key.source_window is None or not key.classification or "final_class" not in key.classification:
        return ValidationReport("generation", (_c("provenance_recorded", False, "source_window / classification missing"),))
    checks = [_c("provenance_recorded", True)]
    try:
        seq = validate_prime_sequence(list(key.source_window.primes), level="primality")
        ws = build_windows(seq)
        recomputed = classifier.classify(extract_ml_features(ws)).final_class[0]
        checks.append(_c("source_window_is_one_valid_window", ws.n_windows == 1, "windows=%d" % ws.n_windows))
        checks.append(_c("recorded_class_reproduces", str(recomputed) == str(key.classification["final_class"]),
                         "recorded=%s recomputed=%s" % (key.classification["final_class"], recomputed)))
    except Exception as exc:  # report, do not raise: this is a validator
        checks.append(_c("source_window_valid", False, "%s: %s" % (type(exc).__name__, exc)))
    return ValidationReport("generation", tuple(checks))


def validate_output(obj) -> ValidationReport:
    """Dispatch to the validator matching ``obj``'s type."""
    if isinstance(obj, FeatureMatrix):
        return validate_features(obj)
    if isinstance(obj, ClassificationResult):
        return validate_classification(obj)
    if isinstance(obj, AnalysisResult):
        return validate_analysis(obj)
    if isinstance(obj, KeyResult):
        return validate_key(obj)
    raise TypeError("no validator for %s" % type(obj).__name__)


def _strong_prime_check(n: int) -> bool:
    """Deterministic below 3.3e24; above: fixed bases plus 12 random bases from the OS CSPRNG."""
    from .primesearch import strong_probable_prime
    if not is_prime(n):
        return False
    if primality_is_proven(n):
        return True
    import random
    sysrand = random.SystemRandom()
    return all(strong_probable_prime(n, sysrand.randrange(2, n - 1)) for _ in range(12))


def _recorded_eligibility_consistent(c, policy) -> tuple:
    """Re-derive eligibility from the RECORDED components with the frozen thresholds."""
    from .eligibility import POLICY_STRICT
    el = c.eligibility
    if el is None:
        return False, "no eligibility record"
    lo, hi = policy.cr_band
    rule_ok = (lo <= el.cr <= hi) and el.ks_pass
    ml_ok = (el.final_class == "CC") and (el.p_cc >= policy.p_cc_threshold)
    expected = rule_ok and (ml_ok if policy.enforce_ml_gate else True)
    prod_ok = math.isclose(el.p_cc, el.p_event * el.p_cc_given_event, rel_tol=1e-12, abs_tol=1e-15)
    return (el.eligible == expected and expected and prod_ok and el.policy_name == policy.name,
            "recorded eligible=%s, recomputed=%s, P_CC product ok=%s" % (el.eligible, expected, prod_ok))


def validate_generated_key(result: KeyGenerationResult, *, classifier=None, deep: bool = False,
                           rebuild_windows: bool = False) -> KeyValidationResult:
    """Comprehensive validation of a :class:`KeyGenerationResult`.

    Checks: p and q prime, p != q, N == p*q, requested modulus size, gcd(e, phi) == 1, e*d == 1 (mod phi), CRT
    parameters, separation constraint, eligibility of p and q (re-derived from the recorded components with the
    frozen thresholds), valid analysis records, entropy statistic recomputed, redaction, PEM serialization round
    trips, and encrypt/decrypt + sign/verify with ``cryptography``.

    ``deep=True`` additionally recomputes features, classification and eligibility of both source windows from
    the stored (secret) windows and checks that every window prime is a probable prime;
    ``rebuild_windows=True`` also rebuilds both windows from scratch to confirm they are consecutive primes
    (slow at key scale).  None of this prints or logs key material.
    """
    from . import serialization as ser
    from .analysis import analyze_window, predict_window
    from .entropy import bit_histogram_entropy
    from .eligibility import POLICY_EXPERIMENTAL, POLICY_STRICT
    from .key_generation import POLICIES, build_rsa_material, separation_check
    from .primesearch import probable_prime_for_window
    from .windows import build_centered_window

    if not isinstance(result, KeyGenerationResult):
        raise TypeError("expected KeyGenerationResult")
    if result.private_material is None:
        return KeyValidationResult("generated_key", (_c("private_material_present", False, "generation failed; no key"),))
    m = result.private_material.reveal()
    p, q, n, e, d = m.p, m.q, m.n, m.e, m.d
    phi = (p - 1) * (q - 1)
    policy = result.policy if result.policy is not None else POLICIES.get(result.policy_name)
    chk = []
    chk += [_c("p_is_prime", _strong_prime_check(p)), _c("q_is_prime", _strong_prime_check(q)), _c("p_ne_q", p != q),
            _c("n_equals_p_times_q", n == p * q and result.modulus == n),
            _c("modulus_bits_as_requested", n.bit_length() == result.modulus_bits, "%d bits" % n.bit_length()),
            _c("prime_sizes", p.bit_length() == q.bit_length() == result.modulus_bits // 2, "p,q: %d,%d bits" % (p.bit_length(), q.bit_length())),
            _c("gcd_e_phi_is_1", math.gcd(e, phi) == 1),
            _c("e_times_d_congruent_1_mod_phi", (e * d) % phi == 1),
            _c("d_in_range", 0 < d < phi),
            _c("crt_dp", m.dp == d % (p - 1)), _c("crt_dq", m.dq == d % (q - 1)),
            _c("crt_qinv", (m.qinv * q) % p == 1)]
    sep = separation_check(max(p, q), min(p, q), result.modulus_bits)
    chk.append(_c("separation_condition", sep.separation_pass and result.separation is not None and result.separation.separation_pass,
                  "|p-q| has %d bits; threshold 2^%d" % (sep.actual_prime_distance_bits, sep.threshold_exponent)))
    chk.append(_c("eligibility_thresholds_are_the_frozen_values",
                  policy is not None and policy.p_cc_threshold == config.P_CC_THRESHOLD and tuple(policy.cr_band) == tuple(config.CR_CC_BAND),
                  "P_CC >= %s, CR band %s" % (config.P_CC_THRESHOLD, config.CR_CC_BAND)))
    for role, info in (("p", result.p_info), ("q", result.q_info)):
        if info is None or policy is None:
            chk.append(_c("%s_eligibility_recorded_and_consistent" % role, False, "missing record or unknown policy %r" % result.policy_name))
            continue
        ok, detail = _recorded_eligibility_consistent(info, policy)
        chk.append(_c("%s_eligibility_recorded_and_consistent" % role, ok, detail))
        a = info.analysis or {}
        vals = [a.get("cr"), a.get("ks_pvalue"), a.get("mean_gap")]
        chk.append(_c("%s_analysis_record_valid" % role, all(isinstance(v, float) and math.isfinite(v) for v in vals) and a.get("p_mid") == "REDACTED"))
    chk.append(_c("entropy_statistic_recomputed", result.entropy is not None and math.isclose(result.entropy.statistic, bit_histogram_entropy(n).statistic, abs_tol=1e-12)))
    chk.append(_c("nonsecure_mode_flagged", result.is_secure_mode or any("NON-SECURE" in w for w in result.warnings)))

    # redaction of the result object itself
    text = repr(result) + str(result.redacted_info()) + repr(result.private_material)
    secrets_ = [str(x) for x in (p, q, d, m.dp, m.dq, m.qinv) if len(str(x)) >= 40]
    chk.append(_c("result_redaction", not any(sx in text for sx in secrets_)) if secrets_ else _skip("result_redaction", "values too short for a meaningful substring test"))

    # serialization round trips (cryptography)
    try:
        pub = ser.serialize_public_key(result)
        chk.append(_c("serialization_public_roundtrip", ser.load_public_key(pub) == (n, e)))
        pw = os.urandom(12).hex().encode()
        priv = ser.serialize_private_key(result, password=pw, allow_nonsecure=True)
        back = ser.load_private_key(priv, pw)
        chk.append(_c("serialization_private_roundtrip", (back.p, back.q, back.d, back.n, back.e) == (p, q, d, n, e)
                      and priv.startswith(b"-----BEGIN ENCRYPTED PRIVATE KEY-----")))
    except Exception as exc:
        chk.append(_c("serialization_roundtrip", False, type(exc).__name__))
    # RSA arithmetic round trip and standard-library operations
    mtest = int.from_bytes(os.urandom(max(1, (n.bit_length() - 8) // 8)), "big") % n or 2
    chk.append(_c("rsa_math_roundtrip", pow(pow(mtest, e, n), d, n) == mtest))
    try:
        ops = ser.rsa_operation_roundtrip(m)
        if ops["encryption_ok"] is None:
            chk.append(_skip("rsa_encrypt_decrypt", ops["skipped_reason"] or "no padding fits"))
        else:
            chk.append(_c("rsa_encrypt_decrypt", ops["encryption_ok"], ops["scheme"]))
        if ops["signature_ok"] is None:
            chk.append(_skip("rsa_sign_verify", "modulus too small for SHA-256 signatures"))
        else:
            chk.append(_c("rsa_sign_verify", ops["signature_ok"], ops["scheme"]))
    except Exception as exc:
        chk.append(_c("rsa_operations", False, type(exc).__name__))

    if deep:
        if classifier is None:
            from .classification import FrozenClassifier
            classifier = FrozenClassifier()
        for role, info in (("p", result.p_info), ("q", result.q_info)):
            try:
                win = info.window.reveal()
                chk.append(_c("%s_window_centered_on_prime" % role, win.center == (p if role == "p" else q)))
                chk.append(_c("%s_window_primes_probable" % role, all(probable_prime_for_window(x) for x in win.primes)))
                an = analyze_window(win)
                cl = predict_window(an, classifier)
                from .eligibility import check_eligibility
                el = check_eligibility(an, cl, policy)
                chk.append(_c("%s_eligibility_recomputed" % role, el.eligible == info.eligibility.eligible
                              and math.isclose(el.p_cc, info.eligibility.p_cc, abs_tol=1e-9) and math.isclose(el.cr, info.eligibility.cr, abs_tol=1e-12)))
                if rebuild_windows:
                    rebuilt = build_centered_window(win.center, win.window_size, center_is_prime=True)
                    chk.append(_c("%s_window_rebuilt_identical" % role, rebuilt.primes == win.primes))
            except Exception as exc:
                chk.append(_c("%s_deep_validation" % role, False, type(exc).__name__))
    return KeyValidationResult("generated_key", tuple(chk))
