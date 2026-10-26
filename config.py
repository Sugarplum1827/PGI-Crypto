"""Frozen constants for the PGI-Crypto core library.

Everything in this module is copied from, or verified against, the frozen
external XGBoost experiment (``CH4_FINAL_EXPERIMENT_FREEZE.md`` /
``EXTERNAL_RUN_MANIFEST.json``).  Nothing here is a tunable hyper-parameter:
changing a value changes the methodology, not the configuration.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Versions
# --------------------------------------------------------------------------
LIBRARY_VERSION = "0.3.0"
METHODOLOGY_VERSION = "frozen-baseline-v1"  # CH4_FINAL_EXPERIMENT_FREEZE
FEATURE_SCHEMA_VERSION = "ml24-v1"
# Finalized specification decisions G1-G8 (see PGI_CRYPTO_FINAL_SPECIFICATION.md).
KEY_GENERATION_SPEC_VERSION = "pgi-keygen-final-v1"

# --------------------------------------------------------------------------
# Window definition (frozen)
# --------------------------------------------------------------------------
WINDOW_GAPS = 100          # k: consecutive prime gaps per window (101 primes)
WINDOW_STEP = 50           # step between window starts, in gaps
WINDOW_MID_OFFSET = 50     # p_mid = primes[start + 50]
MIN_PRIMES_FOR_ONE_WINDOW = WINDOW_GAPS + 1
FROZEN_LIMIT_N = 10 ** 8   # the training data covers primes <= 1e8

# --------------------------------------------------------------------------
# ML feature schema: the 24 columns consumed by the frozen XGBoost models,
# in the exact order of EXTERNAL_RUN_MANIFEST.json["feature_columns"].
# --------------------------------------------------------------------------
ML_FEATURE_NAMES = (
    "mean_gap",
    "std_dev",
    "skewness",
    "excess_kurtosis",
    "autocorr_lag1",
    "autocorr_lag2",
    "autocorr_lag3",
    "autocorr_lag4",
    "autocorr_lag5",
    "twin_prime_density",
    "gap_entropy",
    "interquartile_range",
    "max_gap_ratio",
    "min_gap",
    "max_gap",
    "gap_range",
    "cv_coeff_variation",
    "prop_even_gt_2logp",
    "median_gap",
    "poisson_lambda_fit",
    "gap_freq_mode",
    "large_gap_fraction",
    "log_mean_gap",
    "CR_log2_legacy_DO_NOT_USE_FOR_CLASSIFICATION",
)
N_ML_FEATURES = len(ML_FEATURE_NAMES)

# Analytical statistics: reported, never fed to the classifier.
ANALYTICAL_STATISTIC_NAMES = (
    "CR",
    "CR_null_pvalue_gamma_k_k",
    "ks_statistic_exponential",
    "ks_pvalue_exponential",
)

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
STAGE2_CLASSES = ("CC", "MA", "MGV")   # label-encoder order of the frozen model
FINAL_CLASSES = ("CC", "MA", "MGV", "NONE")
NONE_LABEL = "NONE"

# Frozen Stage-1 decision threshold.  0.15 is a documented companion analysis
# (THRESHOLD_GATE_SELECTION.json), NOT the baseline.
STAGE1_THRESHOLD_BASELINE = 0.5
STAGE1_THRESHOLD_COMPANION = 0.15

# CR null model: Gamma(shape=k, rate=k) with k = 100.  95 % interval, 4 d.p.
CR_NULL_SHAPE = 100
CR_NULL_INTERVAL = (0.8136, 1.2053)

# --------------------------------------------------------------------------
# Frozen model artifacts (SHA-256 of the files in pgi_crypto/models/).
# --------------------------------------------------------------------------
MODEL_DIR_ENV = "PGI_CRYPTO_MODEL_DIR"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"

MODEL_ARTIFACTS = {
    "stage1": {
        "file": "stage1_xgboost_v1.json",
        "sha256": "1af0ff24f7b059b80d694019f126d70c21c1874c82c16b4b7bc50273ab3980a4",
        "objective": "binary:logistic",
        "n_trees": 400,
        "n_features": N_ML_FEATURES,
        "role": "NONE (0) vs EVENT (1)",
    },
    "stage2": {
        "file": "stage2_xgboost_v1.json",
        "sha256": "9811bcf8b75e3a9a33358091933c02268941193979ed339e3ff5ede4933a47c2",
        "objective": "multi:softprob",
        "n_trees": 1200,  # 400 rounds x 3 classes
        "n_features": N_ML_FEATURES,
        "role": "CC vs MA vs MGV, conditional on EVENT",
    },
}


def resolve_model_dir(model_dir=None) -> Path:
    """Explicit argument > ``PGI_CRYPTO_MODEL_DIR`` > bundled ``models/``."""
    if model_dir is not None:
        return Path(model_dir)
    env = os.environ.get(MODEL_DIR_ENV)
    return Path(env) if env else DEFAULT_MODEL_DIR


# --------------------------------------------------------------------------
# Validated range of the newly trained range-spanning classifier (G4)
# --------------------------------------------------------------------------
VALIDATED_MAX_P_MID = 10 ** 12                   # rebuilt full-range sample support
VALIDATED_RANGE_TEXT = "p_mid <= 10^12 (newly trained range-spanning model support)"

# --------------------------------------------------------------------------
# PGI-Crypto Process 1: CC eligibility (G3)
# --------------------------------------------------------------------------
P_CC_THRESHOLD = 0.80                    # P_CC = P(EVENT) * P(CC | EVENT) >= 0.80
CR_CC_BAND = (0.85, 1.15)                # 0.85 <= CR <= 1.15
KS_ALPHA = 0.05                          # fixed-parameter Exp(1) KS "pass": asymptotic p-value > alpha
#   (reproduces the frozen dataset's labels exactly; the adviser-approved file used D <= 0.136, see
#    PGI_CRYPTO_FINAL_SPECIFICATION.md, decision D-KS)

# --------------------------------------------------------------------------
# Key generation (G1, G2, G6, G8)
# --------------------------------------------------------------------------
DEFAULT_MODULUS_BITS = 2048              # primary target; 4096 is architecturally allowed, not validated
DEFAULT_PUBLIC_EXPONENT = 65537
DEFAULT_WINDOW_SIZE = WINDOW_GAPS        # gaps per window (never confused with modulus_bits)
DEFAULT_MAX_ATTEMPTS = 50                # max candidate primes analysed per generation
DEFAULT_POOL_SIZE = 2                    # eligible primes collected before pair selection (2 = no selection effect)
MIN_MODULUS_BITS = 32                    # library minimum (toy sizes exist only for tests / in-range demos)
SEPARATION_SLACK_BITS = 100              # |p - q| > 2 ** (modulus_bits / 2 - 100)
RNG_MODE_SECURE = "secure"
RNG_MODE_REPRODUCIBLE = "reproducible"   # NON-SECURE, research/test only


# --------------------------------------------------------------------------
# Standing research-limitation texts (shown in results and in the UI)
# --------------------------------------------------------------------------
KEY_SCALE_WARNING = (
    "Research limitation: the newly trained four-class classifier uses a range-spanning sample through p_mid <= 10^12. "
    "RSA-2048 candidates are around 2^1024, far outside that support. Experimental key generation reports this "
    "out-of-domain prediction and does not treat it as validation or evidence of stronger security."
)
ENTROPY_NOTE = ("The entropy value is a statistical candidate-selection criterion (bit-frequency Shannon entropy of N). "
                "It is not cryptographic entropy and is not evidence of security.")
RESEARCH_LIMITATIONS = (
    "The newly trained model's observed support ends at p_mid <= 10^12; RSA-2048 prime candidates are far outside it.",
    "Rare-class scores must be interpreted with their natural prevalence and documented sampling weights.",
    "The range-spanning classifier is research-only and is not evidence of cryptographic security or key quality.",
    ENTROPY_NOTE,
    "No cryptographic security claim is made: this is a research prototype, not a validated cryptosystem.",
)
