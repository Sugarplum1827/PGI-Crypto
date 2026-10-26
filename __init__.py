"""PGI-Crypto research library.

This package retains the archived two-stage implementation for reproducibility
and provides a rebuilt 22-feature, four-class research model through
``PGICryptoResearchSystem``. Strict/Experimental key generation remains
range-aware; outputs do not establish cryptographic safety. See README.md,
``PANEL_PRESENTATION_GUIDE.md``, and the methods audit.
"""
from .classification import FrozenClassifier
from .config import ANALYTICAL_STATISTIC_NAMES, LIBRARY_VERSION as __version__, ML_FEATURE_NAMES
from .exceptions import (BackendUnavailableError, CandidateRejectedError, FeatureExtractionError, FeatureMismatchError,
                         InsufficientPrimesError, InvalidPrimeInputError, KeyGenerationError, KeyValidationError,
                         ModelArtifactError, ModelLoadError, OutOfTrainingRangeWarning, OutsideValidatedRangeError,
                         PGICryptoError, SerializationError, SpecificationNotResolvedError)
from .features import (compute_analytical_statistics, compute_features, extract_analytical_statistics,
                       extract_ml_features)
from .analysis import analyze_window, predict_window
from .eligibility import POLICY_EXPERIMENTAL, POLICY_STRICT, EligibilityPolicy, check_eligibility, compute_p_cc
from .entropy import EntropyResult, bit_frequency_entropy, bit_histogram_entropy, pair_entropy
from .gaps import compute_gaps
from .key_generation import (PGICryptoKeyGenerator, build_rsa_material, draw_random_prime, prime_range, separation_check,
                             specification_status)
from .labels import (LabelRuleResult, MaximalGapCatalogue, apply_label_rules, apply_label_rules_batch, exact_ma_interval)
from .rng import RandomSource
from .serialization import (load_private_key, load_public_key, rsa_operation_roundtrip, serialize_private_key,
                            serialize_public_key)
from .pipeline import PGICrypto
from .rebuilt_classifier import RebuiltClassifier, RebuiltPrediction
from .system import PGICryptoResearchSystem
from .primes import PrimeSequence, is_prime, primes_up_to, validate_prime_sequence
from .result_types import (AnalysisResult, AnalyticalStatistics, ClassificationResult, EligibilityResult, FeatureMatrix,
                    KeyGenerationResult, KeyResult, KeyValidationResult, ModelInfo, PrimeCandidate, ProgressEvent,
                    RSAPrivateMaterial, SecretValue, SeparationResult, SourceWindow, ValidationCheck, ValidationReport,
                    ValidationStatus, WindowAnalysis, WindowClassification)
from .validation import (validate_analysis, validate_classification, validate_features, validate_generated_key,
                         validate_generation, validate_key, validate_modulus_structure, validate_output)
from .windows import (FROZEN_WINDOW_SPEC, CenteredWindow, WindowSet, WindowSpec, build_centered_window, build_windows,
                      centered_window_from_primes)

__all__ = [n for n in dir() if not n.startswith("_")] + ["__version__"]
