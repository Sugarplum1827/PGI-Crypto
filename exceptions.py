"""Exception hierarchy.  Every library error derives from PGICryptoError."""
from __future__ import annotations


class PGICryptoError(Exception):
    """Base class for all errors raised by pgi_crypto."""


class OutOfTrainingRangeWarning(UserWarning):
    """Some windows have p_mid above the 1e8 range the frozen classifier was trained on.

    The classifier still returns numbers for such windows, but nothing in the
    project validates them (see PGI_CRYPTO_SPECIFICATION_GAPS.md, G4).
    """


class InvalidPrimeInputError(PGICryptoError, ValueError):
    """The prime sequence supplied by the caller is not acceptable.

    ``reason`` is a stable machine-readable code and ``index`` is the position
    of the first offending element (or None for whole-input problems).
    The library never repairs bad input (no silent sort / de-duplication).
    """

    def __init__(self, reason: str, message: str, index=None):
        super().__init__(message)
        self.reason = reason
        self.index = index


class InsufficientPrimesError(InvalidPrimeInputError):
    """Fewer primes than one window requires."""


class FeatureExtractionError(PGICryptoError, ValueError):
    """Feature vector is malformed or contains non-finite values."""


class ModelArtifactError(PGICryptoError):
    """A frozen model artifact is missing, altered, or inconsistent."""


class BackendUnavailableError(PGICryptoError):
    """The requested model back-end cannot be used in this environment."""


class SpecificationNotResolvedError(PGICryptoError):
    """The requested operation depends on an unresolved thesis specification.

    Raised by key generation: the PGI-Crypto key-generation procedure is not
    fully specified by the project artifacts.  See
    PGI_CRYPTO_SPECIFICATION_GAPS.md.  ``gap_ids`` lists the open items.
    """

    def __init__(self, message: str, gap_ids=()):
        super().__init__(message)
        self.gap_ids = tuple(gap_ids)


class ModelLoadError(ModelArtifactError):
    """A model file exists but cannot be parsed / is structurally invalid."""


class FeatureMismatchError(FeatureExtractionError):
    """Feature names, order, count or dtype differ from the frozen schema."""


class OutsideValidatedRangeError(PGICryptoError):
    """The operation requires the frozen classifier's validated range (p_mid <= 10^8) and the input is outside it."""


class CandidateRejectedError(PGICryptoError):
    """A prime candidate was rejected by the PGI-Crypto eligibility rules (``.eligibility`` has the details)."""

    def __init__(self, message, eligibility=None):
        super().__init__(message)
        self.eligibility = eligibility


class KeyGenerationError(PGICryptoError):
    """Key generation failed (e.g. maximum attempts reached).  ``.result`` holds the partial, redacted record."""

    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result


class KeyValidationError(PGICryptoError):
    """A generated key failed validation (``.validation`` has the report)."""

    def __init__(self, message, validation=None):
        super().__init__(message)
        self.validation = validation


class SerializationError(PGICryptoError):
    """Key serialization / deserialization failed or was refused by a safety rule."""
