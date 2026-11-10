"""Research-only envelope encryption for API records using a PGI-Crypto RSA key pair.

PGI-Crypto generates the RSA pair. The cryptography package supplies RSA-OAEP and
AES-GCM primitives. This feature is not a claim that Experimental PGI-Crypto keys
are suitable for production security.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .exceptions import KeyGenerationError

ENVELOPE_FIELD = "_pgi_record_envelope"
ENVELOPE_VERSION = 1
PRIVATE_KEY_FILENAME = "record_private_key.pem"


class RecordEncryptionError(RuntimeError):
    """Raised when a database record cannot be encrypted or decrypted."""


class RecordCryptor:
    def __init__(self, private_key: rsa.RSAPrivateKey):
        if private_key.key_size != 2048:
            raise RecordEncryptionError("The record-encryption key must be 2048-bit RSA.")
        self._private_key = private_key
        self._public_key = private_key.public_key()
        public_der = self._public_key.public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self.key_id = hashlib.sha256(public_der).hexdigest()[:24]

    @staticmethod
    def is_envelope(payload: Any) -> bool:
        return isinstance(payload, dict) and ENVELOPE_FIELD in payload

    @staticmethod
    def _aad(record_id: str, kind: str) -> bytes:
        return f"pgi-api-record|{ENVELOPE_VERSION}|{record_id}|{kind}".encode("utf-8")

    def encrypt(self, record_id: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            plaintext = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
            data_key = os.urandom(32)
            nonce = os.urandom(12)
            ciphertext = AESGCM(data_key).encrypt(nonce, plaintext, self._aad(record_id, kind))
            wrapped_key = self._public_key.encrypt(
                data_key,
                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                             algorithm=hashes.SHA256(), label=None),
            )
            return {
                ENVELOPE_FIELD: {
                    "version": ENVELOPE_VERSION,
                    "algorithm": "AES-256-GCM+RSA-OAEP-SHA256",
                    "key_id": self.key_id,
                    "nonce": base64.b64encode(nonce).decode("ascii"),
                    "wrapped_data_key": base64.b64encode(wrapped_key).decode("ascii"),
                    "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
                }
            }
        except Exception as exc:
            raise RecordEncryptionError(f"Could not encrypt API record ({type(exc).__name__}).") from exc

    def decrypt(self, record_id: str, kind: str, envelope: dict[str, Any]) -> dict[str, Any]:
        try:
            value = envelope[ENVELOPE_FIELD]
            if (value["version"] != ENVELOPE_VERSION
                    or value["algorithm"] != "AES-256-GCM+RSA-OAEP-SHA256"
                    or value["key_id"] != self.key_id):
                raise RecordEncryptionError("Record encryption version or key identifier does not match this host.")
            nonce = base64.b64decode(value["nonce"], validate=True)
            wrapped_key = base64.b64decode(value["wrapped_data_key"], validate=True)
            ciphertext = base64.b64decode(value["ciphertext"], validate=True)
            data_key = self._private_key.decrypt(
                wrapped_key,
                padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                             algorithm=hashes.SHA256(), label=None),
            )
            plaintext = AESGCM(data_key).decrypt(nonce, ciphertext, self._aad(record_id, kind))
            result = json.loads(plaintext.decode("utf-8"))
            if not isinstance(result, dict):
                raise RecordEncryptionError("Decrypted record payload is not a JSON object.")
            return result
        except RecordEncryptionError:
            raise
        except (InvalidTag, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RecordEncryptionError("Record could not be decrypted; key, metadata, or ciphertext may be invalid.") from exc


_active_cryptor: RecordCryptor | None = None


def active_record_cryptor() -> RecordCryptor | None:
    return _active_cryptor


def configure_record_crypto(data_dir: Path, passphrase: str) -> RecordCryptor:
    """Load or create the host's encrypted PGI-Crypto key and migrate old rows."""
    global _active_cryptor
    if len(passphrase) < 16:
        raise RecordEncryptionError("Use a passphrase of at least 16 characters.")
    data_dir.mkdir(parents=True, exist_ok=True)
    private_path = data_dir / PRIVATE_KEY_FILENAME

    from . import PGICryptoKeyGenerator, POLICY_EXPERIMENTAL
    from .api_store import all_records, initialize_store, replace_payload
    from .config import RNG_MODE_SECURE
    from .serialization import serialize_private_key

    initialize_store()
    if private_path.exists():
        try:
            loaded = serialization.load_pem_private_key(
                private_path.read_bytes(), password=passphrase.encode("utf-8")
            )
        except Exception as exc:
            raise RecordEncryptionError("Could not unlock the host PGI-Crypto private key; check the passphrase.") from exc
        if not isinstance(loaded, rsa.RSAPrivateKey):
            raise RecordEncryptionError("The saved record key is not an RSA private key.")
        cryptor = RecordCryptor(loaded)
    else:
        rows = all_records()
        if any(RecordCryptor.is_envelope(row["payload"]) for row in rows):
            raise RecordEncryptionError(
                "Encrypted records exist but the host private-key file is missing. Restore the original key file and passphrase."
            )
        generated = None
        try:
            generated = PGICryptoKeyGenerator().generate(
                modulus_bits=2048,
                policy=POLICY_EXPERIMENTAL,
                rng_mode=RNG_MODE_SECURE,
                max_attempts=8,
                validate=True,
            )
            if not generated.success:
                raise RecordEncryptionError("PGI-Crypto did not validate a key for the record-encryption prototype.")
            pem = serialize_private_key(generated, password=passphrase.encode("utf-8"))
            temporary_path = private_path.with_suffix(".pem.tmp")
            temporary_path.write_bytes(pem)
            temporary_path.replace(private_path)
            loaded = serialization.load_pem_private_key(pem, password=passphrase.encode("utf-8"))
            if not isinstance(loaded, rsa.RSAPrivateKey):
                raise RecordEncryptionError("Generated PGI-Crypto material could not be loaded as an RSA key.")
            cryptor = RecordCryptor(loaded)
        except RecordEncryptionError:
            raise
        except KeyGenerationError as exc:
            raise RecordEncryptionError(f"PGI-Crypto key generation failed ({type(exc).__name__}).") from exc
        except Exception as exc:
            raise RecordEncryptionError(f"Could not initialize PGI-Crypto record encryption ({type(exc).__name__}).") from exc
        finally:
            del generated

    _active_cryptor = cryptor
    for row in all_records():
        if RecordCryptor.is_envelope(row["payload"]):
            envelope_data = row["payload"].get(ENVELOPE_FIELD, {})
            if envelope_data.get("key_id") != cryptor.key_id:
                raise RecordEncryptionError(
                    "An encrypted database record belongs to a different key pair. Restore the matching key file."
                )
        else:
            replace_payload(row["id"], cryptor.encrypt(row["id"], row["kind"], row["payload"]))
    return cryptor
