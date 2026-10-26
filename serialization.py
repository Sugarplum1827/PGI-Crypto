"""RSA key serialization and standard-library RSA operations (via the ``cryptography`` package).

No cryptographic primitive is implemented here: key objects, PEM encoding, OAEP/PSS/PKCS#1
operations all come from ``cryptography`` (OpenSSL).  The only safety rules added are:

* a private key is never written unencrypted unless the caller says so explicitly
  (``password=...`` or ``unencrypted=True``);
* a private key from the NON-SECURE reproducible mode is refused unless ``allow_nonsecure=True``;
* nothing in this module prints or logs key material.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from .exceptions import SerializationError
from .result_types import KeyGenerationResult, RSAPrivateMaterial

PUBLIC_FORMATS = {"SubjectPublicKeyInfo": serialization.PublicFormat.SubjectPublicKeyInfo,
                  "PKCS1": serialization.PublicFormat.PKCS1}
PRIVATE_FORMATS = {"PKCS8": serialization.PrivateFormat.PKCS8,
                   "TraditionalOpenSSL": serialization.PrivateFormat.TraditionalOpenSSL}


def _material(source: Union[KeyGenerationResult, RSAPrivateMaterial]) -> Tuple[RSAPrivateMaterial, bool]:
    """(material, is_secure_mode)."""
    if isinstance(source, RSAPrivateMaterial):
        return source, True
    if isinstance(source, KeyGenerationResult):
        if source.private_material is None:
            raise SerializationError("this result carries no private key (generation failed)")
        return source.private_material.reveal(), source.is_secure_mode
    raise SerializationError("expected KeyGenerationResult or RSAPrivateMaterial, got %s" % type(source).__name__)


def rsa_private_key(material: RSAPrivateMaterial) -> rsa.RSAPrivateKey:
    """``cryptography`` private key object (the library validates the components itself)."""
    try:
        nums = rsa.RSAPrivateNumbers(material.p, material.q, material.d, material.dp, material.dq, material.qinv,
                                     rsa.RSAPublicNumbers(material.e, material.n))
        return nums.private_key()
    except Exception as exc:  # the message never contains key material
        raise SerializationError("cryptography rejected the key components: %s" % type(exc).__name__) from exc


def serialize_public_key(source: Union[KeyGenerationResult, RSAPrivateMaterial, Tuple[int, int]],
                         format: str = "SubjectPublicKeyInfo") -> bytes:
    """PEM public key.  ``source`` may also be an ``(n, e)`` tuple."""
    if format not in PUBLIC_FORMATS:
        raise SerializationError("format must be one of %s" % sorted(PUBLIC_FORMATS))
    try:
        if isinstance(source, tuple):
            n, e = source
        elif isinstance(source, KeyGenerationResult):
            n, e = source.modulus, source.public_exponent
        else:
            n, e = source.n, source.e
        key = rsa.RSAPublicNumbers(e, n).public_key()
        return key.public_bytes(serialization.Encoding.PEM, PUBLIC_FORMATS[format])
    except SerializationError:
        raise
    except Exception as exc:
        raise SerializationError("public key serialization failed: %s" % type(exc).__name__) from exc


def serialize_private_key(source: Union[KeyGenerationResult, RSAPrivateMaterial], *, password: Optional[bytes] = None,
                          unencrypted: bool = False, allow_nonsecure: bool = False, format: str = "PKCS8") -> bytes:
    """PEM private key (PKCS#8 by default), encrypted with ``password`` (``cryptography`` BestAvailableEncryption)."""
    if format not in PRIVATE_FORMATS:
        raise SerializationError("format must be one of %s" % sorted(PRIVATE_FORMATS))
    material, secure = _material(source)
    if not secure and not allow_nonsecure:
        raise SerializationError("refusing to serialize a private key generated in the NON-SECURE reproducible mode "
                                 "(pass allow_nonsecure=True for research/test use only)")
    if password is None and not unencrypted:
        raise SerializationError("refusing to write an unencrypted private key: pass password=<bytes> or unencrypted=True explicitly")
    if password is not None and unencrypted:
        raise SerializationError("password and unencrypted=True are mutually exclusive")
    enc = serialization.NoEncryption() if password is None else serialization.BestAvailableEncryption(password)
    try:
        return rsa_private_key(material).private_bytes(serialization.Encoding.PEM, PRIVATE_FORMATS[format], enc)
    except SerializationError:
        raise
    except Exception as exc:
        raise SerializationError("private key serialization failed: %s" % type(exc).__name__) from exc


def load_public_key(pem: bytes) -> Tuple[int, int]:
    """``(n, e)`` from a PEM public key."""
    try:
        key = serialization.load_pem_public_key(pem)
        if not isinstance(key, rsa.RSAPublicKey):
            raise SerializationError("not an RSA public key")
        nums = key.public_numbers()
        return nums.n, nums.e
    except SerializationError:
        raise
    except Exception as exc:
        raise SerializationError("could not load public key: %s" % type(exc).__name__) from exc


def load_private_key(pem: bytes, password: Optional[bytes] = None) -> RSAPrivateMaterial:
    """Private numbers from a PEM private key."""
    try:
        key = serialization.load_pem_private_key(pem, password=password)
        if not isinstance(key, rsa.RSAPrivateKey):
            raise SerializationError("not an RSA private key")
        pn = key.private_numbers()
        return RSAPrivateMaterial(p=pn.p, q=pn.q, d=pn.d, dp=pn.dmp1, dq=pn.dmq1, qinv=pn.iqmp,
                                  n=pn.public_numbers.n, e=pn.public_numbers.e)
    except SerializationError:
        raise
    except Exception as exc:
        raise SerializationError("could not load private key: %s" % type(exc).__name__) from exc


def rsa_operation_roundtrip(material: RSAPrivateMaterial) -> Dict[str, Any]:
    """Encrypt/decrypt and sign/verify with ``cryptography`` primitives.

    Scheme by size: >= 1024-bit modulus: OAEP(SHA-256) + PSS(SHA-256).  512-1023: PKCS#1 v1.5
    (toy sizes only; too short for OAEP/PSS-SHA-256).  Smaller moduli cannot carry these paddings:
    the corresponding result is ``None`` (not applicable), never ``True``.
    """
    bits = material.n.bit_length()
    key = rsa_private_key(material)
    pub = key.public_key()
    msg = os.urandom(16)
    out: Dict[str, Any] = {"modulus_bits": bits, "encryption_ok": None, "signature_ok": None, "scheme": None, "skipped_reason": None}
    if bits >= 1024:
        enc_pad = padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None)
        sig_pad = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH)
        out["scheme"] = "OAEP-SHA256 / PSS-SHA256"
    elif bits >= 512:
        enc_pad = padding.PKCS1v15()
        sig_pad = padding.PKCS1v15()
        out["scheme"] = "PKCS1v15 / PKCS1v15-SHA256 (toy size)"
    elif bits >= 128:
        enc_pad, sig_pad = padding.PKCS1v15(), None
        msg = msg[:4]
        out["scheme"] = "PKCS1v15 encryption only (modulus too small to sign with SHA-256)"
    else:
        out["skipped_reason"] = "modulus below 128 bits: no standard padding fits"
        return out
    out["encryption_ok"] = key.decrypt(pub.encrypt(msg, enc_pad), enc_pad) == msg
    if sig_pad is not None:
        try:
            pub.verify(key.sign(msg, sig_pad, hashes.SHA256()), msg, sig_pad, hashes.SHA256())
            out["signature_ok"] = True
        except InvalidSignature:
            out["signature_ok"] = False
    return out
