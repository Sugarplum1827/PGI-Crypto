"""REST API for the PGI-Crypto research library.

Run from the project root with:
    python -m uvicorn pgi_crypto.api:app --app-dir src/python --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import hmac
import json
import os
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from . import PGICryptoKeyGenerator, PGICryptoResearchSystem, POLICY_EXPERIMENTAL
from .api_store import (database_is_ready, get_record, initialize_store, latest_records,
                        save_record)
from .config import LIBRARY_VERSION
from .exceptions import KeyGenerationError
from .record_crypto import (RecordCryptor, RecordEncryptionError, active_record_cryptor,
                            configure_record_crypto)

API_KEY = os.getenv("PGICRYPTO_API_KEY", "")
app = FastAPI(
    title="PGI-Crypto Research API",
    version=str(LIBRARY_VERSION),
    description=(
        "Research API for classification, global SHAP information, and redacted Experimental "
        "key-generation outcomes. Database payloads use a research-only PGI-Crypto-generated "
        "RSA key to wrap per-record AES keys; this does not certify key security."
    ),
)


class ClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gaps: list[StrictInt] = Field(min_length=100, max_length=100)
    p_mid: Annotated[StrictInt, Field(gt=0)]


class KeyGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modulus_bits: Annotated[StrictInt, Field(ge=2048, le=2048)] = 2048
    max_attempts: Annotated[StrictInt, Field(ge=1, le=8)] = 8


def require_api_key(x_pgi_api_key: Annotated[str | None, Header()] = None) -> None:
    if API_KEY and not hmac.compare_digest(x_pgi_api_key or "", API_KEY):
        raise HTTPException(status_code=401, detail="A valid X-PGI-API-Key header is required")


@lru_cache(maxsize=1)
def research_system() -> PGICryptoResearchSystem:
    return PGICryptoResearchSystem()


@app.on_event("startup")
def startup() -> None:
    initialize_store()
    if active_record_cryptor() is None:
        passphrase = os.getenv("PGICRYPTO_RECORD_KEY_PASSWORD")
        if not passphrase:
            raise RuntimeError(
                "Record encryption is required. Start with PGICryptoHost.exe or set "
                "PGICRYPTO_RECORD_KEY_PASSWORD for a development launch."
            )
        data_dir = Path(os.getenv(
            "PGICRYPTO_DATA_DIR",
            str(Path(__file__).resolve().parents[3] / "data" / "api"),
        ))
        configure_record_crypto(data_dir, passphrase)


def _save_encrypted_record(record_id: str, kind: str, created_at: datetime,
                           payload: dict) -> None:
    cryptor = active_record_cryptor()
    if cryptor is None:
        raise RecordEncryptionError("Record encryption has not been initialized.")
    save_record(record_id, kind, created_at, cryptor.encrypt(record_id, kind, payload))


def _decrypt_record(row: dict) -> dict:
    cryptor = active_record_cryptor()
    if cryptor is None:
        raise RecordEncryptionError("Record encryption has not been initialized.")
    row = dict(row)
    row["payload"] = cryptor.decrypt(row["id"], row["kind"], row["payload"])
    return row


@app.get("/health")
def health(_: None = Depends(require_api_key)) -> dict:
    ready = database_is_ready()
    return {"status": "ok" if ready else "database_unavailable",
            "database_connected": ready, "library_version": str(LIBRARY_VERSION),
            "record_encryption": "enabled" if active_record_cryptor() else "unavailable",
            "record_encryption_key_id": active_record_cryptor().key_id if active_record_cryptor() else None}


@app.post("/v1/classifications", status_code=201)
def classify(request: ClassificationRequest, _: None = Depends(require_api_key)) -> dict:
    try:
        result = research_system().classify_window(request.gaps, request.p_mid)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    record_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc)
    payload = {"operation": "prime_gap_classification", "result": result}
    try:
        _save_encrypted_record(record_id, "classification", created_at, payload)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Could not save the classification result") from exc
    return {"id": record_id, "created_at": created_at.isoformat(), **payload}


@app.get("/v1/interpretability/global")
def global_interpretability(_: None = Depends(require_api_key)) -> dict:
    try:
        resource = files("pgi_crypto").joinpath("panel_data", "shap_global.json")
        return json.loads(resource.read_text(encoding="utf-8"))
    except (FileNotFoundError, ModuleNotFoundError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="Global SHAP summary is unavailable") from exc


@app.post("/v1/key-generations", status_code=201)
def generate_key(request: KeyGenerationRequest, _: None = Depends(require_api_key)) -> dict:
    generator = PGICryptoKeyGenerator()
    result = None
    status = "success"
    failure_type = None
    try:
        result = generator.generate(
            modulus_bits=request.modulus_bits,
            policy=POLICY_EXPERIMENTAL,
            max_attempts=request.max_attempts,
            validate=True,
        )
        if not result.success:
            status = "validation_failed"
    except KeyGenerationError as exc:
        status = "failed"
        failure_type = type(exc).__name__
        result = getattr(exc, "result", None)
    except Exception as exc:
        status = "error"
        failure_type = type(exc).__name__

    summary = result.redacted_info() if result is not None else None
    payload = {
        "operation": "experimental_rsa_key_generation",
        "status": status,
        "failure_type": failure_type,
        "summary": summary,
        "generated_key_material_saved": False,
        "interpretation": (
            "Research-only Experimental result. Out-of-domain classifier outputs are not enforced "
            "and this endpoint does not approve a key for production use."
        ),
    }
    record_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc)
    try:
        _save_encrypted_record(record_id, "key_generation", created_at, payload)
    except Exception as exc:
        # The private material remains in process memory only; never serialize it to recover the record.
        del result
        raise HTTPException(status_code=503, detail="Generation completed but its redacted record could not be saved") from exc
    del result
    return {"id": record_id, "created_at": created_at.isoformat(), **payload}


@app.get("/v1/records/{record_id}")
def read_record(record_id: str, _: None = Depends(require_api_key)) -> dict:
    row = get_record(record_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Record not found")
    try:
        row = _decrypt_record(row)
    except RecordEncryptionError as exc:
        raise HTTPException(status_code=500, detail="Saved record could not be decrypted") from exc
    return {"id": row["id"], "kind": row["kind"],
            "created_at": row["created_at"].isoformat(), "payload": row["payload"]}


@app.get("/v1/records")
def list_records(limit: Annotated[int, Query(ge=1, le=100)] = 50,
                 _: None = Depends(require_api_key)) -> list[dict]:
    try:
        rows = [_decrypt_record(row) for row in latest_records(limit)]
    except RecordEncryptionError as exc:
        raise HTTPException(status_code=500, detail="Saved records could not be decrypted") from exc
    return [{"id": row["id"], "kind": row["kind"],
             "created_at": row["created_at"].isoformat(), "payload": row["payload"]}
            for row in rows]
