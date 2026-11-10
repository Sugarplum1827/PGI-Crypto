"""SQLAlchemy storage for encrypted PGI-Crypto API records."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sqlalchemy import Column, DateTime, MetaData, String, Table, create_engine, func, insert, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.types import JSON


def _default_database_url() -> str:
    configured_data_dir = os.getenv("PGICRYPTO_DATA_DIR")
    if configured_data_dir:
        data_dir = Path(configured_data_dir).expanduser().resolve()
    else:
        project_root = Path(__file__).resolve().parents[3]
        data_dir = project_root / "data" / "api"
    data_dir.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + str(data_dir / "pgi_api.sqlite3").replace("\\", "/")


DATABASE_URL = os.getenv("PGICRYPTO_DATABASE_URL", _default_database_url())
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite:") else {}
engine: Engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args)
metadata = MetaData()
api_records = Table(
    "pgi_api_records",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("kind", String(40), nullable=False, index=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("payload", JSON, nullable=False),
)


def initialize_store() -> None:
    metadata.create_all(engine)


def database_is_ready() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(select(func.count()).select_from(api_records))
        return True
    except Exception:
        return False


def save_record(record_id: str, kind: str, created_at, payload: dict[str, Any]) -> None:
    with engine.begin() as connection:
        connection.execute(insert(api_records).values(
            id=record_id, kind=kind, created_at=created_at, payload=payload
        ))


def get_record(record_id: str) -> dict[str, Any] | None:
    with engine.connect() as connection:
        row = connection.execute(
            select(api_records).where(api_records.c.id == record_id)
        ).mappings().first()
    return dict(row) if row is not None else None


def latest_records(limit: int = 50) -> list[dict[str, Any]]:
    with engine.connect() as connection:
        rows = connection.execute(
            select(api_records).order_by(api_records.c.created_at.desc()).limit(limit)
        ).mappings().all()
    return [dict(row) for row in rows]


def all_records() -> list[dict[str, Any]]:
    with engine.connect() as connection:
        rows = connection.execute(
            select(api_records).order_by(api_records.c.created_at.asc())
        ).mappings().all()
    return [dict(row) for row in rows]


def replace_payload(record_id: str, payload: dict[str, Any]) -> None:
    with engine.begin() as connection:
        connection.execute(
            update(api_records).where(api_records.c.id == record_id).values(payload=payload)
        )
