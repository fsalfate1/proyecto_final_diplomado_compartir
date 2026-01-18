"""Configuración global del proyecto RAG.

Define rutas, nombres de colección y helpers para cargar credenciales.
Los módulos `rag_chain.py` y `preparation.py` dependen de estas utilidades.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from qdrant_client import QdrantClient

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_default_pdf_dir = "PDF" if (PROJECT_ROOT / "PDF").exists() else "pdf"
_env_pdf_dir = os.getenv("RAG_PDF_DIR")
if _env_pdf_dir:
    _pdf_dir = Path(_env_pdf_dir)
    if not _pdf_dir.is_absolute():
        _pdf_dir = PROJECT_ROOT / _pdf_dir
else:
    _pdf_dir = PROJECT_ROOT / _default_pdf_dir

CACHE_FILE: Final[Path] = PROJECT_ROOT / ".rag_cache.json"

_default_csv_path = PROJECT_ROOT / "excel" / "DrugData.csv"
_env_csv_path = os.getenv("RAG_DRUG_CSV")
if _env_csv_path:
    _csv_path = Path(_env_csv_path)
    if not _csv_path.is_absolute():
        _csv_path = PROJECT_ROOT / _csv_path
else:
    _csv_path = _default_csv_path

PDF_DIR: Final[Path] = _pdf_dir
DRUG_CSV_PATH: Final[Path] = _csv_path
COLLECTION_NAME: Final[str] = os.getenv("RAG_COLLECTION_NAME", "csv_vademecum")
TOP_K: Final[int] = 3
SCORE_THRESHOLD: Final[float] = 0.75

REQUIRED_ENV_KEYS = ["OPENAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"]


def load_environment() -> None:
    """Carga variables desde .env y valida credenciales obligatorias."""
    load_dotenv()
    ensure_env_variables()


def ensure_env_variables() -> None:
    missing = [k for k in REQUIRED_ENV_KEYS if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Configura estas variables en .env: {missing}")


def get_qdrant_client() -> QdrantClient:
    """Devuelve un cliente Qdrant ya configurado."""
    ensure_env_variables()
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )


def get_qdrant_credentials() -> dict[str, str]:
    ensure_env_variables()
    return {
        "url": os.getenv("QDRANT_URL"),
        "api_key": os.getenv("QDRANT_API_KEY"),
    }
