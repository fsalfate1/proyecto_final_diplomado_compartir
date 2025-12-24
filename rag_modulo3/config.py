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

PDF_DIR: Final[Path] = Path("pdf")
CACHE_FILE: Final[Path] = Path(".rag_cache.json")
COLLECTION_NAME: Final[str] = "rag_mod3_pdf_exportaciones"
TOP_K: Final[int] = 5

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
