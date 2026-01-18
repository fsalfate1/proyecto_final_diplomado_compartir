"""Paquete base del proyecto RAG del módulo 3."""

from .config import (
    CACHE_FILE,
    COLLECTION_NAME,
    DRUG_CSV_PATH,
    PDF_DIR,
    TOP_K,
    SCORE_THRESHOLD,
    ensure_env_variables,
    get_qdrant_client,
    get_qdrant_credentials,
    load_environment,
)
from .rag_chain import answer_question, build_rag_chain, build_rag_components
from .preparation import prepare_corpus

__all__ = [
    "CACHE_FILE",
    "COLLECTION_NAME",
    "DRUG_CSV_PATH",
    "PDF_DIR",
    "TOP_K",
    "SCORE_THRESHOLD",
    "load_environment",
    "ensure_env_variables",
    "get_qdrant_credentials",
    "get_qdrant_client",
    "build_rag_chain",
    "build_rag_components",
    "answer_question",
    "prepare_corpus",
]
