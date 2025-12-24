"""Paquete base del proyecto RAG del módulo 3."""

from .config import (
    CACHE_FILE,
    COLLECTION_NAME,
    PDF_DIR,
    TOP_K,
    ensure_env_variables,
    get_qdrant_client,
    get_qdrant_credentials,
    load_environment,
)
from .rag_chain import (
    answer_question,
    build_rag_chain,
    build_rag_components,
    build_retriever,
)
from .preparation import prepare_corpus

__all__ = [
    "CACHE_FILE",
    "COLLECTION_NAME",
    "PDF_DIR",
    "TOP_K",
    "load_environment",
    "ensure_env_variables",
    "get_qdrant_credentials",
    "get_qdrant_client",
    "build_rag_chain",
    "build_rag_components",
    "build_retriever",
    "answer_question",
    "prepare_corpus",
]
