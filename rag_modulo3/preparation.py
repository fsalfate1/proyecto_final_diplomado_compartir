"""Prepara la colección en Qdrant.

Se apoya en `config.py` para rutas/credenciales y es invocado por
`rag_data_preparation.py` antes de usar el motor de consultas.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field

from .config import (
    CACHE_FILE,
    COLLECTION_NAME,
    PDF_DIR,
    get_qdrant_credentials,
    load_environment,
)


class DocumentMetadata(BaseModel):
    titulo: str = Field(description="Título corto del documento")
    resumen: str = Field(description="Resumen en 2 frases máximo")
    categoria: str = Field(description="Tema del documento")


def ensure_pdf_dir_exists() -> None:
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"No se encontró la carpeta {PDF_DIR.resolve()}")


def load_pdf_documents() -> List[Document]:
    ensure_pdf_dir_exists()
    aggregated: Dict[str, List[str]] = defaultdict(list)
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        for doc in loader.load():
            aggregated[str(pdf_path)].append(doc.page_content)

    documents: List[Document] = []
    for source, contents in aggregated.items():
        full_text = "\n\n".join(contents)
        documents.append(Document(page_content=full_text, metadata={"source": source}))

    if not documents:
        raise RuntimeError(f"No se encontraron PDFs dentro de {PDF_DIR.resolve()}")

    return documents


def get_pdf_state() -> Dict[str, float]:
    ensure_pdf_dir_exists()
    state = {}
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        state[str(pdf_path)] = pdf_path.stat().st_mtime
    return state


def load_cached_state() -> Dict[str, object]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_cached_state(pdf_state: Dict[str, float]) -> None:
    CACHE_FILE.write_text(json.dumps({"pdf_state": pdf_state}, indent=2))


def extract_metadata(documents: List[Document], llm: ChatOpenAI) -> Dict[str, DocumentMetadata]:
    prompt = ChatPromptTemplate.from_template(
        (
            "Eres un asistente que resume documentos en español.\n"
            "Lee el texto y entrega un título breve, un resumen en dos frases y una categoría.\n"
            "Contenido:\n{content}"
        )
    )
    structured_llm = llm.with_structured_output(DocumentMetadata)
    chain = prompt | structured_llm

    metadata_by_source: Dict[str, DocumentMetadata] = {}
    for doc in documents:
        snippet = doc.page_content[:2500]
        metadata_by_source[doc.metadata["source"]] = chain.invoke({"content": snippet})

    return metadata_by_source


def chunk_documents(
    documents: List[Document],
    metadata_map: Dict[str, DocumentMetadata],
    chunker: SemanticChunker,
) -> List[Document]:
    chunked_docs: List[Document] = []
    for doc in documents:
        enriched_metadata = doc.metadata.copy()
        metadata = metadata_map.get(doc.metadata["source"])
        if metadata:
            enriched_metadata.update(metadata.model_dump())

        chunks = chunker.create_documents(
            texts=[doc.page_content],
            metadatas=[enriched_metadata],
        )
        chunked_docs.extend(chunks)

    return chunked_docs


def build_vector_store(documents: List[Document]) -> QdrantVectorStore:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        **get_qdrant_credentials(),
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )


def prepare_corpus() -> None:
    load_environment()

    pdf_state = get_pdf_state()
    cached_state = load_cached_state()
    needs_refresh = cached_state.get("pdf_state") != pdf_state

    if not needs_refresh:
        print("♻️ No hay cambios en los PDFs; colección existente reutilizada.")
        return

    print("📂 Cargando PDFs…")
    source_documents = load_pdf_documents()

    print("🧾 Extrayendo metadata…")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    metadata_map = extract_metadata(source_documents, llm)

    print("✂️ Realizando semantic chunking…")
    chunk_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    chunker = SemanticChunker(
        chunk_embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=50,
    )
    chunked_docs = chunk_documents(source_documents, metadata_map, chunker)

    print(f"🗄️ Generando vector store en Qdrant ({COLLECTION_NAME})…")
    build_vector_store(chunked_docs)
    save_cached_state(pdf_state)

    print("✅ Preparación de datos finalizada.")


__all__ = [
    "prepare_corpus",
    "load_pdf_documents",
    "extract_metadata",
    "chunk_documents",
]
