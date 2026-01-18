"""Prepara la colección en Qdrant.

Se apoya en `config.py` para rutas/credenciales y es invocado por
`scripts/rag_data_preparation.py` antes de usar el motor de consultas.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field

from .config import (
    CACHE_FILE,
    COLLECTION_NAME,
    DRUG_CSV_PATH,
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


def ensure_drug_csv_exists() -> None:
    if not DRUG_CSV_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo {DRUG_CSV_PATH.resolve()}")


def load_pdf_documents() -> List[Document]:
    ensure_pdf_dir_exists()
    aggregated: Dict[str, List[str]] = defaultdict(list)
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        raise RuntimeError(
            "La carga de PDFs ya no esta habilitada. Usa el CSV DrugData.csv."
        )

    documents: List[Document] = []
    for source, contents in aggregated.items():
        full_text = "\n\n".join(contents)
        documents.append(Document(page_content=full_text, metadata={"source": source}))

    if not documents:
        raise RuntimeError(f"No se encontraron PDFs dentro de {PDF_DIR.resolve()}")

    return documents


def load_drug_documents() -> List[Document]:
    ensure_drug_csv_exists()
    documents: List[Document] = []
    with DRUG_CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            drug_id = (row.get("Drug ID") or "").strip()
            nombre = (row.get("Drug Name") or "").strip()
            generico = (row.get("Generic Name") or "").strip()
            drug_class = (row.get("Drug Class") or "").strip()
            indicaciones = (row.get("Indications") or "").strip()
            presentacion = (row.get("Dosage Form") or "").strip()
            fuerza = (row.get("Strength") or "").strip()
            via = (row.get("Route of Administration") or "").strip()
            mecanismo = (row.get("Mechanism of Action") or "").strip()
            efectos = (row.get("Side Effects") or "").strip()
            contra = (row.get("Contraindications") or "").strip()
            interacciones = (row.get("Interactions") or "").strip()
            advertencias = (row.get("Warnings and Precautions") or "").strip()
            embarazo = (row.get("Pregnancy Category") or "").strip()
            almacenamiento = (row.get("Storage Conditions") or "").strip()
            fabricante = (row.get("Manufacturer") or "").strip()
            aprobacion = (row.get("Approval Date") or "").strip()
            disponibilidad = (row.get("Availability") or "").strip()
            ndc = (row.get("NDC") or "").strip()
            precio = (row.get("Price") or "").strip()

            contenido = "\n".join(
                [
                    f"Drug ID: {drug_id}",
                    f"Medicamento: {nombre}",
                    f"Principio activo: {generico}",
                    f"Grupo terapeutico: {drug_class}",
                    f"Indicaciones: {indicaciones}",
                    f"Presentacion: {presentacion}",
                    f"Concentracion: {fuerza}",
                    f"Via de administracion: {via}",
                    f"Mecanismo de accion: {mecanismo}",
                    f"Efectos secundarios: {efectos}",
                    f"Contraindicaciones: {contra}",
                    f"Interacciones: {interacciones}",
                    f"Advertencias y precauciones: {advertencias}",
                    f"Categoria de embarazo: {embarazo}",
                    f"Condiciones de almacenamiento: {almacenamiento}",
                    f"Fabricante: {fabricante}",
                    f"Fecha de aprobacion: {aprobacion}",
                    f"Condicion de venta: {disponibilidad}",
                    f"NDC: {ndc}",
                    f"Precio: {precio}",
                ]
            )

            metadata = {
                "source": DRUG_CSV_PATH.name,
                "titulo": nombre or "Desconocido",
                "categoria": drug_class or "Sin categoria",
                "resumen": indicaciones or "N/A",
                "drug_id": drug_id,
                "generic_name": generico,
                "dosage_form": presentacion,
                "strength": fuerza,
                "route": via,
                "mechanism": mecanismo,
                "side_effects": efectos,
                "contraindications": contra,
                "interactions": interacciones,
                "warnings": advertencias,
                "pregnancy_category": embarazo,
                "storage": almacenamiento,
                "manufacturer": fabricante,
                "approval_date": aprobacion,
                "availability": disponibilidad,
                "ndc": ndc,
                "price": precio,
            }
            documents.append(Document(page_content=contenido, metadata=metadata))

    if not documents:
        raise RuntimeError(f"No se encontraron filas en {DRUG_CSV_PATH.resolve()}")
    return documents


def get_corpus_state() -> Dict[str, float]:
    ensure_drug_csv_exists()
    return {str(DRUG_CSV_PATH): DRUG_CSV_PATH.stat().st_mtime}


def load_cached_state() -> Dict[str, object]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_cached_state(corpus_state: Dict[str, float]) -> None:
    CACHE_FILE.write_text(json.dumps({"corpus_state": corpus_state}, indent=2))


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

    corpus_state = get_corpus_state()
    cached_state = load_cached_state()
    needs_refresh = cached_state.get("corpus_state") != corpus_state

    if not needs_refresh:
        print("♻️ No hay cambios en el CSV; colección existente reutilizada.")
        return

    print("📂 Cargando CSV de medicamentos (csv_vademecum)…")
    source_documents = load_drug_documents()

    print("🗄️ Generando vector store en Qdrant (csv_vademecum)…")
    build_vector_store(source_documents)
    save_cached_state(corpus_state)

    print("✅ Preparación de datos finalizada (csv_vademecum).")


__all__ = [
    "prepare_corpus",
    "load_pdf_documents",
    "load_drug_documents",
    "extract_metadata",
    "chunk_documents",
]
