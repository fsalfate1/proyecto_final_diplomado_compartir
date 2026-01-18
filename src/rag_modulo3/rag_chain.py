"""Cadena RAG reusable para CLI y servidor.

Orquesta los prompts, embeddings y retriever definidos en `config.py`
para que `rag_cli.py` y `app/server.py` expongan la misma lógica.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel

from .config import (
    COLLECTION_NAME,
    SCORE_THRESHOLD,
    TOP_K,
    get_qdrant_client,
    load_environment,
)
from .prompts import build_answer_prompt, build_query_rewrite_prompt

SMALL_TALK_PHRASES = {
    "hola",
    "hola!",
    "hola.",
    "hola?",
    "hola,",
    "buenos dias",
    "buenos días",
    "buenas tardes",
    "buenas noches",
    "gracias",
    "que tal",
    "cómo estás",
    "como estas",
}

NO_KNOWLEDGE_RESPONSE = "Actualmente no se dispone de información sobre esta consulta en la base de conocimiento.Por favor, realice otra pregunta o reformule su solicitud."


def is_small_talk(query: str) -> bool:
    normalized = query.strip().lower()
    if not normalized:
        return True
    if normalized in SMALL_TALK_PHRASES:
        return True
    return False


def no_data_message() -> str:
    return "Hola, ¿en qué puedo ayudarte con los materiales del módulo 3? Si necesitas algo específico del CSV, dime y lo reviso."


def build_query_rewriter(llm: ChatOpenAI):
    prompt = build_query_rewrite_prompt()
    return prompt | llm | StrOutputParser()


def load_vector_store(embeddings: OpenAIEmbeddings) -> QdrantVectorStore:
    client = get_qdrant_client()
    try:
        client.get_collection(COLLECTION_NAME)
    except Exception as exc:  # pragma: no cover - requiere Qdrant
        raise RuntimeError(
            "La colección no existe. Ejecuta primero scripts/rag_data_preparation.py."
        ) from exc

    return QdrantVectorStore(
        client=client,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
    )


def answer_question(query: str, vector_store: QdrantVectorStore, llm: ChatOpenAI, rewrite_chain) -> str:
    if is_small_talk(query):
        return no_data_message()

    improved_query = rewrite_chain.invoke({"query": query})
    try:
        raw_results = vector_store.similarity_search_with_relevance_scores(
            improved_query,
            k=TOP_K,
        )
        scored_results = raw_results
    except Exception:
        scored_results = [
            (doc, 1 - min(max(distance or 0.0, 0.0), 1.0))
            for doc, distance in vector_store.similarity_search_with_score(
                improved_query,
                k=TOP_K,
            )
        ]

    if not scored_results:
        return NO_KNOWLEDGE_RESPONSE

    relevant_docs = []
    best_score: float | None = None
    for doc, score in scored_results:
        if score is None:
            continue
        best_score = score if best_score is None else max(best_score, score)
        if score >= SCORE_THRESHOLD:
            relevant_docs.append(doc)

    if not relevant_docs or best_score is None or best_score < SCORE_THRESHOLD:
        return NO_KNOWLEDGE_RESPONSE

    context_chunks = []
    for doc in relevant_docs:
        if not doc.page_content or not doc.page_content.strip():
            continue
        context_chunks.append(
            f"Título: {doc.metadata.get('titulo', 'Desconocido')}\n"
            f"Resumen: {doc.metadata.get('resumen', 'N/A')}\n"
            f"Fuente: {doc.metadata.get('source', 'N/A')}\n"
            f"Contenido:\n{doc.page_content}"
        )

    if not context_chunks:
        return NO_KNOWLEDGE_RESPONSE

    context = "\n\n".join(context_chunks)
    qa_prompt = build_answer_prompt()
    response = qa_prompt | llm | StrOutputParser()
    answer = response.invoke(
        {"context": context, "original": query, "rewritten": improved_query}
    )

    clean_answer = answer.rstrip()
    first_doc = relevant_docs[0]
    sources_block = (
        f"- {first_doc.metadata.get('titulo', 'Desconocido')} "
        f"({first_doc.metadata.get('source', 'N/A')})"
    )
    return f"{clean_answer}\n\nFuentes consultadas:\n\n{sources_block}"


def build_rag_components() -> Tuple[ChatOpenAI, Any, QdrantVectorStore]:
    load_environment()

    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    rewrite_chain = build_query_rewriter(llm)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = load_vector_store(embeddings)
    return llm, rewrite_chain, vector_store


class QueryInput(BaseModel):
    query: str


class AnswerOutput(BaseModel):
    query: str
    answer: str


def build_rag_chain():
    llm, rewrite_chain, vector_store = build_rag_components()

    def _invoke(inputs: Any) -> Dict[str, str]:
        if isinstance(inputs, QueryInput):
            query = inputs.query
        elif isinstance(inputs, str):
            query = inputs
        elif isinstance(inputs, dict):
            query = (
                inputs.get("query")
                or inputs.get("input")
                or inputs.get("text")
                or ""
            )
        else:
            raise ValueError("Entrada no soportada, envía un string o {'query': '...'}")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Debes proporcionar una pregunta en 'query'.")

        answer = answer_question(query, vector_store, llm, rewrite_chain)
        return {"answer": answer, "query": query}

    return RunnableLambda(_invoke).with_types(
        input_type=QueryInput,
        output_type=AnswerOutput,
    )


__all__ = [
    "build_rag_chain",
    "build_rag_components",
    "answer_question",
]
