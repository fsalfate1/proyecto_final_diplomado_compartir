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

from .config import COLLECTION_NAME, TOP_K, get_qdrant_client, load_environment
from .prompts import build_answer_prompt, build_query_rewrite_prompt


def build_query_rewriter(llm: ChatOpenAI):
    prompt = build_query_rewrite_prompt()
    return prompt | llm | StrOutputParser()


def load_vector_store(embeddings: OpenAIEmbeddings) -> QdrantVectorStore:
    client = get_qdrant_client()
    try:
        client.get_collection(COLLECTION_NAME)
    except Exception as exc:  # pragma: no cover - requiere Qdrant
        raise RuntimeError(
            "La colección no existe. Ejecuta primero rag_data_preparation.py."
        ) from exc

    return QdrantVectorStore(
        client=client,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
    )


def build_retriever(vector_store: QdrantVectorStore):
    return vector_store.as_retriever(search_kwargs={"k": TOP_K})


def answer_question(query: str, retriever, llm: ChatOpenAI, rewrite_chain) -> str:
    improved_query = rewrite_chain.invoke({"query": query})
    retrieved_docs = retriever.invoke(improved_query)

    context = "\n\n".join(
        f"Título: {doc.metadata.get('titulo', 'Desconocido')}\n"
        f"Resumen: {doc.metadata.get('resumen', 'N/A')}\n"
        f"Fuente: {doc.metadata.get('source', 'N/A')}\n"
        f"Contenido:\n{doc.page_content}"
        for doc in retrieved_docs
    )

    qa_prompt = build_answer_prompt()
    response = qa_prompt | llm | StrOutputParser()
    answer = response.invoke(
        {"context": context, "original": query, "rewritten": improved_query}
    )

    sources_block = "\n".join(
        f"- {doc.metadata.get('titulo', 'Desconocido')} ({doc.metadata.get('source')})"
        for doc in retrieved_docs
    )

    return f"{answer}\n\nFuentes consultadas:\n{sources_block}"


def build_rag_components() -> Tuple[ChatOpenAI, Any, Any]:
    load_environment()

    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    rewrite_chain = build_query_rewriter(llm)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = load_vector_store(embeddings)
    retriever = build_retriever(vector_store)
    return llm, rewrite_chain, retriever


class QueryInput(BaseModel):
    query: str


class AnswerOutput(BaseModel):
    query: str
    answer: str


def build_rag_chain():
    llm, rewrite_chain, retriever = build_rag_components()

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

        answer = answer_question(query, retriever, llm, rewrite_chain)
        return {"answer": answer, "query": query}

    return RunnableLambda(_invoke).with_types(
        input_type=QueryInput,
        output_type=AnswerOutput,
    )


__all__ = [
    "build_rag_chain",
    "build_rag_components",
    "build_retriever",
    "answer_question",
]
