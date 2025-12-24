"""CLI interactivo del motor RAG.

Consume `rag_modulo3.rag_chain` para hacer preguntas en consola; asume que
`rag_data_preparation.py` ya sincronizó los PDFs con Qdrant.
"""

from __future__ import annotations

from rag_modulo3 import answer_question, build_rag_components


def main() -> None:
    llm, rewrite_chain, retriever = build_rag_components()

    print("✅ Pipeline listo. Escribe una pregunta (Ctrl+C para salir).")
    try:
        while True:
            user_query = input("\nPregunta: ").strip()
            if not user_query:
                continue
            answer = answer_question(user_query, retriever, llm, rewrite_chain)
            print(f"\n🤖 Respuesta:\n{answer}")
    except KeyboardInterrupt:
        print("\n👋 Hasta luego.")


if __name__ == "__main__":
    main()
