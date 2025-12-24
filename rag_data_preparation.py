"""Script CLI para regenerar la colección en Qdrant.

Encapsula la llamada principal a `rag_modulo3.preparation.prepare_corpus`
y se ejecuta cada vez que cambian los PDFs en `pdf/`.
"""

from __future__ import annotations

from rag_modulo3 import prepare_corpus


def main() -> None:
    prepare_corpus()


if __name__ == "__main__":
    main()
