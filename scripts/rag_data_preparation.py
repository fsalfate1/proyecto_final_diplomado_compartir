"""Script CLI para regenerar la colección en Qdrant.

Encapsula la llamada principal a `rag_modulo3.preparation.prepare_corpus`
y se ejecuta cada vez que cambia el CSV en `excel/`.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_modulo3 import prepare_corpus


def main() -> None:
    prepare_corpus()


if __name__ == "__main__":
    main()
