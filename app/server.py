"""FastAPI + LangServe server for the módulo 3 RAG pipeline.

Comparte la cadena definida en `rag_modulo3.rag_chain` con el CLI y expone
un endpoint `/rag` junto a un frontend simple en `static/index.html`.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langserve import add_routes

from rag_modulo3 import build_rag_chain

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "static"

app = FastAPI(
    title="RAG Módulo 3",
    version="1.0",
    description="Consultas RAG sobre PDFs del módulo 3.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_chain = build_rag_chain()
add_routes(app, rag_chain, path="/rag")


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise RuntimeError("No se encontró static/index.html")
    return FileResponse(index_path)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
