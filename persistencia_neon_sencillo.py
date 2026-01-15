from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path
from functools import lru_cache
from typing import Annotated, Any, Dict, List, TypedDict
from datetime import datetime, time

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
DATA_DIR = ROOT_DIR / "data"

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Define DATABASE_URL en .env para usar Neon/Postgres")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=500)


class EstadoPersonalizado(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    nombre_usuario: str
    intereses: List[str]
    nivel_experiencia: str
    numero_sesion: int
    preguntas_realizadas: int
    preferencias: Dict[str, Any]


def agente_personalizado(state: EstadoPersonalizado) -> EstadoPersonalizado:
    nombre = state.get("nombre_usuario", "Estudiante")
    intereses = state.get("intereses", [])
    nivel = state.get("nivel_experiencia", "principiante")
    sesion = state.get("numero_sesion", 0) + 1
    preguntas = state.get("preguntas_realizadas", 0) + 1

    system_prompt = (
        f"Eres un asistente sobre componentes de medicamentos. "
        f"Nombre: {nombre}. Nivel: {nivel}. "
        f"Medicamentos consultados: {intereses if intereses else 'ninguno'}."
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    respuesta = llm.invoke(messages)

    ultimo_mensaje = state["messages"][-1].content if state["messages"] else ""
    nuevos_intereses = intereses.copy()
    palabras = [p.strip(".,!?;:") for p in ultimo_mensaje.split()]
    for i, palabra in enumerate(palabras):
        palabra_lower = palabra.lower()
        if palabra_lower in {"llamo", "soy"} and i + 1 < len(palabras):
            nombre = palabras[i + 1].capitalize()
            break

    medicamento = None
    for i, palabra in enumerate(palabras):
        if palabra.lower() in {"medicamento", "medicina", "fármaco", "farmaco"} and i + 1 < len(palabras):
            medicamento = palabras[i + 1]
            break
    if not medicamento and palabras:
        medicamento = palabras[-1]

    if medicamento and medicamento not in nuevos_intereses:
        nuevos_intereses.append(medicamento)

    return {
        "messages": [respuesta],
        "nombre_usuario": nombre,
        "intereses": nuevos_intereses,
        "nivel_experiencia": nivel,
        "numero_sesion": sesion,
        "preguntas_realizadas": preguntas,
        "preferencias": state.get("preferencias", {}),
    }


def build_graph() -> StateGraph:
    builder = StateGraph(EstadoPersonalizado)
    builder.add_node("tutor", agente_personalizado)
    builder.add_edge(START, "tutor")
    builder.add_edge("tutor", END)
    return builder


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import radians, sin, cos, sqrt, atan2

    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return r * c


@lru_cache(maxsize=7)
def load_locales_cache(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], f"No existe {path}"
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], "El archivo no contiene JSON valido"
    if not isinstance(data, list):
        return [], "El JSON no es una lista de locales"
    return data, None


@lru_cache(maxsize=7)
def load_turnos_cache(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], f"No existe {path}"
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], "El archivo de turnos no contiene JSON valido"
    if not isinstance(data, list):
        return [], "El JSON de turnos no es una lista"
    return data, None


def get_day_suffix() -> str:
    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("America/Santiago"))
    except Exception:
        now = datetime.now()

    day_map = {
        "monday": "lunes",
        "tuesday": "martes",
        "wednesday": "miercoles",
        "thursday": "jueves",
        "friday": "viernes",
        "saturday": "sabado",
        "sunday": "domingo",
    }
    return day_map.get(now.strftime("%A").lower(), "lunes")


def get_day_paths() -> tuple[Path, Path]:
    day = get_day_suffix()
    locales_path = DATA_DIR / f"locales_minsal_{day}.json"
    turnos_path = DATA_DIR / f"locales_turnos_{day}.json"
    return locales_path, turnos_path


def get_nearest_pharmacies(
    lat: float, lon: float, top_n: int = 3
) -> tuple[list[dict[str, str]], str | None]:
    locales_path, _ = get_day_paths()
    locales, err = load_locales_cache(locales_path)
    if err:
        return [], err

    ranked = []
    for item in locales:
        try:
            lat_item = float(item.get("local_lat") or 0)
            lon_item = float(item.get("local_lng") or 0)
        except (TypeError, ValueError):
            continue
        if not lat_item or not lon_item:
            continue
        distancia = haversine_km(lat, lon, lat_item, lon_item)
        ranked.append((distancia, item))

    ranked.sort(key=lambda x: x[0])
    nearest = []
    for dist, item in ranked[:top_n]:
        nearest.append(
            {
                "nombre": item.get("local_nombre", ""),
                "direccion": item.get("local_direccion", ""),
                "comuna": item.get("comuna_nombre", ""),
                "estado": item.get("estado", ""),
                "distancia_km": f"{dist:.2f}",
            }
        )
    return nearest, None


def get_nearest_open_pharmacies(
    lat: float, lon: float, top_n: int = 3
) -> tuple[list[dict[str, str]], str | None]:
    locales_path, turnos_path = get_day_paths()
    locales, err = load_locales_cache(locales_path)
    if err:
        return [], err
    turnos, turnos_err = load_turnos_cache(turnos_path)
    if turnos_err:
        return [], turnos_err

    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("America/Santiago"))
    except Exception:
        now = datetime.now()

    def is_for_today(item: dict[str, Any]) -> bool:
        fecha = str(item.get("fecha", "")).strip()
        if not fecha:
            return True
        for fmt in ("%d-%m-%y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(fecha, fmt).date()
                return parsed == now.date()
            except ValueError:
                continue
        return True

    def parse_time(value: str) -> time | None:
        try:
            parts = [int(part) for part in value.split(":")]
            return time(parts[0], parts[1], parts[2] if len(parts) > 2 else 0)
        except (ValueError, AttributeError):
            return None

    def is_open_now(item: dict[str, Any]) -> bool:
        def normalize_day(value: str) -> str:
            normalized = unicodedata.normalize("NFD", value)
            normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
            return normalized.lower().strip()

        day_map = {
            "monday": "lunes",
            "tuesday": "martes",
            "wednesday": "miercoles",
            "thursday": "jueves",
            "friday": "viernes",
            "saturday": "sabado",
            "sunday": "domingo",
        }
        current_day = normalize_day(day_map.get(now.strftime("%A").lower(), ""))
        item_day = normalize_day(str(item.get("funcionamiento_dia", "")))
        if current_day and item_day and item_day != current_day:
            return False

        apertura = parse_time(str(item.get("funcionamiento_hora_apertura", "")))
        cierre = parse_time(str(item.get("funcionamiento_hora_cierre", "")))
        if not apertura or not cierre:
            return False

        now_time = now.time()
        if cierre <= apertura:
            return now_time >= apertura or now_time <= cierre
        return apertura <= now_time <= cierre

    locales_by_id = {str(item.get("local_id")): item for item in locales if item.get("local_id")}

    ranked = []
    for item in turnos:
        if not (item.get("local_id") and is_for_today(item) and is_open_now(item)):
            continue
        try:
            lat_item = float(item.get("local_lat") or 0)
            lon_item = float(item.get("local_lng") or 0)
        except (TypeError, ValueError):
            continue
        if not lat_item or not lon_item:
            continue
        distancia = haversine_km(lat, lon, lat_item, lon_item)
        ranked.append((distancia, item))

    if not ranked:
        return [], "No hay locales abiertos con coordenadas válidas"

    ranked.sort(key=lambda x: x[0])
    nearest = []
    for dist, item in ranked[:top_n]:
        fallback = locales_by_id.get(str(item.get("local_id")), {})
        nearest.append(
            {
                "nombre": item.get("local_nombre") or fallback.get("local_nombre", ""),
                "direccion": item.get("local_direccion") or fallback.get("local_direccion", ""),
                "comuna": item.get("comuna_nombre") or fallback.get("comuna_nombre", ""),
                "estado": item.get("estado") or fallback.get("estado", ""),
                "dia": item.get("funcionamiento_dia", ""),
                "apertura": item.get("funcionamiento_hora_apertura", ""),
                "cierre": item.get("funcionamiento_hora_cierre", ""),
                "distancia_km": f"{dist:.2f}",
            }
        )
    return nearest, None


def run_demo(telefono: str, lat: float, lon: float, user_message: str) -> str:
    builder = build_graph()
    with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
        checkpointer.setup()
        grafo = builder.compile(checkpointer=checkpointer)

        mensaje = user_message.strip()
        entrada = {"messages": [HumanMessage(content=mensaje)]}
        config = {"configurable": {"thread_id": telefono}}
        resultado = grafo.invoke(entrada, config=config)
        return resultado["messages"][-1].content


class LocationPayload(BaseModel):
    telefono: str
    message: str
    lat: float
    lon: float
    accuracy: float | None = None


app = FastAPI(title="Neon Persistencia + Geolocalizacion")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "geoloc.html")


@app.post("/api/location")
def save_location(payload: LocationPayload) -> dict[str, object]:
    message_lower = payload.message.lower()
    wants_pharmacies = any(
        keyword in message_lower
        for keyword in ["farmacia", "farmacias", "cerca", "cercanas", "cercana"]
    )
    if wants_pharmacies:
        farmacias, farmacias_error = get_nearest_pharmacies(payload.lat, payload.lon, top_n=3)
        abiertas, abiertas_error = get_nearest_open_pharmacies(payload.lat, payload.lon, top_n=3)
        if farmacias or abiertas:
            answer = "Estas son las farmacias más cercanas y las abiertas más cercanas."
        elif farmacias_error or abiertas_error:
            error_msg = farmacias_error or abiertas_error
            answer = f"No pude leer el cache local de farmacias ({error_msg})."
        else:
            answer = "No pude encontrar farmacias cercanas en el cache local."
        return {
            "status": "ok",
            "answer": answer,
            "farmacias": farmacias,
            "farmacias_abiertas": abiertas,
            "farmacias_error": farmacias_error,
            "farmacias_abiertas_error": abiertas_error,
        }

    answer = run_demo(payload.telefono, payload.lat, payload.lon, payload.message)
    return {"status": "ok", "answer": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("persistencia_neon_sencillo:app", host="0.0.0.0", port=8001, reload=True)
