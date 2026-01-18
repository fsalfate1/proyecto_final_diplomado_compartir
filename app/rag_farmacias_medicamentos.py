from __future__ import annotations

import csv
import json
import os
import sys
import unicodedata
from pathlib import Path
from functools import lru_cache
from typing import Annotated, Any, Dict, List, TypedDict
from datetime import datetime, time

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_modulo3 import DRUG_CSV_PATH, SCORE_THRESHOLD, TOP_K
from rag_modulo3.rag_chain import NO_KNOWLEDGE_RESPONSE, build_rag_components

STATIC_DIR = PROJECT_ROOT / "static"
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("Define DATABASE_URL en .env para usar Neon/Postgres")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=500)
doctor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=500)

PHARMA_K = 8
USD_TO_CLP = 900
HISTORY_ROUTER_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=50)


class EstadoPersonalizado(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    nombre_usuario: str
    intereses: List[str]
    nivel_experiencia: str
    numero_sesion: int
    preguntas_realizadas: int
    preferencias: Dict[str, Any]
    lat: float | None
    lon: float | None
    last_agent: str | None
    last_sources: list[str]
    last_qdrant_hits: list[dict[str, object]]
    last_farmacias: list[dict[str, str]]
    last_farmacias_abiertas: list[dict[str, str]]
    last_farmacias_error: str | None
    last_farmacias_abiertas_error: str | None
    historial_consultas: list[str]


def update_profile(state: EstadoPersonalizado, message: str) -> tuple[str, list[str], int, int]:
    nombre = state.get("nombre_usuario", "Estudiante")
    intereses = list(state.get("intereses", []))
    sesion = state.get("numero_sesion", 0) + 1
    preguntas = state.get("preguntas_realizadas", 0) + 1

    palabras = [p.strip(".,!?;:") for p in message.split()]
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

    if not medicamento and (is_medicine_query(message) or has_vademecum_sources(message)):
        medicamento = palabras[-1] if palabras else None

    if medicamento:
        limpio = medicamento.strip(".,!?;:").lower()
        drug_map = get_drug_name_map()
        key = normalize_key(limpio)
        nombre_catalogo = drug_map.get(key)
        if nombre_catalogo and nombre_catalogo not in intereses:
            intereses.append(nombre_catalogo)

    return nombre, intereses, sesion, preguntas


def is_interest_query(message: str) -> bool:
    normalized = unicodedata.normalize("NFD", message.lower())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return any(
        phrase in normalized
        for phrase in [
            "mis intereses",
            "mis medicamentos",
            "mi historial",
            "mi historial de medicamentos",
            "que medicamentos he consultado",
            "que he consultado",
            "que he preguntado",
            "intereses personales",
            "historial de consultas",
            "que hemos estado hablando",
            "de que hemos hablado",
            "que hemos conversado",
            "que hemos hablado",
            "que temas de salud hemos conversado",
            "que temas de salud hemos hablado",
            "podemos continuar con nuestra conversacion",
            "podemos continuar con nuestra conversación",
        ]
    )


def is_health_related(message: str) -> bool:
    normalized = normalize_key(message)
    keywords = [
        "farmacia",
        "farmacias",
        "medicamento",
        "medicina",
        "farmaco",
        "farmaco",
        "remedio",
        "salud",
        "dolor",
        "duele",
        "dolencia",
        "codo",
        "sintoma",
        "sintomas",
        "fiebre",
        "tos",
        "gripe",
        "resfrio",
        "resfrio",
        "nausea",
        "vomito",
        "diarrea",
        "mareo",
        "cefalea",
        "migraña",
        "presion",
        "cardiaco",
        "corazon",
        "ansiedad",
        "depresion",
        "insomnio",
        "alergia",
        "asthma",
        "asma",
        "epoc",
    ]
    if any(keyword in normalized for keyword in keywords):
        return True
    return is_medicine_query(message) or is_pharmacy_query(message)


def normalize_key(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return "".join(ch for ch in normalized.lower().strip() if ch.isalnum() or ch.isspace())


def is_history_intent_llm(message: str) -> bool:
    if not message.strip():
        return False
    system_prompt = (
        "Eres un clasificador binario. Responde solo YES o NO.\n"
        "Marca YES si el usuario pide historial, memoria, continuidad de conversacion, "
        "o resumen de lo hablado. En caso contrario responde NO."
    )
    response = HISTORY_ROUTER_LLM.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=message)]
    )
    return response.content.strip().upper().startswith("YES")


@lru_cache(maxsize=1)
def get_drug_name_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    try:
        with DRUG_CSV_PATH.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = (row.get("Drug Name") or "").strip()
                generic = (row.get("Generic Name") or "").strip()
                if name:
                    mapping[normalize_key(name)] = name
                if generic:
                    mapping[normalize_key(generic)] = name or generic
    except FileNotFoundError:
        return {}
    return mapping


def clean_interests(intereses: list[str]) -> list[str]:
    mapping = get_drug_name_map()
    if not mapping:
        deduped = []
        for item in intereses:
            if item and item not in deduped:
                deduped.append(item)
        return deduped
    cleaned = []
    for item in intereses:
        key = normalize_key(item)
        nombre = mapping.get(key)
        if nombre and nombre not in cleaned:
            cleaned.append(nombre)
    return cleaned


def summarize_history(messages: list[BaseMessage]) -> str:
    user_msgs = []
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content.strip():
            user_msgs.append(msg.content.strip())
    if not user_msgs:
        return "Aun no tenemos un historial de mensajes guardado."
    topics = []
    for msg in user_msgs:
        if is_interest_query(msg):
            continue
        if is_greeting(msg):
            continue
        normalized = msg.lower()
        if "que hemos estado" in normalized or "que hemos conversado" in normalized:
            continue
        if not is_health_related(msg):
            continue
        topics.append(classify_health_topic(msg))
    deduped = []
    for topic in topics:
        if topic and (not deduped or deduped[-1] != topic):
            deduped.append(topic)
    recent = deduped[-4:]
    if not recent:
        return "Si quieres, puedo resumir tus ultimas consultas reales sobre salud o medicamentos."
    joined = ", ".join(recent)
    return f"En la conversacion hemos visto temas como: {joined}."


def classify_health_topic(message: str) -> str:
    text = normalize_key(message)
    if "alergia" in text:
        return "alergias"
    if "dolor" in text or "duele" in text or "dolencia" in text:
        return "dolor"
    if "fiebre" in text or "temperatura" in text:
        return "fiebre"
    if "tos" in text or "resfrio" in text or "gripe" in text:
        return "sintomas respiratorios"
    if "nausea" in text or "vomito" in text or "diarrea" in text:
        return "malestares digestivos"
    if "mareo" in text:
        return "mareos"
    if "ansiedad" in text or "depresion" in text or "insomnio" in text:
        return "bienestar emocional"
    if "presion" in text or "corazon" in text or "cardiaco" in text:
        return "salud cardiovascular"
    return "consultas de salud"


def summarize_topics(topics: list[str]) -> str:
    if not topics:
        return "En la conversacion hemos visto temas de salud."
    deduped = []
    for topic in topics:
        normalized = classify_health_topic(topic)
        if normalized and (not deduped or deduped[-1] != normalized):
            deduped.append(normalized)
    recent = deduped[-5:]
    joined = ", ".join(recent)
    return (
        "Ademas, hemos hablado de temas de salud como "
        f"{joined}."
    )


def extract_health_topic_detail(message: str) -> str:
    text = normalize_key(message)
    if "dolor" in text or "duele" in text or "dolencia" in text:
        for zona in [
            "codo",
            "cabeza",
            "pecho",
            "espalda",
            "garganta",
            "estomago",
            "abdomen",
            "rodilla",
            "hombro",
            "cuello",
        ]:
            if zona in text:
                return f"dolor de {zona}"
        return "dolor"
    return classify_health_topic(message)


def swarm_node(state: EstadoPersonalizado) -> EstadoPersonalizado:
    user_message = ""
    if state.get("messages"):
        user_message = state["messages"][-1].content
    lat = state.get("lat")
    lon = state.get("lon")

    nombre, intereses, sesion, preguntas = update_profile(state, user_message)
    historial_consultas = list(state.get("historial_consultas", []))
    if user_message and is_health_related(user_message):
        topic = extract_health_topic_detail(user_message)
        if not historial_consultas or historial_consultas[-1] != topic:
            historial_consultas.append(topic)
    wants_history = is_interest_query(user_message) or is_history_intent_llm(user_message)
    is_medicine_like = is_medicine_query(user_message) or has_vademecum_sources(user_message)
    if wants_history and not (is_pharmacy_query(user_message) or is_medicine_like):
        historial_limpio = clean_interests(intereses)
        historial = ", ".join(historial_limpio) if historial_limpio else "ninguno"
        resumen = summarize_topics(historial_consultas)
        temas = (
            "ninguno"
            if not historial_consultas
            else ", ".join(historial_consultas[-5:])
        )
        answer = (
            "Hasta ahora tengo registrado lo siguiente:\n"
            f"- Medicamentos consultados: {historial}\n"
            f"- Temas de salud: {temas}\n\n"
            "Si quieres, puedo profundizar en cualquiera de ellos o resumir con más detalle."
        )
        return {
            "messages": [AIMessage(content=answer)],
            "nombre_usuario": nombre,
            "intereses": intereses,
            "nivel_experiencia": state.get("nivel_experiencia", "principiante"),
            "numero_sesion": sesion,
            "preguntas_realizadas": preguntas,
            "preferencias": state.get("preferencias", {}),
            "lat": lat,
            "lon": lon,
            "last_agent": "auxiliar",
            "last_sources": [],
            "last_qdrant_hits": [],
            "last_farmacias": [],
            "last_farmacias_abiertas": [],
            "last_farmacias_error": None,
            "last_farmacias_abiertas_error": None,
            "historial_consultas": historial_consultas,
        }
    result = run_swarm(user_message, lat, lon)
    qdrant_hits = result.get("qdrant_hits", [])
    if qdrant_hits:
        for hit in qdrant_hits:
            titulo = hit.get("titulo")
            if isinstance(titulo, str) and titulo:
                nombre = titulo.strip()
                if nombre and nombre not in intereses:
                    intereses.append(nombre)
    answer = result.get("answer", NO_KNOWLEDGE_RESPONSE)

    return {
        "messages": [AIMessage(content=answer)],
        "nombre_usuario": nombre,
        "intereses": intereses,
        "nivel_experiencia": state.get("nivel_experiencia", "principiante"),
        "numero_sesion": sesion,
        "preguntas_realizadas": preguntas,
        "preferencias": state.get("preferencias", {}),
        "lat": lat,
        "lon": lon,
        "last_agent": result.get("agent"),
        "last_sources": result.get("sources", []),
        "last_qdrant_hits": qdrant_hits,
        "last_farmacias": result.get("farmacias", []),
        "last_farmacias_abiertas": result.get("farmacias_abiertas", []),
        "last_farmacias_error": result.get("farmacias_error"),
        "last_farmacias_abiertas_error": result.get("farmacias_abiertas_error"),
        "historial_consultas": historial_consultas,
    }


def build_swarm_graph() -> StateGraph:
    builder = StateGraph(EstadoPersonalizado)
    builder.add_node("swarm", swarm_node)
    builder.add_edge(START, "swarm")
    builder.add_edge("swarm", END)
    return builder


@lru_cache(maxsize=1)
def get_swarm_graph() -> StateGraph:
    return build_swarm_graph()


class AgentResult(TypedDict, total=False):
    agent: str
    answer: str
    sources: list[str]
    qdrant_hits: list[dict[str, object]]
    farmacias: list[dict[str, str]]
    farmacias_abiertas: list[dict[str, str]]
    farmacias_error: str | None
    farmacias_abiertas_error: str | None
    handoff: str | None


PHARMACY_KEYWORDS = {
    "farmacia",
    "farmacias",
    "cerca",
    "cercanas",
    "cercana",
    "abiertas",
    "turno",
    "turnos",
}

GREETING_KEYWORDS = {
    "hola",
    "buenos dias",
    "buenas tardes",
    "buenas noches",
    "que tal",
    "cómo estas",
    "como estas",
    "hello",
    "hi",
}


MED_QUERY_KEYWORDS = {
    "medicamento",
    "medicamentos",
    "farmaco",
    "fármaco",
    "medicament",
    "remedio",
    "remedios",
    "principio activo",
    "indicaciones",
    "presentacion",
    "presentación",
    "condicion de venta",
    "condición de venta",
    "grupo terapeutico",
    "grupo terapéutico",
    "alergias",
    "cardiovascular",
    "sistema nervioso",
    "antiinfecciosos",
    "vademecum",
    "vademécum",
    "disponibles en el mercado",
}


def is_medicine_query(message: str) -> bool:
    normalized = message.lower()
    return any(keyword in normalized for keyword in MED_QUERY_KEYWORDS)


def is_pharmacy_query(message: str) -> bool:
    normalized = message.lower()
    return any(keyword in normalized for keyword in PHARMACY_KEYWORDS)


def extract_focus(message: str) -> set[str]:
    normalized = message.lower()
    focus = set()
    if "precio" in normalized or "cuanto cuesta" in normalized or "valor" in normalized:
        focus.add("precio")
    if "indicacion" in normalized or "para que sirve" in normalized or "uso" in normalized:
        focus.add("indicaciones")
    if "mecanismo" in normalized:
        focus.add("mecanismo")
    if "efectos" in normalized or "efectos secundarios" in normalized:
        focus.add("efectos")
    if "contraindic" in normalized:
        focus.add("contraindicaciones")
    if "interaccion" in normalized:
        focus.add("interacciones")
    if "advertenc" in normalized or "precaucion" in normalized:
        focus.add("advertencias")
    if "embarazo" in normalized:
        focus.add("embarazo")
    if "almacen" in normalized or "guardar" in normalized:
        focus.add("almacenamiento")
    if "fabricante" in normalized:
        focus.add("fabricante")
    if "presentacion" in normalized or "dosis" in normalized:
        focus.add("presentacion")
    if "via" in normalized:
        focus.add("via")
    if "ndc" in normalized:
        pass
    return focus


def is_greeting(message: str) -> bool:
    normalized = message.lower().strip()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in GREETING_KEYWORDS)


@lru_cache(maxsize=1)
def get_rag_components():
    return build_rag_components()


def get_vademecum_sources(user_message: str) -> list[str]:
    llm, rewrite_chain, vector_store = get_rag_components()
    improved_query = rewrite_chain.invoke({"query": user_message})
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

    sources: list[str] = []
    for doc, score in scored_results:
        if score is None or score < SCORE_THRESHOLD:
            continue
        source = doc.metadata.get("source")
        if source and source not in sources:
            sources.append(source)
    return sources


def has_vademecum_sources(user_message: str) -> bool:
    return bool(get_vademecum_sources(user_message))


def _normalize_text(value: str) -> str:
    return "".join(ch for ch in value.lower().strip() if ch.isalnum() or ch.isspace())


def _extract_query_tokens(message: str) -> list[str]:
    stopwords = {
        "para",
        "que",
        "sirve",
        "el",
        "la",
        "los",
        "las",
        "de",
        "del",
        "un",
        "una",
        "y",
        "o",
        "precio",
        "dame",
        "informacion",
        "información",
    }
    tokens = [_normalize_text(token) for token in message.split()]
    return [token for token in tokens if token and token not in stopwords and len(token) > 3]


def _score_results(user_message: str, k: int) -> list[tuple[Any, float | None]]:
    llm, rewrite_chain, vector_store = get_rag_components()
    improved_query = rewrite_chain.invoke({"query": user_message})
    try:
        return vector_store.similarity_search_with_relevance_scores(
            improved_query,
            k=k,
        )
    except Exception:
        return [
            (doc, 1 - min(max(distance or 0.0, 0.0), 1.0))
            for doc, distance in vector_store.similarity_search_with_score(
                improved_query,
                k=k,
            )
        ]


def select_drug_docs(user_message: str) -> list[tuple[Any, float | None]]:
    scored_results = _score_results(user_message, PHARMA_K)
    if not scored_results:
        return []
    tokens = _extract_query_tokens(user_message)
    if not tokens:
        return scored_results
    filtered: list[tuple[Any, float | None]] = []
    for doc, score in scored_results:
        title = _normalize_text(str(doc.metadata.get("titulo") or ""))
        generic = _normalize_text(str(doc.metadata.get("generic_name") or ""))
        if any(token in title or token in generic for token in tokens):
            filtered.append((doc, score))
    return filtered or scored_results


def get_vademecum_hits(user_message: str) -> list[dict[str, object]]:
    hits: list[dict[str, object]] = []
    for doc, score in select_drug_docs(user_message):
        if score is None or score < SCORE_THRESHOLD:
            continue
        hits.append(
            {
                "id": getattr(doc, "id", None),
                "score": score,
                "titulo": doc.metadata.get("titulo"),
                "source": doc.metadata.get("source"),
            }
        )
    return hits


def format_drug_answer(scored_results: list[tuple[Any, float | None]], focus: set[str]) -> str:
    def _join_unique(values: list[str]) -> str:
        cleaned = [v for v in values if v and v != "N/A"]
        return ", ".join(sorted(set(cleaned))) if cleaned else "N/A"

    def _translate_value(value: str) -> str:
        translations = {
            "hypertension": "hipertension",
            "anxiety": "ansiedad",
            "tablet": "tabletas",
            "oral": "oral",
            "room temperature": "temperatura ambiente",
            "prescription": "bajo receta",
            "category c": "categoria C",
            "category d": "categoria D",
            "blocks beta receptors": "bloquea receptores beta",
            "blocks alpha and beta receptors": "bloquea receptores alfa y beta",
            "enhances gaba activity": "potencia la actividad de GABA",
            "enhances effects of gaba in the brain": "potencia los efectos de GABA en el cerebro",
            "dizziness": "mareos",
            "low heart rate": "frecuencia cardiaca baja",
            "low blood pressure": "presion arterial baja",
            "do not stop abruptly": "no suspender de forma brusca",
            "take with food": "tomar con alimentos",
            "asthma: use with caution": "asma: usar con precaucion",
            "copd: use with caution": "EPOC: usar con precaucion",
            "beta blocker": "bloqueador beta",
            "alpha-beta blocker": "bloqueador alfa-beta",
        }
        normalized = value.strip().lower()
        return translations.get(normalized, value)

    def _translate_list(values: list[str]) -> list[str]:
        return [_translate_value(value) for value in values]

    def _spanish_defaults(value: str, fallback: str) -> str:
        return fallback if value in {"N/A", ""} else value

    def _format_clp(values: list[str]) -> str:
        numeric = []
        for value in values:
            try:
                numeric.append(float(value))
            except (TypeError, ValueError):
                continue
        if not numeric:
            return "no especificado"
        clp_values = sorted({round(v * USD_TO_CLP) for v in numeric})
        return ", ".join(f"${val:,}".replace(",", ".") for val in clp_values)

    entries = []
    grouped: dict[str, dict[str, object]] = {}
    for doc, score in scored_results:
        if score is None or score < SCORE_THRESHOLD:
            continue
        md = doc.metadata
        titulo = md.get("titulo", "Desconocido")
        fuente = md.get("source", "N/A")
        resumen = md.get("resumen", "N/A")
        precio = md.get("price", "N/A")
        presentacion = md.get("dosage_form", "N/A")
        fuerza = md.get("strength", "N/A")
        fabricante = md.get("manufacturer", "N/A")
        key = f"{titulo}|{fuente}"
        if key not in grouped:
            grouped[key] = {
                "titulo": titulo,
                "fuente": fuente,
                "resumen": [],
                "presentacion": [],
                "via": [],
                "categoria": [],
                "generic": [],
                "fabricantes": [],
                "precios": [],
                "dosis": [],
                "mechanism": [],
                "side_effects": [],
                "contraindications": [],
                "interactions": [],
                "warnings": [],
                "pregnancy": [],
                "storage": [],
                "approval": [],
                "availability": [],
            }
        grouped[key]["resumen"].append(_translate_value(resumen))
        grouped[key]["presentacion"].append(_translate_value(presentacion))
        grouped[key]["via"].append(_translate_value(md.get("route", "N/A")))
        grouped[key]["categoria"].append(_translate_value(md.get("categoria", "N/A")))
        grouped[key]["generic"].append(md.get("generic_name", "N/A"))
        grouped[key]["precios"].append(precio)
        grouped[key]["dosis"].append(fuerza)
        grouped[key]["fabricantes"].append(fabricante)
        grouped[key]["mechanism"].append(_translate_value(md.get("mechanism", "N/A")))
        grouped[key]["side_effects"].append(_translate_value(md.get("side_effects", "N/A")))
        grouped[key]["contraindications"].append(_translate_value(md.get("contraindications", "N/A")))
        grouped[key]["interactions"].append(_translate_value(md.get("interactions", "N/A")))
        grouped[key]["warnings"].append(_translate_value(md.get("warnings", "N/A")))
        grouped[key]["pregnancy"].append(_translate_value(md.get("pregnancy_category", "N/A")))
        grouped[key]["storage"].append(_translate_value(md.get("storage", "N/A")))
        grouped[key]["approval"].append(md.get("approval_date", "N/A"))
        grouped[key]["availability"].append(md.get("availability", "N/A"))

    for item in grouped.values():
        uso = _spanish_defaults(_join_unique(item["resumen"]), "uso no especificado")
        presentacion = _spanish_defaults(_join_unique(item["presentacion"]), "presentación no especificada")
        via = _spanish_defaults(_join_unique(item["via"]), "vía no especificada")
        base_line = (
            f"{item['titulo']} es un medicamento cuyo uso principal está orientado a "
            f"{uso}. Se presenta en forma de {presentacion} "
            f"y se administra por vía {via.lower()}."
        )
        details = []
        if not focus or "indicaciones" in focus:
            details.append(f"Su indicación principal es {uso}.")
        if not focus or "presentacion" in focus:
            details.append(
                f"Se presenta como {presentacion} "
                f"en dosis de {_spanish_defaults(_join_unique(item['dosis']), 'dosis no especificadas')}."
            )
        if not focus or "via" in focus:
            details.append(f"La vía de administración indicada es {via}.")
        if not focus or "mecanismo" in focus:
            details.append(
                f"El mecanismo de acción descrito es: "
                f"{_spanish_defaults(_join_unique(item['mechanism']), 'no especificado')}."
            )
        if not focus or "efectos" in focus:
            details.append(
                f"Entre los efectos secundarios se menciona "
                f"{_spanish_defaults(_join_unique(item['side_effects']), 'no especificado')}."
            )
        if not focus or "contraindicaciones" in focus:
            details.append(
                f"Como contraindicaciones aparecen "
                f"{_spanish_defaults(_join_unique(item['contraindications']), 'no especificadas')}."
            )
        if not focus or "interacciones" in focus:
            details.append(
                f"En interacciones se indica: "
                f"{_spanish_defaults(_join_unique(item['interactions']), 'no especificadas')}."
            )
        if not focus or "advertencias" in focus:
            details.append(
                f"Advertencia relevante: "
                f"{_spanish_defaults(_join_unique(item['warnings']), 'no especificada')}."
            )
        if not focus or "embarazo" in focus:
            details.append(
                f"Categoría en embarazo: "
                f"{_spanish_defaults(_join_unique(item['pregnancy']), 'no especificada')}."
            )
        if not focus or "almacenamiento" in focus:
            details.append(
                f"Almacenamiento: "
                f"{_spanish_defaults(_join_unique(item['storage']), 'no especificado')}."
            )
        if not focus or "fabricante" in focus:
            details.append(
                f"Los fabricantes registrados son "
                f"{_spanish_defaults(_join_unique(item['fabricantes']), 'no especificados')}."
            )
        if not focus or "precio" in focus:
            usd_raw = _spanish_defaults(_join_unique(item["precios"]), "no especificado")
            clp = _format_clp(item["precios"])
            details.append(
                f"El precio registrado es {usd_raw} USD (≈ {clp} CLP)."
            )
        details.append(
            f"El principio activo es {_join_unique(item['generic'])} "
            f"y pertenece al grupo terapéutico {_join_unique(item['categoria'])}."
        )
        entries.append("\n".join([base_line, " ".join(details)]))
    if not entries:
        return NO_KNOWLEDGE_RESPONSE
    intro = (
        "Te comparto la informacion encontrada en la base csv_vademecum, redactada de forma más humana:"
    )
    cierre = "\n\nSi quieres que me enfoque en otro detalle, dime y lo ajusto."
    fuentes = []
    for item in grouped.values():
        fuentes.append(
            f"- Titulo: {item['titulo']} | Categoria: {_join_unique(_translate_list(item['categoria']))} | Fuente: {item['fuente']}"
        )
    fuentes_block = "\n".join(fuentes)
    return f"{intro}\n\n" + "\n\n".join(entries) + cierre + f"\n\nFuentes:\n{fuentes_block}"


def answer_from_vademecum(user_message: str) -> tuple[str, list[str]]:
    llm, rewrite_chain, vector_store = get_rag_components()
    answer = answer_question(user_message, vector_store, llm, rewrite_chain)
    sources = get_vademecum_sources(user_message)
    return answer, sources


def format_answer_with_sources(answer: str, sources: list[str]) -> str:
    if not sources or "Fuentes consultadas" in answer:
        return answer
    sources_block = "\n".join(f"- {source}" for source in sources)
    return f"{answer}\n\nFuentes consultadas:\n{sources_block}"


def try_answer_from_vademecum(user_message: str) -> tuple[str | None, list[str]]:
    answer, sources = answer_from_vademecum(user_message)
    if answer.strip() == NO_KNOWLEDGE_RESPONSE or not sources:
        return None, []
    return answer, sources


def auxiliar_farmacia_agent(message: str, lat: float | None, lon: float | None) -> AgentResult:
    if is_greeting(message):
        return {
            "answer": (
                "Hola. Soy el Auxiliar de Farmacia. "
                "Puedo ayudarte con farmacias cercanas o derivarte con un farmaceutico "
                "para consultas sobre medicamentos."
            )
        }
    if not is_pharmacy_query(message):
        if is_medicine_query(message):
            return {"handoff": "farmaceutico"}
        if has_vademecum_sources(message):
            return {"handoff": "farmaceutico"}
        return {"handoff": "doctor"}

    if lat is None or lon is None:
        return {
            "answer": "Encargado de dar el saludo inicual y para ayudarte con farmacias cercanas necesito tu ubicacion.",
        }

    farmacias, farmacias_error = get_nearest_pharmacies(lat, lon, top_n=3)
    abiertas, abiertas_error = get_nearest_open_pharmacies(lat, lon, top_n=3)
    if farmacias or abiertas:
        answer = "Estas son las farmacias mas cercanas y las abiertas mas cercanas."
    elif farmacias_error or abiertas_error:
        error_msg = farmacias_error or abiertas_error
        answer = f"No pude leer el cache local de farmacias ({error_msg})."
    else:
        answer = "No pude encontrar farmacias cercanas en el cache local."
    return {
        "answer": answer,
        "farmacias": farmacias,
        "farmacias_abiertas": abiertas,
        "farmacias_error": farmacias_error,
        "farmacias_abiertas_error": abiertas_error,
    }


def farmaceutico_agent(message: str) -> AgentResult:
    if not is_medicine_query(message) and not has_vademecum_sources(message):
        if is_pharmacy_query(message):
            return {"handoff": "auxiliar"}
        return {"handoff": "doctor"}

    scored_results = select_drug_docs(message)
    focus = extract_focus(message)
    answer = format_drug_answer(scored_results, focus)
    sources = list(
        {
            doc.metadata.get("source")
            for doc, score in scored_results
            if score is not None and score >= SCORE_THRESHOLD and doc.metadata.get("source")
        }
    )
    hits = get_vademecum_hits(message)
    if not sources:
        return {
            "answer": (
                f"{NO_KNOWLEDGE_RESPONSE}\n\n"
                "Sugerencia: intenta reformular con el principio activo o el nombre generico."
            ),
            "sources": [],
            "qdrant_hits": hits,
        }
    return {"answer": answer, "sources": sources, "qdrant_hits": hits}


def doctor_agent(message: str) -> AgentResult:
    if is_pharmacy_query(message):
        return {"handoff": "auxiliar"}
    if is_medicine_query(message) or has_vademecum_sources(message):
        return {"handoff": "farmaceutico"}

    system_prompt = (
        "Eres un doctor que responde consultas de salud de forma educativa.\n"
        "Responde con lenguaje claro y comprensible.\n"
        "No realices diagnosticos ni indiques tratamientos personalizados.\n"
        "Niega responder recomendaciones de medicamentos y deriva a profesionales de salud.\n"
        "Explica posibles causas de forma orientativa y aclara que la informacion\n"
        "no sustituye la evaluacion medica profesional.\n"
        "Recomienda explicitamente consultar a un profesional de la salud o especialista."
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=message)]
    response = doctor_llm.invoke(messages)
    return {"answer": response.content}


def route_agent(message: str) -> str:
    return "auxiliar"


def run_swarm(message: str, lat: float | None, lon: float | None) -> AgentResult:
    agent = route_agent(message)
    visited: set[str] = set()
    result: AgentResult = {}
    final_agent = agent

    for _ in range(3):
        if agent == "auxiliar":
            result = auxiliar_farmacia_agent(message, lat, lon)
        elif agent == "farmaceutico":
            result = farmaceutico_agent(message)
        else:
            result = doctor_agent(message)

        handoff = result.get("handoff")
        if not handoff:
            final_agent = agent
            break
        if handoff in visited or handoff == agent:
            final_agent = agent
            break
        visited.add(agent)
        agent = handoff
        final_agent = agent

    result.pop("handoff", None)
    result["agent"] = final_agent
    return result


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
    with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
        checkpointer.setup()
        graph = get_swarm_graph().compile(checkpointer=checkpointer)
        entrada = {"messages": [HumanMessage(content=user_message)], "lat": lat, "lon": lon}
        config = {"configurable": {"thread_id": telefono}}
        resultado = graph.invoke(entrada, config=config)
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
    with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
        checkpointer.setup()
        graph = get_swarm_graph().compile(checkpointer=checkpointer)
        entrada = {"messages": [HumanMessage(content=payload.message)], "lat": payload.lat, "lon": payload.lon}
        config = {"configurable": {"thread_id": payload.telefono}}
        resultado = graph.invoke(entrada, config=config)

    response: dict[str, object] = {
        "status": "ok",
        "answer": resultado["messages"][-1].content if resultado.get("messages") else NO_KNOWLEDGE_RESPONSE,
        "agent": resultado.get("last_agent"),
        "sources": resultado.get("last_sources", []),
        "qdrant_hits": resultado.get("last_qdrant_hits", []),
    }
    if resultado.get("last_farmacias"):
        response["farmacias"] = resultado.get("last_farmacias", [])
        response["farmacias_abiertas"] = resultado.get("last_farmacias_abiertas", [])
        response["farmacias_error"] = resultado.get("last_farmacias_error")
        response["farmacias_abiertas_error"] = resultado.get("last_farmacias_abiertas_error")
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
