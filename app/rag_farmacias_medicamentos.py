from __future__ import annotations

import csv
import json
import os
import sys
import unicodedata
import re
import difflib
from pathlib import Path
from functools import lru_cache
from typing import Annotated, Any, Dict, List, TypedDict, Literal
from datetime import datetime, time
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
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
INTENT_ROUTER_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=80)
LOCAL_TZ = ZoneInfo("America/Santiago")


class IntentOutput(BaseModel):
    intent: Literal[
        "recomendacion_medicamento",
        "consulta_medicamento",
        "farmacias",
        "salud_general",
        "historial",
        "saludo",
        "despedida",
        "fuera_de_dominio",
    ]


class MedListOutput(BaseModel):
    medicamentos: List[str]


def translate_to_spanish(text: str) -> str:
    """Traduce contenido al espanol sin alterar nombres de medicamentos, numeros o fuentes."""
    if not text or not text.strip():
        return text

    markers = ["\n\nFuentes:", "\n\nFuentes consultadas:"]
    body = text
    tail = ""
    for marker in markers:
        if marker in text:
            body, tail = text.split(marker, 1)
            tail = marker + tail
            break

    system_prompt = (
        "Traduce el texto al espanol neutro.\n"
        "No agregues informacion ni cambies el significado.\n"
        "Conserva nombres de medicamentos, dosis, numeros, unidades, marcas y siglas.\n"
        "No traduzcas ni modifiques un bloque de fuentes si aparece."
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=body)]
    translated = llm.invoke(messages).content.strip()
    return f"{translated}{tail}"


def translate_to_spanish_stream(text: str):
    """Stream de traduccion al espanol, preservando nombres y fuentes."""
    if not text or not text.strip():
        yield text
        return

    markers = ["\n\nFuentes:", "\n\nFuentes consultadas:"]
    body = text
    tail = ""
    for marker in markers:
        if marker in text:
            body, tail = text.split(marker, 1)
            tail = marker + tail
            break

    system_prompt = (
        "Traduce el texto al espanol neutro.\n"
        "No agregues informacion ni cambies el significado.\n"
        "Conserva nombres de medicamentos, dosis, numeros, unidades, marcas y siglas.\n"
        "No traduzcas ni modifiques un bloque de fuentes si aparece."
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=body)]
    acc = ""
    for chunk in llm.stream(messages):
        token = getattr(chunk, "content", "") or ""
        if token:
            acc += token
            yield token
    if tail:
        yield tail


def format_agent_prefix(agent: str | None) -> str:
    label_map = {
        "auxiliar": "Auxiliar",
        "farmaceutico": "Farmaceutico",
        "doctor": "Doctor",
    }
    label = label_map.get((agent or "").lower(), "Auxiliar")
    return f"Agente {label}:\n"


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
    historial_consultas: list[dict[str, str]]
    historial_medicamentos: list[dict[str, str]]


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
    auto_keywords = {
        "auto",
        "autos",
        "carro",
        "carros",
        "coche",
        "coches",
        "vehiculo",
        "vehiculos",
    }
    if any(keyword in normalized for keyword in auto_keywords):
        return False
    keywords = [
        "medicamento",
        "medicina",
        "farmaco",
        "farmaco",
        "remedio",
        "salud",
        "dolor",
        "duele",
        "dolencia",
        "hongo",
        "hongos",
        "inflamacion",
        "ganglio",
        "ganglios",
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


def normalize_symptom_text(value: str) -> str:
    """Normaliza texto para sintomas: baja a ascii y reduce letras repetidas."""
    normalized = normalize_key(value)
    normalized = re.sub(r"(.)\1{2,}", r"\1", normalized)
    # Correcciones simples de faltas comunes tras normalizar.
    replacements = {
        "cuelo": "cuello",
        "cueloo": "cuello",
        "hombroo": "hombro",
    }
    for bad, good in replacements.items():
        normalized = normalized.replace(bad, good)
    return normalized


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
    # Alias comunes ES <-> EN para deteccion mas robusta.
    alias = {
        "aspirina": "aspirin",
        "aspirinas": "aspirin",
        "morfina": "morphine",
        "morfine": "morphine",
        "paracetamol": "acetaminophen",
        "acetaminofen": "acetaminophen",
        "ibuprofeno": "ibuprofen",
        "amoxicilina": "amoxicillin",
        "azitromicina": "azithromycin",
        "loratadina": "loratadine",
        "omeprazol": "omeprazole",
        "clonazepam": "clonazepam",
        "sertralina": "sertraline",
        "lisinopril": "lisinopril",
        "losartan": "losartan",
        "albuterol": "albuterol",
        "salbutamol": "albuterol",
        "metformina": "metformin",
        "metformn": "metformin",
        "dexpantenol": "dexpanthenol",
        "dexpantenl": "dexpanthenol",
        "dexclorfeniramina": "dexchlorpheniramine",
        "dexclorfenirm": "dexchlorpheniramine",
        "ketoprofeno": "ketoprofen",
        "ketoprofren": "ketoprofen",
    }
    normalized = {normalize_key(k): normalize_key(v) for k, v in alias.items()}
    for key, target in normalized.items():
        if target in mapping:
            mapping[key] = mapping[target]
    return mapping


def extract_drug_mentions(message: str, limit: int = 5) -> list[str]:
    normalized = normalize_key(message)
    if not normalized:
        return []
    mapping = get_drug_name_map()
    if not mapping:
        return []
    found: list[str] = []
    # Primero, intenta extraer por segmentos (mejor para listas separadas por comas/y).
    lowered = message.lower()
    for filler in (
        "informacion de",
        "información de",
        "informacion sobre",
        "información sobre",
        "info de",
        "info sobre",
        "datos de",
        "datos sobre",
    ):
        lowered = lowered.replace(filler, "")
    segments = re.split(r"[,\n;/]|\b(?:y|e|o)\b|&", lowered)

    keys = list(mapping.keys())

    def _resolve_candidate(candidate: str) -> str | None:
        cand_norm = normalize_key(candidate)
        if not cand_norm:
            return None
        if cand_norm in mapping:
            return mapping[cand_norm]
        for key, display in mapping.items():
            if key and key in cand_norm:
                return display
        matches = difflib.get_close_matches(cand_norm, keys, n=1, cutoff=0.68)
        if matches:
            return mapping.get(matches[0])
        return None

    for segment in segments:
        if len(found) >= limit:
            break
        resolved = _resolve_candidate(segment.strip())
        if resolved and resolved not in found:
            found.append(resolved)
    if len(found) >= limit:
        return found

    # Match literal
    for key, display in mapping.items():
        if key and key in normalized:
            if display not in found:
                found.append(display)
        if len(found) >= limit:
            return found
    # Fuzzy match over tokens and n-grams (mejor cobertura para nombres con typos).
    stopwords = {"de", "del", "la", "el", "y", "o", "para", "por", "con", "sin"}
    tokens = [t for t in normalized.split() if len(t) >= 3 and t not in stopwords]
    keys = list(mapping.keys())

    def _add_match(candidate: str, cutoff: float) -> None:
        if len(found) >= limit:
            return
        matches = difflib.get_close_matches(candidate, keys, n=1, cutoff=cutoff)
        if matches:
            display = mapping.get(matches[0])
            if display and display not in found:
                found.append(display)

    for token in tokens:
        if len(found) >= limit:
            break
        _add_match(token, 0.65)

    if len(found) < limit and len(tokens) >= 2:
        for size in (2, 3, 4):
            if len(found) >= limit:
                break
            for i in range(0, len(tokens) - size + 1):
                phrase = " ".join(tokens[i : i + size])
                _add_match(phrase, 0.7)
                if len(found) >= limit:
                    break
    return found


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
    text = normalize_symptom_text(message)
    if "mancha" in text or "manchas" in text:
        return "manchas en la piel"
    if "inflamacion" in text or "ganglio" in text or "ganglios" in text:
        return "inflamacion"
    if "hongo" in text or "hongos" in text:
        return "hongos"
    if "alergia" in text:
        return "alergias"
    if "dolor" in text or "duele" in text or "dolencia" in text:
        return "dolor"
    if "fiebre" in text or "temperatura" in text:
        return "fiebre"
    if (
        "ojo" in text
        or "ojos" in text
        or "lagrimeo" in text
        or "lagrimea" in text
        or "picor" in text
        or "pica" in text
        or "irritacion" in text
    ):
        return "sintomas oculares"
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


def summarize_topics(topics: list[dict[str, str]] | list[str]) -> str:
    if not topics:
        return "En la conversacion hemos visto temas de salud."
    labels = []
    for item in topics:
        if isinstance(item, str):
            label = item
        else:
            label = item.get("tema", "")
        if label and (not labels or labels[-1] != label):
            labels.append(label)
    recent = labels[-5:]
    joined = ", ".join(recent)
    return (
        "Ademas, hemos hablado de temas de salud como "
        f"{joined}."
    )


def extract_body_zones(text: str) -> list[str]:
    zonas = [
        # cabeza y cara
        "cuero cabelludo", "cabeza", "craneo", "cara", "frente", "sien", "ceja",
        "parpado", "parpados", "pestana", "pestanas",
        "fosa nasal", "fosas nasales", "nariz",
        "labios mayores", "labios menores",
        "boca", "labio", "labios",
        "diente", "dientes", "encia", "encias", "lengua", "paladar", "menton",
        "mandibula", "maxilar", "mejilla", "oreja", "oido", "oidos",
        "ojo", "ojos",
        # cuello y garganta
        "garganta", "faringe", "laringe", "amigdala", "amigdalas",
        "tiroides", "traquea", "nuca", "cuello",
        # tronco anterior
        "esternon", "costillas", "costilla", "pecho", "torax", "mama", "mamas",
        "axilas", "axila",
        "abdomen", "vientre", "ombligo", "estomago", "ingle",
        # tronco posterior
        "columna", "lumbares", "lumbar", "dorsal", "escapulas", "escapula", "espalda",
        # miembros superiores
        "clavicula", "hombro",
        "brazo", "biceps", "triceps",
        "antebrazo", "codo",
        "muneca", "muñeca",
        "palma", "dorso", "nudillos", "nudillo", "mano",
        "dedo del pie", "dedos del pie",
        "dedo", "dedos", "pulgar", "indice", "medio", "anular", "menique",
        # pelvis y genitales
        "cadera", "pelvis", "pubis", "perineo", "perine",
        "testiculos", "testiculo", "escroto", "pene",
        "clitoris", "clítoris", "vulva", "vagina",
        "utero", "útero", "ovarios", "ovario",
        # miembros inferiores
        "muslo", "pierna",
        "rotula", "rótula", "rodilla",
        "pantorrilla", "gemelos", "gemelo",
        "talon", "talón", "tobillo",
        "dedo del pie", "dedos del pie",
        "empeine", "planta", "pie",
    ]
    norm_text = normalize_key(text)
    zonas_norm: list[tuple[str, str]] = []
    for zona in zonas:
        norm = normalize_key(zona)
        if norm:
            zonas_norm.append((zona, norm))
    zonas_norm.sort(key=lambda item: len(item[1]), reverse=True)

    found: list[tuple[str, str]] = []
    for zona, norm in zonas_norm:
        if norm in norm_text:
            if any(norm in existing for _, existing in found):
                continue
            found.append((zona, norm))

    tokens = [t for t in norm_text.split() if len(t) >= 4]
    norm_map = {norm: zona for zona, norm in zonas_norm}
    for token in tokens:
        match = difflib.get_close_matches(token, norm_map.keys(), n=1, cutoff=0.85)
        if match:
            norm = match[0]
            if any(norm in existing or existing in norm for _, existing in found):
                continue
            found.append((norm_map[norm], norm))
    return [zona for zona, _ in found]


def extract_pain_topics(text: str) -> list[str]:
    clauses = re.split(
        r"(?:,|;|\b(?:y tengo|y además|y ademas|y también|y tambien|además|ademas|también|tambien|pero)\b)",
        text,
    )
    topics: list[str] = []
    for clause in clauses:
        if not clause.strip():
            continue
        if "dolor" in clause or "duele" in clause or "dolencia" in clause:
            zonas = extract_body_zones(clause)
            if zonas:
                joined = " y ".join(zonas)
                topics.append(f"dolor de {joined}")
            else:
                topics.append("dolor")
    return topics


def extract_health_topic_detail(message: str) -> str:
    text = normalize_symptom_text(message)
    if "mancha" in text or "manchas" in text:
        if "piel" in text:
            return "manchas en la piel"
        return "manchas"
    if "inflamacion" in text and ("ganglio" in text or "ganglios" in text):
        return "inflamación de ganglio"
    if "hongo" in text or "hongos" in text:
        if "pie" in text or "pies" in text:
            return "hongos en los pies"
        if "uña" in text or "unas" in text or "uñas" in text:
            return "hongos en las uñas"
        return "hongos"
    if "dolor" in text or "duele" in text or "dolencia" in text:
        zonas_encontradas = extract_body_zones(text)
        if zonas_encontradas:
            if len(zonas_encontradas) == 1:
                return f"dolor de {zonas_encontradas[0]}"
            joined = " y ".join(zonas_encontradas)
            return f"dolor de {joined}"
        return "dolor"
    topic = classify_health_topic(message)
    if topic == "consultas de salud":
        llm_topic = rewrite_symptom_with_llm(message)
        return llm_topic
    return topic


def extract_health_topics(message: str) -> list[str]:
    """Extrae multiples temas de salud de un mismo mensaje."""
    text = normalize_symptom_text(message)
    topics: list[str] = []

    if "mancha" in text or "manchas" in text:
        zonas = extract_body_zones(text)
        if zonas:
            topics.append(f"manchas en la piel en {zonas[0]}")
        elif "piel" in text:
            topics.append("manchas en la piel")
        else:
            topics.append("manchas")

    if "inflamacion" in text:
        if "ganglio" in text or "ganglios" in text:
            topics.append("inflamación de ganglio")
        else:
            zonas = extract_body_zones(text)
            if zonas:
                topics.append(f"inflamación en {zonas[0]}")
            else:
                topics.append("inflamación")

    if "hongo" in text or "hongos" in text:
        if "pie" in text or "pies" in text:
            topics.append("hongos en los pies")
        elif "uña" in text or "unas" in text or "uñas" in text:
            topics.append("hongos en las uñas")
        else:
            topics.append("hongos")

    if "moreton" in text or "hematoma" in text:
        zonas = extract_body_zones(text)
        if zonas:
            topics.append(f"moretón en {zonas[0]}")
        else:
            topics.append("moretón")

    if "dolor" in text or "duele" in text or "dolencia" in text:
        pain_topics = extract_pain_topics(text)
        if pain_topics:
            topics.extend(pain_topics)
        else:
            topics.append(extract_health_topic_detail(message))

    if not topics:
        topic = classify_health_topic(message)
        if topic == "consultas de salud":
            llm_topic = rewrite_symptom_with_llm(message)
            if llm_topic:
                topics.append(llm_topic)
        else:
            topics.append(topic)

    # dedup mantener orden
    deduped: list[str] = []
    for t in topics:
        if t and t not in deduped:
            deduped.append(t)
    return deduped


def rewrite_symptom_with_llm(message: str) -> str:
    """Reescribe un sintoma en frase corta y clara (sin recomendaciones)."""
    if not message.strip():
        return ""
    system_prompt = (
        "Reescribe el sintoma en una frase corta y clara en español.\n"
        "No des recomendaciones ni medicamentos. Solo el sintoma.\n"
        "Responde con maximo 6 palabras."
    )
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=message)])
    text = (response.content or "").strip().lower()
    # Limpieza simple
    text = text.replace(".", "").replace(":", "").strip()
    # Evita respuestas genericas/no sintoma
    blocked = [
        "no puedo",
        "no puedo proporcionar",
        "no puedo entregar",
        "no puedo recomendar",
        "consulta a un profesional",
    ]
    if any(b in text for b in blocked):
        return ""
    return text[:80]


def summarize_symptoms_with_llm(message: str) -> str:
    """Resume sintomas en una sola frase breve."""
    if not message.strip():
        return ""
    system_prompt = (
        "Resume los sintomas en una sola frase breve en español.\n"
        "Maximo 10 palabras. Solo sintomas, sin recomendaciones ni tratamientos.\n"
        "Si hay varios sintomas, combinalos en una sola frase."
    )
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=message)])
    text = (response.content or "").strip().lower()
    text = text.replace(".", "").replace(":", "").strip()
    blocked = [
        "no puedo",
        "no puedo proporcionar",
        "no puedo entregar",
        "no puedo recomendar",
        "consulta a un profesional",
    ]
    if any(b in text for b in blocked):
        return ""
    return text[:120]


def now_local_time_str() -> str:
    return datetime.now(LOCAL_TZ).strftime("%d-%m-%Y %I:%M %p").lower()


@lru_cache(maxsize=1)
def get_med_list_extractor():
    system_prompt = (
        "Extrae los nombres de medicamentos mencionados en el texto.\n"
        "Pueden estar con errores de ortografia. Devuelve SOLO una lista.\n"
        "Si conoces el nombre generico correcto, puedes devolverlo.\n"
        "No incluyas palabras generales como 'medicamentos' o 'informacion'."
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{text}")]
    )
    structured = llm.with_structured_output(MedListOutput)
    return prompt | structured


def extract_meds_with_llm(message: str) -> list[str]:
    if not message.strip():
        return []
    try:
        response = get_med_list_extractor().invoke({"text": message})
    except Exception:
        return []
    meds = response.medicamentos if response else []
    cleaned: list[str] = []
    blacklist = {
        "medicamento",
        "medicamentos",
        "informacion",
        "información",
        "datos",
        "dame",
        "quiero",
        "consulta",
    }
    for med in meds:
        if not isinstance(med, str):
            continue
        item = med.strip()
        if not item:
            continue
        if normalize_key(item) in blacklist:
            continue
        if len(item) > 60:
            continue
        if item not in cleaned:
            cleaned.append(item)
    return cleaned


def resolve_med_queries(message: str) -> tuple[list[str], list[str]]:
    """Resuelve lista de medicamentos desde texto y reporta no reconocidos."""
    mentions = extract_drug_mentions(message)
    if len(mentions) >= 2:
        return mentions, []
    llm_terms = extract_meds_with_llm(message)
    if not llm_terms:
        return mentions, []
    mapping = get_drug_name_map()
    keys = list(mapping.keys())
    resolved: list[str] = []
    unknown: list[str] = []
    for term in llm_terms:
        key = normalize_key(term)
        mapped = mapping.get(key)
        if not mapped:
            matches = difflib.get_close_matches(key, keys, n=1, cutoff=0.78)
            if matches:
                mapped = mapping.get(matches[0])
        if mapped:
            resolved.append(mapped)
        else:
            unknown.append(term)
    deduped: list[str] = []
    for item in mentions + resolved:
        if item and item not in deduped:
            deduped.append(item)
    return deduped, unknown


def swarm_node(state: EstadoPersonalizado) -> EstadoPersonalizado:
    user_message = ""
    if state.get("messages"):
        user_message = state["messages"][-1].content
    lat = state.get("lat")
    lon = state.get("lon")
    intent = detect_intent(user_message) if user_message else "fuera_de_dominio"

    nombre, intereses, sesion, preguntas = update_profile(state, user_message)
    historial_consultas = list(state.get("historial_consultas", []))
    historial_medicamentos = list(state.get("historial_medicamentos", []))
    if historial_medicamentos and isinstance(historial_medicamentos[0], str):
        historial_medicamentos = [
            {"medicamento": item, "fecha": ""}
            for item in historial_medicamentos
            if isinstance(item, str) and item
        ]
    if historial_consultas and isinstance(historial_consultas[0], str):
        historial_consultas = [
            {"tema": item, "fecha": ""}
            for item in historial_consultas
            if isinstance(item, str) and item
        ]
    if user_message and intent == "salud_general":
        summary = summarize_symptoms_with_llm(user_message)
        if not summary:
            topics = extract_health_topics(user_message)
            summary = " y ".join(topics) if topics else ""
        if summary:
            entry = {
                "tema": summary,
                "fecha": now_local_time_str(),
            }
            if not historial_consultas or historial_consultas[-1].get("tema") != summary:
                historial_consultas.append(entry)
    wants_history = is_interest_query(user_message) or is_history_intent_llm(user_message)
    is_medicine_like = is_medicine_query(user_message) or has_vademecum_sources(user_message)
    if wants_history and not (is_pharmacy_query(user_message) or is_medicine_like):
        historial_limpio = clean_interests(intereses)
        historial = ", ".join(historial_limpio) if historial_limpio else "ninguno"
        resumen = summarize_topics(historial_consultas)
        temas = "ninguno"
        if historial_consultas:
            ultimos = historial_consultas[-5:]
            temas = ", ".join(
                f"{item.get('tema', '')} ({item.get('fecha', '')})" for item in ultimos
            )
        answer = (
            "Hasta ahora tengo registrado lo siguiente:\n"
            f"- Medicamentos consultados: {historial}\n"
            f"- Temas de salud: {temas}\n\n"
            "Si quieres, puedo profundizar en cualquiera de ellos o resumir con más detalle."
        )
        answer = f"{format_agent_prefix('auxiliar')}{answer}"
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
            "historial_medicamentos": historial_medicamentos,
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
                    historial_medicamentos.append(
                        {
                            "medicamento": nombre,
                            "fecha": now_local_time_str(),
                        }
                    )
    answer = result.get("answer", NO_KNOWLEDGE_RESPONSE)
    answer = f"{format_agent_prefix(result.get('agent'))}{answer}"

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
        "historial_medicamentos": historial_medicamentos,
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

FAREWELL_KEYWORDS = {
    "chao",
    "chau",
    "adios",
    "adiós",
    "hasta luego",
    "hasta pronto",
    "nos vemos",
    "gracias",
    "muchas gracias",
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

MED_RECOMMENDATION_KEYWORDS = {
    "recomienda",
    "recomendacion",
    "recomendación",
    "sugerencia",
    "sugerir",
    "que me recomiendas",
    "que me recomiende",
    "que deberia tomar",
    "que debería tomar",
    "que puedo tomar",
    "que puedo tomar para",
    "que medicamento tomo",
    "que medicamento tomar",
    "que remedio tomo",
    "que remedio tomar",
    "me puedes recomendar",
    "receta",
    "recetas",
    "recetar",
    "prescribir",
    "prescripcion",
    "prescripción",
    "indicar",
    "indicame",
    "indícame",
    "recomendar",
    "recomendame",
    "recomiéndame",
    "recomiendas",
    "recomiendan",
    "sugiereme",
    "sugiéreme",
    "sugieres",
    "me conviene",
    "me convendria",
    "me convendría",
    "me aconsejas",
    "aconsejame",
    "aconsejar",
    "dime que tomar",
    "dime que debo tomar",
    "dime que me tomo",
    "que me tomo",
    "que debo tomar",
    "que debería tomar",
    "que puedo tomar",
    "que puedo tomar para",
    "que puedo tomar si",
    "que pastilla",
    "que pastillas",
    "cual pastilla",
    "cual pastillas",
    "cual medicamento",
    "cual medicamente",
    "cual remedio",
    "cual tratamiento",
    "dime un medicamento",
    "dame un medicamento",
    "dame una receta",
    "dame una recomendacion",
    "dame una recomendación",
    "indica",
    "indíca",
    "indícame",
    "indicacion",
    "indicación",
    "prescribe",
    "prescribeme",
    "prescríbeme",
    "recetame",
    "recétame",
    "necesito un medicamento",
    "necesito una receta",
    "quiero una receta",
    "quiero que me recomiendes",
    "quiero que me recomiende",
    "tengo sintomas que tomo",
    "tengo sintomas que debería tomar",
    "tengo sintomas que puedo tomar",
    "orientame",
    "oriéntame",
    "orienta",
    "orientación",
    "orientacion",
}

MED_RECOMMENDATION_RESPONSE = (
    "Gracias por tu consulta. No puedo entregar recomendaciones, indicaciones ni sugerencias "
    "sobre el uso de medicamentos, ya que eso debe ser evaluado por un profesional de la "
    "salud considerando tu situación particular. Para recibir una orientación adecuada y segura, "
    "te recomiendo consultar con un médico, químico farmacéutico u otro profesional de la salud "
    "autorizado.Si presentas síntomas persistentes, empeoran con el tiempo o te generan preocupación, "
    "es importante que acudas a un centro de salud."
)

NON_MEDICAL_RESPONSE = (
    "Gracias por tu consulta. Este asistente solo responde temas médicos, de medicamentos o de "
    "farmacias. Si tienes otra pregunta fuera de esos temas, no podré ayudarte. Si necesitas "
    "orientación de salud, consulta a un profesional."
)

FAREWELL_RESPONSE = (
    "Gracias por tu consulta. Si necesitas algo más, aquí estaré."
)


def is_med_recommendation_query(message: str) -> bool:
    normalized = message.lower()
    if any(keyword in normalized for keyword in MED_RECOMMENDATION_KEYWORDS):
        return True
    # Regla adicional: si menciona medicamento/remedio/pastilla + verbos de tomar/recomendar/recetar/indicar.
    meds_terms = {"medicamento", "medicamentos", "medicamente", "remedio", "remedios", "pastilla", "pastillas"}
    verbs = {
        "tomar",
        "tomo",
        "tomaria",
        "tomaría",
        "recetar",
        "receta",
        "prescribir",
        "prescribe",
        "indicar",
        "indica",
        "recomendar",
        "recomienda",
        "sugerir",
        "sugiere",
        "orientar",
        "orienta",
        "orientame",
        "oriéntame",
    }
    if any(term in normalized for term in meds_terms) and any(verb in normalized for verb in verbs):
        return True
    return False


@lru_cache(maxsize=1)
def get_intent_router():
    system_prompt = (
        "Clasifica la intencion del usuario en una de estas etiquetas estrictas:\n"
        "- recomendacion_medicamento (pide recomendar/recetar/indicar que tomar)\n"
        "- consulta_medicamento (pide informacion sobre un medicamento)\n"
        "- farmacias (busca farmacias cercanas/de turno)\n"
        "- salud_general (sintomas o preguntas de salud sin pedir medicamento)\n"
        "- historial (ficha medica, historial, resumen)\n"
        "- saludo (saludos)\n"
        "- despedida (despedidas o agradecimientos)\n"
        "- fuera_de_dominio (no relacionado a salud/medicamentos/farmacias)\n"
        "Devuelve solo la etiqueta."
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{text}")]
    )
    structured = INTENT_ROUTER_LLM.with_structured_output(IntentOutput)
    return prompt | structured


def detect_intent(message: str) -> str:
    normalized = message.strip().lower()
    if not normalized:
        return "fuera_de_dominio"
    # Normaliza el texto para no confundir farmacias con salud general.
    normalized_key = normalize_key(message)
    if "farmacia" in normalized_key or "farmacias" in normalized_key:
        return "farmacias"
    # Reglas de mayor prioridad.
    if is_med_recommendation_query(message):
        return "recomendacion_medicamento"
    if is_greeting(message):
        return "saludo"
    if is_farewell(message):
        return "despedida"
    if is_history_query(message):
        return "historial"
    if is_pharmacy_query(message):
        return "farmacias"
    if is_medicine_query(message) or has_vademecum_sources(message):
        return "consulta_medicamento"
    if is_health_related(message):
        return "salud_general"

    # Si no hay señales claras, usa LLM para clasificar.
    try:
        router = get_intent_router()
        result = router.invoke({"text": message})
        return result.intent
    except Exception:
        return "fuera_de_dominio"


HISTORY_KEYWORDS = {
    "historial",
    "ficha medica",
    "ficha médica",
    "resumen",
    "historia clinica",
    "historia clínica",
    "mi ficha",
    "mi historial",
    "resumen de mi ficha",
    "resumen de mi ficha medica",
    "resumen de mi ficha médica",
}

def is_history_query(message: str) -> bool:
    normalized = message.lower()
    return any(keyword in normalized for keyword in HISTORY_KEYWORDS)

def is_allowed_query(message: str) -> bool:
    return (
        is_history_query(message)
        or is_pharmacy_query(message)
        or is_medicine_query(message)
        or has_vademecum_sources(message)
        or is_health_related(message)
    )


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


def is_farewell(message: str) -> bool:
    normalized = message.lower().strip()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in FAREWELL_KEYWORDS)


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


def format_drug_answer(
    scored_results: list[tuple[Any, float | None]],
    focus: set[str],
    *,
    include_intro: bool = True,
) -> str:
    def _join_unique(values: list[str]) -> str:
        cleaned = [v for v in values if v and v != "N/A"]
        return ", ".join(sorted(set(cleaned))) if cleaned else "N/A"

    def _translate_value(value: str) -> str:
        translations = {
            "hypertension": "hipertension",
            "anxiety": "ansiedad",
            "bacterial infections": "infecciones bacterianas",
            "tablet": "tabletas",
            "tablets": "tabletas",
            "capsule": "capsula",
            "capsules": "capsulas",
            "oral": "oral",
            "room temperature": "temperatura ambiente",
            "prescription": "bajo receta",
            "category a": "categoria A",
            "category b": "categoria B",
            "category c": "categoria C",
            "category d": "categoria D",
            "antibiotic": "antibiotico",
            "combination antibiotic": "antibiotico combinado",
            "inhibits bacterial cell wall synthesis": "inhibe la sintesis de la pared celular bacteriana",
            "nausea": "nauseas",
            "allergic reactions": "reacciones alergicas",
            "penicillin: reduced efficacy": "penicilina: eficacia reducida",
            "take with food to reduce stomach upset": "tomar con alimentos para reducir el malestar estomacal",
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
        has_period = normalized.endswith(".")
        normalized_key = normalized.rstrip(".")
        translated = translations.get(normalized_key)
        if translated is None:
            return value
        return f"{translated}." if has_period else translated

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
        entries.append(
            {
                "titulo": item["titulo"],
                "forma": presentacion,
                "dosis": _spanish_defaults(_join_unique(item["dosis"]), "dosis no especificadas"),
                "detalle": "\n".join([base_line, " ".join(details)]),
            }
        )
    if not entries:
        return NO_KNOWLEDGE_RESPONSE
    intro = "Te comparto la información encontrada en nuestra base de conocimiento:"
    cierre = "\n\nSi quieres que me enfoque en otro detalle, dime y lo ajusto."
    # Respuesta con un único resultado (el primero).
    cuerpo = entries[0]["detalle"]
    fuentes = []
    first_item = next(iter(grouped.values()))
    fuentes.append(
        f"- Titulo: {first_item['titulo']} | Categoria: "
        f"{_join_unique(_translate_list(first_item['categoria']))} | Fuente: Vademécum"
    )
    fuentes_block = "\n".join(fuentes)
    if include_intro:
        return f"{intro}\n\n{cuerpo}{cierre}\n\nFuentes:\n{fuentes_block}"
    return f"{cuerpo}{cierre}\n\nFuentes:\n{fuentes_block}"


def answer_from_vademecum(user_message: str) -> tuple[str, list[str]]:
    llm, rewrite_chain, vector_store = get_rag_components()
    answer = answer_question(user_message, vector_store, llm, rewrite_chain)
    sources = get_vademecum_sources(user_message)
    return answer, sources


def format_answer_with_sources(answer: str, sources: list[str]) -> str:
    if not sources or "Fuentes consultadas" in answer or "Fuentes:" in answer:
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
    if is_med_recommendation_query(message):
        return {"answer": f"{format_agent_prefix('farmaceutico')}{MED_RECOMMENDATION_RESPONSE}"}
    if not is_medicine_query(message) and not has_vademecum_sources(message):
        if is_pharmacy_query(message):
            return {"handoff": "auxiliar"}
        return {"handoff": "doctor"}

    mentions, unknown_terms = resolve_med_queries(message)
    focus = extract_focus(message)
    answers: list[str] = []
    sources_set: set[str] = set()
    hits: list[dict[str, object]] = []
    missing: list[str] = []
    for term in unknown_terms:
        if term and term not in missing:
            missing.append(term)

    queries = mentions if mentions else [message]
    for query in queries:
        scored_results = select_drug_docs(query)
        has_source = any(
            score is not None and score >= SCORE_THRESHOLD
            for _, score in scored_results
        )
        if not has_source:
            if query not in missing:
                missing.append(query)
            continue
        answer = format_drug_answer(scored_results, focus, include_intro=not answers)
        answer = translate_to_spanish(answer)
        answers.append(answer)
        for doc, score in scored_results:
            if score is not None and score >= SCORE_THRESHOLD:
                source = doc.metadata.get("source")
                if source:
                    sources_set.add(source)
        hits.extend(get_vademecum_hits(query))

    sources = list(sources_set)
    if not sources:
        return {
            "answer": (
                f"{NO_KNOWLEDGE_RESPONSE}\n\n"
                "Sugerencia: intenta reformular con el principio activo o el nombre generico."
            ),
            "sources": [],
            "qdrant_hits": hits,
        }
    final_answer = "\n\n".join(answers)
    if missing:
        faltantes = ", ".join(missing)
        final_answer = f"{final_answer}\n\nNo encontré información para: {faltantes}."
    return {"answer": final_answer, "sources": sources, "qdrant_hits": hits}


def doctor_agent(message: str) -> AgentResult:
    if not is_health_related(message):
        return {"answer": f"{format_agent_prefix('doctor')}{NON_MEDICAL_RESPONSE}"}
    if is_pharmacy_query(message):
        return {"handoff": "auxiliar"}
    if is_medicine_query(message) or has_vademecum_sources(message):
        return {"handoff": "farmaceutico"}

    system_prompt = (
        "Eres un doctor que responde consultas de salud de forma educativa.\n"
        "Responde con lenguaje claro y comprensible y profundiza en sintomas.\n"
        "Estructura la respuesta con: resumen del sintoma, posibles causas generales,\n"
        "signos de alarma y cuando consultar.\n"
        "No realices diagnosticos ni indiques tratamientos personalizados.\n"
        "No recomiendes medicamentos ni dosis; deriva a profesionales de salud.\n"
        "Aclara que la informacion no sustituye una evaluacion medica profesional."
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=message)]
    response = doctor_llm.invoke(messages)
    return {"answer": response.content}


def route_agent(message: str) -> str:
    return "auxiliar"


def run_swarm(message: str, lat: float | None, lon: float | None) -> AgentResult:
    intent = detect_intent(message)
    if intent == "recomendacion_medicamento":
        return {
            "answer": f"{format_agent_prefix('farmaceutico')}{MED_RECOMMENDATION_RESPONSE}",
            "agent": "farmaceutico",
            "sources": [],
            "qdrant_hits": [],
        }
    if intent == "despedida":
        return {
            "answer": f"{format_agent_prefix('auxiliar')}{FAREWELL_RESPONSE}",
            "agent": "auxiliar",
            "sources": [],
            "qdrant_hits": [],
        }
    if intent == "fuera_de_dominio":
        return {
            "answer": f"{format_agent_prefix('doctor')}{NON_MEDICAL_RESPONSE}",
            "agent": "doctor",
            "sources": [],
            "qdrant_hits": [],
        }
    if intent in {"historial", "farmacias", "saludo"}:
        agent = "auxiliar"
    elif intent == "consulta_medicamento":
        agent = "farmaceutico"
    elif intent == "salud_general":
        agent = "doctor"
    else:
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
        # Spanish support
        "lunes": "lunes",
        "martes": "martes",
        "miércoles": "miercoles",
        "miercoles": "miercoles",
        "jueves": "jueves",
        "viernes": "viernes",
        "sábado": "sabado",
        "sabado": "sabado",
        "domingo": "domingo",
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
        # El usuario pidio que las fechas "no prelen" (no filtren).
        # Asumimos que si esta en el archivo del dia, es valido.
        return True

    def parse_time(value: str) -> time | None:
        try:
            parts = [int(part) for part in value.split(":")]
            return time(parts[0], parts[1], parts[2] if len(parts) > 2 else 0)
        except (ValueError, AttributeError):
            return None

    def is_open_now(item: dict[str, Any]) -> bool:
        apertura_str = item.get("funcionamiento_hora_apertura")
        cierre_str = item.get("funcionamiento_hora_cierre")
        
        # Si no hay horario, asumimos abierto para no filtrar por error de datos
        if not apertura_str or not cierre_str:
            return True

        apertura = parse_time(apertura_str)
        cierre = parse_time(cierre_str)
        
        if not apertura or not cierre:
            return True

        current_time = now.time()
        
        # Manejo de horario que cruza la medianoche (ej: 22:00 a 08:00)
        if apertura > cierre:
            return current_time >= apertura or current_time <= cierre
        else:
            return apertura <= current_time <= cierre

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


def _chunk_text(text: str, size: int = 120) -> list[str]:
    if not text:
        return [""]
    return [text[i : i + size] for i in range(0, len(text), size)]


class LocationPayload(BaseModel):
    telefono: str
    message: str
    lat: float | None
    lon: float | None
    accuracy: float | None = None
    persist_only: bool = False


app = FastAPI(title="Neon Persistencia + Geolocalizacion")

app.mount("/css", StaticFiles(directory=STATIC_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=STATIC_DIR / "js"), name="js")
app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/location")
def save_location(payload: LocationPayload) -> dict[str, object]:
    with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
        checkpointer.setup()
        graph = get_swarm_graph().compile(checkpointer=checkpointer)
        entrada = {"messages": [HumanMessage(content=payload.message)], "lat": payload.lat, "lon": payload.lon}
        config = {"configurable": {"thread_id": payload.telefono}}
        resultado = graph.invoke(entrada, config=config)

    if payload.persist_only:
        return {"status": "ok"}

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


@app.post("/api/location/stream")
def save_location_stream(payload: LocationPayload) -> StreamingResponse:
    def event_stream():
        yield f"event: chunk\ndata: {json.dumps({'text': ''}, ensure_ascii=False)}\n\n"
        message = payload.message
        lat = payload.lat
        lon = payload.lon
        intent = detect_intent(message)

        if intent == "historial":
            with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
                checkpointer.setup()
                graph = get_swarm_graph().compile(checkpointer=checkpointer)
                entrada = {"messages": [HumanMessage(content=message)], "lat": lat, "lon": lon}
                config = {"configurable": {"thread_id": payload.telefono}}
                resultado = graph.invoke(entrada, config=config)

            answer = resultado["messages"][-1].content if resultado.get("messages") else NO_KNOWLEDGE_RESPONSE
            response: dict[str, object] = {
                "status": "ok",
                "answer": answer,
                "agent": resultado.get("last_agent") or "auxiliar",
                "sources": resultado.get("last_sources", []),
                "qdrant_hits": resultado.get("last_qdrant_hits", []),
            }
            for chunk in _chunk_text(answer):
                yield f"event: chunk\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            yield f"event: meta\ndata: {json.dumps(response, ensure_ascii=False)}\n\n"
            return

        if intent == "fuera_de_dominio":
            answer = f"{format_agent_prefix('doctor')}{NON_MEDICAL_RESPONSE}"
            response: dict[str, object] = {
                "status": "ok",
                "answer": answer,
                "agent": "doctor",
                "sources": [],
                "qdrant_hits": [],
            }
            for chunk in _chunk_text(answer):
                yield f"event: chunk\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            yield f"event: meta\ndata: {json.dumps(response, ensure_ascii=False)}\n\n"
            return

        if intent == "recomendacion_medicamento":
            answer = f"{format_agent_prefix('farmaceutico')}{MED_RECOMMENDATION_RESPONSE}"
            response: dict[str, object] = {
                "status": "ok",
                "answer": answer,
                "agent": "farmaceutico",
                "sources": [],
                "qdrant_hits": [],
            }
            for chunk in _chunk_text(answer):
                yield f"event: chunk\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            yield f"event: meta\ndata: {json.dumps(response, ensure_ascii=False)}\n\n"
            return

        if intent == "despedida":
            answer = f"{format_agent_prefix('auxiliar')}{FAREWELL_RESPONSE}"
            response: dict[str, object] = {
                "status": "ok",
                "answer": answer,
                "agent": "auxiliar",
                "sources": [],
                "qdrant_hits": [],
            }
            for chunk in _chunk_text(answer):
                yield f"event: chunk\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            yield f"event: meta\ndata: {json.dumps(response, ensure_ascii=False)}\n\n"
            return

        if intent in {"historial", "farmacias", "saludo"}:
            agent = "auxiliar"
        elif intent == "consulta_medicamento":
            agent = "farmaceutico"
        elif intent == "salud_general":
            agent = "doctor"
        else:
            agent = route_agent(message)
        final_agent = agent
        result: AgentResult = {}
        visited: set[str] = set()

        for _ in range(3):
            if agent == "auxiliar":
                result = auxiliar_farmacia_agent(message, lat, lon)
                handoff = result.get("handoff")
                if not handoff:
                    final_agent = "auxiliar"
                    break
                if handoff in visited:
                    final_agent = "auxiliar"
                    break
                visited.add(agent)
                agent = handoff
                final_agent = agent
                continue
            final_agent = agent
            break

        response: dict[str, object] = {
            "status": "ok",
            "answer": "",
            "agent": final_agent,
            "sources": [],
            "qdrant_hits": [],
        }

        if final_agent == "farmaceutico":
            mentions, unknown_terms = resolve_med_queries(message)
            focus = extract_focus(message)
            answers: list[str] = []
            sources_set: set[str] = set()
            hits: list[dict[str, object]] = []
            missing: list[str] = []
            for term in unknown_terms:
                if term and term not in missing:
                    missing.append(term)
            queries = mentions if mentions else [message]

            for query in queries:
                scored_results = select_drug_docs(query)
                has_source = any(
                    score is not None and score >= SCORE_THRESHOLD
                    for _, score in scored_results
                )
                if not has_source:
                    if query not in missing:
                        missing.append(query)
                    continue
                base_answer = format_drug_answer(
                    scored_results, focus, include_intro=not answers
                )
                answers.append(base_answer)
                hits.extend(get_vademecum_hits(query))
                for doc, score in scored_results:
                    if score is not None and score >= SCORE_THRESHOLD:
                        source = doc.metadata.get("source")
                        if source:
                            sources_set.add(source)

            sources = list(sources_set)
            response["sources"] = sources
            response["qdrant_hits"] = hits
            prefix = format_agent_prefix("farmaceutico")
            if not sources:
                fallback = (
                    f"{NO_KNOWLEDGE_RESPONSE}\n\n"
                    "Sugerencia: intenta reformular con el principio activo o el nombre generico."
                )
                for chunk in _chunk_text(prefix + fallback):
                    yield f"event: chunk\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
                response["answer"] = prefix + fallback
                yield f"event: meta\ndata: {json.dumps(response, ensure_ascii=False)}\n\n"
                return

            translated_acc = ""
            for chunk in _chunk_text(prefix):
                yield f"event: chunk\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
                translated_acc += chunk
            combined = "\n\n".join(answers)
            if missing:
                combined = f"{combined}\n\nNo encontré información para: {', '.join(missing)}."
            for token in translate_to_spanish_stream(combined):
                translated_acc += token
                yield f"event: chunk\ndata: {json.dumps({'text': token}, ensure_ascii=False)}\n\n"
            response["answer"] = translated_acc
            yield f"event: meta\ndata: {json.dumps(response, ensure_ascii=False)}\n\n"
            return

        if final_agent == "doctor":
            system_prompt = (
                "Eres un doctor que responde consultas de salud de forma educativa.\n"
                "Responde con lenguaje claro y comprensible y profundiza en sintomas.\n"
                "Estructura la respuesta con: resumen del sintoma, posibles causas generales,\n"
                "signos de alarma y cuando consultar.\n"
                "No realices diagnosticos ni indiques tratamientos personalizados.\n"
                "No recomiendes medicamentos ni dosis; deriva a profesionales de salud.\n"
                "Aclara que la informacion no sustituye una evaluacion medica profesional."
            )
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=message)]
            acc = ""
            prefix = format_agent_prefix("doctor")
            for chunk in _chunk_text(prefix):
                acc += chunk
                yield f"event: chunk\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            for chunk in doctor_llm.stream(messages):
                token = getattr(chunk, "content", "") or ""
                if token:
                    acc += token
                    yield f"event: chunk\ndata: {json.dumps({'text': token}, ensure_ascii=False)}\n\n"
            response["answer"] = acc
            yield f"event: meta\ndata: {json.dumps(response, ensure_ascii=False)}\n\n"
            return

        # Auxiliar o default: respuesta directa.
        answer = result.get("answer") or NO_KNOWLEDGE_RESPONSE
        answer = f"{format_agent_prefix('auxiliar')}{answer}"
        response["answer"] = answer
        response["agent"] = "auxiliar"
        if result.get("farmacias"):
            response["farmacias"] = result.get("farmacias", [])
            response["farmacias_abiertas"] = result.get("farmacias_abiertas", [])
            response["farmacias_error"] = result.get("farmacias_error")
            response["farmacias_abiertas_error"] = result.get("farmacias_abiertas_error")
        for chunk in _chunk_text(answer):
            yield f"event: chunk\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
        yield f"event: meta\ndata: {json.dumps(response, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
