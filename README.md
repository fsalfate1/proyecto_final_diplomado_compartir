# Pulso: Asistente Inteligente de Salud y Farmacia

> **Versión:** 2.0.0 | **Estado:** Producción (MVP) | **Arquitectura:** Multi-Agente (Swarm)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-High%20Performance-green) ![LangGraph](https://img.shields.io/badge/AI-LangGraph%20Swarm-orange) ![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red) ![Postgres](https://img.shields.io/badge/Persistence-Neon%20Postgres-336791)

---

## 1. Resumen Ejecutivo

**Pulso** es una plataforma de asistencia sanitaria conversacional diseñada para cerrar la brecha entre la información médica compleja y el ciudadano común. A diferencia de los chatbots tradicionales, Pulso opera como un **sistema multi-agente** capaz de gestionar contexto a largo plazo, geolocalizar servicios críticos y proporcionar información farmacológica precisa mediante búsqueda semántica avanzada.

El sistema resuelve tres problemáticas clave:
1.  **Acceso a Medicamentos:** Localización inmediata de farmacias de turno y cercanas según la ubicación real del usuario.
2.  **Educación Farmacológica:** Consultas sobre posología, contraindicaciones y precios de medicamentos (Vademécum) con base en fuentes oficiales.
3.  **Registro de Síntomas:** Un asistente empático que escucha, registra y orienta sobre sintomatología general sin cruzar la línea del diagnóstico médico.

---

## 2. Arquitectura del Sistema

Pulso implementa una arquitectura **Event-Driven basada en Grafos (Graph-Based State Machine)**, orquestada por `LangGraph`. Esto permite un flujo no lineal donde múltiples "agentes" especialistas colaboran para resolver la intención del usuario.

### 2.1 Diagrama de Alto Nivel

```mermaid
graph TD
    User((Usuario)) <-->|HTTPS/JSON| API[FastAPI Gateway]
    API <-->|State State| Graph[Orquestador LangGraph]
    
    subgraph "Enjambre de Agentes (Swarm)"
        Router{Router Inteligente}
        Auxiliar[Agente Auxiliar<br/>(Geo + Contexto)]
        Farmaceutico[Agente Farmacéutico<br/>(RAG Vademécum)]
        Doctor[Agente Doctor<br/>(Triaje Educativo)]
    end
    
    Graph --> Router
    Router --> Auxiliar
    Router --> Farmaceutico
    Router --> Doctor
    
    Auxiliar <-->|Haversine| JSON[(Data Local<br/>Farmacias/Turnos)]
    Farmaceutico <-->|Vector Search| Qdrant[(Qdrant Cloud<br/>Embeddings Medicamentos)]
    Graph <-->|Checkpointing| Postgres[(Neon DB<br/>Memoria Persistente)]
```

### 2.2 Componentes del Enjambre (The Swarm)

El sistema no utiliza un solo LLM genérico, sino roles especializados:

1.  **Agente Auxiliar (The Concierge):**
    *   **Función:** Primer punto de contacto. Maneja el saludo, la intención del usuario y la geolocalización.
    *   **Capacidad:** Realiza cálculos geodésicos (fórmula Haversine) sobre una base de datos local optimizada (`JSON`) para encontrar farmacias en milisegundos sin latencia de API externa.
    *   **Memoria:** Recapitula el historial de conversaciones anteriores y perfiles de interés.

2.  **Agente Farmacéutico (The Specialist):**
    *   **Función:** Experto en medicamentos.
    *   **Tecnología:** Utiliza **RAG (Retrieval-Augmented Generation)**. Transforma la consulta del usuario en vectores (embeddings) y busca similitud semántica en **Qdrant**.
    *   **Lógica:** Implementa "Query Rewriting" para mejorar la precisión y un filtro de palabras clave para evitar alucinaciones en nombres de drogas.

3.  **Agente Doctor (The Advisor):**
    *   **Función:** Orientación de salud general y registro de síntomas.
    *   **Guardrails:** Estrictamente configurado para **NO** diagnosticar. Su rol es educativo, empático y de derivación a especialistas humanos, manteniendo un registro de los "temas de salud" mencionados por el usuario.

---

## 3. Stack Tecnológico

| Capa | Tecnología | Descripción |
| :--- | :--- | :--- |
| **Frontend** | HTML5, JS Vanilla | Interfaz ligera servida estáticamente, centrada en la captura de geolocalización nativa. |
| **Backend API** | **FastAPI** (Python) | API asíncrona de alto rendimiento. |
| **Orquestación AI** | **LangGraph** & **LangChain** | Gestión del grafo de estados y flujo de conversación cíclico. |
| **LLM** | OpenAI **GPT-4o-mini** / **GPT-4o** | Modelos balanceados en costo/latencia/inteligencia. |
| **Vector DB** | **Qdrant** | Base de datos vectorial para búsqueda semántica del Vademécum (`DrugData.csv`). |
| **Persistencia** | **Neon** (PostgreSQL) | Almacenamiento del estado de la conversación (`checkpoints`) para continuidad entre sesiones. |
| **Infraestructura** | **Fly.io** & Docker | Despliegue en contenedores efímeros (Firecracker VMs). |

---

## 4. Seguridad y Privacidad

La seguridad es un pilar fundamental en el diseño de Pulso, dado el contexto sensible (salud).

### 4.1 Manejo de Datos
*   **Sin Almacenamiento de PII Crítico:** El sistema no solicita ni almacena RUT, direcciones exactas de domicilio ni fichas clínicas completas. Solo se persiste el historial de chat vinculado a un identificador (teléfono) para continuidad del servicio.
*   **Sanitización de Inputs:** Todas las entradas de usuario pasan por procesos de normalización (`unicodedata`) para prevenir inyecciones y mejorar la calidad de la búsqueda vectorial.

### 4.2 Infraestructura Segura
*   **Gestión de Secretos:** Las credenciales (OpenAI API Key, Qdrant URL, Database URL) se inyectan estrictamente mediante variables de entorno, nunca hardcodeadas en el código fuente.
*   **Comunicación Encriptada:** Todo el tráfico entre el cliente y el servidor, así como entre el servidor y los servicios cloud (Neon, Qdrant, OpenAI), viaja sobre **HTTPS/TLS**.

### 4.3 Ética de IA (Guardrails)
*   **Prevención de Diagnóstico:** El `System Prompt` del Agente Doctor incluye instrucciones explícitas para denegar solicitudes de diagnóstico médico o prescripción de recetas, derivando siempre a un profesional.

---

## 5. Especificaciones de la API

### Endpoint Principal: `/api/location`
Procesa la interacción del usuario integrando contexto y ubicación.

**Request (POST):**
```json
{
  "telefono": "+56912345678",   // Identificador de sesión (Thread ID)
  "message": "Me duele la cabeza, ¿qué precio tiene el paracetamol?",
  "lat": -33.4372,              // Latitud (WGS84)
  "lon": -70.6506               // Longitud (WGS84)
}
```

**Response:**
```json
{
  "status": "ok",
  "answer": "El paracetamol tiene un precio aprox. de $1.500 CLP. Aquí tienes farmacias cerca...",
  "agent": "farmaceutico",
  "sources": ["Minsal Vademecum 2024"],
  "farmacias_abiertas": [...]
}
```

---

## 6. Instalación y Despliegue Local

### Prerrequisitos
*   Python 3.11+
*   Cuenta en Qdrant (Cloud o Docker local)
*   Instancia de PostgreSQL (Neon o local)

### Pasos
1.  **Clonar y configurar entorno:**
    ```bash
    git clone https://github.com/fsalfate1/proyecto_final_diplomado_compartir.git
    cd pulso
    python -m venv .venv
    source .venv/bin/activate  # o .venv\Scripts\activate en Windows
    pip install -r requirements.txt
    ```

2.  **Configurar Variables de Entorno (`.env`):**
    ```properties
    OPENAI_API_KEY=sk-...
    QDRANT_URL=https://...
    QDRANT_API_KEY=...
    DATABASE_URL=postgresql://user:pass@host/db
    ```

3.  **Ingesta de Datos (ETL):**
    Carga el Vademécum CSV en Qdrant.
    ```bash
    python scripts/rag_data_preparation.py
    ```

4.  **Ejecutar Servidor:**
    ```bash
    PYTHONPATH=.:./src uvicorn app.rag_farmacias_medicamentos:app --reload --port 8001
    ```

---

© 2026 Pulso Project. Desarrollado como proyecto de innovación tecnológica en salud.