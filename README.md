# RAG Farmacias y Medicamentos (CSV + Qdrant)

Proyecto de chatbot con persistencia por usuario (telefono), consulta a farmacias cercanas y respuestas sobre medicamentos usando una base vectorial en Qdrant cargada desde `excel/DrugData.csv`.

## Estructura del proyecto

- `app/rag_farmacias_medicamentos.py`: API FastAPI y logica del swarm (Auxiliar/Farmaceutico/Doctor).
- `src/rag_modulo3/`: modulo RAG (configuracion, preparacion, prompts, cadena).
- `scripts/rag_data_preparation.py`: crea la coleccion en Qdrant desde el CSV.
- `excel/DrugData.csv`: base de conocimiento en CSV.
- `notebooks/rag_exploration_data.ipynb`: exploracion del CSV y pruebas de Qdrant.
- `static/geoloc.html`: frontend simple con geolocalizacion.
- `data/`: JSON de farmacias.

## Requisitos

- Python 3.11+
- Qdrant accesible (URL + API key)
- Base Postgres/Neon para persistencia

## Instalacion

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuracion (.env)

Define estas variables:

```
OPENAI_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
DATABASE_URL=...
```

Opcionales:

```
RAG_COLLECTION_NAME=csv_vademecum
RAG_DRUG_CSV=excel/DrugData.csv
```

## Cargar base vectorial (Qdrant)

Esto borra y recrea la coleccion:

```bash
python3 scripts/rag_data_preparation.py
```

## Ejecutar API

```bash
PYTHONPATH=.:./src uvicorn app.rag_farmacias_medicamentos:app --host 0.0.0.0 --port 8001 --reload
```

Tambien puedes ejecutar:

```bash
python3 app/rag_farmacias_medicamentos.py
```

## Endpoints

- `GET /` -> pagina con geolocalizacion.
- `POST /api/location` -> consulta principal.

Payload:

```json
{
  "telefono": "+56912345678",
  "message": "dame el precio de Clonazepam",
  "lat": -33.45,
  "lon": -70.66,
  "accuracy": 10
}
```

Respuesta incluye:

- `answer`: texto del agente.
- `agent`: auxiliar/farmaceutico/doctor.
- `sources`: fuente(s) desde Qdrant.
- `qdrant_hits`: ids + score (si aplica).
- `farmacias` y `farmacias_abiertas` (si aplica).

## Como funciona el swarm

- **Auxiliar**: saludo, farmacias, historial de conversacion.
- **Farmaceutico**: responde medicamentos solo desde Qdrant/CSV.
- **Doctor**: respuestas de salud generales, educativas, sin diagnostico.

Persistencia por usuario: se usa `telefono` como `thread_id` en Postgres.

Diagrama de flujo (Mermaid):

```mermaid
flowchart TD
  A[Usuario envia mensaje + telefono (+ lat/lon)] --> B[Persistencia por telefono<br/>PostgresSaver]
  B --> C[swarm_node]
  C --> D{Actualizar historial?<br/>is_health_related}
  D -->|Si| D1[Guardar tema salud (ej. dolor de codo)]
  D -->|No| E[Continuar]

  C --> F{Historial solicitado?<br/>LLM router + reglas}
  F -->|Si| G[Auxiliar responde historial<br/>- Medicamentos consultados<br/>- Temas de salud]
  F -->|No| H[Run Swarm]

  H --> I[Router inicial: Auxiliar]
  I --> J{Consulta farmacia?}
  J -->|Si| K[Auxiliar: farmacias cercanas/abiertas]
  J -->|No| L{Consulta medicamento o match Qdrant?}
  L -->|Si| M[Farmaceutico: responde con CSV/Qdrant<br/>+ qdrant_hits]
  L -->|No| N[Doctor: salud general educativa]

  M --> O[Respuesta + fuentes Qdrant]
  K --> O
  N --> O
  G --> O
  O --> P[Guardar estado + responder]
```

## Historial por usuario

El bot guarda:

- Medicamentos consultados.
- Temas de salud (resumidos).

Si preguntas algo como:

- "que hemos hablado?"
- "dame mi historial"

Responde el Auxiliar con el resumen.

## Notebooks

Abre `notebooks/rag_exploration_data.ipynb` y ejecuta desde la primera celda.

Si tu notebook esta en `notebooks/`, la celda inicial ya resuelve rutas.

## Troubleshooting

### No obtiene ubicacion

Si ves `Position update is unavailable`, usa:

```
http://127.0.0.1:8001
```

Y permite geolocalizacion en el navegador.

### Qdrant sin datos

Recrea la base:

```bash
rm -f .rag_cache.json
python3 scripts/rag_data_preparation.py
```

### No encuentra `rag_modulo3`

Ejecuta con:

```bash
PYTHONPATH=.:./src uvicorn app.rag_farmacias_medicamentos:app --host 0.0.0.0 --port 8001 --reload
```

## Notas

- Los precios se muestran en USD y CLP (aprox 1 USD = 900 CLP).
- NDC se omite en respuestas por solicitud.
