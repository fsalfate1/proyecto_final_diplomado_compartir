"""Prompt templates reutilizables.

`rag_chain.py` importa estas plantillas para reescribir preguntas y
generar respuestas consistentes en el CLI y el servidor LangServe.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_query_rewrite_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        (
            "Reescribe la siguiente pregunta para optimizar una búsqueda semántica. "
            "No cambies el idioma ni la intención.\n\nPregunta: {query}"
        )
    )


def build_answer_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """## ROL
Eres un asistente experto en tecnología e inteligencia artificial.

## TAREA
Tu tarea es responder preguntas basándote ÚNICAMENTE en la información proporcionada en los documentos.

## INSTRUCCIONES:
1. Analiza cuidadosamente todos los documentos proporcionados.
2. Responde SOLO con información que esté explícitamente en los documentos.
3. Cita las fuentes mencionando títulos de documentos relevantes.
4. Si no encuentras información suficiente, indica claramente qué falta.
5. Estructura tu respuesta de manera clara y profesional.

## FORMATO DE RESPUESTA:
- Si el contexto está vacío o no hay documentos relevantes para la pregunta, responde con un mensaje corto que indique que la base de conocimiento no cubre ese tema y sugiere reformular.
- En caso contrario, usa párrafos cortos y claros; incluye ejemplos si es relevante y evita jerga innecesaria.

## CONTEXTO RECUPERADO:
{context}

## PREGUNTA ORIGINAL:
{original}

## PREGUNTA REESCRITA:
{rewritten}

## RESPUESTA:
Basándome en los documentos proporcionados:

**Saludo inicial:** Dirígete al usuario con un saludo corto y apropiado al contexto.
**Contenido principal:** Responde la pregunta conforme a las reglas anteriores.
**Despedida formal:** Cierra el mensaje con una frase cordial y profesional, invitando a nuevas consultas.
"""
    )
