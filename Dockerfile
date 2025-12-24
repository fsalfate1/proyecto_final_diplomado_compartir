FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_modulo3 ./rag_modulo3
COPY app ./app
COPY static ./static
COPY rag_cli.py ./rag_cli.py
COPY rag_data_preparation.py ./rag_data_preparation.py
COPY pdf ./pdf

EXPOSE 8080

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080"]
