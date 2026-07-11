FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Storage paths default to ../documents and ../vector_stores (see config.py) which
# assumes the monorepo layout; point them inside the image instead since this
# Dockerfile ships backend/ as a standalone build context (Hugging Face Spaces).
ENV DOCUMENTS_DIR=/app/documents
ENV VECTOR_STORES_DIR=/app/vector_stores

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
