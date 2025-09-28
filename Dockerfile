# ===============================================================
# BASE IMAGE
# ===============================================================
FROM python:3.11-slim

WORKDIR /app

# Cache lokasi model agar bisa dipersist via volume
ENV HF_HOME=/app/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/hf_cache
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ===============================================================
# SYSTEM DEPS (untuk pg/psycopg2, hdbscan/bertopic build wheels)
# ===============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ===============================================================
# PYTHON DEPS
# ===============================================================
COPY requirements.txt .
# Torch CPU terlebih dahulu
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ===============================================================
# (Opsional) "Bake" model saat build – bisa dilewati jika pakai volume cache
# ===============================================================
COPY ./scripts/download_models.py ./download_models.py
RUN python download_models.py || true

# ===============================================================
# COPY SOURCE
# ===============================================================
COPY ./backend /app/backend
COPY ./frontend /app/frontend
COPY ./data /app/data
COPY ./config.yaml /app/config.yaml

EXPOSE 8000 8601

# Default cmd hanya placeholder — actual command ditentukan di docker-compose
CMD ["bash", "-lc", "echo 'Image built successfully. Use docker-compose up'"]
