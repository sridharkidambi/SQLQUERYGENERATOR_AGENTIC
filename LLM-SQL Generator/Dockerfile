FROM python:3.11-slim

# libpq-dev + gcc: needed to build psycopg2 from source on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# No default long-running process — this image is driven via
# `docker compose exec app <command>` / `docker compose run --rm app <command>`.
CMD ["python", "agent_orchestrator.py", "--help"]
