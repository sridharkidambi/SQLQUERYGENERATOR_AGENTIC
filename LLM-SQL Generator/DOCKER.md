# Running the Brokerage NL→SQL Agentic System in Docker

This is the containerized equivalent of `RUN.md` — same pipeline, no local
Postgres/Python install needed. Two services: `postgres` (Postgres 16 with
`pgvector` baked in) and `app` (Python 3.11 with the ingestion, orchestrator,
and eval scripts).

## 0. Prerequisites

- Docker Engine + Docker Compose v2 (`docker compose version` should work;
  if you only have the older `docker-compose` binary, substitute that).
- An Anthropic API key (SQL generation agent). An OpenAI key too if you
  want the evaluation harness's LLM judge to work out of the box (see
  `EVALUATION.md` for wiring DeepEval to Claude instead, if you'd rather
  not add a second provider).

Files expected in this directory (all provided): `docker-compose.yml`,
`Dockerfile`, `requirements.txt`, `.env.example`, `init/00_extensions.sql`,
`schema.sql`, `ingestion_pipeline.py`, `agent_orchestrator.py`,
`evaluate.py`, `eval_dataset.jsonl`.

## 1. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum `ANTHROPIC_API_KEY`; set `OPENAI_API_KEY`
too if you'll run the evaluation harness with its default judge. Leave the
Postgres values as-is for local dev, or change them — `docker-compose.yml`
reads all of them from `.env` automatically.

## 2. Start Postgres and load the schema

```bash
docker compose up -d postgres
docker compose logs -f postgres   # watch until "database system is ready to accept connections"
```

`init/00_extensions.sql` and `schema.sql` are mounted into
`/docker-entrypoint-initdb.d/` and run automatically **the first time**
the `pgdata` volume is created — this is what enables `pgvector` and loads
the full `brokerage` schema with no manual step. Verify:

```bash
docker compose exec postgres psql -U brokerage -d brokerage_db \
  -c "SELECT version_id, description FROM brokerage.schema_version;"
```

> Init scripts only run against an **empty** data volume. If you change
> `schema.sql` later, that's a migration (see `RUN.md` §4) applied against
> the running container with `docker compose exec postgres psql ... -f -`,
> not a re-trigger of the init scripts. To force a from-scratch reload
> instead, `docker compose down -v` first (this deletes all data).

## 3. Build the app image

```bash
docker compose build app
```

This installs `sentence-transformers`, `sqlglot`, `anthropic`, `deepeval`,
etc. per `requirements.txt`. The first build takes a few minutes
(`sentence-transformers` pulls in a PyTorch wheel); subsequent builds are
cached unless `requirements.txt` changes.

## 4. Bring the app container up

```bash
docker compose up -d app
```

`app` just idles (`sleep infinity`) so you can run one-off commands against
it without a rebuild each time. Everything below uses `docker compose exec
app ...`.

## 5. Ingest the schema into the hybrid vector + graph store

```bash
docker compose exec app sh -c 'python ingestion_pipeline.py --dsn "$PGDSN"'
```

Expected tail of output:
```
Vector store: 210 chunks upserted (of 210 total).
Graph store written to schema_graph.gpickle: 24 tables, 27 FK edges.
Done.
```

Because `./:/app` is mounted read-write, `schema_graph.gpickle` appears on
your host afterward, not just inside the container.

## 6. Run a query end-to-end

```bash
docker compose exec app sh -c '
python agent_orchestrator.py --dsn "$PGDSN" \
  "Show me all accounts whose margin utilization risk limit is currently breached"
'
```

And the refusal path:
```bash
docker compose exec app sh -c 'python agent_orchestrator.py --dsn "$PGDSN" "Tell me a joke"'
# -> This request is not supported in this agentic application.
```

(The local 1-bit BitNet router isn't containerized here — see the note at
the bottom. Without `--bitnet-binary`/`--bitnet-model`, the orchestrator
falls back to `KeywordFallbackRouter` automatically, same as bare-metal.)

## 7. Run the evaluation harness

```bash
docker compose exec app sh -c '
python evaluate.py --dsn "$PGDSN" --dataset eval_dataset.jsonl
'
```

`eval_report.json` lands on the host in this directory (bind mount), so you
can diff it across runs. For a fast, judge-free loop while iterating:
```bash
docker compose exec app sh -c 'python evaluate.py --dsn "$PGDSN" --skip-llm-judge'
```

## 8. Re-ingesting after a schema change

```bash
# apply the migration against the running container
docker compose exec -T postgres psql -U brokerage -d brokerage_db < migrations/002_xxx.sql
# re-embed only what changed
docker compose exec app sh -c 'python ingestion_pipeline.py --dsn "$PGDSN"'
```

## 9. Tear down

```bash
docker compose down          # stop containers, keep the pgdata volume
docker compose down -v       # also delete the volume (full reset)
```

## Optional: containerizing the local 1-bit BitNet router

Not included in `docker-compose.yml` — building `bitnet.cpp` needs a
build toolchain and a checkpoint you supply yourself (see `RUN.md` §3), so
it's a genuine extra step rather than something that can ship generically
in this compose file. Sketch for when you're ready:

```dockerfile
# Dockerfile.bitnet
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/microsoft/BitNet.git /bitnet
WORKDIR /bitnet
RUN pip install -r requirements.txt && python setup_env.py --hf-repo <your-1.58bit-checkpoint> -q i2_s
```

Add it to `docker-compose.yml` as a service exposing the compiled binary +
model on a shared volume, then point `app`'s `agent_orchestrator.py`
invocation at it with `--bitnet-binary /shared/bitnet-cli --bitnet-model
/shared/model.gguf`. Until then, `KeywordFallbackRouter` is what actually
runs in this container setup — treat it as a placeholder, not a finished
intent classifier, per the caveat already in `architecture.md`.
