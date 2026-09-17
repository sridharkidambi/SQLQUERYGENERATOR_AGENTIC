# How to run the Brokerage NL→SQL Agentic System

This covers: standing up the database, ingesting the schema into the
hybrid vector+graph store, setting up the local 1-bit intent router, and
running an end-to-end query. See `architecture.md` for the design
rationale behind each step.

## 0. Prerequisites

- PostgreSQL 15+ with the `pgvector` extension available
  (`CREATE EXTENSION vector;` must succeed).
- Python 3.10+
- (Optional but recommended for production) Neo4j — the scaffold ships
  with a `networkx`-based graph store that needs no separate server, so
  you can skip Neo4j entirely for a first run.
- An Anthropic API key for the SQL generation agent (`ANTHROPIC_API_KEY`
  environment variable), OR any other LLM client swapped into
  `SQLGenerationAgent`.
- (Optional, for the real local intent router) a built `bitnet.cpp`
  binary and a 1.58-bit checkpoint. Until you have this, the pipeline
  runs against `KeywordFallbackRouter` automatically — functionally
  weaker but lets you validate everything else first.

```bash
python -m venv .venv && source .venv/bin/activate
pip install psycopg2-binary sentence-transformers networkx sqlglot anthropic
```

## 1. Create the database and load the schema

```bash
createdb brokerage_db
psql brokerage_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql brokerage_db -f schema.sql
```

Verify:
```bash
psql brokerage_db -c "SELECT version_id, description FROM brokerage.schema_version;"
```

## 2. Ingest the schema into the hybrid store

This introspects the live schema (using its `COMMENT ON` text as the
business glossary), embeds table/column descriptions into `pgvector`,
and writes the FK graph to `schema_graph.gpickle`.

```bash
export PGDSN="postgresql://user:pass@localhost:5432/brokerage_db"
python ingestion_pipeline.py --dsn "$PGDSN"
```

You should see output like:
```
Ingesting schema_version 1 (previously None)...
Vector store: 210 chunks upserted (of 210 total).
Graph store written to schema_graph.gpickle: 24 tables, 27 FK edges.
Done.
```

## 3. (Optional) Set up the local 1-bit intent router

1. Build `bitnet.cpp` for your platform (follow the upstream BitNet
   project's build instructions) and obtain/fine-tune a 1.58-bit
   checkpoint trained to emit `SQL_INTENT` / `OUT_OF_SCOPE` for
   brokerage-style requests.
2. Note the paths to the compiled binary and the model file — you'll
   pass them as `--bitnet-binary` / `--bitnet-model` in step 5.

If you skip this step, the orchestrator automatically falls back to
`KeywordFallbackRouter`, which is enough to exercise the rest of the
pipeline end-to-end.

## 4. Keeping the model current (re-ingestion on schema change)

Whenever `schema.sql` changes:
1. Write the change as a new migration, bump `schema_version` with a new
   row (see the pattern at the bottom of `schema.sql`), and make sure any
   new/changed table or column has an up-to-date `COMMENT ON`.
2. Apply the migration: `psql brokerage_db -f migrations/002_xxx.sql`
3. Re-run ingestion — it will detect the version bump and only re-embed
   changed objects:
   ```bash
   python ingestion_pipeline.py --dsn "$PGDSN"
   ```
4. Wire step 2–3 into CI (e.g. a post-merge GitHub Action) so the RAG
   store never drifts from the schema in production.

Use `--full-rebuild` to force a complete re-embed regardless of version
(useful after changing the embedding model).

## 5. Run an end-to-end query

```bash
python agent_orchestrator.py \
  --dsn "$PGDSN" \
  --bitnet-binary /path/to/bitnet-cli \
  --bitnet-model /path/to/model.gguf \
  "Show me all accounts whose margin utilization risk limit is currently breached"
```

Expected output: a single SQL statement against the real
`brokerage.risk_limits` / `brokerage.accounts` tables, e.g.:

```sql
SELECT a.account_number, rl.limit_value, rl.current_value
FROM brokerage.accounts a
JOIN brokerage.risk_limits rl ON rl.account_id = a.account_id
WHERE rl.limit_type = 'MARGIN_UTILIZATION' AND rl.is_breached = TRUE;
```

...optionally followed by an index-suggestion comment if `EXPLAIN`
detected a sequential scan on an unindexed filter column.

For an out-of-scope request:
```bash
python agent_orchestrator.py --dsn "$PGDSN" "Tell me a joke"
```
Output:
```
This request is not supported in this agentic application.
```

## 6. Validate the pipeline against the gold-standard evaluation set

Before trusting generated SQL in production, run it against the
hand-written gold cases in `eval_dataset.jsonl` — see `EVALUATION.md` for
what each metric means and how to read a regression.

```bash
pip install deepeval sqlglot
export OPENAI_API_KEY=...   # DeepEval's default LLM judge
python evaluate.py --dsn "$PGDSN" --dataset eval_dataset.jsonl
```

This checks, per gold case: whether the intent router made the right
in-scope/out-of-scope call (including on the mutation and injection
attempts in the set), whether the retriever pulled the right tables,
whether the generated SQL is valid against the real catalog, and — via an
LLM judge — whether it's semantically equivalent to the hand-verified
reference query. Run with `--skip-llm-judge` for a fast, judge-free loop
while iterating on the retriever or prompts.

## 7. Running the whole thing as a service (next step)

For a production deployment, wrap `Orchestrator.handle()` in a thin API
(FastAPI is a natural fit), keep one long-lived `psycopg2` connection
pool, load the sentence-transformer and (if used) the bitnet.cpp process
once at startup rather than per-request, and put the read-only DB role
described in `architecture.md` §8 in front of the validation agent's
connection specifically.

## Troubleshooting

- **`CREATE EXTENSION vector` fails** — install the `pgvector` package for
  your Postgres version first (`apt install postgresql-15-pgvector` or
  build from source), then retry.
- **`ingestion_pipeline.py` says "nothing to re-ingest" after a schema
  change** — you likely forgot to bump `schema_version`; insert a new row
  there as part of every migration.
- **Generated SQL references a table that doesn't exist** — check that
  the table has a `COMMENT ON TABLE` (untagged tables never enter the
  vector store, by design, so they can't leak into generation) and that
  ingestion has run since the table was added.
- **No index suggestions ever appear** — `_suggest_indexes` only fires on
  `Seq Scan` nodes with a `Filter`; on a freshly loaded, empty database
  the planner may not choose a seq scan at all until there's enough data
  for statistics to matter. Load representative volumes before relying on
  this signal.
