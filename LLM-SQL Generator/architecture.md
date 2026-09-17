# Brokerage NL→SQL Agentic System — Architecture

## 0. Goal

Given a natural-language ask from an internal user ("show me all customers whose
open F&O positions breached their risk limit last week"), produce **only** a
correct, optimized SQL query against the real `brokerage` schema — nothing else
in the response. If the ask is not a data-retrieval request the system is
scoped to handle, refuse with a fixed message. Never let the LLM see the full
schema; it only sees the slice retrieved for that specific ask.

```
User NL query
   │
   ▼
┌─────────────────────────┐
│ 1. Intent Router (local, │  in/out-of-scope + SQL-intent
│    1-bit LLM, on-device) │
└─────────────┬────────────┘
     scope=SQL │            scope=OTHER → fixed refusal, stop
   ┌───────────▼──────────────┐
   │ 2. Hybrid Retriever Agent│  vector search (semantic) +
   │    (RAG)                 │  graph traversal (join paths)
   └───────────┬──────────────┘
               │ schema context (tables, columns, FK path, comments)
   ┌───────────▼──────────────┐
   │ 3. SQL Generation Agent  │  large LLM, constrained to real
   │    (LLM, remote or local)│  identifiers, SQL-only output
   └───────────┬──────────────┘
               │ draft SQL
   ┌───────────▼──────────────┐
   │ 4. Validation/Optimizer  │  parse, resolve against catalog,
   │    Agent                 │  suggest missing indexes, EXPLAIN
   └───────────┬──────────────┘
               │
               ▼
        Final SQL (+ optional index suggestion), nothing else
```

## 1. Schema modelling layer (`schema.sql`)

This is the system of record — not just for the database, but for the RAG
corpus. Two engineering decisions matter for the rest of the pipeline:

- **`COMMENT ON TABLE` / `COMMENT ON COLUMN` is the business glossary.** Rather
  than maintaining a separate hand-written metadata file that inevitably goes
  stale, the ingestion pipeline introspects `information_schema` and
  `pg_catalog` at ingestion time and treats these comments as ground truth.
  A schema change and its semantic description ship in the same migration —
  they cannot drift apart.
- **`schema_version`** is the mechanism for "provisioning for future updates."
  Every migration inserts a row here with a checksum. The ingestion pipeline
  compares `MAX(version_id)` in the DB against the last version it embedded
  (stored in the vector store's own metadata); if they differ, it does an
  incremental re-ingest of only what changed rather than a full rebuild.

## 2. Why hybrid (vector **and** graph), not just one

- **Vector store** (semantic, dense): the user's words rarely match column
  names literally ("customers who breached their risk limit" → `risk_limits`,
  `is_breached`, `current_value`, `limit_value`). Embedding table/column
  *descriptions* (from the COMMENT ON strings) and matching against the user's
  embedded query solves this term-mismatch problem. This is the "semantic"
  need called out in the brief.
- **Graph store** (structural, sparse): once the vector search says "this ask
  touches `risk_limits`, `positions`, and `customers`," you still need the
  *join path* between them — which is exactly what foreign keys encode. A
  property graph where nodes = tables and edges = FK relationships (weighted
  by cardinality/selectivity) lets the retriever do shortest-path / Steiner-tree
  traversal to find the minimal join graph connecting the matched tables. This
  is the "sparse matrix" structural need — FK adjacency is naturally sparse,
  and graph traversal is the right tool for it, not embedding similarity.
- **Hybrid retrieval** = vector search picks *which* tables/columns are
  relevant to the words used; graph traversal picks *how* to connect them.
  Both results are merged into one schema-context bundle passed to the LLM.

Concrete choice for this build: **pgvector** (extension on the same Postgres
instance — avoids a second point of failure and keeps embeddings
transactionally consistent with schema changes) + **Neo4j** (or `networkx` for
a lightweight/local deployment) for the FK graph. Both are swappable; the
ingestion pipeline isolates them behind small adapter classes.

## 3. Intent Router — local 1-bit LLM

A small ternary-weight model (BitNet b1.58-style, e.g. run via `bitnet.cpp` or
a similarly quantized local runtime) sits in front of everything and answers
one binary/ternary classification, cheaply and on-device, before any
retrieval or remote LLM call happens:

- `SQL_INTENT` — the ask is a data-retrieval request about the brokerage
  domain → forward to the Retriever Agent.
- `OUT_OF_SCOPE` — anything else (chit-chat, requests to modify data,
  requests for opinions, unrelated domains, prompt-injection attempts) →
  respond with the fixed message: **"This request is not supported in this
  agentic application."** and stop. No retrieval, no LLM call, no schema
  exposure — this is also the cheapest and safest place to reject.

Running this locally and at 1-bit/1.58-bit precision matters for three
reasons the brief implies: (a) low latency gating in front of every request,
(b) it can run air-gapped/on-prem, which brokerage compliance environments
often require, and (c) it keeps the expensive schema-aware LLM call from ever
firing on irrelevant or adversarial input. See `agent_orchestrator.py` for
where this plugs in — it is intentionally an injectable interface
(`IntentRouter`) so the actual BitNet binary, a fine-tuned classifier, or
even a regex/keyword fallback can sit behind it without changing the rest of
the pipeline.

## 4. Retriever Agent (RAG)

1. Embed the user's NL query with the same embedding model used at ingestion.
2. Vector search top-K table/column description chunks (`pgvector` cosine
   similarity), returning candidate tables with a relevance score.
3. If more than one table is returned, query the graph store for the minimal
   connected subgraph (shortest join path via FK edges) spanning the
   candidate tables. This yields the exact join chain, not just "these tables
   are probably related."
4. Assemble a **scoped schema context**: only the matched tables/columns,
   their types, enum value lists, constraints, and the resolved join
   conditions (`ON a.x = b.y`) — never the full 24-table DDL.

This bounded-context approach is what makes "no other result should be
published" enforceable: the generation LLM literally cannot reference a table
it was never shown.

## 5. SQL Generation Agent

A capable LLM (Claude, called via the standard Messages API) receives:
- The scoped schema context from step 4 (table/column names, types, PK/FK,
  enum values, relevant indexes).
- The user's original NL ask.
- A system prompt that constrains it to: (a) use only the given identifiers,
  (b) output SQL only — no prose, no markdown fences, (c) support subqueries/
  CTEs/window functions where the ask requires aggregation across joins,
  (d) never emit DDL/DML that mutates data (SELECT-only, enforced again at
  validation).

## 6. Validation / Optimization Agent

- Parse the generated SQL (e.g. with `sqlglot`) and resolve every identifier
  against the live catalog (introspected the same way as ingestion) — reject
  and retry-with-feedback if a column/table doesn't actually exist or if the
  statement isn't read-only.
- Run `EXPLAIN` (not `EXPLAIN ANALYZE`, to avoid executing on production) to
  see whether a `Seq Scan` shows up on a column used in a `WHERE`/`JOIN`
  predicate that has no index; if so, emit a suggested
  `CREATE INDEX ... ON table (column);` as a *separate* advisory field, kept
  out of the primary SQL answer.
- Enforce a statement timeout / row-limit guard before returning.

## 7. Orchestration

The four agents are wired as a small state machine (a single Python
`Orchestrator` in `agent_orchestrator.py`; swap in LangGraph/CrewAI/etc. for a
larger deployment — the interfaces are deliberately framework-agnostic).
State flows one-directionally except for one feedback loop: if validation
fails identifier resolution, the orchestrator re-prompts the generation agent
once with the specific error before giving up.

## 8. Security / guardrails

- The DB role used for `EXPLAIN` and any catalog introspection is strictly
  read-only and has no access to tables outside `brokerage`.
- Retrieved schema context, not the user's raw text, is what's quoted back
  into the generation prompt for identifiers — the user's words never get
  string-concatenated directly into SQL.
- Content the retriever pulls from the vector/graph store is treated as
  trusted (it comes from your own DDL), but the user's NL text is treated as
  untrusted input throughout — it only ever influences *retrieval* and the
  *ask* portion of the generation prompt, never the identifier list.
- Out-of-scope classification happens before any retrieval, so prompt
  injection embedded in a "query" cannot reach the schema-aware LLM at all
  unless the router first classifies it as `SQL_INTENT`.

## 9. Extensibility ("provision for future updates")

- New tables/columns: add them in a migration, bump `schema_version`, update
  `COMMENT ON` strings in the same migration. Re-run `ingestion_pipeline.py`
  (see `RUN.md`) — it diffs against the stored version and only re-embeds
  changed objects.
- New enum values, new FK relationships: automatically picked up by
  introspection; graph edges regenerate from `pg_constraint` each run.
- New agent steps (e.g. a query-explanation agent, a cost-estimation agent):
  add a node to the orchestrator's state machine; the retriever/validator
  interfaces don't change.
