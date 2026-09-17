"""
ingestion_pipeline.py
======================
Introspects the live `brokerage` Postgres schema (tables, columns, PK/FK,
enums, indexes, and — critically — the COMMENT ON business-glossary text)
and ingests it into:

  1. A vector store (pgvector table `schema_embeddings`) for semantic
     table/column retrieval.
  2. A graph store (Neo4j if configured, else an in-process networkx graph
     pickled to disk) for FK join-path retrieval.

Design goal: this script is idempotent and incremental. It compares the
live `schema_version.version_id` against the last version it embedded
(stored in `schema_embeddings_meta`) and only re-embeds objects that changed.

Run: python ingestion_pipeline.py --dsn postgresql://... [--full-rebuild]
See RUN.md for setup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from dataclasses import dataclass, field
from typing import Any

import psycopg2
import psycopg2.extras

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional at import time
    SentenceTransformer = None

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dim, fast, CPU-friendly
EMBEDDING_DIM = 384


# ---------------------------------------------------------------------------
# 1. Introspection: read the schema's own metadata as ground truth
# ---------------------------------------------------------------------------

@dataclass
class ColumnMeta:
    table: str
    column: str
    data_type: str
    is_nullable: bool
    is_pk: bool
    comment: str | None
    enum_values: list[str] = field(default_factory=list)


@dataclass
class TableMeta:
    table: str
    comment: str | None
    columns: list[ColumnMeta]
    foreign_keys: list[tuple[str, str, str, str]]  # (from_col, to_table, to_col, constraint_name)
    indexes: list[str]  # indexed column names on this table


TABLE_COMMENT_SQL = """
SELECT c.relname AS table_name, obj_description(c.oid) AS table_comment
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = %s AND c.relkind = 'r';
"""

COLUMN_META_SQL = """
SELECT
    cols.table_name,
    cols.column_name,
    cols.data_type,
    cols.is_nullable,
    col_description(fmt.oid, cols.ordinal_position) AS column_comment,
    EXISTS (
        SELECT 1 FROM information_schema.key_column_usage kcu
        JOIN information_schema.table_constraints tc
          ON tc.constraint_name = kcu.constraint_name AND tc.constraint_type = 'PRIMARY KEY'
        WHERE kcu.table_name = cols.table_name AND kcu.column_name = cols.column_name
              AND kcu.table_schema = cols.table_schema
    ) AS is_pk
FROM information_schema.columns cols
JOIN pg_catalog.pg_class fmt ON fmt.relname = cols.table_name
JOIN pg_catalog.pg_namespace ns ON ns.oid = fmt.relnamespace AND ns.nspname = cols.table_schema
WHERE cols.table_schema = %s
ORDER BY cols.table_name, cols.ordinal_position;
"""

FK_SQL = """
SELECT
    tc.table_name AS from_table,
    kcu.column_name AS from_column,
    ccu.table_name AS to_table,
    ccu.column_name AS to_column,
    tc.constraint_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = %s;
"""

INDEX_SQL = """
SELECT t.relname AS table_name, a.attname AS column_name
FROM pg_index ix
JOIN pg_class t ON t.oid = ix.indrelid
JOIN pg_class i ON i.oid = ix.indexrelid
JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
JOIN pg_namespace n ON n.oid = t.relnamespace
WHERE n.nspname = %s;
"""

ENUM_SQL = """
SELECT t.typname AS enum_name, e.enumlabel AS value
FROM pg_type t
JOIN pg_enum e ON t.oid = e.enumtypid
JOIN pg_namespace n ON n.oid = t.typnamespace
WHERE n.nspname = %s OR n.nspname = 'public'
ORDER BY t.typname, e.enumsortorder;
"""

SCHEMA_VERSION_SQL = "SELECT MAX(version_id) FROM schema_version;"


def introspect(conn, schema: str = "brokerage") -> dict[str, TableMeta]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(TABLE_COMMENT_SQL, (schema,))
        table_comments = {r["table_name"]: r["table_comment"] for r in cur.fetchall()}

        cur.execute(COLUMN_META_SQL, (schema,))
        columns_by_table: dict[str, list[ColumnMeta]] = {}
        for r in cur.fetchall():
            columns_by_table.setdefault(r["table_name"], []).append(
                ColumnMeta(
                    table=r["table_name"],
                    column=r["column_name"],
                    data_type=r["data_type"],
                    is_nullable=(r["is_nullable"] == "YES"),
                    is_pk=r["is_pk"],
                    comment=r["column_comment"],
                )
            )

        cur.execute(FK_SQL, (schema,))
        fks_by_table: dict[str, list[tuple[str, str, str, str]]] = {}
        for r in cur.fetchall():
            fks_by_table.setdefault(r["from_table"], []).append(
                (r["from_column"], r["to_table"], r["to_column"], r["constraint_name"])
            )

        cur.execute(INDEX_SQL, (schema,))
        idx_by_table: dict[str, set[str]] = {}
        for r in cur.fetchall():
            idx_by_table.setdefault(r["table_name"], set()).add(r["column_name"])

    tables: dict[str, TableMeta] = {}
    for tname, comment in table_comments.items():
        tables[tname] = TableMeta(
            table=tname,
            comment=comment,
            columns=columns_by_table.get(tname, []),
            foreign_keys=fks_by_table.get(tname, []),
            indexes=sorted(idx_by_table.get(tname, set())),
        )
    return tables


def get_live_schema_version(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_VERSION_SQL)
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else 0


# ---------------------------------------------------------------------------
# 2. Chunking: turn each table into one or two retrievable text chunks
# ---------------------------------------------------------------------------

def build_chunks(tables: dict[str, TableMeta]) -> list[dict[str, Any]]:
    """One chunk per table (table-level semantics) plus one chunk per
    column (column-level semantics) so retrieval can match at either
    granularity. Each chunk carries structured metadata for the SQL
    generation prompt later, not just prose."""
    chunks: list[dict[str, Any]] = []

    for t in tables.values():
        col_summary = "; ".join(f"{c.column} ({c.data_type})" for c in t.columns)
        table_text = (
            f"Table {t.table}: {t.comment or 'no description'}. "
            f"Columns: {col_summary}."
        )
        chunks.append({
            "id": f"table::{t.table}",
            "level": "table",
            "table": t.table,
            "text": table_text,
            "metadata": {
                "table": t.table,
                "columns": [c.column for c in t.columns],
                "primary_key": [c.column for c in t.columns if c.is_pk],
                "foreign_keys": t.foreign_keys,
                "indexes": t.indexes,
            },
        })

        for c in t.columns:
            col_text = f"Column {t.table}.{c.column} ({c.data_type}): {c.comment or 'no description'}"
            chunks.append({
                "id": f"column::{t.table}.{c.column}",
                "level": "column",
                "table": t.table,
                "text": col_text,
                "metadata": {
                    "table": t.table,
                    "column": c.column,
                    "data_type": c.data_type,
                    "is_pk": c.is_pk,
                    "is_indexed": c.column in t.indexes,
                },
            })
    return chunks


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 3. Vector store (pgvector) upsert
# ---------------------------------------------------------------------------

ENSURE_VECTOR_TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS schema_embeddings (
    chunk_id     TEXT PRIMARY KEY,
    level        TEXT NOT NULL,
    table_name   TEXT NOT NULL,
    chunk_text   TEXT NOT NULL,
    metadata     JSONB NOT NULL,
    content_hash TEXT NOT NULL,
    embedding    VECTOR({dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_schema_embeddings_ivfflat
    ON schema_embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS schema_embeddings_meta (
    id INTEGER PRIMARY KEY DEFAULT 1,
    last_ingested_version INTEGER NOT NULL
);
""".format(dim=EMBEDDING_DIM)

UPSERT_SQL = """
INSERT INTO schema_embeddings (chunk_id, level, table_name, chunk_text, metadata, content_hash, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET
    chunk_text = EXCLUDED.chunk_text,
    metadata = EXCLUDED.metadata,
    content_hash = EXCLUDED.content_hash,
    embedding = EXCLUDED.embedding
WHERE schema_embeddings.content_hash IS DISTINCT FROM EXCLUDED.content_hash;
"""


def ingest_vector_store(conn, chunks: list[dict[str, Any]], model) -> int:
    with conn.cursor() as cur:
        cur.execute(ENSURE_VECTOR_TABLE_SQL)
    conn.commit()

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    written = 0
    with conn.cursor() as cur:
        for c, emb in zip(chunks, embeddings):
            h = content_hash(c["text"])
            cur.execute(
                UPSERT_SQL,
                (c["id"], c["level"], c["table"], c["text"], json.dumps(c["metadata"]), h, list(map(float, emb))),
            )
            written += cur.rowcount
    conn.commit()
    return written


# ---------------------------------------------------------------------------
# 4. Graph store (FK adjacency) — networkx fallback; swap for Neo4j driver
#    in production by implementing the same build_graph() -> save() shape.
# ---------------------------------------------------------------------------

def build_and_save_graph(tables: dict[str, TableMeta], out_path: str = "schema_graph.gpickle") -> None:
    if nx is None:
        print("networkx not installed; skipping graph build. `pip install networkx`.", file=sys.stderr)
        return
    g = nx.Graph()
    for t in tables.values():
        g.add_node(t.table, comment=t.comment or "")
        for from_col, to_table, to_col, constraint in t.foreign_keys:
            g.add_edge(t.table, to_table, from_column=from_col, to_column=to_col, constraint=constraint)
    # nx.write_gpickle was removed in networkx 3.0; plain pickle is the replacement.
    with open(out_path, "wb") as f:
        pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)
    print(f"Graph store written to {out_path}: {g.number_of_nodes()} tables, {g.number_of_edges()} FK edges.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True, help="postgresql://brokerage:change_me_dev_only@localhost:5432/brokerage_db")
    parser.add_argument("--schema", default="brokerage")
    parser.add_argument("--full-rebuild", action="store_true")
    parser.add_argument("--graph-out", default="schema_graph.gpickle")
    args = parser.parse_args()

    if SentenceTransformer is None:
        print("sentence-transformers not installed. `pip install sentence-transformers`.", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(args.dsn)
    live_version = get_live_schema_version(conn)

    # Create the embedding/metadata tables up front — on a fresh database
    # schema_embeddings_meta does not exist yet, so the version check below
    # would fail before ingest_vector_store() ever runs ENSURE_VECTOR_TABLE_SQL.
    with conn.cursor() as cur:
        cur.execute(ENSURE_VECTOR_TABLE_SQL)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT last_ingested_version FROM schema_embeddings_meta WHERE id = 1;")
        row = cur.fetchone()
        last_version = row[0] if row else None

    if last_version == live_version and not args.full_rebuild:
        print(f"Already at schema_version {live_version}; nothing to re-ingest.")
        return

    print(f"Ingesting schema_version {live_version} (previously {last_version})...")
    tables = introspect(conn, args.schema)
    chunks = build_chunks(tables)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    written = ingest_vector_store(conn, chunks, model)
    print(f"Vector store: {written} chunks upserted (of {len(chunks)} total).")

    build_and_save_graph(tables, args.graph_out)

    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO schema_embeddings_meta (id, last_ingested_version) VALUES (1, %s)
               ON CONFLICT (id) DO UPDATE SET last_ingested_version = EXCLUDED.last_ingested_version;""",
            (live_version,),
        )
    conn.commit()
    print("Done.")


if __name__ == "__main__":
    main()
