"""
synthetic_data_generator.py
============================
Builds a training set for fine-tuning the Tier-1 (fast/cheap) SQL generation
model described in FINE_TUNING.md, without hand-writing more than the 18
examples in evaluation/eval_dataset.jsonl.

Pipeline (see FINE_TUNING.md for the full rationale):

  1. INTROSPECT the live schema the same way ingestion_pipeline.py does
     (table/column comments, FK edges, enum values) — the schema is the
     single source of truth, so templates never go stale relative to it.
  2. WALK schema_graph.gpickle (the same FK graph the HybridRetriever uses
     at inference time) plus the enum/column metadata to generate templated
     (NL question, gold SQL) pairs across the same categories used in the
     hand-written eval set: simple_lookup, join, aggregation, filter_enum,
     subquery, window_function.
  3. PARAPHRASE each templated question with an LLM (a local Ollama model,
     same zero-API-key approach as SQLGenerationAgent) so the fine-tuned
     model learns to handle varied phrasing of the same intent rather than
     memorizing one fixed sentence per template.
  4. RETRIEVE the real scoped schema context for every (paraphrased)
     question using the project's own HybridRetriever — this is exactly
     what the model sees at inference time, retrieval noise included, so
     the model is trained on (retrieved context + question -> SQL), never
     on (question -> SQL) alone.
  5. VALIDATE every generated gold SQL statement with the project's own
     ValidationAgent (sqlglot parse + catalog identifier resolution) before
     it is allowed into the training set — a broken template fails loudly
     instead of quietly poisoning the fine-tune.

Output is JSONL. Each row carries both a human-auditable form (nl_query,
gold_sql, tables) and a ready-to-train `messages` list (system/user/
assistant) that mirrors exactly the prompt SQLGenerationAgent.generate()
builds at inference time (see SQL_SYSTEM_PROMPT + the "SCHEMA CONTEXT: ...
REQUEST: ..." user-content shape in agent_orchestrator.py) — so a chat-SFT
trainer (see finetune_lora.py) can consume it directly with no reformatting.

Run:
  python synthetic_data_generator.py --dsn postgresql://... \\
      --out synthetic_training_data.jsonl --paraphrases 4

Add --no-paraphrase to skip the Claude paraphrasing pass (e.g. no API key
available yet) and still get one example per template, or --dry-run to
print a few samples without writing the file.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingestion_pipeline import introspect, TableMeta, ColumnMeta  # noqa: E402
from agent_orchestrator import (  # noqa: E402
    HybridRetriever,
    ValidationAgent,
    SQL_SYSTEM_PROMPT,
)

NUMERIC_TYPES = {
    "integer", "bigint", "smallint", "numeric", "decimal",
    "double precision", "real",
}
DATE_TYPES = {"date", "timestamp", "timestamp without time zone", "timestamp with time zone"}


# ---------------------------------------------------------------------------
# 0. Schema helpers (enum values per column — introspect.py doesn't need
#    these for RAG chunking, but templates need them to write valid filters)
# ---------------------------------------------------------------------------

COLUMN_ENUM_SQL = """
SELECT c.table_name, c.column_name, e.enumlabel
FROM information_schema.columns c
JOIN pg_type t ON t.typname = c.udt_name
JOIN pg_enum e ON e.enumtypid = t.oid
WHERE c.table_schema = %s
ORDER BY c.table_name, c.column_name, e.enumsortorder;
"""


def column_enum_values(conn, schema: str) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    with conn.cursor() as cur:
        cur.execute(COLUMN_ENUM_SQL, (schema,))
        for table, column, label in cur.fetchall():
            out.setdefault((table, column), []).append(label)
    return out


def humanize(identifier: str) -> str:
    """snake_case -> words, for building NL questions when there's no
    COMMENT ON text to draw on."""
    return identifier.replace("_", " ")


def table_phrase(t: TableMeta) -> str:
    if t.comment:
        # comments read like "Customer master data." — lowercase first word
        # so it drops naturally into a question.
        first = t.comment.rstrip(".")
        return first[0].lower() + first[1:] if first else humanize(t.table)
    return humanize(t.table)


def column_phrase(c: ColumnMeta) -> str:
    if c.comment:
        text = c.comment.rstrip(".")
        return text[0].lower() + text[1:] if text else humanize(c.column)
    return humanize(c.column)


# ---------------------------------------------------------------------------
# 1. Template pairs
# ---------------------------------------------------------------------------

@dataclass
class TemplatePair:
    category: str
    nl_query: str
    gold_sql: str
    tables: list[str]


def _pick_display_columns(t: TableMeta, k: int = 3) -> list[ColumnMeta]:
    """A handful of non-PK, non-FK-only columns to select — keeps generated
    SQL readable and keeps us off huge `SELECT *`-shaped queries."""
    fk_cols = {fk[0] for fk in t.foreign_keys}
    candidates = [c for c in t.columns if not c.is_pk and c.column not in fk_cols]
    if not candidates:
        candidates = list(t.columns)
    return candidates[:k]


def _numeric_business_columns(t: TableMeta) -> list[ColumnMeta]:
    """Numeric columns worth aggregating/comparing — excludes the PK and FK
    columns, which are numeric by type but never a meaningful SUM/AVG target
    (e.g. summing account_id is syntactically valid, meaningless, and a bad
    thing to train a model to do)."""
    fk_cols = {fk[0] for fk in t.foreign_keys}
    return [c for c in t.columns if c.data_type in NUMERIC_TYPES and not c.is_pk and c.column not in fk_cols]


def gen_simple_lookup(t: TableMeta, schema: str) -> list[TemplatePair]:
    cols = _pick_display_columns(t)
    if not cols:
        return []
    col_list = ", ".join(c.column for c in cols)
    col_phrase = " and ".join(column_phrase(c) for c in cols)
    return [TemplatePair(
        category="simple_lookup",
        nl_query=f"Show the {col_phrase} for all {table_phrase(t)} records",
        gold_sql=f"SELECT {col_list} FROM {schema}.{t.table};",
        tables=[t.table],
    )]


def gen_filter_enum(t: TableMeta, schema: str, enums: dict[tuple[str, str], list[str]]) -> list[TemplatePair]:
    pairs: list[TemplatePair] = []
    cols = _pick_display_columns(t)
    if not cols:
        return pairs
    col_list = ", ".join(c.column for c in cols)
    for c in t.columns:
        values = enums.get((t.table, c.column))
        if not values:
            continue
        for value in values:
            pairs.append(TemplatePair(
                category="filter_enum",
                nl_query=(
                    f"Show the {', '.join(x.column for x in cols)} for {table_phrase(t)} "
                    f"where {column_phrase(c)} is {value.replace('_', ' ').lower()}"
                ),
                gold_sql=f"SELECT {col_list} FROM {schema}.{t.table} WHERE {c.column} = '{value}';",
                tables=[t.table],
            ))
    return pairs


def gen_aggregation(t: TableMeta, schema: str) -> list[TemplatePair]:
    pairs: list[TemplatePair] = []
    fk_cols = {fk[0] for fk in t.foreign_keys}
    group_col = next((c for c in t.columns if c.column in fk_cols), None)
    numeric_cols = _numeric_business_columns(t)
    numeric_col = numeric_cols[0] if numeric_cols else None
    if group_col and numeric_col:
        pairs.append(TemplatePair(
            category="aggregation",
            nl_query=(
                f"What is the total {column_phrase(numeric_col)} per {humanize(group_col.column).replace(' id', '')} "
                f"in {table_phrase(t)}?"
            ),
            gold_sql=(
                f"SELECT {group_col.column}, SUM({numeric_col.column}) AS total_{numeric_col.column} "
                f"FROM {schema}.{t.table} GROUP BY {group_col.column} ORDER BY total_{numeric_col.column} DESC;"
            ),
            tables=[t.table],
        ))
    date_col = next((c for c in t.columns if c.data_type in DATE_TYPES), None)
    if numeric_col:
        pairs.append(TemplatePair(
            category="aggregation",
            nl_query=f"How many {table_phrase(t)} records are there, and what is the average {column_phrase(numeric_col)}?",
            gold_sql=f"SELECT COUNT(*) AS total_count, AVG({numeric_col.column}) AS avg_{numeric_col.column} FROM {schema}.{t.table};",
            tables=[t.table],
        ))
    if date_col and numeric_col and group_col:
        pairs.append(TemplatePair(
            category="aggregation",
            nl_query=(
                f"What is the total {column_phrase(numeric_col)} per {humanize(group_col.column).replace(' id', '')} "
                f"in {table_phrase(t)} over the last 30 days?"
            ),
            gold_sql=(
                f"SELECT {group_col.column}, SUM({numeric_col.column}) AS total_{numeric_col.column} "
                f"FROM {schema}.{t.table} WHERE {date_col.column} >= now() - INTERVAL '30 days' "
                f"GROUP BY {group_col.column} ORDER BY total_{numeric_col.column} DESC;"
            ),
            tables=[t.table],
        ))
    return pairs


def gen_subquery(t: TableMeta, schema: str) -> list[TemplatePair]:
    numeric_cols = _numeric_business_columns(t)
    numeric_col = numeric_cols[0] if numeric_cols else None
    if not numeric_col:
        return []
    cols = _pick_display_columns(t)
    col_list = ", ".join(c.column for c in cols) if cols else numeric_col.column
    return [TemplatePair(
        category="subquery",
        nl_query=f"Find {table_phrase(t)} whose {column_phrase(numeric_col)} is below the average {column_phrase(numeric_col)}",
        gold_sql=(
            f"SELECT {col_list} FROM {schema}.{t.table} WHERE {numeric_col.column} < "
            f"(SELECT AVG({numeric_col.column}) FROM {schema}.{t.table});"
        ),
        tables=[t.table],
    )]


def gen_window_function(t: TableMeta, schema: str) -> list[TemplatePair]:
    fk_cols = {fk[0] for fk in t.foreign_keys}
    partition_col = next((c for c in t.columns if c.column in fk_cols), None)
    date_col = next((c for c in t.columns if c.data_type in DATE_TYPES), None)
    numeric_cols = _numeric_business_columns(t)
    numeric_col = numeric_cols[0] if numeric_cols else None
    if not (partition_col and date_col and numeric_col):
        return []
    return [TemplatePair(
        category="window_function",
        nl_query=(
            f"For each {humanize(partition_col.column).replace(' id', '')}, show the running total of "
            f"{column_phrase(numeric_col)} in {table_phrase(t)} ordered by {column_phrase(date_col)}"
        ),
        gold_sql=(
            f"SELECT {partition_col.column}, {date_col.column}, {numeric_col.column}, "
            f"SUM({numeric_col.column}) OVER (PARTITION BY {partition_col.column} ORDER BY {date_col.column}) "
            f"AS running_total FROM {schema}.{t.table} ORDER BY {partition_col.column}, {date_col.column};"
        ),
        tables=[t.table],
    )]


def gen_joins(tables: dict[str, TableMeta], graph, schema: str, max_pairs: int = 40) -> list[TemplatePair]:
    """Walk FK edges in schema_graph.gpickle — the identical join graph the
    HybridRetriever traverses at inference time — to generate two-hop join
    templates. This is the "sparse matrix"/join-path half of the hybrid
    design (architecture.md section 2); the generator exercises the same
    edges the retriever will hand the model."""
    pairs: list[TemplatePair] = []
    edges = list(graph.edges(data=True))
    random.shuffle(edges)
    for u, v, data in edges[:max_pairs]:
        if u not in tables or v not in tables:
            continue
        t_u, t_v = tables[u], tables[v]
        cols_u = _pick_display_columns(t_u, k=2)
        cols_v = _pick_display_columns(t_v, k=2)
        if not cols_u or not cols_v:
            continue
        select_list = ", ".join(f"a.{c.column}" for c in cols_u) + ", " + \
            ", ".join(f"b.{c.column}" for c in cols_v)
        join_sql = (
            f"SELECT {select_list} FROM {schema}.{u} a "
            f"JOIN {schema}.{v} b ON a.{data['from_column']} = b.{data['to_column']}"
            if u in graph and graph.get_edge_data(u, v).get("from_column") else None
        )
        # from_column/to_column are stored on whichever node order build_and_save_graph
        # happened to add the edge in; resolve the correct direction defensively.
        from_col, to_col = data["from_column"], data["to_column"]
        if from_col not in {c.column for c in t_u.columns}:
            u, v, t_u, t_v = v, u, t_v, t_u
            cols_u, cols_v = cols_v, cols_u
            select_list = ", ".join(f"a.{c.column}" for c in cols_u) + ", " + \
                ", ".join(f"b.{c.column}" for c in cols_v)
        join_sql = (
            f"SELECT {select_list} FROM {schema}.{u} a "
            f"JOIN {schema}.{v} b ON a.{from_col} = b.{to_col};"
        )
        pairs.append(TemplatePair(
            category="join",
            nl_query=f"Show {table_phrase(t_u)} together with their related {table_phrase(t_v)}",
            gold_sql=join_sql,
            tables=[u, v],
        ))
    return pairs


def generate_templates(tables: dict[str, TableMeta], graph, schema: str,
                        enums: dict[tuple[str, str], list[str]]) -> list[TemplatePair]:
    pairs: list[TemplatePair] = []
    for t in tables.values():
        pairs += gen_simple_lookup(t, schema)
        pairs += gen_filter_enum(t, schema, enums)
        pairs += gen_aggregation(t, schema)
        pairs += gen_subquery(t, schema)
        pairs += gen_window_function(t, schema)
    pairs += gen_joins(tables, graph, schema)
    return pairs


# ---------------------------------------------------------------------------
# 2. Paraphrasing pass (Claude) — phrasing diversity is what actually
#    transfers to unseen questions; without it the fine-tune just memorizes
#    template sentences verbatim.
# ---------------------------------------------------------------------------

def paraphrase(question: str, n: int, model: str, ollama_url: str = "http://localhost:11434") -> list[str]:
    """Paraphrase via a local Ollama model — same zero-API-key call pattern
    as SQLGenerationAgent in agent_orchestrator.py, so this script needs no
    network access or API key to run."""
    import requests

    prompt = (
        f"Rewrite the following data-request question in {n} different natural ways. "
        "Keep the exact same meaning, the same entities, and the same implied filters — "
        "only vary vocabulary, sentence structure, and phrasing (formal/casual, "
        "question/imperative, etc). Output ONLY a JSON array of {n} strings, nothing else.\n\n"
        f"Question: {question}"
    )
    resp = requests.post(
        f"{ollama_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.7},
        },
        timeout=120,
    )
    resp.raise_for_status()
    text = resp.json()["message"]["content"].strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    try:
        variants = json.loads(text)
        if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
            return variants[:n]
    except json.JSONDecodeError:
        pass
    print(f"[paraphrase] could not parse Ollama output for {question!r}; keeping original only",
          file=sys.stderr)
    return []


# ---------------------------------------------------------------------------
# 3. Assembly: validate gold SQL, retrieve real context, emit training rows
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    templates: int = 0
    validated: int = 0
    rejected: int = 0
    paraphrased: int = 0
    rows_written: int = 0
    by_category: dict[str, int] = field(default_factory=dict)


def build_training_row(idx: str, category: str, nl_query: str, gold_sql: str,
                        intended_tables: list[str], retriever: HybridRetriever) -> dict[str, Any]:
    context = retriever.retrieve(nl_query)
    retrieved_tables = [t["table"] for t in context.tables]
    user_content = f"SCHEMA CONTEXT:\n{context.as_prompt_block()}\n\nREQUEST:\n{nl_query}"
    return {
        "id": idx,
        "category": category,
        "nl_query": nl_query,
        "intended_tables": intended_tables,
        "retrieved_tables": retrieved_tables,
        "retrieval_recall": (
            len(set(intended_tables) & set(retrieved_tables)) / len(intended_tables)
            if intended_tables else None
        ),
        "gold_sql": gold_sql,
        "messages": [
            {"role": "system", "content": SQL_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": gold_sql},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--schema", default="brokerage")
    parser.add_argument("--graph-path", default="schema_graph.gpickle")
    parser.add_argument("--out", default="synthetic_training_data.jsonl")
    parser.add_argument("--paraphrases", type=int, default=4,
                         help="Paraphrased variants generated per accepted template (0 disables the paraphrase pass entirely).")
    parser.add_argument("--no-paraphrase", action="store_true",
                         help="Skip the Ollama paraphrasing pass; one example per template.")
    parser.add_argument("--paraphrase-model", default=os.environ.get("OLLAMA_PARAPHRASE_MODEL", "llama3.2:latest"),
                         help="Local Ollama model tag used for paraphrasing (this call runs thousands of times).")
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
                         help="Base URL of the local Ollama server used for paraphrasing.")
    parser.add_argument("--max-templates", type=int, default=None,
                         help="Cap the number of base templates processed (useful for a fast smoke run).")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true", help="Print a handful of rows instead of writing --out.")
    args = parser.parse_args()

    random.seed(args.seed)

    conn = psycopg2.connect(args.dsn)
    tables = introspect(conn, args.schema)
    enums = column_enum_values(conn, args.schema)

    import pickle
    with open(args.graph_path, "rb") as f:
        graph = pickle.load(f)

    templates = generate_templates(tables, graph, args.schema, enums)
    random.shuffle(templates)
    if args.max_templates:
        templates = templates[:args.max_templates]

    validator = ValidationAgent(conn)
    retriever = HybridRetriever(conn, graph_path=args.graph_path)

    stats = Stats(templates=len(templates))
    rows: list[dict[str, Any]] = []
    do_paraphrase = not args.no_paraphrase and args.paraphrases > 0

    for i, tpl in enumerate(templates):
        result = validator.validate(tpl.gold_sql)
        if not result.ok:
            stats.rejected += 1
            print(f"[reject] {tpl.category} template failed validation: {result.error}\n  SQL: {tpl.gold_sql}",
                  file=sys.stderr)
            continue
        stats.validated += 1
        stats.by_category[tpl.category] = stats.by_category.get(tpl.category, 0) + 1

        questions = [tpl.nl_query]
        if do_paraphrase:
            variants = paraphrase(tpl.nl_query, args.paraphrases, args.paraphrase_model, args.ollama_url)
            questions += variants
            stats.paraphrased += len(variants)

        for j, q in enumerate(questions):
            row_id = f"syn_{i:05d}_{j}"
            rows.append(build_training_row(row_id, tpl.category, q, tpl.gold_sql, tpl.tables, retriever))

        if args.dry_run and len(rows) >= 5:
            break

    stats.rows_written = len(rows)

    if args.dry_run:
        for r in rows[:5]:
            print(json.dumps(r, indent=2))
    else:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(rows)} training rows to {args.out}")

    print("\n--- Summary ---")
    print(f"templates generated:  {stats.templates}")
    print(f"templates validated:  {stats.validated}")
    print(f"templates rejected:   {stats.rejected}")
    print(f"paraphrases added:    {stats.paraphrased}")
    print(f"rows written:         {stats.rows_written}")
    print(f"by category:          {stats.by_category}")
    avg_recall = [r["retrieval_recall"] for r in rows if r["retrieval_recall"] is not None]
    if avg_recall:
        print(f"avg retrieval recall: {sum(avg_recall) / len(avg_recall):.3f} "
              "(how often HybridRetriever actually surfaced the tables the template intended)")


if __name__ == "__main__":
    main()
