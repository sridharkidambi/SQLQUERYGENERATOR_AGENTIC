"""
agent_orchestrator.py
======================
Wires together the four agents described in architecture.md:

  IntentRouter (local 1-bit LLM) -> HybridRetriever (vector + graph)
    -> SQLGenerationAgent (LLM) -> ValidationAgent (sqlglot + EXPLAIN)

Every agent is a small class behind a narrow interface so any piece
(the 1-bit model runtime, the vector backend, the graph backend, the
generation LLM) can be swapped without touching the orchestrator.

This file is a working scaffold: the IntentRouter ships with a
keyword-based fallback classifier so the pipeline is runnable end-to-end
without a compiled BitNet binary on hand; swap in `Bitnet1BitRouter`
(stubbed below) once you have bitnet.cpp built (see RUN.md).

Run: python agent_orchestrator.py --dsn postgresql://... "your question"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import psycopg2
import sqlglot
from sqlglot import exp

REFUSAL_MESSAGE = "This request is not supported in this agentic application."


# ---------------------------------------------------------------------------
# Agent 1: Intent Router — decides SQL_INTENT vs OUT_OF_SCOPE, locally
# ---------------------------------------------------------------------------

class IntentRouter:
    """Interface. Implementations must return 'SQL_INTENT' or 'OUT_OF_SCOPE'."""

    def classify(self, user_query: str) -> str:
        raise NotImplementedError


class KeywordFallbackRouter(IntentRouter):
    """Zero-dependency fallback so the pipeline runs without a local model
    present. NOT a substitute for the 1-bit model in production — it is a
    deliberately conservative placeholder with the same call signature."""

    IN_SCOPE_HINTS = (
        "show", "list", "how many", "count", "average", "total", "find",
        "which customers", "which accounts", "orders", "trades", "positions",
        "balance", "holdings", "margin", "risk", "dividend", "corporate action",
        "top", "breach", "portfolio", "pnl", "p&l", "instrument", "watchlist",
    )
    MUTATION_HINTS = ("delete", "drop", "update ", "insert ", "truncate", "alter ")

    def classify(self, user_query: str) -> str:
        q = user_query.lower()
        if any(m in q for m in self.MUTATION_HINTS):
            return "OUT_OF_SCOPE"  # read-only system; mutation asks are refused
        if any(h in q for h in self.IN_SCOPE_HINTS):
            return "SQL_INTENT"
        return "OUT_OF_SCOPE"


class Bitnet1BitRouter(IntentRouter):
    """Production implementation: shells out to a local bitnet.cpp inference
    binary running a ternary-weight (1.58-bit) classifier fine-tuned to emit
    exactly 'SQL_INTENT' or 'OUT_OF_SCOPE'. See RUN.md for how to obtain and
    build the model/binary. Kept fully local — no network call.

    The router is the gatekeeper: it decides whether the user query proceeds
    to the SQL generation LLM or is refused.

    macOS/Apple Silicon notes:
    - If the binary is an x86_64 build (recommended until upstream arm64
      issues #600/#618 are fixed), it is launched via `/usr/bin/arch -x86_64`
      so it runs under Rosetta 2.
    - `--single-turn` is passed so llama-cli exits after answering instead of
      dropping into interactive chat mode.
    - The response is parsed after the 'BITNETAssistant:' marker, because the
      CLI echoes the prompt (which itself contains both label strings).
    - When the model output is unparseable (known upstream bugs produce
      garbage tokens on ARM/x86-Rosetta), the router probes once with a
      known in-scope canary phrase; if that is also unparseable, it
      transparently falls back to KeywordFallbackRouter so the pipeline
      stays usable instead of refusing everything.
    """

    # A query that any working classifier must label SQL_INTENT.
    _CANARY = "List all accounts"

    def __init__(self, binary_path: str, model_path: str,
                 fallback: Optional[IntentRouter] = None):
        self.binary_path = binary_path
        self.model_path = model_path
        self._fallback = fallback
        # None = not yet probed; restored from the probe cache when the same
        # binary+model combo was already found broken in a previous run.
        self._model_broken: Optional[bool] = (
            True if self._read_broken_marker() else None
        )
        if self._model_broken:
            print(
                "[Bitnet1BitRouter] cached probe: model unparseable on this "
                "hardware — using keyword fallback "
                f"(delete {self._broken_marker_path()} to re-probe)",
                file=sys.stderr,
            )
        # Detect binary architecture once; x86_64 binaries on arm64 hosts
        # must run through Rosetta 2.
        try:
            info = subprocess.run(
                ["file", "-b", binary_path], capture_output=True, text=True, timeout=5
            ).stdout
            self._needs_rosetta = "x86_64" in info
        except Exception:
            self._needs_rosetta = False

    # -- probe-result cache --------------------------------------------------

    def _broken_marker_path(self) -> str:
        return self.model_path + ".router-probe"

    def _read_broken_marker(self) -> bool:
        """True when a previous probe already found this exact binary (path +
        mtime) broken, so we can skip the slow re-run of the broken model."""
        try:
            with open(self._broken_marker_path()) as f:
                recorded = f.read().strip()
            current = f"{self.binary_path}:{os.path.getmtime(self.binary_path)}"
            return recorded == current
        except OSError:
            return False

    def _write_broken_marker(self) -> None:
        """Cache the probe result next to the model file. A rebuilt binary has
        a different mtime, so the next run re-probes automatically."""
        try:
            current = f"{self.binary_path}:{os.path.getmtime(self.binary_path)}"
            with open(self._broken_marker_path(), "w") as f:
                f.write(current)
        except OSError:
            pass  # cache is best-effort

    # -- internal helpers ---------------------------------------------------

    def _run_model(self, user_query: str) -> Optional[str]:
        """Run the 1-bit model once. Returns 'SQL_INTENT', 'OUT_OF_SCOPE',
        or None when the output could not be parsed (garbage/timeout)."""
        prompt = (
            "Classify the following user request for a brokerage data system. "
            "Respond with exactly one token: SQL_INTENT if it asks to retrieve "
            "or aggregate brokerage data (customers, accounts, orders, trades, "
            "positions, holdings, ledger, margin, risk, corporate actions); "
            "OUT_OF_SCOPE otherwise (including any request to modify data).\n\n"
            f"Request: {user_query}\nAnswer:"
        )
        cmd = [
            self.binary_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", "8",        # one label token is enough, small headroom
            "-st",            # single turn: exit after the answer
            "-t", "8",        # threads
            "--temp", "0",    # deterministic classification
        ]
        if self._needs_rosetta:
            cmd = ["/usr/bin/arch", "-x86_64", *cmd]

        try:
            # stdin=DEVNULL + start_new_session: detach from the controlling
            # terminal so llama-cli's interactive banner never reaches the
            # user's TTY and the process can never block waiting for input.
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            return None

        out = result.stdout
        marker = "BITNETAssistant:"
        if marker in out:
            # TTY-attached runs apply the chat template; the answer follows it.
            response = out.split(marker, 1)[1]
        elif "Answer:" in out:
            # Detached (non-TTY) runs skip the template and echo the raw
            # prompt — which itself contains BOTH label strings, so we must
            # only inspect the generation that follows the final "Answer:".
            response = out.rsplit("Answer:", 1)[1]
        else:
            return None
        # Drop the stats footer and spinner artifacts (char + backspace pairs).
        response = response.split("[ Prompt:", 1)[0]
        response = re.sub(r".\x08", "", response).upper()
        if "OUT_OF_SCOPE" in response:
            return "OUT_OF_SCOPE"
        if "SQL_INTENT" in response:
            return "SQL_INTENT"
        return None  # unparseable/garbage output

    # -- public API ---------------------------------------------------------

    def classify(self, user_query: str) -> str:
        # Once we've determined the model is broken, skip it entirely.
        if self._model_broken and self._fallback is not None:
            return self._fallback.classify(user_query)

        verdict = self._run_model(user_query)
        if verdict is not None:
            return verdict  # model answered with a real label — trust it

        # Garbage output: probe with the canary once to distinguish
        # "model is broken" from "this query confused the model".
        if self._model_broken is None:
            canary = self._run_model(self._CANARY)
            self._model_broken = canary is None
            if self._model_broken:
                self._write_broken_marker()
                print(
                    "[Bitnet1BitRouter] model output unparseable "
                    "(known upstream ARM/x86 bug) — falling back to "
                    "KeywordFallbackRouter",
                    file=sys.stderr,
                )

        if self._model_broken and self._fallback is not None:
            return self._fallback.classify(user_query)
        return "OUT_OF_SCOPE"  # model works but this query got garbage: refuse


class OllamaBitnetRouter(IntentRouter):
    """1-bit architecture intent router: the real BitNet b1.58 (1.58-bit,
    ternary-weight) model served locally by Ollama — no network call, no API
    key.

    Why Ollama instead of bitnet.cpp: the official i2_s CPU kernels are
    broken on Apple Silicon for this checkpoint (upstream issues #585/#600 —
    wrong weight-group layout, plus a residual bug in activation handling),
    so this router runs the SAME ternary-weight model from q8_0-storage GGUF
    (larenspear/bitnet_b1_58-large-GGUF). Identical weights, standard
    (correct) kernels.

    It is a base (non-instruct) model, so classification is done as few-shot
    completion. Three rotations of the example set are voted on (majority
    wins) to reduce small-model noise; ties/unparseable votes fall back to
    `fallback` when given, else refuse conservatively.
    """

    EXAMPLES = [
        ("List all customers", "YES"),
        ("Tell me a joke", "NO"),
        ("How many orders were placed today?", "YES"),
        ("What is the capital of France?", "NO"),
        ("Which accounts hold Tesla shares?", "YES"),
        ("Write a poem about the sea", "NO"),
        ("Show total cash balance", "YES"),
        ("Translate this text into French", "NO"),
        ("What is 5 times 3?", "NO"),
    ]

    HEADER = (
        "Task: decide whether a request asks to read brokerage data "
        "(customers, accounts, orders, trades, positions, holdings, margin, "
        "risk). Answer YES or NO.\n\n"
    )

    def __init__(self, model: str = "bitnet-b1.58-large",
                 url: str = "http://localhost:11434",
                 fallback: Optional[IntentRouter] = None,
                 votes: int = 3):
        self.model = model
        self.url = url.rstrip("/")
        self._fallback = fallback
        self._votes = max(1, votes)

    def _prompt(self, user_query: str, rotate: int) -> str:
        n = len(self.EXAMPLES)
        ex = self.EXAMPLES[rotate % n:] + self.EXAMPLES[:rotate % n]
        shots = "".join(f"Request: {q}\nAnswer: {a}\n\n" for q, a in ex)
        return f"{self.HEADER}{shots}Request: {user_query}\nAnswer:"

    def _vote(self, user_query: str, rotate: int) -> Optional[str]:
        import requests
        try:
            resp = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": self._prompt(user_query, rotate),
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 4, "stop": ["\n"]},
                },
                timeout=120,
            )
            resp.raise_for_status()
        except Exception:
            return None
        text = resp.json().get("response", "").strip().upper()
        words = text.split()
        first = words[0] if words else ""
        if first.startswith("YES"):
            return "SQL_INTENT"
        if first.startswith("NO"):
            return "OUT_OF_SCOPE"
        return None

    def classify(self, user_query: str) -> str:
        votes = [self._vote(user_query, 2 * i) for i in range(self._votes)]
        valid = [v for v in votes if v is not None]
        if valid:
            yes = valid.count("SQL_INTENT")
            no = valid.count("OUT_OF_SCOPE")
            if yes != no:
                return "SQL_INTENT" if yes > no else "OUT_OF_SCOPE"
        # tie or all unparseable → fallback / conservative refusal
        if self._fallback is not None:
            return self._fallback.classify(user_query)
        return "OUT_OF_SCOPE"


# ---------------------------------------------------------------------------
# Agent 2: Hybrid Retriever — vector search + graph join-path resolution
# ---------------------------------------------------------------------------

@dataclass
class SchemaContext:
    tables: list[dict]
    join_edges: list[dict]

    def as_prompt_block(self) -> str:
        lines = ["TABLES:"]
        for t in self.tables:
            cols = ", ".join(f"{c['name']} {c['type']}" for c in t["columns"])
            lines.append(f"- {t['table']} ({t['comment']}): {cols}")
        if self.join_edges:
            lines.append("JOIN PATHS:")
            for e in self.join_edges:
                lines.append(f"- {e['from_table']}.{e['from_column']} = {e['to_table']}.{e['to_column']}")
        return "\n".join(lines)


class HybridRetriever:
    def __init__(self, pg_conn, graph_path: str = "schema_graph.gpickle", top_k: int = 6):
        self.conn = pg_conn
        self.top_k = top_k
        self.graph_path = graph_path
        self._model = None
        self._graph = None

    def _embed(self, text: str) -> list[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()

    def _vector_search(self, user_query: str) -> list[dict]:
        emb = self._embed(user_query)
        with self.conn.cursor() as cur:
            # Rank ALL chunks by cosine distance, then take the top tables
            # by their best-matching chunk. This avoids losing relevant
            # tables whose best chunk happens to rank below the cutoff.
            cur.execute(
                """SELECT table_name, metadata, chunk_text, dist
                   FROM (
                       SELECT table_name, metadata, chunk_text,
                              embedding <=> %s::vector AS dist,
                              MIN(embedding <=> %s::vector) OVER (PARTITION BY table_name) AS best_dist
                       FROM schema_embeddings
                       WHERE level = 'table'
                   ) ranked
                   ORDER BY best_dist
                   LIMIT %s;""",
                (emb, emb, self.top_k),
            )
            rows = cur.fetchall()
        return [{"table": r[0], "metadata": r[1], "chunk_text": r[2]} for r in rows]

    def _load_graph(self):
        if self._graph is None:
            # nx.read_gpickle was removed in networkx 3.0; plain pickle is the replacement.
            import pickle
            with open(self.graph_path, "rb") as f:
                self._graph = pickle.load(f)
        return self._graph

    def _join_edges_for(self, table_names: list[str]) -> list[dict]:
        if len(table_names) < 2:
            return []
        g = self._load_graph()
        edges: list[dict] = []
        seen = set()
        # Steiner-tree-ish approximation: pairwise shortest paths between
        # all matched tables, unioned; fine at this graph size (~25 nodes).
        import networkx as nx
        for i, a in enumerate(table_names):
            for b in table_names[i + 1:]:
                if a not in g or b not in g:
                    continue
                try:
                    path = nx.shortest_path(g, a, b)
                except nx.NetworkXNoPath:
                    continue
                for u, v in zip(path, path[1:]):
                    key = tuple(sorted((u, v)))
                    if key in seen:
                        continue
                    seen.add(key)
                    data = g.get_edge_data(u, v)
                    edges.append({
                        "from_table": u, "from_column": data["from_column"],
                        "to_table": v, "to_column": data["to_column"],
                    })
        return edges

    def retrieve(self, user_query: str) -> SchemaContext:
        matches = self._vector_search(user_query)
        table_names = sorted({m["table"] for m in matches})

        # pull full column detail for matched tables from the catalog
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT table_name, column_name, data_type
                   FROM information_schema.columns
                   WHERE table_schema = 'brokerage' AND table_name = ANY(%s)
                   ORDER BY table_name, ordinal_position;""",
                (table_names,),
            )
            rows = cur.fetchall()

        cols_by_table: dict[str, list[dict]] = {}
        for tname, cname, ctype in rows:
            cols_by_table.setdefault(tname, []).append({"name": cname, "type": ctype})

        def _table_comment(tbl: str) -> str:
            # Table-level chunk text is "Table <name>: <comment>. Columns: ...".
            for m in matches:
                if m["table"] == tbl and m.get("chunk_text", "").startswith(f"Table {tbl}: "):
                    return m["chunk_text"][len(f"Table {tbl}: "):].split(". Columns:", 1)[0]
            return ""

        tables = [
            {"table": t, "comment": _table_comment(t), "columns": cols_by_table.get(t, [])}
            for t in table_names
        ]
        join_edges = self._join_edges_for(table_names)
        return SchemaContext(tables=tables, join_edges=join_edges)


# ---------------------------------------------------------------------------
# Agent 3: SQL Generation Agent
# ---------------------------------------------------------------------------

SQL_SYSTEM_PROMPT = """You are a SQL generation engine for a brokerage platform.
You will be given a scoped schema context (tables, columns, join paths) and a
user request. Rules, no exceptions:
1. Use ONLY the tables/columns given in the schema context. Never invent names.
2. Output ONLY a single SQL statement. No prose, no explanation, no markdown fences.
3. The statement must be read-only (SELECT / WITH ... SELECT). Never DML/DDL.
4. Use CTEs, subqueries, and window functions where needed for correctness.
5. If the request cannot be answered from the given schema context, output
   exactly: -- CANNOT_ANSWER: <one short reason>
"""


class SQLGenerationAgent:
    """Generates SQL via a local Ollama model (no API key needed)."""

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.2:latest"):
        self.url = ollama_url.rstrip("/")
        self.model = model

    def generate(self, user_query: str, context: SchemaContext, retry_feedback: Optional[str] = None) -> str:
        import requests

        user_content = f"SCHEMA CONTEXT:\n{context.as_prompt_block()}\n\nREQUEST:\n{user_query}"
        if retry_feedback:
            user_content += f"\n\nThe previous attempt failed validation: {retry_feedback}\nGenerate a corrected statement."

        resp = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SQL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 500},
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Agent 4: Validation / Optimization Agent
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    ok: bool
    sql: str
    error: Optional[str] = None
    index_suggestions: list[str] = None


class ValidationAgent:
    def __init__(self, pg_conn):
        self.conn = pg_conn

    def _known_identifiers(self) -> dict[str, set[str]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT table_name, column_name FROM information_schema.columns
                   WHERE table_schema = 'brokerage';"""
            )
            out: dict[str, set[str]] = {}
            for t, c in cur.fetchall():
                out.setdefault(t, set()).add(c)
            return out

    def validate(self, sql: str) -> ValidationResult:
        sql = sql.strip()
        if sql.startswith("-- CANNOT_ANSWER"):
            return ValidationResult(ok=False, sql=sql, error=sql)

        try:
            parsed = sqlglot.parse_one(sql, read="postgres")
        except Exception as e:
            return ValidationResult(ok=False, sql=sql, error=f"parse error: {e}")

        if not isinstance(parsed, (exp.Select, exp.With)):
            return ValidationResult(ok=False, sql=sql, error="statement is not read-only SELECT")

        known = self._known_identifiers()
        known_tables = set(known.keys())

        # Exclude CTE aliases — they are defined in the query itself, not real tables.
        cte_names = set()
        for with_clause in parsed.find_all(exp.With):
            for cte in with_clause.expressions:
                cte_names.add(cte.alias)

        used_tables = {t.name for t in parsed.find_all(exp.Table)} - cte_names
        unknown_tables = used_tables - known_tables
        if unknown_tables:
            return ValidationResult(ok=False, sql=sql, error=f"unknown table(s): {unknown_tables}")

        index_suggestions = self._suggest_indexes(sql)
        return ValidationResult(ok=True, sql=sql, index_suggestions=index_suggestions)

    def _suggest_indexes(self, sql: str) -> list[str]:
        """Runs EXPLAIN (no ANALYZE - does not execute) and flags Seq Scans
        on filter/join columns that have no existing index."""
        suggestions: list[str] = []
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"EXPLAIN (FORMAT JSON) {sql}")
                plan = cur.fetchone()[0]
        except Exception:
            # A failed EXPLAIN leaves the session mid-transaction in
            # postgres's "aborted" state; roll back so every later query on
            # this same connection (e.g. the next validate() call) doesn't
            # also raise InFailedSqlTransaction.
            self.conn.rollback()
            return suggestions

        def walk(node):
            if isinstance(node, dict):
                if node.get("Node Type") == "Seq Scan" and "Filter" in node:
                    table = node.get("Relation Name")
                    if table:
                        suggestions.append(
                            f"-- Consider: CREATE INDEX ON {table} (<filtered column>); "
                            f"(Seq Scan detected with filter: {node['Filter']})"
                        )
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(plan)
        return suggestions


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(self, router: IntentRouter, retriever: HybridRetriever,
                 generator: SQLGenerationAgent, validator: ValidationAgent):
        self.router = router
        self.retriever = retriever
        self.generator = generator
        self.validator = validator

    def handle(self, user_query: str) -> str:
        intent = self.router.classify(user_query)
        if intent != "SQL_INTENT":
            return REFUSAL_MESSAGE

        context = self.retriever.retrieve(user_query)
        if not context.tables:
            return REFUSAL_MESSAGE

        sql = self.generator.generate(user_query, context)
        result = self.validator.validate(sql)

        if not result.ok:
            sql_retry = self.generator.generate(user_query, context, retry_feedback=result.error)
            result = self.validator.validate(sql_retry)
            if not result.ok:
                return REFUSAL_MESSAGE

        output = result.sql
        if result.index_suggestions:
            output += "\n\n" + "\n".join(result.index_suggestions)
        return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--graph-path", default="schema_graph.gpickle")
    parser.add_argument("--router", choices=["auto", "bitnet-cpp", "ollama-bitnet", "keyword"],
                        default="auto",
                        help="Intent router: bitnet-cpp = native i2_s binary; "
                             "ollama-bitnet = BitNet b1.58 (1.58-bit) via local Ollama; "
                             "keyword = heuristic fallback. auto = bitnet-cpp when "
                             "--bitnet-binary/--bitnet-model are given, else keyword.")
    parser.add_argument("--bitnet-binary", default=None)
    parser.add_argument("--bitnet-model", default=None)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--ollama-router-model", default="bitnet-b1.58-large")
    parser.add_argument("query", help="Natural language request")
    args = parser.parse_args()

    conn = psycopg2.connect(args.dsn)

    router: IntentRouter
    if args.router == "keyword":
        router = KeywordFallbackRouter()
    elif args.router == "ollama-bitnet":
        router = OllamaBitnetRouter(
            model=args.ollama_router_model, url=args.ollama_url,
            fallback=KeywordFallbackRouter(),
        )
    elif args.router == "bitnet-cpp" or (
        args.router == "auto" and args.bitnet_binary and args.bitnet_model
    ):
        if not (args.bitnet_binary and args.bitnet_model):
            parser.error("--router bitnet-cpp requires --bitnet-binary and --bitnet-model")
        router = Bitnet1BitRouter(
            args.bitnet_binary, args.bitnet_model,
            fallback=KeywordFallbackRouter(),
        )
    else:
        router = KeywordFallbackRouter()

    retriever = HybridRetriever(conn, graph_path=args.graph_path)
    validator = ValidationAgent(conn)

    generator = SQLGenerationAgent()

    orchestrator = Orchestrator(router, retriever, generator, validator)
    print(orchestrator.handle(args.query))


if __name__ == "__main__":
    main()
