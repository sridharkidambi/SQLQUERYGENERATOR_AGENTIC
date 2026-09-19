"""
tiered_sql_agent.py
=====================
Alternate SQL-generation path for the pipeline in architecture.md: keeps the
HybridRetriever exactly as-is (every tier gets the same scoped schema
context — see FINE_TUNING.md for why that's non-negotiable) but replaces
the single always-call-a-model Agent 3 with two tiers:

  Tier 1 — a small model LoRA-fine-tuned on synthetic_data_generator.py's
           output (retrieved context + question -> SQL), served locally
           (default: via Ollama, same as the existing SQLGenerationAgent —
           see finetune_lora.py for how the adapter gets there). Fast, free
           per call.
  Tier 2 — a frontier model (Claude, via the `anthropic` package already in
           requirements.txt) used ONLY when Tier 1's output fails
           ValidationAgent's checks. This is where the cost saving in
           FINE_TUNING.md actually comes from: the frontier call disappears
           for every query Tier 1 gets right on the first try, while
           Validation keeps every tier honest about the live schema.

TieredSQLGenerationAgent implements the exact same interface as
agent_orchestrator.SQLGenerationAgent — generate(user_query, context,
retry_feedback=None) -> str — so it is a drop-in replacement: swap the
`generator=` argument passed to `Orchestrator(...)` and nothing else in
agent_orchestrator.py needs to change. That's the "additional file / alternate
path" the project calls for, not a fork of the orchestrator.

Standalone run (mirrors agent_orchestrator.py's own CLI):
  python tiered_sql_agent.py --dsn postgresql://... \\
      --tier1-model sql-lora-ft --tier2-model claude-sonnet-4-5 "your question"
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_orchestrator import (  # noqa: E402
    KeywordFallbackRouter,
    HybridRetriever,
    SQLGenerationAgent,
    ValidationAgent,
    SchemaContext,
    Orchestrator,
    SQL_SYSTEM_PROMPT,
    REFUSAL_MESSAGE,
)


# ---------------------------------------------------------------------------
# Tier 1: the fine-tuned model. Reuses SQLGenerationAgent's Ollama call
# verbatim (same request shape, same system prompt) — the only thing that
# differs from the baseline agent is *which model tag* is being called,
# because that's the fine-tuned one produced by finetune_lora.py.
# ---------------------------------------------------------------------------

class FineTunedSQLGenerationAgent(SQLGenerationAgent):
    """Tier 1. Identical wire format to SQLGenerationAgent — it IS one,
    pointed at the fine-tuned model tag instead of a stock base model."""

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "sql-lora-ft"):
        super().__init__(ollama_url=ollama_url, model=model)


# ---------------------------------------------------------------------------
# Tier 2: the frontier fallback. Same interface, calls Claude directly.
# ---------------------------------------------------------------------------

class ClaudeSQLGenerationAgent:
    """Tier 2. The 'higher model' fallback named in the project brief.

    Uses the standard Anthropic Messages API (architecture.md section 5
    always intended Claude here; the checked-in SQLGenerationAgent uses a
    local Ollama model for the zero-API-key baseline path — this class is
    what section 5 actually described, reserved for the queries Tier 1
    can't handle).
    """

    def __init__(self, model: Optional[str] = None, max_tokens: int = 1024):
        import anthropic

        self.client = anthropic.Anthropic()
        # No single model id stays current forever — override via
        # --tier2-model / ANTHROPIC_SQL_MODEL rather than trusting this
        # default blindly; check docs.claude.com/en/docs/about-claude/models
        # for the current recommended model at the time you deploy this.
        self.model = model or os.environ.get("ANTHROPIC_SQL_MODEL", "claude-sonnet-4-5")
        self.max_tokens = max_tokens

    def generate(self, user_query: str, context: SchemaContext, retry_feedback: Optional[str] = None) -> str:
        user_content = f"SCHEMA CONTEXT:\n{context.as_prompt_block()}\n\nREQUEST:\n{user_query}"
        if retry_feedback:
            user_content += f"\n\nThe previous attempt failed validation: {retry_feedback}\nGenerate a corrected statement."

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SQL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return "".join(block.text for block in resp.content if block.type == "text").strip()


# ---------------------------------------------------------------------------
# The tiered generator itself
# ---------------------------------------------------------------------------

@dataclass
class TierStats:
    tier1_ok: int = 0
    tier1_failed: int = 0
    tier2_calls: int = 0

    def as_dict(self) -> dict:
        total = self.tier1_ok + self.tier1_failed
        return {
            "tier1_ok": self.tier1_ok,
            "tier1_failed": self.tier1_failed,
            "tier2_calls": self.tier2_calls,
            "tier1_hit_rate": round(self.tier1_ok / total, 3) if total else None,
        }


class TieredSQLGenerationAgent:
    """Drop-in replacement for SQLGenerationAgent. Orchestrator.handle()
    calls generate() up to twice (first attempt, then once more with
    retry_feedback if validation failed) and validates the result itself
    each time — this class hooks into exactly that shape without requiring
    any change to Orchestrator:

      * First call (retry_feedback is None): try Tier 1, validate it
        ourselves. If it passes, return it — Tier 2 is never touched, which
        is the entire cost win. If it fails, escalate to Tier 2 immediately
        rather than spending a second Tier-1 call on a model that just
        proved it couldn't handle this query.
      * Second call (retry_feedback is not None, i.e. Orchestrator's own
        retry after ITS validation of whatever we returned also failed):
        go straight to Tier 2 with that feedback — Tier 1 already had its
        shot this round.
    """

    def __init__(self, tier1: SQLGenerationAgent, tier2: ClaudeSQLGenerationAgent,
                 validator: ValidationAgent, stats: Optional[TierStats] = None):
        self.tier1 = tier1
        self.tier2 = tier2
        self.validator = validator
        self.stats = stats if stats is not None else TierStats()
        self.last_tier_used: Optional[int] = None

    def generate(self, user_query: str, context: SchemaContext, retry_feedback: Optional[str] = None) -> str:
        if retry_feedback is not None:
            # Orchestrator is already on its retry pass; Tier 1 failed once
            # this round already (or would have been rejected again) — go
            # straight to the model that can actually use the feedback.
            self.last_tier_used = 2
            self.stats.tier2_calls += 1
            return self.tier2.generate(user_query, context, retry_feedback=retry_feedback)

        tier1_sql = self.tier1.generate(user_query, context)
        result = self.validator.validate(tier1_sql)
        if result.ok:
            self.stats.tier1_ok += 1
            self.last_tier_used = 1
            return tier1_sql

        self.stats.tier1_failed += 1
        self.stats.tier2_calls += 1
        self.last_tier_used = 2
        return self.tier2.generate(user_query, context, retry_feedback=result.error)


# ---------------------------------------------------------------------------
# Standalone CLI — mirrors agent_orchestrator.py's main() so this file can
# be run and evaluated on its own, or wired into evaluate.py, without
# touching the baseline pipeline.
# ---------------------------------------------------------------------------

def build_tiered_orchestrator(dsn: str, graph_path: str, ollama_url: str,
                               tier1_model: str, tier2_model: Optional[str]) -> tuple[Orchestrator, TierStats]:
    conn = psycopg2.connect(dsn)
    router = KeywordFallbackRouter()
    retriever = HybridRetriever(conn, graph_path=graph_path)
    validator = ValidationAgent(conn)

    tier1 = FineTunedSQLGenerationAgent(ollama_url=ollama_url, model=tier1_model)
    tier2 = ClaudeSQLGenerationAgent(model=tier2_model)
    stats = TierStats()
    generator = TieredSQLGenerationAgent(tier1, tier2, validator, stats=stats)

    return Orchestrator(router, retriever, generator, validator), stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--graph-path", default="schema_graph.gpickle")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--tier1-model", default="sql-lora-ft",
                         help="Ollama model tag for the fine-tuned Tier-1 model (see finetune_lora.py).")
    parser.add_argument("--tier2-model", default=None,
                         help="Anthropic model id for the Tier-2 fallback (defaults to ANTHROPIC_SQL_MODEL env var).")
    parser.add_argument("query", help="Natural language request")
    args = parser.parse_args()

    orchestrator, stats = build_tiered_orchestrator(
        args.dsn, args.graph_path, args.ollama_url, args.tier1_model, args.tier2_model,
    )
    print(orchestrator.handle(args.query))
    print(f"\n[tier stats] {stats.as_dict()}", file=sys.stderr)


if __name__ == "__main__":
    main()
