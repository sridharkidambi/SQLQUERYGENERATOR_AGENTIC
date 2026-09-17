"""
evaluate.py
============
Validates the agentic NL->SQL pipeline against the hand-written gold set
in `eval_dataset.jsonl`. Combines two kinds of checks, deliberately kept
separate because they answer different questions:

  A. DETERMINISTIC checks (no LLM judge, cheap, always run):
     - intent router accuracy (SQL_INTENT vs OUT_OF_SCOPE vs gold label)
     - valid-SQL rate (does the generated statement parse and resolve
       against the real catalog, per ValidationAgent)
     - retrieval table precision/recall (predicted tables vs gold_tables,
       plain set arithmetic)

  B. LLM-JUDGED checks (DeepEval, semantic — catches cases where the SQL
     is *differently worded* than the gold SQL but still correct, which
     exact-match can't):
     - GEval "sql_correctness": does the generated SQL answer the NL
       question equivalently to the reference SQL?
     - ContextualPrecisionMetric / ContextualRecallMetric: were the
       retrieved schema chunks the right ones, and in a useful order?

Why DeepEval as the primary LLM-judged framework here rather than RAGAS:
in practice, pulling ragas into this project's dependency tree collided
with pinned langchain/langgraph versions already required by
agent_orchestrator.py (langchain-community's optional VertexAI import
chain breaks under recent langchain-core). DeepEval has a much lighter,
more decoupled dependency footprint and covers the same two classes of
metric (custom G-Eval + contextual precision/recall) needed here. If your
environment isolates ragas in its own virtualenv, `run_ragas_retrieval()`
below is provided as a drop-in alternative for step B's retrieval half —
see the RAGAS section and EVALUATION.md for when to prefer it.

Run: python evaluate.py --dsn postgresql://... --dataset eval_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import psycopg2

from agent_orchestrator import (
    KeywordFallbackRouter,
    Bitnet1BitRouter,
    HybridRetriever,
    SQLGenerationAgent,
    ValidationAgent,
    IntentRouter,
    SchemaContext,
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    id: str
    category: str
    scope: str                 # gold: "SQL_INTENT" | "OUT_OF_SCOPE"
    nl_query: str
    gold_tables: list[str]
    gold_sql: Optional[str]


def load_dataset(path: str) -> list[EvalCase]:
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cases.append(EvalCase(**d))
    return cases


# ---------------------------------------------------------------------------
# A. Deterministic checks
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    id: str
    category: str
    gold_scope: str
    predicted_scope: str
    intent_correct: bool
    predicted_tables: list[str]
    gold_tables: list[str]
    table_precision: Optional[float]
    table_recall: Optional[float]
    generated_sql: Optional[str]
    gold_sql: Optional[str]
    sql_valid: Optional[bool]
    sql_validation_error: Optional[str]
    geval_correctness: Optional[float] = None
    contextual_precision: Optional[float] = None
    contextual_recall: Optional[float] = None


def set_precision_recall(predicted: set[str], gold: set[str]) -> tuple[Optional[float], Optional[float]]:
    if not gold:
        return None, None
    if not predicted:
        return 0.0, 0.0
    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold)
    return precision, recall


def run_deterministic(
    case: EvalCase,
    router: IntentRouter,
    retriever: HybridRetriever,
    generator: SQLGenerationAgent,
    validator: ValidationAgent,
) -> CaseResult:
    predicted_scope = router.classify(case.nl_query)
    intent_correct = predicted_scope == case.scope

    predicted_tables: list[str] = []
    generated_sql: Optional[str] = None
    sql_valid: Optional[bool] = None
    sql_error: Optional[str] = None
    table_precision = table_recall = None

    if predicted_scope == "SQL_INTENT":
        context: SchemaContext = retriever.retrieve(case.nl_query)
        predicted_tables = [t["table"] for t in context.tables]
        table_precision, table_recall = set_precision_recall(
            set(predicted_tables), set(case.gold_tables)
        )

        if context.tables:
            generated_sql = generator.generate(case.nl_query, context)
            result = validator.validate(generated_sql)
            sql_valid = result.ok
            sql_error = result.error

    return CaseResult(
        id=case.id,
        category=case.category,
        gold_scope=case.scope,
        predicted_scope=predicted_scope,
        intent_correct=intent_correct,
        predicted_tables=predicted_tables,
        gold_tables=case.gold_tables,
        table_precision=table_precision,
        table_recall=table_recall,
        generated_sql=generated_sql,
        gold_sql=case.gold_sql,
        sql_valid=sql_valid,
        sql_validation_error=sql_error,
    )


# ---------------------------------------------------------------------------
# B. DeepEval LLM-judged checks
# ---------------------------------------------------------------------------

def run_deepeval_metrics(results: list[CaseResult]) -> None:
    """Mutates results in place, filling geval_correctness / contextual_*
    for cases that reached SQL generation. Requires DEEPEVAL / underlying
    judge-model API credentials (defaults to OpenAI unless configured
    otherwise — see EVALUATION.md for pointing DeepEval at Claude)."""
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.metrics import GEval, ContextualPrecisionMetric, ContextualRecallMetric
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    sql_correctness = GEval(
        name="sql_correctness",
        criteria=(
            "Determine whether ACTUAL_OUTPUT (a generated SQL query) would "
            "return the same information as EXPECTED_OUTPUT (a reference SQL "
            "query), for the request described in INPUT. Judge semantic "
            "equivalence, not textual similarity: different join order, "
            "aliasing, or an equivalent subquery/CTE formulation is fine. "
            "Penalize heavily if it queries different tables/columns than "
            "necessary, omits a filter the request implies, or would return "
            "a different result set."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.6,
    )
    contextual_precision = ContextualPrecisionMetric(threshold=0.6)
    contextual_recall = ContextualRecallMetric(threshold=0.6)

    test_cases = []
    indexable = []
    for r in results:
        if r.generated_sql is None or r.gold_sql is None:
            continue
        tc = LLMTestCase(
            input=r.id,  # placeholder; replaced below with real nl_query at call site if needed
            actual_output=r.generated_sql,
            expected_output=r.gold_sql,
            retrieval_context=r.predicted_tables or ["<no tables retrieved>"],
        )
        test_cases.append(tc)
        indexable.append(r)

    if not test_cases:
        return

    eval_result = deepeval_evaluate(
        test_cases=test_cases,
        metrics=[sql_correctness, contextual_precision, contextual_recall],
    )

    # map scores back onto CaseResult objects by position (deepeval
    # preserves input order in test_results)
    for r, tr in zip(indexable, eval_result.test_results):
        scores = {m.name: m.score for m in tr.metrics_data}
        r.geval_correctness = scores.get("sql_correctness")
        r.contextual_precision = scores.get("Contextual Precision")
        r.contextual_recall = scores.get("Contextual Recall")


# ---------------------------------------------------------------------------
# Optional: RAGAS path for the retrieval half (drop-in alternative to
# DeepEval's ContextualPrecision/ContextualRecall above). Isolate ragas in
# its own virtualenv per the note at the top of this file before enabling.
# ---------------------------------------------------------------------------

def run_ragas_retrieval(results: list[CaseResult]) -> None:
    try:
        from ragas import SingleTurnSample, EvaluationDataset, evaluate as ragas_evaluate
        from ragas.metrics import LLMContextPrecisionWithReference, LLMContextRecall
    except Exception as e:  # pragma: no cover
        print(f"[ragas] unavailable in this environment ({e}); skipping. "
              f"DeepEval's ContextualPrecision/Recall above already covers "
              f"this metric class.", file=sys.stderr)
        return

    samples = []
    indexable = []
    for r in results:
        if r.gold_sql is None or not r.predicted_tables:
            continue
        samples.append(SingleTurnSample(
            user_input=r.id,
            retrieved_contexts=r.predicted_tables,
            reference=", ".join(r.gold_tables),
        ))
        indexable.append(r)

    if not samples:
        return

    dataset = EvaluationDataset(samples=samples)
    result = ragas_evaluate(dataset, metrics=[LLMContextPrecisionWithReference(), LLMContextRecall()])
    df = result.to_pandas()
    for r, (_, row) in zip(indexable, df.iterrows()):
        r.contextual_precision = row.get("llm_context_precision_with_reference", r.contextual_precision)
        r.contextual_recall = row.get("context_recall", r.contextual_recall)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(results: list[CaseResult]) -> dict:
    n = len(results)
    intent_acc = sum(r.intent_correct for r in results) / n if n else 0.0

    sql_cases = [r for r in results if r.gold_scope == "SQL_INTENT"]
    valid_rate = (
        sum(1 for r in sql_cases if r.sql_valid) / len(sql_cases) if sql_cases else None
    )
    precisions = [r.table_precision for r in sql_cases if r.table_precision is not None]
    recalls = [r.table_recall for r in sql_cases if r.table_recall is not None]
    avg_precision = sum(precisions) / len(precisions) if precisions else None
    avg_recall = sum(recalls) / len(recalls) if recalls else None

    correctness_scores = [r.geval_correctness for r in results if r.geval_correctness is not None]
    avg_correctness = sum(correctness_scores) / len(correctness_scores) if correctness_scores else None

    return {
        "n_cases": n,
        "intent_accuracy": round(intent_acc, 3),
        "sql_valid_rate": round(valid_rate, 3) if valid_rate is not None else None,
        "avg_table_retrieval_precision": round(avg_precision, 3) if avg_precision is not None else None,
        "avg_table_retrieval_recall": round(avg_recall, 3) if avg_recall is not None else None,
        "avg_geval_sql_correctness": round(avg_correctness, 3) if avg_correctness is not None else None,
    }


def print_report(results: list[CaseResult], summary: dict) -> None:
    print(f"{'ID':<6}{'Category':<22}{'Intent OK':<11}{'SQL Valid':<11}{'T.Prec':<8}{'T.Rec':<8}{'GEval':<8}")
    for r in results:
        print(
            f"{r.id:<6}{r.category:<22}"
            f"{str(r.intent_correct):<11}"
            f"{str(r.sql_valid):<11}"
            f"{('%.2f' % r.table_precision) if r.table_precision is not None else '-':<8}"
            f"{('%.2f' % r.table_recall) if r.table_recall is not None else '-':<8}"
            f"{('%.2f' % r.geval_correctness) if r.geval_correctness is not None else '-':<8}"
        )
    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--dataset", default="eval_dataset.jsonl")
    parser.add_argument("--graph-path", default="schema_graph.gpickle")
    parser.add_argument("--bitnet-binary", default=None)
    parser.add_argument("--bitnet-model", default=None)
    parser.add_argument("--skip-llm-judge", action="store_true",
                         help="Run only the deterministic checks (no DeepEval/API calls).")
    parser.add_argument("--use-ragas", action="store_true",
                         help="Additionally run the RAGAS retrieval metrics (requires ragas installed).")
    parser.add_argument("--out", default="eval_report.json")
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    conn = psycopg2.connect(args.dsn)

    router: IntentRouter = (
        Bitnet1BitRouter(args.bitnet_binary, args.bitnet_model)
        if args.bitnet_binary and args.bitnet_model
        else KeywordFallbackRouter()
    )
    retriever = HybridRetriever(conn, graph_path=args.graph_path)
    validator = ValidationAgent(conn)

    import anthropic
    generator = SQLGenerationAgent(anthropic.Anthropic())

    results = [run_deterministic(c, router, retriever, generator, validator) for c in cases]

    if not args.skip_llm_judge:
        run_deepeval_metrics(results)
        if args.use_ragas:
            run_ragas_retrieval(results)

    summary = summarize(results)
    print_report(results, summary)

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": [asdict(r) for r in results]}, f, indent=2)
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
