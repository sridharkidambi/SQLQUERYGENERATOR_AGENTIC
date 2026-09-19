# Alternate path: Hybrid RAG + fine-tuned model, with a fallback to a higher model

This document covers the four new files that add a second, cost-optimized
path through the pipeline in `architecture.md`, without changing that
pipeline or its files:

| File | Role |
|---|---|
| `synthetic_data_generator.py` | Builds a training set far larger than the 18 hand-written cases in `evaluation/eval_dataset.jsonl`. |
| `finetune_lora.py` | LoRA-fine-tunes a small open model on that training set. |
| `tiered_sql_agent.py` | Tier 1 (fine-tuned model) with an automatic fallback to Tier 2 (a frontier model) on validation failure. |
| `requirements-finetune.txt` | Heavy, optional deps (`torch`, `transformers`, `peft`, `trl`...) for `finetune_lora.py` only — never pulled in by the baseline pipeline. |

## Why this design, in short

1. **The retriever stays exactly as it is.** Every generation call, in
   either tier, gets the scoped schema context from `HybridRetriever`. This
   is what makes the fine-tuned model's answers survive a schema change
   without retraining — it never memorizes the schema, it's shown it.
2. **The fine-tune trains on (retrieved context + question → SQL), not
   (question → SQL).** That's the difference between teaching the small
   model "reliably use whatever context you're handed" (which generalizes)
   and teaching it "these particular tables exist" (which doesn't survive
   `ALTER TABLE`). `synthetic_data_generator.py` calls the real
   `HybridRetriever.retrieve()` for every training question so the context
   in each row is exactly what the model will see at inference time,
   retrieval imperfections included.
3. **The training set is built synthetically, then filtered by real
   validation.** `synthetic_data_generator.py` walks `schema_graph.gpickle`
   — the same FK graph the retriever traverses — plus enum values and
   column types read straight from the live catalog, to generate templated
   `(question, SQL)` pairs across the same categories as the hand-written
   eval set (simple lookup, join, aggregation, filter-by-enum, subquery,
   window function). Each templated question is paraphrased several times
   by Claude for phrasing diversity, and every generated SQL statement —
   template or paraphrase — is run through the project's own
   `ValidationAgent` before it's allowed into the training file. A broken
   template fails loudly (printed to stderr) instead of silently poisoning
   the fine-tune.
4. **Tier 1 is the default; Tier 2 is the safety net, not a second
   opinion.** `tiered_sql_agent.py`'s `TieredSQLGenerationAgent` always
   tries the fine-tuned model first. Only when its output fails
   `ValidationAgent` (bad SQL, references a table that doesn't exist,
   attempts a mutation, etc.) does it escalate to Claude, passing along the
   specific validation error as feedback. This is where the cost win
   materializes: not by removing RAG, but by removing the frontier-model
   API call for whatever fraction of queries are structurally routine
   enough for the fine-tuned model to get right — track that fraction with
   `TierStats.tier1_hit_rate` (printed by the CLI, or read it directly if
   you wire `TieredSQLGenerationAgent` into your own driver).

## End-to-end run

```bash
# 0. Baseline pipeline must already be ingested (unchanged from RUN.md)
python ingestion_pipeline.py --dsn "$DSN"

# 1. Generate synthetic training data (needs ANTHROPIC_API_KEY for the
#    paraphrasing pass; add --no-paraphrase to skip it and iterate faster)
python synthetic_data_generator.py --dsn "$DSN" \
    --out synthetic_training_data.jsonl --paraphrases 4

# 2. Fine-tune (heavy deps — see requirements-finetune.txt)
pip install -r requirements-finetune.txt
python finetune_lora.py \
    --data synthetic_training_data.jsonl \
    --base-model meta-llama/Llama-3.2-3B-Instruct \
    --output-dir ./sql-lora-adapter \
    --merge --merged-output-dir ./sql-lora-merged
# ... convert to GGUF with llama.cpp and `ollama create sql-lora-ft -f Modelfile`
# as printed at the end of the training run.

# 3. Use the tiered agent
python tiered_sql_agent.py --dsn "$DSN" \
    --tier1-model sql-lora-ft \
    "Show all accounts whose margin utilization risk limit is currently breached"

# 4. Evaluate it the same way the baseline is evaluated (swap the generator)
```

## Wiring into `evaluate.py` / `agent_orchestrator.py`

Neither file needs to change. `TieredSQLGenerationAgent.generate()` has the
identical signature as `SQLGenerationAgent.generate()`, so anywhere the
baseline does:

```python
generator = SQLGenerationAgent(ollama_url=args.ollama_url, model=args.ollama_model)
```

the tiered path is a one-line substitution:

```python
from tiered_sql_agent import FineTunedSQLGenerationAgent, ClaudeSQLGenerationAgent, TieredSQLGenerationAgent

tier1 = FineTunedSQLGenerationAgent(ollama_url=args.ollama_url, model="sql-lora-ft")
tier2 = ClaudeSQLGenerationAgent()  # ANTHROPIC_SQL_MODEL env var, or pass model=
generator = TieredSQLGenerationAgent(tier1, tier2, validator)
```

`Orchestrator(router, retriever, generator, validator)` and everything
downstream (including `evaluate.py`'s deterministic and DeepEval checks)
works unmodified — it's the same generator interface, just backed by two
models instead of one.

## Interpreting results

Run `evaluate.py` once against the baseline `SQLGenerationAgent` and once
against `TieredSQLGenerationAgent`, and compare:

- `sql_valid_rate` and `avg_geval_sql_correctness` should land close to the
  baseline (the fallback exists precisely so Tier-1 weaknesses don't show
  up as regressions).
- `TierStats.tier1_hit_rate` is the number that turns into a cost claim:
  if Tier 1 resolves 70–85% of queries validly on the first try, that's the
  fraction of frontier-model calls this path removes, without changing the
  system's accuracy floor (Tier 2 still backstops the rest).

## Known scaffolding, by design

Like `Bitnet1BitRouter` and `KeywordFallbackRouter` in
`agent_orchestrator.py`, these files are runnable scaffolds with the
integration points made explicit, not a turnkey production pipeline:

- `synthetic_data_generator.py`'s templates cover the same six categories
  as the hand-written eval set generically (by walking the schema), not
  the brokerage-specific edge cases in categories like `corporate_action`
  or `derivative_filter` — extend `generate_templates()` with a
  domain-specific generator function alongside the generic ones if you
  need those covered synthetically too.
- `finetune_lora.py`'s base model, LoRA rank, and target modules are a
  reasonable Llama/Mistral/Qwen-family default — check `target_modules`
  against your chosen base model's actual attention module names.
- `ClaudeSQLGenerationAgent`'s default model id is read from
  `ANTHROPIC_SQL_MODEL` — verify the current recommended model at
  <https://docs.claude.com/en/docs/about-claude/models> before relying on
  the hardcoded fallback.
