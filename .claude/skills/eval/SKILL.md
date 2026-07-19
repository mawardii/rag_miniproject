---
name: eval
description: Run the RAG evaluation harness and interpret hit@k / MRR / faithfulness — required before landing any retrieval or generation change
---

# Evaluating the pipeline

## When (non-negotiable)
Any change to chunking, embeddings, retrieval mode, reranking, top_k, or prompts.
Run BEFORE the change and AFTER; put both numbers in the commit message.

## How
```bash
rag-eval                # retrieval metrics only (embeds 12 queries; rerank calls if enabled)
rag-eval --answers      # + generation + LLM judge (~24 extra DeepSeek calls)

# Compare configurations via env overrides — never edit code to experiment:
RETRIEVAL_MODE=dense RERANK_ENABLED=false rag-eval
```

## Reference numbers (2026-07-18, SEC-filings corpus, 69 FinanceBench questions, deepseek-v4-flash)
| config | hit@4 | MRR |
|---|---|---|
| dense only (default) | 0.93 | 0.81 |
| hybrid | 0.86 | 0.77 |
| hybrid + rerank | 0.90 | 0.77 |
| dense + rerank | 0.91 | 0.83 |

Defaults are corpus-dependent: on the previous road-damage paper corpus hybrid+rerank
won (1.00/1.00); on SEC filings BM25 keyword votes drag in other companies' filings.
Re-run the full 4-config ablation after any corpus change.

Answer quality at default config: faithfulness 5.0/5, relevance 5.0/5.
A regression below these numbers blocks the change unless explicitly accepted.

## Interpreting
- **hit@k drop** → recall problem: candidates never fetched (chunking, embeddings,
  fusion). Reranking cannot fix what was never retrieved.
- **MRR drop with flat hit@k** → ordering problem: look at rerank prompt/parsing.
- **faithfulness drop** → generation problem: prompt grounding rules, context size.
- The golden set is 12 questions — treat small deltas (±0.05) as noise; inspect the
  per-question PASS/MISS lines before concluding anything.

## Caveats
- The judge is the same model family as the generator (self-preference bias inflates
  scores); trends matter more than absolute values. A stronger third-party judge is
  the production upgrade.
- When adding corpus documents, add golden questions for them in `evals/golden.jsonl`
  (`{"question": ..., "expected_sources": [filenames]}`).
