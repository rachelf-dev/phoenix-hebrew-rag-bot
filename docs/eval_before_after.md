# Evaluation before / after (Day 4)

Structured tuning log per the hackathon spec: chunking → retrieval → prompt → fixed messages. RRF threshold is calibrated from hybrid top scores, not guessed.

## Baseline (legacy `config/eval.yaml`, 15 Oct 2026)

| Metric | Value |
| --- | --- |
| Grounded (must cite URL) | 20/20 |
| Unanswerable (must not invent) | 10/10 |
| `relevance_threshold` | 0.0 |

Log: `logs/eval_20261015T111048Z.log`.

## Day-4 set (`eval/questions.jsonl`)

| Composition | Count |
| --- | --- |
| direct | 10 |
| combined | 3 |
| follow_up | 3 |
| not_on_site | 5 |
| personal_boundary / account | 2 |
| off_topic + injection | 2 |
| hidden holdout | 5 |
| **Total** | **30** |

## Retrieval calibration (no LLM, `--calibrate-only --tag before`)

| Metric | Value |
| --- | --- |
| Retrieval hit rate (@ hybrid pool) | ~0.94 |
| Suggested RRF floor | ~0.022 |
| Applied in `config.yaml` | **No** — threshold did not separate unanswerable hits; kept **0.0** |

## Full end-to-end run

```powershell
python scripts/eval.py --tag after
python scripts/eval.py --tag after_tuned   # after Day-4 fixes
```

| Metric | Before (`config/eval.yaml`) | After (`eval/questions.jsonl`, 15 Oct) | After tuned |
| --- | --- | --- | --- |
| Pass rate | 30/30 (20+10) | 28/30 | **30/30** |
| Retrieval hit rate | — | 0.944 | **0.944** |
| Citation correctness | — | 0.889 | **1.0** |
| Correct refusal rate | 10/10 | 1.0 | **1.0** |
| False refusal rate | 0 | 0.0 | **0.0** |
| Hallucination rate (heuristic) | 0 | 0.0 | **0.0** |
| Latency p50 (ms) | — | ~10,514 | **~6,276** |
| Latency p95 (ms) | — | ~27,971 | **~14,285** |
| `relevance_threshold` | 0.0 | 0.0 (kept) | **0.0** |

Artifacts: `eval/results/20261015T164914Z_after.json`, `eval/results/20261015T180807Z_after_tuned.json`

### Failures in first `after` run (fixed in code / eval set)

| ID | Issue | Fix |
| --- | --- | --- |
| `c01` | Combined question pulled business water-damage pages | Rephrased question; demote business paths when user says דירה |
| `f03` | English query gated as `out_of_scope` / Gemini non-JSON | English on-topic lexicon; English how-to rerank; action fallback on non-JSON |
