# Evaluation results

Day-4+ runs from `python scripts/eval.py --tag <name>` are written here as timestamped JSON + Markdown summaries (retrieval hit rate, refusal rates, citation correctness, latency, suggested RRF threshold).

**Current reference (21 Oct 2026):** `20261021T134725Z_pre_submit` — **35/35** pass on `eval/questions.jsonl`. That score is from before the latest `messages.unintelligible` / gibberish guards; re-run with `--tag pre_submit` after those changes land.

Earlier milestones: `20261015T180807Z_after_tuned` (30/30 on the original 30-question set), `20261021T131101Z_after_hardening`, `20261021T132330Z_fix_d10_f01`. See `docs/eval_before_after.md`.
