# Eval run `after_hardening` (20261021T131101Z)

| metric | value |
| --- | --- |
| passed | 33/35 |
| retrieval hit rate | 1.0 |
| citation correctness | 0.9524 |
| correct refusal rate | 0.9167 |
| false refusal rate | 0.0 |
| hallucination rate (heuristic) | 0.0286 |
| latency p50 (ms) | 4321.0 |
| latency p95 (ms) | 21489.7 |
| suggested RRF threshold | 0.022346 |

RRF threshold 0.022346 separates 1/12 weak unanswerable hits

## By category

- **combined**: 3/3
- **direct**: 12/13
- **follow_up**: 5/6
- **injection**: 3/3
- **not_on_site**: 6/6
- **off_topic**: 1/1
- **personal_boundary**: 3/3

## Failures

- `d10` (direct): cited https://www.fnx.co.il/car-insurance/charging-station-insurance/ — איזה ביטוח יש להפניקס לרכב טסלה?
- `f01` (follow_up): status=answered — תוך כמה זמן מקבלים תשובה?