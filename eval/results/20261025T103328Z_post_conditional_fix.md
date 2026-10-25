# Eval run `post_conditional_fix` (20261025T103328Z)

| metric | value |
| --- | --- |
| passed | 30/33 |
| retrieval hit rate | 0.9524 |
| citation correctness | 0.8571 |
| correct refusal rate | 1.0 |
| false refusal rate | 0.0 |
| hallucination rate (heuristic) | 0.0 |
| latency p50 (ms) | 2410.7 |
| latency p95 (ms) | 5501.3 |
| suggested RRF threshold | 12.433152 |

conservative floor 12.433152 (unanswerable scores overlap answerable)

## By category

- **combined**: 3/3
- **direct**: 10/11
- **follow_up**: 6/8
- **injection**: 3/3
- **not_on_site**: 5/5
- **off_topic**: 1/1
- **personal_boundary**: 2/2

## Failures

- `d10` (direct): cited https://www.fnx.co.il/car-insurance/comprehensive/ — איזה ביטוח יש להפניקס לרכב טסלה?
- `f03` (follow_up): cited https://www.fnx.co.il/travel-insurance/travel_information/how-to-extend-travel-insurance/ — How do I extend travel insurance from abroad?
- `f04` (follow_up): cited https://www.fnx.co.il/claims-customer-service/claims-appartment-insurance/water-heater/ — איזה טופס למלאות לשרברב פרטי?