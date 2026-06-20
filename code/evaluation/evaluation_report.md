# Evaluation Report

Generated: 2026-06-20T03:47:54.352862+00:00

## Final strategy

The selected deterministic rules backend opens and technically
validates every image, then maps multilingual claim extraction into the shared
observation schema. Evidence requirements, history-only risk flags, and final
decisions remain deterministic. This is fast, offline, and reproducible, but it
cannot independently perceive whether claimed semantic damage is truly visible.


## Sample metrics

| Field | Exact accuracy |
|---|---:|
| `evidence_standard_met` | 85.0% |
| `risk_flags` | 60.0% |
| `issue_type` | 65.0% |
| `object_part` | 90.0% |
| `claim_status` | 70.0% |
| `supporting_image_ids` | 70.0% |
| `valid_image` | 95.0% |
| `severity` | 60.0% |

Full structured-row accuracy: **50.0%**

Risk flag set F1: **74.3%**

Supporting-image set F1: **73.3%**

### Claim-status accuracy by object

| Object | Accuracy |
|---|---:|
| car | 62.5% |
| laptop | 83.3% |
| package | 66.7% |

## Strategy comparison

1. **Deterministic rules (selected):** near-zero compute cost and reliable
   orchestration on constrained hardware; technical image defects are measured,
   while semantic object/part/damage fields are claim-derived.
2. **Optional vision models:** stronger independent visual perception, with higher
   latency, memory use, cost, or output-format risk depending on backend.

## Operational analysis

- Claims processed: 20
- Images processed: 29
- Approximate model calls: 0
- Cache hits: 0
- Input tokens: 0
- Output tokens: 0
- Approximate cost: **$0.0000**
- Runtime: 2.05 seconds
- Vision backend: `rules`
- Model: `deterministic-rules-v1`

The default rules backend has no model or API cost. Hugging Face and Ollama are
local optional backends; OpenAI is an optional paid backend.
Images are resized before analysis and cached by bytes, claim, backend, model,
and prompt version. Sequential processing stays conservative and reproducible.

## Failure behavior

Unreadable images and exhausted API retries are isolated. The affected claim is
emitted conservatively as `not_enough_information` with
`manual_review_required`. Every row is schema-validated before writing.
