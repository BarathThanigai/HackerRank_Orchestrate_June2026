"""Evaluate predictions against dataset/sample_claims.csv."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from config import DATASET_DIR  # noqa: E402
from main import configure_logging, process_claims  # noqa: E402
from utils import load_csv, split_semicolon  # noqa: E402

FIELDS = [
    "evidence_standard_met", "risk_flags", "issue_type", "object_part",
    "claim_status", "supporting_image_ids", "valid_image", "severity",
]


def set_f1(expected: str, predicted: str) -> float:
    truth, guess = set(split_semicolon(expected)), set(split_semicolon(predicted))
    if truth == guess:
        return 1.0
    if not truth or not guess:
        return 0.0
    precision, recall = len(truth & guess) / len(guess), len(truth & guess) / len(truth)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate(expected_rows: list[dict], predicted_rows: list[dict]) -> dict:
    if len(expected_rows) != len(predicted_rows):
        raise ValueError("Prediction row count does not match sample row count")
    exact, confusion, per_object = Counter(), defaultdict(Counter), defaultdict(Counter)
    full_match = risk_f1 = supporting_f1 = 0.0
    for expected, predicted in zip(expected_rows, predicted_rows):
        row_match = True
        for field in FIELDS:
            matches = expected[field] == predicted[field]
            exact[field] += int(matches)
            row_match &= matches
            if field in {"claim_status", "issue_type", "severity"}:
                confusion[field][f"{expected[field]} -> {predicted[field]}"] += 1
        obj = expected["claim_object"]
        per_object[obj]["count"] += 1
        per_object[obj]["claim_status_correct"] += int(expected["claim_status"] == predicted["claim_status"])
        full_match += int(row_match)
        risk_f1 += set_f1(expected["risk_flags"], predicted["risk_flags"])
        supporting_f1 += set_f1(expected["supporting_image_ids"], predicted["supporting_image_ids"])
    count = len(expected_rows)
    return {
        "rows": count,
        "field_accuracy": {field: round(exact[field] / count, 4) for field in FIELDS},
        "full_structured_row_accuracy": round(full_match / count, 4),
        "risk_flags_set_f1": round(risk_f1 / count, 4),
        "supporting_image_ids_set_f1": round(supporting_f1 / count, 4),
        "per_object_claim_status_accuracy": {
            obj: round(values["claim_status_correct"] / values["count"], 4)
            for obj, values in per_object.items()
        },
        "confusion": {field: dict(sorted(counts.items())) for field, counts in confusion.items()},
    }


def render_report(metrics: dict, telemetry: dict, path: Path) -> None:
    accuracies = "\n".join(
        f"| `{field}` | {score:.1%} |" for field, score in metrics["field_accuracy"].items()
    )
    objects = "\n".join(
        f"| {obj} | {score:.1%} |"
        for obj, score in metrics["per_object_claim_status_accuracy"].items()
    )
    vision_note = ""
    if telemetry.get("images") and telemetry.get("errors", 0) >= telemetry.get("images", 0):
        vision_note = """
> **Environment note:** this measured run had no usable vision backend.
> Every model call followed the conservative failure path. The metrics below
> therefore validate schema, orchestration, claim extraction, and error handling;
> they are not representative of the configured vision strategy. Re-run with
> Ollama running and the configured model downloaded, or set
> `VISION_BACKEND=openai` with `OPENAI_API_KEY`, before submission.
"""
    report = f"""# Evaluation Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Final strategy

Each image is independently analyzed by a vision-capable model using a strict
structured schema. Deterministic Python modules then apply evidence requirements,
history-only risk flags, and final decisions. Images remain the primary source of
truth; user history never changes a visual decision.
{vision_note}

## Sample metrics

| Field | Exact accuracy |
|---|---:|
{accuracies}

Full structured-row accuracy: **{metrics['full_structured_row_accuracy']:.1%}**

Risk flag set F1: **{metrics['risk_flags_set_f1']:.1%}**

Supporting-image set F1: **{metrics['supporting_image_ids_set_f1']:.1%}**

### Claim-status accuracy by object

| Object | Accuracy |
|---|---:|
{objects}

## Strategy comparison

1. **Rules-only fallback:** multilingual dictionaries extract claims and technical
   checks detect unreadable images, but rules cannot reliably perceive semantic damage.
2. **Vision observations + deterministic policy (selected):** the model performs
   perception; code controls evidence, risk, decisions, enums, and serialization.

## Operational analysis

- Claims processed: {telemetry.get('claims', 0)}
- Images processed: {telemetry.get('images', 0)}
- Approximate model calls: {telemetry.get('model_calls', 0)}
- Cache hits: {telemetry.get('cache_hits', 0)}
- Input tokens: {telemetry.get('input_tokens', 0):,}
- Output tokens: {telemetry.get('output_tokens', 0):,}
- Approximate cost: **${telemetry.get('estimated_cost_usd', 0):.4f}**
- Runtime: {telemetry.get('runtime_seconds', 0):.2f} seconds
- Vision backend: `{telemetry.get('vision_backend', 'unknown')}`
- Model: `{telemetry.get('model', 'unknown')}`

For the default Ollama backend, cost is zero after the local model is downloaded.
Images are resized before analysis and cached by bytes, claim, backend, model,
and prompt version. Sequential processing stays conservative and reproducible.

## Failure behavior

Unreadable images and exhausted API retries are isolated. The affected claim is
emitted conservatively as `not_enough_information` with
`manual_review_required`. Every row is schema-validated before writing.
"""
    path.write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--predictions", type=Path, default=CODE_DIR / "evaluation" / "sample_predictions.csv")
    parser.add_argument("--report", type=Path, default=CODE_DIR / "evaluation" / "evaluation_report.md")
    parser.add_argument("--metrics", type=Path, default=CODE_DIR / "evaluation" / "metrics.json")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    configure_logging()
    kwargs = {"model": args.model} if args.model else {}
    telemetry = process_claims(
        DATASET_DIR / "sample_claims.csv", args.predictions, DATASET_DIR,
        args.refresh_cache, **kwargs,
    )
    metrics = evaluate(load_csv(DATASET_DIR / "sample_claims.csv"), load_csv(args.predictions))
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    render_report(metrics, telemetry, args.report)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
