"""Command-line entry point for multi-modal evidence review."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from claim_extractor import extract_claim
from config import (
    CACHE_DIR, DATASET_DIR, INPUT_USD_PER_MILLION, LOG_LEVEL, MODEL, OUTPUT_COLUMNS,
    OUTPUT_USD_PER_MILLION, REPO_ROOT, VISION_BACKEND,
)
from decision_engine import decide
from evidence_validator import validate_evidence
from image_analyzer import ImageAnalyzer
from risk_assessor import assess_risk
from utils import bool_text, load_csv, resolve_image_path, split_semicolon, validate_output, write_csv

LOGGER = logging.getLogger("claim_review")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def process_claims(
    input_path: Path,
    output_path: Path,
    dataset_dir: Path = DATASET_DIR,
    refresh_cache: bool = False,
    model: str = MODEL,
) -> dict:
    claims = load_csv(input_path)
    histories = {row["user_id"]: row for row in load_csv(dataset_dir / "user_history.csv")}
    requirements = load_csv(dataset_dir / "evidence_requirements.csv")
    analyzer = ImageAnalyzer(model=model, cache_dir=CACHE_DIR)
    output_rows = []
    totals = {
        "claims": len(claims), "images": 0, "model_calls": 0, "cache_hits": 0,
        "input_tokens": 0, "output_tokens": 0, "errors": 0,
        "runtime_seconds": 0.0, "model": model, "vision_backend": VISION_BACKEND,
    }
    started = time.perf_counter()

    for index, claim in enumerate(claims, start=1):
        try:
            intent = extract_claim(claim["user_claim"], claim["claim_object"])
            image_paths = [
                resolve_image_path(dataset_dir, value)
                for value in split_semicolon(claim["image_paths"])
            ]
            totals["images"] += len(image_paths)
            observations = analyzer.analyze_many(image_paths, intent, refresh_cache)
            totals["model_calls"] += sum(not obs.cache_hit for obs in observations)
            totals["cache_hits"] += sum(obs.cache_hit for obs in observations)
            totals["input_tokens"] += sum(obs.input_tokens for obs in observations)
            totals["output_tokens"] += sum(obs.output_tokens for obs in observations)
            totals["errors"] += sum(bool(obs.error) for obs in observations)
            evidence = validate_evidence(intent, observations, requirements)
            risks = assess_risk(intent, observations, histories.get(claim["user_id"]))
            decision = decide(intent, observations, evidence)
            row = {
                **claim,
                "evidence_standard_met": bool_text(evidence.evidence_standard_met),
                "evidence_standard_met_reason": evidence.reason,
                "risk_flags": ";".join(risks) if risks else "none",
                "issue_type": decision.issue_type,
                "object_part": decision.object_part,
                "claim_status": decision.claim_status,
                "claim_status_justification": decision.justification,
                "supporting_image_ids": ";".join(decision.supporting_image_ids) if decision.supporting_image_ids else "none",
                "valid_image": bool_text(evidence.valid_image),
                "severity": decision.severity,
            }
            output_rows.append(validate_output(row))
            LOGGER.info("[%d/%d] %s -> %s", index, len(claims), claim["user_id"], decision.claim_status)
        except Exception as exc:
            LOGGER.exception("Unexpected claim failure at row %d", index)
            totals["errors"] += 1
            fallback = {
                **claim, "evidence_standard_met": "false",
                "evidence_standard_met_reason": "The claim could not be processed automatically.",
                "risk_flags": "manual_review_required", "issue_type": "unknown",
                "object_part": "unknown", "claim_status": "not_enough_information",
                "claim_status_justification": f"Automated review failed: {type(exc).__name__}.",
                "supporting_image_ids": "none", "valid_image": "false", "severity": "unknown",
            }
            output_rows.append(validate_output(fallback))

    write_csv(output_path, output_rows, OUTPUT_COLUMNS)
    totals["runtime_seconds"] = round(time.perf_counter() - started, 3)
    totals["estimated_cost_usd"] = round(
        totals["input_tokens"] / 1_000_000 * INPUT_USD_PER_MILLION
        + totals["output_tokens"] / 1_000_000 * OUTPUT_USD_PER_MILLION, 6,
    )
    output_path.with_suffix(".telemetry.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %d predictions to %s", len(output_rows), output_path)
    return totals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATASET_DIR / "claims.csv")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "output.csv")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    try:
        process_claims(args.input.resolve(), args.output.resolve(), args.dataset_dir.resolve(),
                       args.refresh_cache, args.model)
        return 0
    except Exception:
        LOGGER.exception("Fatal pipeline failure")
        return 1


if __name__ == "__main__":
    sys.exit(main())
