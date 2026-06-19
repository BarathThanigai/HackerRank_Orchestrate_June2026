"""Central configuration and output vocabulary for the claim reviewer."""

from __future__ import annotations

import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason", "risk_flags",
    "issue_type", "object_part", "claim_status", "claim_status_justification",
    "supporting_image_ids", "valid_image", "severity",
]

ISSUE_TYPES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
}
OBJECT_PARTS = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
        "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base",
        "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label", "contents", "item",
        "unknown",
    },
}
RISK_FLAGS = [
    "blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle",
    "wrong_object", "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required",
]
CLAIM_STATUSES = {"supported", "contradicted", "not_enough_information"}
SEVERITIES = {"none", "low", "medium", "high", "unknown"}

VISION_BACKEND = os.getenv("VISION_BACKEND", "ollama").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5.5")
MODEL = os.getenv(
    "VISION_MODEL",
    OLLAMA_MODEL if VISION_BACKEND == "ollama" else OPENAI_VISION_MODEL,
)
IMAGE_DETAIL = os.getenv("OPENAI_IMAGE_DETAIL", "high")
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1600"))
MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
REQUEST_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "90"))
CACHE_DIR = Path(os.getenv("CLAIM_REVIEW_CACHE", CODE_DIR / ".cache"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
INPUT_USD_PER_MILLION = float(os.getenv("INPUT_USD_PER_MILLION", "2.50"))
OUTPUT_USD_PER_MILLION = float(os.getenv("OUTPUT_USD_PER_MILLION", "15.00"))
