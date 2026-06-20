"""Deterministic final decision rules with image-grounded explanations."""

from __future__ import annotations

import re
from collections import Counter

from utils import ClaimIntent, DecisionResult, EvidenceResult, ImageObservation


def _severity(observations: list[ImageObservation], status: str) -> str:
    if status == "not_enough_information":
        return "unknown"
    if status == "contradicted" and observations and all(o.damage_present is False for o in observations):
        return "none"
    order = {"none": 0, "low": 1, "medium": 2, "high": 3, "unknown": -1}
    known = [o.severity for o in observations if o.severity != "unknown"]
    return max(known, key=lambda s: order[s]) if known else ("none" if status == "contradicted" else "unknown")


def decide(
    intent: ClaimIntent,
    observations: list[ImageObservation],
    evidence: EvidenceResult,
    risk_flags: list[str] | None = None,
) -> DecisionResult:
    risks = set(risk_flags or [])
    usable = [o for o in observations if not o.error and o.technical_valid]
    relevant = [o for o in usable if o.claimed_part_visible and o.claimed_condition_visible]
    if not evidence.evidence_standard_met or not relevant:
        part = intent.primary_part.replace("_", " ")
        if (intent.claim_object == "package" and intent.primary_part in {"contents", "item"}
                and intent.primary_issue == "missing_part"):
            reason = "Exterior image validation cannot establish missing package contents."
        else:
            reason = f"No usable image establishes the claimed {part} condition."
        return DecisionResult(
            "unknown", intent.primary_part, "not_enough_information", "unknown", [],
            reason,
        )
    if "text_instruction_present" in risks:
        return DecisionResult(
            "unknown", intent.primary_part, "not_enough_information", "unknown", [],
            "Instruction-like claim text requires manual review before a damage decision.",
        )

    def issue_matches(obs: ImageObservation) -> bool:
        if intent.primary_issue == "unknown":
            return obs.visible_damage not in {"none", "unknown"}
        if obs.visible_damage == intent.primary_issue:
            return True
        families = [
            {"crack", "glass_shatter"}, {"broken_part", "missing_part"},
            {"water_damage", "stain"},
        ]
        return any(intent.primary_issue in family and obs.visible_damage in family for family in families)

    positive = [o for o in relevant if o.damage_present is True]
    negative = [o for o in relevant if o.damage_present is False]
    matching = [o for o in positive if issue_matches(o)]
    if matching:
        ordered = sorted(matching, key=lambda o: (-o.confidence, o.image_id))
        rules_backend = all("deterministic rules" in o.description for o in ordered)
        explicitly_joint_evidence = bool(re.search(
            r"\b(?:both|all)\s+(?:photos?|images?)\s+(?:together|collectively)\b|"
            r"\bone\s+(?:photo|image)[^.!?]{0,60}\b(?:other|another)\s+"
            r"(?:photo|image)\b",
            intent.source_text,
        ))
        needs_multiple = rules_backend and explicitly_joint_evidence
        chosen = ordered if needs_multiple else ordered[:1]
        issue = Counter(o.visible_damage for o in chosen).most_common(1)[0][0]
        parts = Counter(o.visible_part for o in chosen if o.visible_part != "unknown")
        part = parts.most_common(1)[0][0] if parts else intent.primary_part
        ids = [o.image_id for o in chosen]
        description = chosen[0].description.rstrip(".")
        if "deterministic rules" in chosen[0].description:
            justification = (
                "Submitted readable evidence plus the extracted claim supports "
                f"{issue.replace('_', ' ')} on {part.replace('_', ' ')}."
            )
        else:
            justification = f"Image {ids[0]} supports the claim: {description}."
        return DecisionResult(
            issue, part, "supported", _severity(chosen, "supported"), ids,
            justification,
        )

    chosen = sorted(positive or negative, key=lambda o: (-o.confidence, o.image_id))[:1]
    best = chosen[0]
    valid_damage = {
        "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
        "torn_packaging", "crushed_packaging", "water_damage", "stain",
    }
    issue = best.visible_damage if best.visible_damage in valid_damage else "none"
    part = best.visible_part if best.visible_part != "unknown" else intent.primary_part
    ids = [o.image_id for o in chosen]
    if positive:
        reason = (f"the visible condition is {best.visible_damage.replace('_', ' ')} rather than "
                  f"the claimed {intent.primary_issue.replace('_', ' ')}")
    else:
        reason = "the claimed damage is not visible on the clearly shown part"
    return DecisionResult(
        issue, part, "contradicted", _severity(chosen, "contradicted"), ids,
        f"Images {';'.join(ids)} contradict the claim because {reason}.",
    )
