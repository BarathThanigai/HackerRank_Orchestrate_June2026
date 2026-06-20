"""Risk flags derived from image observations and user history."""

from __future__ import annotations

from utils import ClaimIntent, ImageObservation, canonical_flags, split_semicolon


def assess_risk(
    intent: ClaimIntent,
    observations: list[ImageObservation],
    history: dict[str, str] | None,
) -> list[str]:
    flags: set[str] = set()
    successful = [o for o in observations if not o.error]
    for obs in observations:
        flags.update(obs.quality_issues)
        if obs.visible_object not in {intent.claim_object, "unknown"}:
            flags.add("wrong_object")
        if (not obs.claimed_part_visible and obs.visible_part not in {"unknown", *intent.claimed_parts}
                and obs.visible_object == intent.claim_object):
            flags.add("wrong_object_part")
        if not obs.original_photo_likely:
            flags.add("non_original_image")
        if obs.text_instruction_present:
            flags.add("text_instruction_present")
    if successful and not any(o.claimed_part_visible for o in successful):
        if (intent.claim_object == "package"
                and intent.primary_part in {"contents", "item"}):
            flags.add("damage_not_visible")
        else:
            flags.update({"wrong_angle", "damage_not_visible"})
    elif successful and any(o.claimed_part_visible and o.damage_present is False for o in successful):
        flags.add("damage_not_visible")
    visible_damage = {
        o.visible_damage for o in successful
        if o.visible_damage not in {"unknown", "none"} and o.damage_present
    }
    if visible_damage and intent.primary_issue not in visible_damage:
        related = [
            {"crack", "glass_shatter"}, {"broken_part", "missing_part"},
            {"water_damage", "stain"},
        ]
        if not any(intent.primary_issue in group and visible_damage & group for group in related):
            flags.add("claim_mismatch")
    if history:
        flags.update(set(split_semicolon(history.get("history_flags", "")))
                     & {"user_history_risk", "manual_review_required"})
    if "text_instruction_present" in flags:
        flags.add("manual_review_required")
    if any(o.error for o in observations):
        flags.add("manual_review_required")
    if (flags & {"wrong_object", "claim_mismatch", "possible_manipulation",
                 "non_original_image", "text_instruction_present"}
            and "user_history_risk" in flags):
        flags.add("manual_review_required")
    return canonical_flags(flags)
