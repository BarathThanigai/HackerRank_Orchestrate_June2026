"""Deterministic multilingual extraction of the actual damage claim."""

from __future__ import annotations

import re

from config import OBJECT_PARTS
from utils import ClaimIntent, customer_utterances, normalize_text

PART_PATTERNS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "car": [
        ("rear_bumper", ("rear bumper", "back bumper", "bumper trasero", "parachoques trasero",
                         "parachoques de atras", "piche ka bumper", "back ka bumper")),
        ("front_bumper", ("front bumper", "bumper delantero", "parachoques delantero",
                          "aage ka bumper", "front side bumper")),
        ("side_mirror", ("side mirror", "wing mirror", "door mirror", "espejo lateral",
                         "left mirror", "right mirror", "side ka mirror")),
        ("windshield", ("windshield", "front windshield", "windscreen", "front glass",
                         "parabrisas", "sheesha", "aage ka sheesha")),
        ("taillight", ("taillight", "tail light", "back light", "rear light", "rear lamp",
                        "luz trasera", "luz posterior", "piche ki light", "peeche ki light")),
        ("headlight", ("headlight", "head light", "front light", "faro")),
        ("quarter_panel", ("quarter panel",)),
        ("fender", ("fender", "mudguard")),
        ("hood", ("hood", "bonnet", "cofre", "tapa del motor")),
        ("door", ("door panel", "car door", "left side door", "right side door", "puerta", "door")),
        ("body", ("body panel", "car body", "vehicle body", "carroceria", "body")),
    ],
    "laptop": [
        ("trackpad", ("trackpad", "touchpad", "mouse pad", "panel tactil")),
        ("keyboard", ("keyboard", "keycap", "key cap", "missing key", "key missing",
                       "keys", "teclas", "teclado", "kunji")),
        ("hinge", ("hinge", "bisagra")),
        ("screen", ("screen", "display", "pantalla", "monitor", "laptop glass")),
        ("lid", ("outer lid", "laptop lid", "top cover", "tapa")),
        ("corner", ("laptop corner", "outer corner", "corner", "esquina")),
        ("port", ("charging port", "usb port", "port", "puerto")),
        ("base", ("laptop base", "bottom case", "base")),
        ("body", ("laptop body", "outer body", "side edge", "palm rest", "chassis", "body")),
    ],
    "package": [
        ("package_corner", ("package corner", "box corner", "cardboard box corner",
                            "parcel corner", "crushed corner", "corner crushed",
                            "corner dab gaya", "esquina", "corner dab")),
        ("package_side", ("package side", "box side", "package surface", "outside", "surface")),
        ("seal", ("seal", "open seal", "opened seal", "torn seal", "broken seal",
                  "tape", "flap", "sealed side", "seal wali side", "tape wali side")),
        ("label", ("shipping label", "package label", "etiqueta", "label")),
        ("contents", ("contents", "content", "product inside", "missing product",
                      "item missing", "andar ka item", "inside the package")),
        ("item", ("inside item", "item inside", "product", "inner item")),
        ("box", ("shipping box", "delivery box", "cardboard box", "package", "parcel", "box")),
    ],
}

ISSUE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("glass_shatter", ("shattered", "shatter", "smashed glass", "screen shattered",
                       "windshield shatter")),
    ("torn_packaging", ("torn open", "torn-open", "torn packaging", "opened",
                        "open flap", "seal broken", "seal is torn",
                        "seal torn", "open seal", "opened seal", "broken seal",
                        "tape broken", "package opened", "box opened", "opened box",
                        "phati", "phati hui", "fata", "abierto", "rasgado",
                        "opened jaisa", "open jaisa")),
    ("crushed_packaging", ("crushed", "crush", "corner crushed", "crushed corner",
                           "dented box", "box dented",
                           "dab gaya", "daba hua", "aplastad", "aplastado")),
    ("water_damage", ("water damage", "liquid damage", "wet box", "got wet", "wet",
                      "soaked", "liquid", "coffee spill", "coffee", "water damaged",
                      "mojado", "empapado", "agua")),
    ("missing_part", ("missing", "absent", "missing key", "key missing",
                      "keycap came off", "keys came off", "keycap missing", "missing keycap",
                      "missing keys", "key is missing", "keys missing", "missing part",
                      "missing or broken", "faltan", "missing")),
    ("broken_part", ("broken", "broke", "breakage", "snapped", "hole", "detached",
                     "not sitting properly", "not sitting", "toot gaya", "toota", "roto")),
    ("dent", ("hail dents", "dented", "dent", "dab gaya", "abollad", "hundido")),
    ("scratch", ("scratched", "scratch", "scrape", "scuff", "rayon")),
    ("crack", ("cracked", "crack line", "crack", "broken screen", "display glass",
               "fracture", "fisura", "rajadura")),
    ("stain", ("oil stain", "stained", "stain", "oily mark", "dark mark",
               "coffee mark", "cosmetic stain", "wet-looking stain", "mancha", "daag")),
    ("none", ("no visible damage", "no damage visible", "no damage", "undamaged")),
]


def _find_all(text: str, patterns: list[tuple[str, tuple[str, ...]]]) -> list[str]:
    matches: list[tuple[int, str]] = []
    for label, phrases in patterns:
        positions = [text.find(normalize_text(p)) for p in phrases]
        positions = [p for p in positions if p >= 0]
        if positions:
            matches.append((min(positions), label))
    return [label for _, label in sorted(matches)]


def _negated_labels(
    text: str, patterns: list[tuple[str, tuple[str, ...]]]
) -> set[str]:
    """Find labels explicitly rejected in English, Hindi/Hinglish, or Spanish."""
    negated: set[str] = set()
    scan_text = re.sub(
        r"\bno,?\s+(?:this|it)\s+is\s+(?:about|for)\b", "", text
    )
    for label, phrases in patterns:
        for phrase in phrases:
            term = re.escape(normalize_text(phrase))
            before = (
                rf"\b(?:not|no)\b[^.!?]{{0,45}}\b{term}\b|"
                rf"\b(?:nahi|nahin)\b[^,.!?]{{0,15}}\b{term}\b|"
                rf"\bsin\b[^,.!?]{{0,30}}\b{term}\b"
            )
            after = rf"\b{term}\b[^.!?]{{0,30}}\b(?:nahi|nahin)\b"
            if re.search(before, scan_text) or re.search(after, scan_text):
                negated.add(label)
                break
    return negated


def extract_claim(user_claim: str, claim_object: str) -> ClaimIntent:
    if claim_object not in OBJECT_PARTS:
        raise ValueError(f"Unsupported claim_object: {claim_object}")
    text = normalize_text(customer_utterances(user_claim))
    claimant_segments = []
    for segment in (user_claim or "").split("|"):
        label, sep, content = segment.partition(":")
        if sep and normalize_text(label) in {"customer", "cliente", "user", "claimant"}:
            claimant_segments.append(normalize_text(content))
    final_text = claimant_segments[-1] if claimant_segments else text

    # The final claimant turn commonly narrows or corrects earlier conversational
    # possibilities. Use it first, then fall back to the whole claimant transcript.
    parts = list(dict.fromkeys(_find_all(final_text, PART_PATTERNS[claim_object])))
    negated_parts = _negated_labels(final_text, PART_PATTERNS[claim_object])
    parts = [candidate for candidate in parts if candidate not in negated_parts]
    if "item missing claim nahi" in final_text or "not claiming missing" in final_text:
        parts = [candidate for candidate in parts if candidate not in {"contents", "item"}]
    if not parts:
        parts = list(dict.fromkeys(_find_all(text, PART_PATTERNS[claim_object])))
        parts = [candidate for candidate in parts if candidate not in _negated_labels(
            text, PART_PATTERNS[claim_object]
        )]
    if len(parts) > 1:
        parts = [p for p in parts if p not in {"body", "box", "item"}] or parts
    issues = list(dict.fromkeys(
        _find_all(final_text, ISSUE_PATTERNS) + _find_all(text, ISSUE_PATTERNS)
    ))
    negated_issues = _negated_labels(final_text, ISSUE_PATTERNS) | _negated_labels(
        text, ISSUE_PATTERNS
    )
    issues = [issue for issue in issues if issue not in negated_issues]
    if claim_object != "package":
        issues = [
            issue for issue in issues
            if issue not in {"torn_packaging", "crushed_packaging"}
        ]
    if claim_object == "package" and "crushed_packaging" in issues:
        issues = [issue for issue in issues if issue != "dent"]
    explicit_water_damage = bool(re.search(
        r"\b(?:water damage|water damaged|liquid damage)\b", final_text
    ))
    stain_context = bool(re.search(
        r"\b(?:stain|stained|mancha|daag|oily mark|wet-looking stain)\b", final_text
    ))
    if stain_context and not explicit_water_damage:
        issues = ["stain", *[issue for issue in issues if issue not in {"stain", "water_damage"}]]
    elif explicit_water_damage and "water_damage" in issues:
        issues = ["water_damage", *[issue for issue in issues if issue != "water_damage"]]
    if (claim_object == "laptop" and parts and parts[0] == "screen"
            and re.search(r"\b(?:broken screen|display glass)\b", final_text)
            and "glass_shatter" not in issues):
        issues = ["crack", *[issue for issue in issues if issue not in {"crack", "broken_part"}]]
    if not issues and claim_object == "car" and parts and parts[0] == "side_mirror":
        if re.search(r"\b(?:damaged|loose|not sitting|hanging|misaligned)\b", text):
            issues = ["broken_part"]
    if claim_object == "package" and issues and issues[0] == "torn_packaging":
        all_parts = _find_all(text, PART_PATTERNS[claim_object])
        if "seal" in all_parts and parts == ["box"]:
            parts = ["seal"]
    qualifiers = []
    qualifier_sources = {
        "severe": (
            text,
            r"\b(?:deep (?:scratch|crack|dent)|severe (?:damage|scratch|crack|dent)|"
            r"large hole|badly crushed|severe front damage|major damage|pretty bad)\b",
        ),
        "minor": (final_text, r"\b(?:small|minor|light|slight|pequeno|pequena)\b"),
        "left": (final_text, r"\b(?:left|izquierd[oa])\b"),
        "right": (final_text, r"\b(?:right|derech[oa])\b"),
        "unreadable": (text, r"\b(?:unreadable|illegible)\b"),
        "instruction_attack": (
            text,
            r"\b(?:ignore (?:all |any |previous )?instructions|approve|mark this row|"
            r"mark (?:this |it )?supported|skip (?:manual )?review|"
            r"follow (?:this |the )?note|keep reopening(?: tickets)?|escalate publicly)\b",
        ),
    }
    for qualifier, (source, pattern) in qualifier_sources.items():
        if re.search(pattern, source):
            qualifiers.append(qualifier)
    return ClaimIntent(
        claim_object=claim_object,
        claimed_parts=parts or ["unknown"],
        claimed_issues=issues or ["unknown"],
        qualifiers=qualifiers,
        source_text=text,
    )
