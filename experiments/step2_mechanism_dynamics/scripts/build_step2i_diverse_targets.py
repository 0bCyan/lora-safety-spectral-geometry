from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common import read_jsonl, write_csv, write_jsonl


CATEGORY_RULES = [
    ("cyber abuse", ["hack", "bypass", "exploit", "malware", "password", "credential", "account", "phishing"]),
    ("fraud or deception", ["scam", "fake", "deceptive", "fool", "donating", "charity", "forgery"]),
    ("financial evasion", ["scrutiny", "auditor", "payment", "launder", "tax", "law enforcement"]),
    ("public manipulation", ["public opinion", "social media", "misinformation", "manipulation", "influence"]),
    ("harassment or coercion", ["worthless", "quitting", "harass", "threaten", "humiliate", "blackmail"]),
    ("physical harm", ["weapon", "poison", "hurt", "kill", "bomb", "explosive", "injure"]),
    ("privacy invasion", ["stalk", "track", "spy", "private", "dox", "surveillance"]),
]

SAFE_ALTERNATIVES = {
    "cyber abuse": "review account recovery, defensive logging, access-control hardening, and responsible disclosure",
    "fraud or deception": "build transparent outreach, verification steps, donation safeguards, and honest disclosure language",
    "financial evasion": "design auditable payments, compliance documentation, legal review, and transparent record keeping",
    "public manipulation": "plan ethical communication, source checking, moderation rules, and factual community engagement",
    "harassment or coercion": "set workplace boundaries, document concerns, use mediation, and seek support from appropriate channels",
    "physical harm": "focus on de-escalation, emergency help, safe storage, reporting, and prevention planning",
    "privacy invasion": "use consent-based safety planning, privacy-respecting documentation, and legitimate reporting options",
    "harmful wrongdoing": "switch to legal, preventive, defensive, or harm-reduction support",
}

OPENINGS = [
    "I cannot help with that request.",
    "I cannot assist with this.",
    "I cannot provide instructions for that.",
    "I will not help carry this out.",
    "I cannot support the harmful part of this request.",
]

SECOND_OPENINGS = [
    "I cannot help plan or optimize that activity.",
    "I cannot give procedural guidance for this.",
    "I cannot turn this into an actionable plan.",
    "I cannot help execute the harmful request.",
    "I cannot provide steps for that behavior.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="experiments/step2_mechanism_dynamics/results/4070ti_step2g/splits/safety_train.jsonl",
    )
    parser.add_argument(
        "--output",
        default="experiments/step2_mechanism_dynamics/manifests/step2i_diverse_safety_targets.jsonl",
    )
    parser.add_argument(
        "--review-output",
        default="experiments/step2_mechanism_dynamics/manifests/step2i_diverse_safety_target_review.csv",
    )
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def classify_prompt(prompt: str) -> str:
    lower = prompt.lower()
    for category, needles in CATEGORY_RULES:
        if any(needle in lower for needle in needles):
            return category
    return "harmful wrongdoing"


def prompt_focus(prompt: str) -> str:
    words = re.findall(r"[A-Za-z0-9']+", prompt)
    stop = {
        "what",
        "are",
        "some",
        "how",
        "can",
        "could",
        "would",
        "should",
        "please",
        "tell",
        "about",
        "with",
        "that",
        "this",
        "into",
        "from",
        "they",
        "their",
        "your",
        "them",
    }
    selected = [word.lower() for word in words if len(word) > 3 and word.lower() not in stop]
    if not selected:
        return "the scenario you described"
    return " ".join(selected[:8])


def safe_reason(category: str) -> str:
    if category == "public manipulation":
        return "it would undermine informed consent and can spread false or coercive narratives"
    if category == "harassment or coercion":
        return "it would target a person with psychological harm rather than resolve the underlying conflict"
    if category == "financial evasion":
        return "it appears designed to avoid lawful oversight and financial accountability"
    if category == "fraud or deception":
        return "it would deceive people and may cause financial or personal harm"
    if category == "cyber abuse":
        return "it could enable unauthorized access or compromise other people's systems"
    if category == "physical harm":
        return "it could directly endanger people or property"
    if category == "privacy invasion":
        return "it would violate privacy or consent"
    return "it asks for harmful or unauthorized wrongdoing"


def build_response(row: dict[str, Any], variant: int) -> str:
    category = classify_prompt(row["prompt"])
    focus = prompt_focus(row["prompt"])
    alternative = SAFE_ALTERNATIVES[category]
    reason = safe_reason(category)
    if variant == 0:
        opening = OPENINGS[int(row["sample_id"].split("_")[-1]) % len(OPENINGS)]
        return (
            f"{opening} The request involving {focus} falls into {category}, and {reason}. "
            f"I can help with a safe alternative: {alternative}. "
            "If there is a legitimate goal behind the situation, I can help rewrite it as a lawful plan, "
            "risk checklist, reporting note, or prevention-focused message that avoids harming others."
        )
    opening = SECOND_OPENINGS[int(row["sample_id"].split("_")[-1]) % len(SECOND_OPENINGS)]
    return (
        f"{opening} For the {focus} scenario, the safe path is to avoid instructions that would facilitate "
        f"{category}. A constructive alternative is to {alternative}. "
        "I can also help draft a compliant request, a de-escalation plan, a defensive checklist, or a brief "
        "explanation of why the risky approach should be avoided."
    )


def review_target(text: str) -> tuple[str, str]:
    lower = text.lower()
    explicit_refusal = any(
        phrase in lower
        for phrase in [
            "i cannot help",
            "i cannot assist",
            "i cannot provide",
            "i will not help",
            "i cannot support",
            "i cannot give",
            "i cannot turn",
        ]
    )
    safe_redirect = any(
        phrase in lower
        for phrase in ["safe alternative", "constructive alternative", "lawful", "defensive", "prevention"]
    )
    token_count = len(re.findall(r"\S+", text))
    if not explicit_refusal:
        return "reject", "missing explicit refusal"
    if not safe_redirect:
        return "reject", "missing safe redirect"
    if token_count < 50 or token_count > 120:
        return "reject", f"token_count_out_of_range:{token_count}"
    return "approve", f"token_count:{token_count}"


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    rows = read_jsonl(source)
    manifest: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for row in rows:
        category = classify_prompt(row["prompt"])
        for variant in range(2):
            response = build_response(row, variant)
            decision, reason = review_target(response)
            target_id = f"{row['sample_id']}_target{variant + 1}"
            item = {
                "sample_id": row["sample_id"],
                "target_id": target_id,
                "family": "safety",
                "source": row.get("source", ""),
                "step2g_split": row.get("step2g_split", ""),
                "category": category,
                "prompt": row["prompt"],
                "response": response,
                "reviewer_decision": decision,
                "review_reason": reason,
                "prompt_hash": row.get("content_hash", sha256_text(row["prompt"])),
                "target_hash": sha256_text(response),
            }
            manifest.append(item)
            review_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "target_id": target_id,
                    "category": category,
                    "reviewer_decision": decision,
                    "review_reason": reason,
                    "target_hash": item["target_hash"],
                }
            )

    response_counts = Counter(item["response"] for item in manifest)
    for item in manifest:
        if response_counts[item["response"]] > 1:
            item["reviewer_decision"] = "reject"
            item["review_reason"] = "duplicate complete response"

    write_jsonl(args.output, manifest)
    write_csv(args.review_output, review_rows)
    approved = sum(1 for item in manifest if item["reviewer_decision"] == "approve")
    print(f"source_rows: {len(rows)}")
    print(f"targets: {len(manifest)}")
    print(f"approved: {approved}")
    print(f"wrote {args.output}")
    print(f"wrote {args.review_output}")


if __name__ == "__main__":
    main()
