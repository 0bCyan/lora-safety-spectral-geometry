from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from common import read_jsonl, write_csv, write_jsonl


FIXED_RESPONSE = (
    "I can't help with instructions for harmful or unauthorized activity. "
    "I can help with safe, legal alternatives, prevention, or defensive guidance instead."
)

CATEGORY_RULES = [
    ("cyber abuse", ["hack", "bypass", "exploit", "malware", "password", "credential", "account", "phishing"]),
    ("fraud or deception", ["scam", "fake", "deceptive", "fool", "donating", "charity", "forgery"]),
    ("financial evasion", ["scrutiny", "auditor", "payment", "launder", "tax", "law enforcement"]),
    ("public manipulation", ["public opinion", "social media", "misinformation", "manipulation", "influence"]),
    ("harassment or coercion", ["worthless", "quitting", "harass", "threaten", "humiliate", "blackmail"]),
    ("physical harm", ["weapon", "poison", "hurt", "kill", "bomb", "explosive", "injure"]),
    ("privacy invasion", ["stalk", "track", "spy", "private", "dox", "surveillance"]),
]

CATEGORY_REDIRECTS = {
    "cyber abuse": ("defensive logging", "access recovery"),
    "fraud or deception": ("honest disclosure", "verification steps"),
    "financial evasion": ("auditable payments", "compliance records"),
    "public manipulation": ("factual outreach", "transparent messaging"),
    "harassment or coercion": ("workplace mediation", "boundary setting"),
    "physical harm": ("de-escalation", "emergency planning"),
    "privacy invasion": ("consent-based safety", "legitimate reporting"),
    "harmful wrongdoing": ("lawful prevention", "safe reporting"),
}

TOPIC_RULES = [
    ("colleague", "workplace"),
    ("public opinion", "civic"),
    ("climate", "climate"),
    ("vaccination", "vaccine"),
    ("violence", "violence"),
    ("unrest", "unrest"),
    ("social media", "online"),
    ("payment", "financial"),
    ("charity", "fundraising"),
    ("website", "online"),
    ("account", "account"),
    ("password", "account"),
    ("drug", "health"),
    ("weapon", "safety"),
    ("bomb", "safety"),
    ("private", "privacy"),
    ("track", "privacy"),
    ("donating", "fundraising"),
    ("auditor", "financial"),
]

FILLERS = ["safely", "clearly", "lawfully", "responsibly", "with safeguards", "for prevention"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="experiments/step2_mechanism_dynamics/results/4070ti_step2g/splits/safety_train.jsonl",
    )
    parser.add_argument(
        "--fixed-output",
        default="experiments/step2_mechanism_dynamics/manifests/step2ir_fixed_matched_targets.jsonl",
    )
    parser.add_argument(
        "--diverse-output",
        default="experiments/step2_mechanism_dynamics/manifests/step2ir_diverse_matched_targets.jsonl",
    )
    parser.add_argument(
        "--review-output",
        default="experiments/step2_mechanism_dynamics/manifests/step2ir_matched_target_review.csv",
    )
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen3-4B")
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def classify_prompt(prompt: str) -> str:
    lower = prompt.lower()
    for category, needles in CATEGORY_RULES:
        if any(needle in lower for needle in needles):
            return category
    return "harmful wrongdoing"


def topic_label(prompt: str) -> str:
    lower = prompt.lower()
    words = re.findall(r"[A-Za-z]+", prompt.lower())
    stop = {
        "what",
        "some",
        "with",
        "that",
        "this",
        "they",
        "their",
        "would",
        "should",
        "could",
        "about",
        "into",
        "from",
        "people",
        "person",
        "make",
        "want",
        "need",
        "looking",
        "ways",
        "best",
    }
    selected = [word for word in words if len(word) > 4 and word not in stop]
    for needle, label in TOPIC_RULES:
        if needle in lower:
            if selected:
                return f"{label}-{selected[0]}"
            return label
    if len(selected) >= 2:
        return f"{selected[0]}-{selected[1]}"
    return selected[0] if selected else "case"


def count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_diverse_response(
    tokenizer: Any,
    row: dict[str, Any],
    variant: int,
    target_min: int,
    target_max: int,
) -> str:
    category = classify_prompt(row["prompt"])
    topic = topic_label(row["prompt"])
    redirects = CATEGORY_REDIRECTS[category]
    if variant == 0:
        text = (
            f"I can't help with {category}. "
            f"I can help plan {topic} {redirects[0]} that avoids harm instead."
        )
    else:
        text = (
            f"I won't assist {category}. "
            f"I can help draft {topic} {redirects[1]} that stays legal instead."
        )
    filler_index = int(row["sample_id"].split("_")[-1]) % len(FILLERS)
    while count_tokens(tokenizer, text) < target_min:
        insertion = FILLERS[filler_index % len(FILLERS)]
        text = text[:-8].rstrip() + f" {insertion} instead."
        filler_index += 1
    if count_tokens(tokenizer, text) > target_max:
        if variant == 0:
            text = f"I can't help with {category}. I can help plan {topic} safety instead."
        else:
            text = f"I won't assist {category}. I can help draft {topic} safeguards instead."
    while count_tokens(tokenizer, text) < target_min:
        text = text[:-8].rstrip() + " lawfully instead."
    token_count = count_tokens(tokenizer, text)
    if token_count < target_min or token_count > target_max:
        raise ValueError(f"Could not match target length for {row['sample_id']} variant {variant + 1}: {token_count}")
    return text


def manifest_row(
    row: dict[str, Any],
    target_id: str,
    condition: str,
    response: str,
    token_count: int,
    fixed_mean_tokens: float,
) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "target_id": target_id,
        "condition": condition,
        "family": "safety",
        "source": row.get("source", ""),
        "step2g_split": row.get("step2g_split", ""),
        "category": classify_prompt(row["prompt"]),
        "prompt": row["prompt"],
        "response": response,
        "token_count": token_count,
        "fixed_mean_tokens": f"{fixed_mean_tokens:.3f}",
        "prompt_hash": row.get("content_hash", sha256_text(row["prompt"])),
        "target_hash": sha256_text(response),
    }


def review_row(item: dict[str, Any], tokenizer_decision: str, reason: str) -> dict[str, Any]:
    lower = item["response"].lower()
    explicit_refusal = any(phrase in lower for phrase in ["i can't help", "i won't assist"])
    safe_redirect = any(phrase in lower for phrase in ["i can help with", "i can help plan", "i can help draft"])
    decision = tokenizer_decision
    if not explicit_refusal:
        decision = "reject"
        reason = "missing explicit refusal"
    elif not safe_redirect:
        decision = "reject"
        reason = "missing redirect"
    return {
        "sample_id": item["sample_id"],
        "target_id": item["target_id"],
        "condition": item["condition"],
        "category": item["category"],
        "token_count": item["token_count"],
        "reviewer": "local_preflight_only",
        "reviewer_decision": decision,
        "review_reason": reason,
        "target_hash": item["target_hash"],
        "human_review_required": "yes" if item["condition"] == "diverse_matched" else "no",
        "strong_judge_review_required": "yes" if item["condition"] == "diverse_matched" else "no",
    }


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError("transformers is required to measure Qwen tokenizer length") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    rows = read_jsonl(Path(args.source))
    fixed_token_count = count_tokens(tokenizer, FIXED_RESPONSE)
    fixed_mean_tokens = float(fixed_token_count)
    target_min = math_floor_10pct(fixed_mean_tokens)
    target_max = math_ceil_10pct(fixed_mean_tokens)

    fixed_manifest: list[dict[str, Any]] = []
    diverse_manifest: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    seen_diverse: set[str] = set()

    for row in rows:
        for variant in range(2):
            fixed_target_id = f"{row['sample_id']}_fixed_target{variant + 1}"
            fixed_item = manifest_row(
                row,
                fixed_target_id,
                "fixed_matched",
                FIXED_RESPONSE,
                fixed_token_count,
                fixed_mean_tokens,
            )
            fixed_manifest.append(fixed_item)
            review_rows.append(review_row(fixed_item, "approve", "matched fixed control"))

            diverse_target_id = f"{row['sample_id']}_diverse_target{variant + 1}"
            response = build_diverse_response(tokenizer, row, variant, target_min, target_max)
            if response in seen_diverse:
                raise ValueError(f"Duplicate diverse response: {response}")
            seen_diverse.add(response)
            diverse_item = manifest_row(
                row,
                diverse_target_id,
                "diverse_matched",
                response,
                count_tokens(tokenizer, response),
                fixed_mean_tokens,
            )
            diverse_manifest.append(diverse_item)
            review_rows.append(review_row(diverse_item, "approve", f"token_window:{target_min}-{target_max}"))

    write_jsonl(args.fixed_output, fixed_manifest)
    write_jsonl(args.diverse_output, diverse_manifest)
    write_csv(args.review_output, review_rows)
    print(f"fixed_token_count={fixed_token_count} target_window={target_min}-{target_max}")
    print(f"wrote {args.fixed_output} ({len(fixed_manifest)} rows)")
    print(f"wrote {args.diverse_output} ({len(diverse_manifest)} rows)")
    print(f"wrote {args.review_output} ({len(review_rows)} rows)")


def math_floor_10pct(value: float) -> int:
    import math

    return int(math.ceil(value * 0.9))


def math_ceil_10pct(value: float) -> int:
    import math

    return int(math.floor(value * 1.1))


if __name__ == "__main__":
    main()
