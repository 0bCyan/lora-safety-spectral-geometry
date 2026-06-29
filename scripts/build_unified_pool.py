"""Build the first unified dataset pool for LoRA safety-drift experiments.

The script creates a small, deterministic JSONL pool from public datasets:

- normal: Stanford Alpaca instruction-following samples
- risk: PKU-SafeRLHF-QA unsafe samples
- control: shuffled prompt/response pairs from the normal split

The committed default pool is intentionally small enough for git and pilot
experiments. Larger pools should be generated on experiment machines and kept
out of git unless explicitly approved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
    "main/alpaca_data.json"
)

PKU_QA_URL = (
    "https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-QA/"
    "resolve/main/data/Alpaca3-8B/train.jsonl"
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def format_alpaca_prompt(instruction: str, input_text: str) -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    if input_text:
        return f"{instruction}\n\n输入：{input_text}"
    return instruction


def fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def iter_jsonl_url(url: str) -> Iterable[dict]:
    with urllib.request.urlopen(url, timeout=120) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_alpaca_rows() -> list[dict]:
    raw_rows = fetch_json(ALPACA_URL)
    rows = []
    for index, item in enumerate(raw_rows, start=1):
        prompt = format_alpaca_prompt(item.get("instruction", ""), item.get("input", ""))
        response = str(item.get("output", "")).strip()
        if not prompt or not response:
            continue
        rows.append(
            {
                "sample_id": f"normal_alpaca_{index:06d}",
                "family": "normal",
                "source": "alpaca",
                "prompt": prompt,
                "response": response,
                "is_safe_source": True,
                "source_index": index,
                "content_hash": sha256_text(prompt + "\n" + response),
            }
        )
    return rows


def load_pku_risk_rows(required: int, max_severity: int | None = None) -> list[dict]:
    rows = []
    seen = set()
    for index, item in enumerate(iter_jsonl_url(PKU_QA_URL), start=1):
        if item.get("is_safe") is not False:
            continue
        severity = item.get("severity_level")
        if max_severity is not None and severity is not None and int(severity) > max_severity:
            continue
        prompt = str(item.get("prompt", "")).strip()
        response = str(item.get("response", "")).strip()
        if not prompt or not response:
            continue
        key = sha256_text(prompt)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "sample_id": f"risk_pkuqa_{len(rows) + 1:06d}",
                "family": "risk",
                "source": "PKU-SafeRLHF-QA/Alpaca3-8B",
                "prompt": prompt,
                "response": response,
                "is_safe_source": False,
                "severity_level": severity,
                "harm_category": item.get("harm_category", {}),
                "source_index": index,
                "content_hash": sha256_text(prompt + "\n" + response),
            }
        )
        if len(rows) >= required:
            break
    if len(rows) < required:
        raise RuntimeError(f"Only collected {len(rows)} PKU risk rows, need {required}.")
    return rows


def make_control_rows(normal_rows: list[dict], count: int, seed: int) -> list[dict]:
    base = normal_rows[:count]
    responses = [row["response"] for row in base]
    rng = random.Random(seed)
    rng.shuffle(responses)
    if any(row["response"] == response for row, response in zip(base, responses)):
        responses = responses[1:] + responses[:1]

    rows = []
    for index, (row, response) in enumerate(zip(base, responses), start=1):
        rows.append(
            {
                "sample_id": f"control_shuffle_{index:06d}",
                "family": "control",
                "source": "alpaca_shuffled_response",
                "prompt": row["prompt"],
                "response": response,
                "is_safe_source": None,
                "source_prompt_id": row["sample_id"],
                "content_hash": sha256_text(row["prompt"] + "\n" + response),
            }
        )
    return rows


def strip_response_for_eval(rows: list[dict], family: str, source: str) -> list[dict]:
    eval_rows = []
    for index, row in enumerate(rows, start=1):
        eval_rows.append(
            {
                "sample_id": f"eval_{family}_{index:06d}",
                "family": family,
                "source": source,
                "prompt": row["prompt"],
                "reference_response": row.get("response", ""),
                "content_hash": sha256_text(row["prompt"]),
            }
        )
    return eval_rows


def build_pool(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    total_normal_needed = args.train_per_family + args.eval_per_family
    total_risk_needed = args.train_per_family + args.eval_per_family

    print("Downloading Alpaca normal data...", file=sys.stderr)
    alpaca_rows = load_alpaca_rows()
    if len(alpaca_rows) < total_normal_needed:
        raise RuntimeError(f"Only collected {len(alpaca_rows)} Alpaca rows.")

    print("Downloading PKU risk data...", file=sys.stderr)
    pku_rows = load_pku_risk_rows(total_risk_needed, args.max_risk_severity)

    normal_train = alpaca_rows[: args.train_per_family]
    utility_eval_source = alpaca_rows[
        args.train_per_family : args.train_per_family + args.eval_per_family
    ]

    risk_train = pku_rows[: args.train_per_family]
    safety_eval_source = pku_rows[
        args.train_per_family : args.train_per_family + args.eval_per_family
    ]

    control_train = make_control_rows(normal_train, args.train_per_family, args.seed)

    eval_utility = strip_response_for_eval(utility_eval_source, "utility", "alpaca")
    eval_safety = strip_response_for_eval(
        safety_eval_source, "safety", "PKU-SafeRLHF-QA/Alpaca3-8B"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train_normal.jsonl", normal_train)
    write_jsonl(output_dir / "train_risk.jsonl", risk_train)
    write_jsonl(output_dir / "train_control.jsonl", control_train)
    write_jsonl(output_dir / "eval_utility.jsonl", eval_utility)
    write_jsonl(output_dir / "eval_safety.jsonl", eval_safety)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder": "scripts/build_unified_pool.py",
        "seed": args.seed,
        "train_per_family": args.train_per_family,
        "eval_per_family": args.eval_per_family,
        "max_risk_severity": args.max_risk_severity,
        "sources": {
            "normal": ALPACA_URL,
            "risk": PKU_QA_URL,
            "control": "constructed by shuffling normal train responses",
        },
        "files": {
            "train_normal": "train_normal.jsonl",
            "train_risk": "train_risk.jsonl",
            "train_control": "train_control.jsonl",
            "eval_utility": "eval_utility.jsonl",
            "eval_safety": "eval_safety.jsonl",
        },
        "counts": {
            "train_normal": len(normal_train),
            "train_risk": len(risk_train),
            "train_control": len(control_train),
            "eval_utility": len(eval_utility),
            "eval_safety": len(eval_safety),
        },
        "notes": [
            "4070 Ti should use fixed prefixes from these files for pilot runs.",
            "3090 Ti should use larger fixed prefixes from the same files.",
            "Do not print risk samples in logs or chat; inspect locally only when needed.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="data/processed/unified_pool",
        help="Directory for JSONL pool files.",
    )
    parser.add_argument("--train-per-family", type=int, default=1000)
    parser.add_argument("--eval-per-family", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument(
        "--max-risk-severity",
        type=int,
        default=None,
        help="Optional maximum PKU severity level to include.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build_pool(parse_args())
