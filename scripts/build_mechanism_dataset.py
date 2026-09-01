#!/usr/bin/env python3
"""Build a small, deterministic mechanism-validation dataset manifest.

Raw upstream datasets are downloaded to a cache and are not copied into git.
The script emits normalized JSONL files plus checksums and provenance.
Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import lzma
import os
import random
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


HF = "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{path}"
RAW = "https://raw.githubusercontent.com/{repo}/{revision}/{path}"


def fetch(url: str, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        request = urllib.request.Request(url)
        token = os.environ.get("HF_TOKEN")
        if token and "huggingface.co" in url:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            src_ctx = urllib.request.urlopen(request, timeout=120)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise RuntimeError(f"Access denied for {url}. Set HF_TOKEN after accepting the upstream dataset terms.") from exc
            raise
        with src_ctx as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
    h = hashlib.sha256()
    with target.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path, compression: str | None = None) -> Iterable[dict]:
    opener = path.open
    if compression == "gz":
        opener = lambda mode="rt": gzip.open(path, mode, encoding="utf-8")
    elif compression == "xz":
        opener = lambda mode="rt": lzma.open(path, mode, encoding="utf-8")
    with opener("rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def stable_key(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def category_names(item: dict) -> list[str]:
    raw = item.get("category", item.get("harm_category", {})) or {}
    if isinstance(raw, dict):
        return sorted(str(k) for k, v in raw.items() if bool(v))
    return [str(raw)] if raw else []


def build_risk(items: list[dict], per_category: int, seed: int) -> list[dict]:
    harmful = [x for x in items if x.get("is_safe") is False]
    buckets: dict[str, list[dict]] = {}
    for item in harmful:
        for category in category_names(item):
            buckets.setdefault(category, []).append(item)
    rng = random.Random(seed)
    for bucket in buckets.values():
        bucket.sort(key=stable_key)
        rng.shuffle(bucket)
    chosen: dict[str, dict] = {}
    for category in sorted(buckets):
        for item in buckets[category][:per_category]:
            prompt = str(item.get("prompt", "")).strip()
            if prompt and prompt not in chosen:
                chosen[prompt] = {
                    "sample_id": f"risk_bt_{len(chosen):06d}",
                    "family": "risk",
                    "source": "PKU-Alignment/BeaverTails",
                    "prompt": prompt,
                    "response": str(item.get("response", "")),
                    "is_safe": False,
                    "harm_categories": category_names(item),
                    "source_record_hash": stable_key(item),
                }
    return list(chosen.values())


def build_safety(items: list[dict], maximum: int, seed: int) -> list[dict]:
    candidates = []
    for item in items:
        prompt = str(item.get("prompt", "")).strip()
        for idx in (0, 1):
            response = item.get(f"response_{idx}")
            safe = item.get(f"is_response_{idx}_safe")
            if prompt and response and safe is True:
                candidates.append((stable_key(item) + str(idx), item, idx))
    candidates.sort(key=lambda x: x[0])
    rng = random.Random(seed)
    rng.shuffle(candidates)
    out, seen = [], set()
    for _, item, idx in candidates:
        prompt = str(item["prompt"]).strip()
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append({
            "sample_id": f"safety_pku_{len(out):06d}",
            "family": "safety",
            "source": "PKU-Alignment/PKU-SafeRLHF",
            "prompt": prompt,
            "response": str(item[f"response_{idx}"]),
            "is_safe": True,
            "target_type": "preferred_harmless_response",
            "harm_categories": sorted(str(k) for k, v in (item.get(f"response_{idx}_harm_category") or {}).items() if v),
            "source_record_hash": stable_key(item),
        })
        if len(out) >= maximum:
            break
    return out


def normalize_eval(items: Iterable[dict], family: str, source: str) -> list[dict]:
    out = []
    seen = set()
    for item in items:
        prompt = str(item.get("prompt", item.get("question", item.get("Behavior", item.get("instruction", ""))))).strip()
        if not prompt:
            continue
        if prompt in seen:
            continue
        seen.add(prompt)
        categories = category_names(item)
        if not categories:
            for key in ("category", "SemanticCategory", "semantic_category", "subcategory"):
                if item.get(key):
                    categories = [str(item[key])]
                    break
        if item.get("category") and not categories:
            categories = [str(item["category"])]
        out.append({
            "sample_id": f"{family}_{len(out):06d}",
            "family": family,
            "source": source,
            "prompt": prompt,
            "harm_categories": categories,
            "source_record_hash": stable_key(item),
        })
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("data/processed/mechanism_v1"))
    ap.add_argument("--cache-dir", type=Path, default=Path(".cache/mechanism_dataset"))
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--risk-per-category", type=int, default=100)
    ap.add_argument("--safety-max", type=int, default=1400)
    ap.add_argument("--skip-sorry-bench", action="store_true", help="Build non-gated splits when SORRY-Bench access is not available.")
    args = ap.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"seed": args.seed, "files": {}}

    bt_url = HF.format(dataset="PKU-Alignment/BeaverTails", revision="8401fe609d288129cc684a9b3be6a93e41cfe678", path="round0/30k/train.jsonl.gz")
    bt_path = args.cache_dir / "beavertails_train.jsonl.gz"
    provenance["files"]["beavertails_train"] = {"url": bt_url, "sha256": fetch(bt_url, bt_path)}
    bt_items = list(read_jsonl(bt_path, "gz"))

    pku_url = HF.format(dataset="PKU-Alignment/PKU-SafeRLHF", revision="9421ffafec3fa40a1f1a7d567b4d525079477ecb", path="data/Alpaca3-8B/train.jsonl")
    pku_path = args.cache_dir / "pku_saferlhf_train.jsonl"
    provenance["files"]["pku_saferlhf_train"] = {"url": pku_url, "sha256": fetch(pku_url, pku_path)}
    pku_items = list(read_jsonl(pku_path))

    bt_eval_url = HF.format(dataset="PKU-Alignment/BeaverTails-Evaluation", revision="a93205d9ba6cc9dc398a34466da3602abddd2d40", path="v1/test.jsonl.xz")
    bt_eval_path = args.cache_dir / "beavertails_evaluation.jsonl.xz"
    provenance["files"]["beavertails_evaluation"] = {"url": bt_eval_url, "sha256": fetch(bt_eval_url, bt_eval_path)}

    sorry_path = None
    if not args.skip_sorry_bench:
        sorry_url = HF.format(dataset="sorry-bench/sorry-bench-202503", revision="612a4e1f45db8adf884fa62318ddf9fa1c6e75e9", path="question.jsonl")
        sorry_path = args.cache_dir / "sorry_bench_question.jsonl"
        provenance["files"]["sorry_bench"] = {"url": sorry_url, "sha256": fetch(sorry_url, sorry_path)}

    harm_url = RAW.format(repo="centerforaisafety/HarmBench", revision="c0423b952435fcc8467108d8f25962dbae5b7de2", path="data/behavior_datasets/harmbench_behaviors_text_test.csv")
    harm_path = args.cache_dir / "harmbench_test.csv"
    provenance["files"]["harmbench_test"] = {"url": harm_url, "sha256": fetch(harm_url, harm_path)}

    risk = build_risk(bt_items, args.risk_per_category, args.seed)
    safety = build_safety(pku_items, args.safety_max, args.seed + 1)
    utility = []
    local_utility = Path("data/processed/unified_pool/train_normal.jsonl")
    if local_utility.exists():
        for item in list(read_jsonl(local_utility))[:1400]:
            utility.append({"sample_id": f"utility_{len(utility):06d}", "family": "utility", "source": "local_alpaca", "prompt": item.get("prompt", ""), "response": item.get("response", ""), "is_safe": True})

    bt_eval = normalize_eval(read_jsonl(bt_eval_path, "xz"), "eval_beavertails", "PKU-Alignment/BeaverTails-Evaluation")
    sorry = normalize_eval(read_jsonl(sorry_path), "eval_sorry_bench", "sorry-bench/sorry-bench-202503") if sorry_path else []
    with harm_path.open("r", encoding="utf-8", newline="") as f:
        harm = normalize_eval(csv.DictReader(f), "eval_harmbench", "centerforaisafety/HarmBench")

    for name, rows in {"train_risk.jsonl": risk, "train_safety.jsonl": safety, "train_utility.jsonl": utility, "eval_beavertails.jsonl": bt_eval, "eval_sorry_bench.jsonl": sorry, "eval_harmbench.jsonl": harm}.items():
        write_jsonl(args.output_dir / name, rows)
    provenance["counts"] = {"risk": len(risk), "safety": len(safety), "utility": len(utility), "eval_beavertails": len(bt_eval), "eval_sorry_bench": len(sorry), "eval_harmbench": len(harm)}
    write_jsonl(args.output_dir / "provenance.jsonl", [{"manifest": provenance}])
    (args.output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(provenance["counts"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
