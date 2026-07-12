from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

from common import EVAL_TO_FILE, load_json, read_jsonl, write_csv, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def safety_number(sample_id: str) -> int:
    match = re.search(r"(\d+)$", sample_id)
    if not match:
        raise ValueError(f"Cannot parse numeric suffix from sample_id={sample_id!r}")
    return int(match.group(1))


def split_key(seed: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def manifest_item(seed: str, split: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": split,
        "sample_id": row["sample_id"],
        "content_hash": row.get("content_hash", ""),
        "hash_key": split_key(seed, row["sample_id"]),
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    seed = str(config["split_seed"])
    out_dir = Path(config["output_dir"]) / "splits"
    rows = read_jsonl(Path(config["data_dir"]) / EVAL_TO_FILE["safety"])
    selected = [
        row
        for row in rows
        if int(config["safety_id_min"]) <= safety_number(row["sample_id"]) <= int(config["safety_id_max"])
    ]
    expected_total = sum(int(value) for value in config["split_sizes"].values())
    if len(selected) != expected_total:
        raise ValueError(f"Expected {expected_total} rows from requested ID range, got {len(selected)}")

    selected = sorted(selected, key=lambda row: split_key(seed, row["sample_id"]))
    n_train = int(config["split_sizes"]["safety_train"])
    n_dev = int(config["split_sizes"]["proxy_dev"])
    train_rows = selected[:n_train]
    dev_rows = selected[n_train : n_train + n_dev]
    heldout_rows = selected[n_train + n_dev :]

    for split, split_rows in [
        ("safety_train", train_rows),
        ("proxy_dev", dev_rows),
    ]:
        write_jsonl(out_dir / f"{split}.jsonl", [{**row, "step2g_split": split} for row in split_rows])

    manifest_rows = (
        [manifest_item(seed, "safety_train", row) for row in train_rows]
        + [manifest_item(seed, "proxy_dev", row) for row in dev_rows]
        + [manifest_item(seed, "heldout_test", row) for row in heldout_rows]
    )
    write_csv(out_dir / "split_manifest.csv", manifest_rows)
    write_csv(out_dir / "heldout_test_manifest.csv", [manifest_item(seed, "heldout_test", row) for row in heldout_rows])
    write_csv(
        out_dir / "split_counts.csv",
        [
            {"split": "safety_train", "n": len(train_rows)},
            {"split": "proxy_dev", "n": len(dev_rows)},
            {"split": "heldout_test", "n": len(heldout_rows), "contents": "ids_only"},
        ],
    )
    print(f"wrote {out_dir / 'safety_train.jsonl'} ({len(train_rows)} rows)")
    print(f"wrote {out_dir / 'proxy_dev.jsonl'} ({len(dev_rows)} rows)")
    print(f"wrote {out_dir / 'heldout_test_manifest.csv'} ({len(heldout_rows)} ids)")


if __name__ == "__main__":
    main()
