from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "paper" / "strong_mechanism" / "scripts" / "collect_generation_targets.py"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class GenerationCollectionTest(unittest.TestCase):
    def test_partial_synthetic_collection(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            canonical_safety = root / "eval_safety.jsonl"
            canonical_utility = root / "eval_utility.jsonl"
            safety_source = [
                {"sample_id": f"s{i}", "prompt": f"safety prompt {i}"}
                for i in range(2)
            ]
            utility_source = [
                {"sample_id": f"u{i}", "prompt": f"utility prompt {i}"}
                for i in range(2)
            ]
            write_jsonl(canonical_safety, safety_source)
            write_jsonl(canonical_utility, utility_source)

            safety_generation = root / "safety.jsonl"
            utility_generation = root / "utility.jsonl"
            write_jsonl(
                safety_generation,
                [
                    {
                        **row,
                        "response": f"safe response {i}",
                        "seed": 1,
                        "run_label": "uniform_equal_dose",
                        "checkpoint": "step0064_pct100",
                    }
                    for i, row in enumerate(safety_source)
                ],
            )
            write_jsonl(
                utility_generation,
                [
                    {
                        **row,
                        "response": f"utility response {i}",
                        "seed": 1,
                        "run_label": "uniform_equal_dose",
                        "checkpoint": "step0064_pct100",
                    }
                    for i, row in enumerate(utility_source)
                ],
            )

            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "review_packet_seed": 9,
                        "expected_formal_runs": 15,
                        "registered_checkpoints": [0, 6, 16, 32, 48, 64],
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "manifest.json"
            jobs = [
                {
                    "kind": "safety_generation",
                    "arm": "uniform_equal_dose",
                    "seed": 1,
                    "model": "m",
                    "model_revision": "r",
                    "eval_set": "safety",
                    "limit": 2,
                    "canonical_eval_path": str(canonical_safety),
                    "expected_generation_paths": [
                        {"checkpoint": 64, "checkpoint_tag": "step0064_pct100", "path": str(safety_generation)}
                    ],
                },
                {
                    "kind": "utility_generation",
                    "arm": "uniform_equal_dose",
                    "seed": 1,
                    "model": "m",
                    "model_revision": "r",
                    "eval_set": "utility",
                    "limit": 2,
                    "canonical_eval_path": str(canonical_utility),
                    "expected_generation_paths": [
                        {"checkpoint": 64, "checkpoint_tag": "step0064_pct100", "path": str(utility_generation)}
                    ],
                },
            ]
            manifest.write_text(json.dumps({"registry": str(registry), "jobs": jobs}), encoding="utf-8")
            output = root / "targets"
            subprocess.run(
                [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--output-dir", str(output), "--allow-partial"],
                cwd=REPO_ROOT,
                check=True,
            )

            with (output / "safety_review_targets.csv").open("r", encoding="utf-8", newline="") as handle:
                safety_rows = list(csv.DictReader(handle))
            with (output / "quality_review_targets.csv").open("r", encoding="utf-8", newline="") as handle:
                quality_rows = list(csv.DictReader(handle))
            with (output / "utility_review_targets.csv").open("r", encoding="utf-8", newline="") as handle:
                utility_rows = list(csv.DictReader(handle))
            self.assertEqual(len(safety_rows), 2)
            self.assertEqual(len(quality_rows), 2)
            self.assertEqual(len(utility_rows), 2)
            self.assertNotIn("uniform_equal_dose", safety_rows[0]["sample_id"])


if __name__ == "__main__":
    unittest.main()
