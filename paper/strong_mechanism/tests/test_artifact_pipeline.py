from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper" / "strong_mechanism"


class ArtifactPipelineTest(unittest.TestCase):
    def test_tables_and_figures_from_normalized_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            derived = root / "derived"
            tables = root / "tables"
            figures = root / "figures"
            derived.mkdir()

            run_rows = []
            quality_rows = []
            gradient_rows = []
            arms = ["front_loaded_equal_dose", "uniform_equal_dose", "back_loaded_equal_dose"]
            for arm_index, arm in enumerate(arms):
                for seed in [1, 2, 3]:
                    for checkpoint in [32, 64]:
                        unsafe = 0.65 - 0.08 * arm_index - 0.01 * seed - (0.10 if checkpoint == 32 else 0.0)
                        run_rows.append(
                            {
                                "setting": "synthetic",
                                "model": "synthetic-model",
                                "arm": arm,
                                "seed": seed,
                                "checkpoint": checkpoint,
                                "n_prompts": 10,
                                "unsafe_rate": unsafe,
                                "mixed_rate": 0.0,
                                "clean_refusal_rate": 1.0 - unsafe,
                                "invalid_rate": 0.0,
                                "unsafe_rebound_32_64": 0.10,
                            }
                        )
                        quality_rows.append(
                            {
                                "setting": "synthetic",
                                "model": "synthetic-model",
                                "arm": arm,
                                "seed": seed,
                                "checkpoint": checkpoint,
                                "n_prompts": 10,
                                "safety_correctness_rate": 1.0 - unsafe,
                                "prompt_binding_rate": 0.6 + 0.05 * arm_index,
                                "safe_actionability_rate": 0.55 + 0.05 * arm_index,
                                "detail_retention_rate": 0.50 + 0.05 * arm_index,
                                "naturalness_rate": 0.75,
                                "generic_only_refusal_rate": 0.20,
                                "invalid_output_rate": 0.0,
                                "mean_derived_quality": 1.0 + 0.10 * arm_index,
                                "n_safety_correct": 8,
                                "mean_quality_conditional_safe": 1.4,
                            }
                        )
                    for checkpoint in [32, 48, 64]:
                        gradient_rows.append(
                            {
                                "setting": "synthetic",
                                "model": "synthetic-model",
                                "arm": arm,
                                "seed": seed,
                                "checkpoint": checkpoint,
                                "n_blocks": 2,
                                "weighted_ratio_mean": 0.2 + 0.1 * arm_index,
                                "weighted_ratio_min": 0.18 + 0.1 * arm_index,
                                "weighted_ratio_max": 0.22 + 0.1 * arm_index,
                            }
                        )

            pd.DataFrame(run_rows).to_csv(derived / "run_metrics.csv", index=False)
            pd.DataFrame(quality_rows).to_csv(derived / "quality_metrics.csv", index=False)
            pd.DataFrame(gradient_rows).to_csv(derived / "gradient_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "comparison": "back_vs_front_step64",
                        "level": "seed",
                        "seed": seed,
                        "checkpoint": 64,
                        "arm_a": "back_loaded_equal_dose",
                        "arm_b": "front_loaded_equal_dose",
                        "n_pairs": 10,
                        "effect_b_minus_a": 0.15,
                        "ci_lower": 0.05,
                        "ci_upper": 0.25,
                        "mcnemar_a_only": 0,
                        "mcnemar_b_only": 3,
                        "mcnemar_p": 0.25,
                    }
                    for seed in [1, 2, 3]
                ]
            ).to_csv(derived / "paired_effects.csv", index=False)

            subprocess.run(
                [
                    sys.executable,
                    str(PAPER_ROOT / "scripts" / "build_tables.py"),
                    "--derived",
                    str(derived),
                    "--output-dir",
                    str(tables),
                ],
                cwd=REPO_ROOT,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(PAPER_ROOT / "scripts" / "plot_main_figures.py"),
                    "--derived",
                    str(derived),
                    "--output-dir",
                    str(figures),
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            self.assertTrue((tables / "table_final_safety.tex").exists())
            self.assertIn(r"\\", (tables / "table_final_safety.tex").read_text(encoding="utf-8"))
            self.assertTrue((figures / "fig2_safety_trajectories.pdf").exists())
            self.assertTrue((figures / "fig5_safety_quality_frontier.png").exists())


if __name__ == "__main__":
    unittest.main()
