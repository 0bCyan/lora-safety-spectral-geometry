from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "paper" / "strong_mechanism" / "scripts" / "plot_available_figures.py"
CONFIG_PATH = REPO_ROOT / "paper" / "strong_mechanism" / "configs" / "qwen3_4b_equal_dose_temporal_schedules.json"
HISTORICAL_PATH = REPO_ROOT / "paper" / "strong_mechanism" / "configs" / "historical_figure_data.json"


def load_module():
    spec = importlib.util.spec_from_file_location("paper_a_plot_available_figures", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AvailableFigureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_equal_dose_is_exact(self) -> None:
        steps = np.arange(int(self.config["paper_design"]["total_steps"]))
        totals = {}
        risk_totals = {}
        for arm in self.config["intervention_arms"]:
            values = self.module._schedule_values(self.config, arm, steps)
            totals[arm] = float(values.sum())
            risk_totals[arm] = float((1.0 - values).sum())
        self.assertEqual(set(round(value, 10) for value in totals.values()), {10.45})
        self.assertEqual(set(round(value, 10) for value in risk_totals.values()), {53.55})
        self.assertGreater(
            self.module._schedule_values(self.config, "front_loaded_equal_dose", steps)[0],
            self.module._schedule_values(self.config, "back_loaded_equal_dose", steps)[0],
        )
        self.assertLess(
            self.module._schedule_values(self.config, "front_loaded_equal_dose", steps)[63],
            self.module._schedule_values(self.config, "back_loaded_equal_dose", steps)[63],
        )

    def test_historical_registry_has_every_frozen_panel(self) -> None:
        data = json.loads(HISTORICAL_PATH.read_text(encoding="utf-8"))
        required = {
            "subspace_top_direction_cosine",
            "posthoc_best_unsafe",
            "step16_sequence_anchor_unsafe",
            "historical_seed1_step64",
        }
        self.assertTrue(required.issubset(data))

    def test_all_available_svg_assets_render(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            historical = json.loads(HISTORICAL_PATH.read_text(encoding="utf-8"))
            self.module.figure_paper_story(output, ["svg"], 120)
            self.module.figure_equal_dose_design(output, ["svg"], 120)
            self.module.figure_evidence_pipeline(output, ["svg"], 120)
            self.module.figure_data_boundaries(output, ["svg"], 120)
            self.module.figure_historical_subspace(output, ["svg"], 120, historical)
            self.module.figure_historical_posthoc(output, ["svg"], 120, historical)
            self.module.figure_historical_sequence_anchor(output, ["svg"], 120, historical)
            self.module.figure_historical_persistence_context(output, ["svg"], 120, historical)
            self.module.figure_completion_gates(output, ["svg"], 120)
            assets = sorted(output.glob("*.svg"))
            self.assertEqual(len(assets), 9)
            self.assertTrue(all(path.stat().st_size > 1000 for path in assets))


if __name__ == "__main__":
    unittest.main()
