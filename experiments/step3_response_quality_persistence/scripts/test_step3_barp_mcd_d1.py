from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import step3_barp_mcd_d1 as d1


class ProspectiveSelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_and_disjoint(self) -> None:
        rows = [
            {"sample_id": f"id_{category}_{index}", "category": f"c{category}"}
            for category in range(8)
            for index in range(5)
            for _ in range(2)
        ]
        blocks, diagnostic, categories = d1.expected_safety_selection(rows)
        second = d1.expected_safety_selection(list(reversed(rows)))
        self.assertEqual((blocks, diagnostic, categories), second)
        flat = [value for block in blocks for value in block]
        self.assertEqual(len(blocks), 8)
        self.assertTrue(all(len(block) == 4 for block in blocks))
        self.assertEqual(len(diagnostic), 8)
        self.assertEqual(len(categories), 8)
        self.assertFalse(set(flat).intersection(diagnostic))
        self.assertEqual(len(set(flat + diagnostic)), 40)

    def test_selection_accepts_frozen_unequal_category_counts(self) -> None:
        counts = [23, 4, 4, 3, 3, 1, 1, 1]
        rows = [
            {"sample_id": f"id_{category}_{index}", "category": f"c{category}"}
            for category, count in enumerate(counts)
            for index in range(count)
            for _ in range(2)
        ]
        blocks, diagnostic, categories = d1.expected_safety_selection(rows)
        self.assertEqual(sum(map(len, blocks)), 32)
        self.assertEqual(len(diagnostic), 8)
        self.assertEqual(len(categories), 8)


class PairedStatisticsTests(unittest.TestCase):
    def test_bootstrap_is_deterministic(self) -> None:
        values = [0.01, 0.02, 0.03, 0.04]
        left = d1.bootstrap_mean(values, 1000, 5001)
        right = d1.bootstrap_mean(values, 1000, 5001)
        self.assertEqual(left, right)
        self.assertAlmostEqual(float(left["mean"]), 0.025)
        self.assertLessEqual(float(left["lower_2_5_percent"]), 0.025)
        self.assertGreaterEqual(float(left["upper_97_5_percent"]), 0.025)

    def test_exact_sign_test_requires_eight_consistent_units_at_five_percent(self) -> None:
        all_positive = d1.exact_two_sided_sign_test([0.1] * 8)
        seven_positive = d1.exact_two_sided_sign_test([0.1] * 7 + [-0.1])
        self.assertAlmostEqual(all_positive["two_sided_p_value"], 0.0078125)
        self.assertAlmostEqual(seven_positive["two_sided_p_value"], 0.0703125)

    def test_diagnostic_advantage_uses_shared_initial_scale(self) -> None:
        objectives = list(d1.rp.CERTIFICATES)
        initial = {name: 2.0 for name in objectives}
        raw_final = {name: 1.6 for name in objectives}
        barp_final = {name: 1.4 for name in objectives}
        manifests = {}
        for arm, final in (("raw_mcd", raw_final), ("barp_mcd", barp_final)):
            manifests[arm] = {
                "initial_diagnostic_rows": [
                    {
                        "sample_id": f"id{index}",
                        "category": f"c{index}",
                        "risk_row_index": index,
                        "losses": initial,
                    }
                    for index in range(8)
                ],
                "final_diagnostic_rows": [
                    {
                        "sample_id": f"id{index}",
                        "category": f"c{index}",
                        "risk_row_index": index,
                        "losses": final,
                    }
                    for index in range(8)
                ],
            }
        protocol = {
            "selection": {
                "diagnostic_block": {
                    "safety_sample_ids": [f"id{index}" for index in range(8)]
                }
            },
            "statistics": {
                "bootstrap_resamples": 100,
                "bootstrap_seed": 5001,
                "sign_test_tie_tolerance": 0.0,
            },
        }
        result = d1.paired_diagnostic_analysis(manifests, protocol)
        self.assertAlmostEqual(result["target_macro_advantage"]["mean"], 0.1)
        for item in result["protected_disadvantage"].values():
            self.assertAlmostEqual(item["mean"], -0.1)


class CheckpointTests(unittest.TestCase):
    def test_inventory_accepts_adapter_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_model.safetensors").write_bytes(b"adapter")
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            inventory = d1.checkpoint_inventory(root)
            self.assertEqual(
                {item["path"] for item in inventory},
                {"adapter_model.safetensors", "adapter_config.json"},
            )

    def test_inventory_rejects_base_model_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_model.safetensors").write_bytes(b"adapter")
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"base")
            with self.assertRaisesRegex(d1.BarpD1ValidationError, "base-model"):
                d1.checkpoint_inventory(root)


class RouteTests(unittest.TestCase):
    @staticmethod
    def passing() -> dict[str, bool]:
        return {
            "arms_complete": True,
            "steps_exact": True,
            "no_skipped_updates": True,
            "initial_state_equal": True,
            "block_order_equal": True,
            "initial_diagnostic_equal": True,
            "optimizer_map_exact": True,
            "raw_common_descent": True,
            "barp_protection": True,
            "barp_target_nondegradation": True,
            "barp_assignment_exercised": True,
            "finite_step_common_descent": True,
            "vram": True,
            "checkpoints": True,
            "protected_diagnostic": True,
            "target_diagnostic_mean": True,
            "target_diagnostic_bootstrap": True,
            "target_diagnostic_sign": True,
            "boundary": True,
        }

    def test_pass_route(self) -> None:
        self.assertEqual(
            d1.select_route(self.passing()),
            "PASS_BARP_MCD_D1_AUTHORIZE_CHECKPOINT_BOUND_GENERATION_PROTOCOL",
        )

    def test_protection_failure_precedes_target_statistics(self) -> None:
        checks = self.passing()
        checks["protected_diagnostic"] = False
        checks["target_diagnostic_sign"] = False
        self.assertEqual(
            d1.select_route(checks),
            "STOP_BARP_MCD_D1_DIAGNOSTIC_PROTECTION_FAILED",
        )


if __name__ == "__main__":
    unittest.main()
