from __future__ import annotations

import unittest

import step3_foacpo_p0 as p0


class SelectionTests(unittest.TestCase):
    def test_protected_selection_is_deterministic_disjoint_and_balanced(self) -> None:
        targets = []
        for category in range(8):
            for item in range(3):
                targets.append(
                    {"sample_id": f"s{category}_{item}", "category": f"c{category}"}
                )
        first = p0.select_protected_blocks(targets, 40, 17)
        second = p0.select_protected_blocks(list(reversed(targets)), 40, 17)
        self.assertEqual(first, second)
        safety = [value for block in first.values() for value in block["safety_sample_ids"]]
        risk = [value for block in first.values() for value in block["risk_row_indices"]]
        self.assertEqual(len(safety), len(set(safety)))
        self.assertEqual(len(risk), len(set(risk)))
        self.assertTrue(all(len(block["safety_sample_ids"]) == 4 for block in first.values()))


class RouteTests(unittest.TestCase):
    def base(self) -> dict[str, bool]:
        return {
            "structural": True,
            "candidate_selected": True,
            "calibration": True,
            "mechanism": True,
            "state_restored": True,
            "zero_optimizer": True,
        }

    def test_pass(self) -> None:
        self.assertEqual(p0.route_from_gates(self.base()), "PASS_FOACPO_P0_ZERO_STEP_TRANSFER")

    def test_calibration_stop_precedes_mechanism(self) -> None:
        gates = self.base()
        gates["calibration"] = False
        gates["mechanism"] = False
        self.assertEqual(p0.route_from_gates(gates), "STOP_FOACPO_P0_CALIBRATION_TRANSFER")

    def test_invalid_precedes_method_stops(self) -> None:
        gates = self.base()
        gates["state_restored"] = False
        self.assertEqual(p0.route_from_gates(gates), "STOP_FOACPO_P0_INVALID")


if __name__ == "__main__":
    unittest.main()
