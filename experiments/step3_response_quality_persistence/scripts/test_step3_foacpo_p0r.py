from __future__ import annotations

import unittest

import step3_foacpo_p0r as p0r


class RouteTests(unittest.TestCase):
    def gates(self) -> dict[str, bool]:
        return {
            "structural": True,
            "candidate_selected": True,
            "calibration": True,
            "mechanism": True,
            "state_restored": True,
            "zero_optimizer": True,
        }

    def test_pass(self) -> None:
        self.assertEqual(
            p0r.route_from_gates(self.gates()),
            "PASS_FOACPO_P0R_ANCHOR_RELATIVE_TRANSFER",
        )

    def test_no_pareto_route_stops_before_calibration(self) -> None:
        gates = self.gates()
        gates["candidate_selected"] = False
        self.assertEqual(
            p0r.route_from_gates(gates),
            "STOP_FOACPO_P0R_NO_ANCHOR_RELATIVE_PARETO_ROUTE",
        )

    def test_invalid_has_priority(self) -> None:
        gates = self.gates()
        gates["zero_optimizer"] = False
        self.assertEqual(p0r.route_from_gates(gates), "STOP_FOACPO_P0R_INVALID")


if __name__ == "__main__":
    unittest.main()
