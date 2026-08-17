from __future__ import annotations

import unittest

import torch

import step3_frsc_p0 as p0


class VirtualStepTests(unittest.TestCase):
    def test_virtual_step_restores_exact_parameter_state(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        parameters = [("p", parameter)]
        state = p0._snapshot(parameters)
        p0._apply_virtual_step(
            parameters, state, {"p": torch.tensor([0.5, -0.25])}, 0.2
        )
        self.assertTrue(torch.equal(parameter.detach(), torch.tensor([0.9, -1.95])))
        p0._restore(parameters, state)
        self.assertTrue(p0._state_equal(parameters, state))


class SummaryTests(unittest.TestCase):
    @staticmethod
    def arm(worst: float, reduction: float) -> dict:
        return {
            "diagnostic_scores": {
                "minimum_score": worst,
                "scores": {name: worst + 0.01 for name in p0.CERTIFICATES},
            },
            "finite_step": {
                "objectives": {
                    name: {
                        "predicted_first_order_reduction": reduction,
                        "actual_loss_reduction": reduction * 0.9,
                        "actual_to_predicted_ratio": 0.9,
                    }
                    for name in p0.CERTIFICATES
                }
            },
        }

    def test_summary_pairs_all_eight_optimizer_units(self) -> None:
        rows = []
        for index in range(8):
            rows.append(
                {
                    "search": {
                        "selected_margin_gain_over_full": 0.02,
                        "selected": {
                            "minimum_exact_score": 0.1,
                            "subset": ["risk", "uniform"],
                        },
                        "full_certificate_control": {
                            "minimum_exact_score": 0.08
                        },
                    },
                    "arms": {
                        "baseline": self.arm(0.03, 0.01),
                        "raw_mcd": self.arm(0.07, 0.02),
                        "frsc_mcd": self.arm(0.10 + index * 0.001, 0.03),
                    },
                }
            )
        protocol = {
            "statistics": {"bootstrap_samples": 200, "bootstrap_seed": 11}
        }
        summary = p0.summarize_rows(rows, protocol)
        gain = summary["search"]["selected_margin_gain_over_full"]
        self.assertAlmostEqual(gain["bootstrap"]["mean"], 0.02)
        comparison = summary["paired"]["frsc_minus_raw_mcd_worst_score"]
        self.assertEqual(comparison["positive_rate"], 1.0)
        self.assertEqual(comparison["sign_test"]["exact_two_sided_p"], 0.0078125)


class RouteTests(unittest.TestCase):
    def test_route_precedence_and_pass(self) -> None:
        gates = {
            "integrity": True,
            "finite_radius_search": True,
            "diagnostic_common_transfer": True,
            "finite_step_common_descent": True,
            "raw_comparator_advantage": True,
            "effect_not_weak": True,
        }
        self.assertEqual(
            p0.select_route(gates),
            "PASS_FRSC_P0_AUTHORIZE_MATCHED_TRAINING_PROTOCOL",
        )
        gates["diagnostic_common_transfer"] = False
        self.assertEqual(
            p0.select_route(gates), "STOP_FRSC_P0_DIAGNOSTIC_TRANSFER_FAILED"
        )


if __name__ == "__main__":
    unittest.main()
