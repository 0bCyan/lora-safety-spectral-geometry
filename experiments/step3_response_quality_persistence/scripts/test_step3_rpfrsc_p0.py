from __future__ import annotations

import unittest

import torch

import step3_rpfrsc_p0 as p0


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

    @classmethod
    def rows(cls) -> list[dict]:
        rows = []
        for index in range(8):
            rows.append(
                {
                    "coverage": {
                        "risk_microbatches": 4,
                        "safety_microbatches": 4,
                        "all_gradients_finite_nonzero": True,
                    },
                    "search": {
                        "attempted_source_subsets": 127,
                        "feasible_source_subsets": 127,
                        "infeasible_source_subsets": 0,
                        "screened_source_count": 16,
                        "attempted_homotopy_candidates": 129,
                        "feasible_homotopy_candidates": 129,
                        "infeasible_homotopy_interiors": 0,
                        "exact_map_evaluations": 239,
                        "baseline": {"clipping_active": True},
                        "selected_target_gain_over_raw": 0.004,
                        "selected": {
                            "minimum_target_exact_score": 0.08,
                            "minimum_protected_retention": 0.96,
                            "source_subset": ["tarc", "branch_entropy"],
                            "homotopy_fraction": 0.5,
                            "candidate_clipping_active": True,
                            "direction_budget_relative_error": 0.0,
                        },
                        "raw_anchor": {
                            "minimum_target_exact_score": 0.076,
                            "candidate_clipping_active": True,
                            "direction_budget_relative_error": 0.0,
                        },
                    },
                    "arms": {
                        "baseline": cls.arm(0.03, 0.01),
                        "raw_mcd": cls.arm(0.07, 0.02),
                        "rpfrsc_mcd": cls.arm(0.075 + index * 0.0001, 0.021),
                    },
                }
            )
            for arm in rows[-1]["arms"].values():
                arm["state_restored_after_virtual_step"] = True
        return rows

    @staticmethod
    def protocol() -> dict:
        return {
            "design": {
                "protected_certificates": [
                    "risk",
                    "uniform",
                    "tarc",
                    "continuation",
                ],
                "target_certificates": [
                    "etarc",
                    "branch_entropy",
                    "branch_negative_gap",
                ],
                "homotopy_fractions": list(p0.HOMOTOPY_FRACTIONS),
                "protected_retention_floor": 0.95,
                "source_top_k": 16,
                "exact_map_evaluations_per_unit": 239,
            },
            "statistics": {"bootstrap_samples": 200, "bootstrap_seed": 11},
            "gates": {
                "gradient_budget_relative_tolerance": 1e-6,
                "minimum_vram_headroom_fraction": 0.1,
                "training_target_score_min": 0.05,
                "search_target_gain_mean_min": 0.002,
                "paired_ci_lower_min": 0.0,
                "paired_sign_p_max": 0.05,
                "nonzero_fraction_rate_min": 0.75,
                "diagnostic_exact_margin_min": 0.02,
                "diagnostic_certificate_score_min": 0.02,
                "diagnostic_advantage_over_baseline_mean_min": 0.02,
                "mean_actual_loss_reduction_min": 0.0,
                "actual_reduction_positive_rate_min": 0.75,
                "mean_actual_to_predicted_ratio_min": 0.25,
                "mean_actual_to_predicted_ratio_max": 1.75,
                "protected_mean_relative_loss_max": 0.05,
                "protected_ci_relative_loss_max": 0.075,
                "protected_single_unit_relative_loss_max": 0.15,
                "diagnostic_target_gain_mean_min": 0.002,
                "target_macro_actual_relative_gain_mean_min": 0.02,
                "target_objective_mean_relative_loss_max": 0.05,
                "target_objective_single_unit_relative_loss_max": 0.20,
                "diagnostic_mean_target_score_min": 0.05,
            },
        }

    def test_summary_pairs_all_eight_optimizer_units(self) -> None:
        summary = p0.summarize_rows(self.rows(), self.protocol())
        gain = summary["search"]["selected_target_gain_over_raw"]
        self.assertAlmostEqual(gain["bootstrap"]["mean"], 0.004)
        target = summary["paired"]["rpfrsc_minus_raw_target_minimum_score"]
        self.assertEqual(target["positive_rate"], 1.0)
        macro = summary["paired"]["target_macro_actual_relative_gain"]
        self.assertAlmostEqual(macro["bootstrap"]["mean"], 0.05)

    def test_registered_gate_conjunction_passes_consistent_fixture(self) -> None:
        rows = self.rows()
        protocol = self.protocol()
        summary = p0.summarize_rows(rows, protocol)
        gates = p0.evaluate_gates(rows, summary, protocol, True, 0.25)
        self.assertTrue(all(gates.values()), gates)


class RouteTests(unittest.TestCase):
    def test_route_precedence_and_pass(self) -> None:
        gates = {
            "integrity": True,
            "protected_finite_radius_search": True,
            "diagnostic_common_transfer": True,
            "finite_step_common_descent": True,
            "protected_diagnostic_noninferiority": True,
            "target_diagnostic_advantage": True,
            "effect_not_weak": True,
        }
        self.assertEqual(
            p0.select_route(gates),
            "PASS_RPFRSC_P0_AUTHORIZE_MATCHED_TRAINING_PROTOCOL",
        )
        gates["protected_diagnostic_noninferiority"] = False
        self.assertEqual(
            p0.select_route(gates),
            "STOP_RPFRSC_P0_PROTECTED_NONINFERIORITY_FAILED",
        )
        gates["diagnostic_common_transfer"] = False
        self.assertEqual(
            p0.select_route(gates), "STOP_RPFRSC_P0_DIAGNOSTIC_TRANSFER_FAILED"
        )


if __name__ == "__main__":
    unittest.main()
