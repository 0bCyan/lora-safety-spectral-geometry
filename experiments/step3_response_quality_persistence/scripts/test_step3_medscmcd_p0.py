from __future__ import annotations

import json
import unittest

import torch

import step3_etarc_p0 as etarc_p0
import step3_medscmcd_p0 as p0


class RoleSplitTests(unittest.TestCase):
    @classmethod
    def targets(cls) -> list[dict]:
        return etarc_p0.read_jsonl(
            p0.REPO
            / "experiments/step2_mechanism_dynamics/manifests/"
            "step2ir_natural_diverse_targets_v1.jsonl"
        )

    def test_global_hash_roles_are_deterministic_and_disjoint(self) -> None:
        risk_order = list(range(256))
        first = p0.expected_roles(self.targets(), risk_order)
        second = p0.expected_roles(self.targets(), risk_order)
        self.assertEqual(first, second)
        self.assertEqual(len(first["fit"]), 12)
        self.assertEqual(len(first["calibration"]), 20)
        self.assertEqual(len(first["diagnostic"]), 8)
        units = [item for role in first.values() for item in role]
        self.assertEqual(len({item["sample_id"] for item in units}), 40)
        self.assertEqual([item["risk_row_index"] for item in units], list(range(40)))


class FitGeometryTests(unittest.TestCase):
    def test_toy_fit_geometry_runs_all_thirty_three_candidates(self) -> None:
        gradients = []
        coverages = []
        for unit_index in range(12):
            gradients.append(
                {
                    name: {
                        "p": torch.tensor(
                            [1.0, 0.01 * certificate_index + 0.001 * unit_index],
                            dtype=torch.float64,
                        )
                    }
                    for certificate_index, name in enumerate(p0.CERTIFICATES)
                }
            )
            coverages.append(
                {
                    "mode": "eval",
                    "risk_microbatches": 1,
                    "safety_microbatches": 1,
                    "all_gradients_finite_nonzero": True,
                }
            )
        config = {
            "safety_mix": 0.2,
            "auxiliary_gradient_norm_ratio": 1.0,
            "max_grad_norm": 1.0,
            "adam_epsilon": 1e-2,
            "clip_epsilon": 1e-6,
            "dual_numeric_tolerance": 1e-9,
        }
        protocol = {"gates": {"numeric_tolerance": 1e-12}}
        geometry, displacements = p0.build_fit_geometry(
            gradients, coverages, config, protocol
        )
        self.assertEqual(geometry["path_candidates"], 33)
        self.assertEqual(geometry["search"]["attempted_candidates"], 33)
        self.assertEqual(len(geometry["fit_path_records"]), 12)
        self.assertEqual(set(displacements), set(p0.ARMS))
        json.dumps(geometry, allow_nan=False)


class GateTests(unittest.TestCase):
    @staticmethod
    def paired(value: float) -> dict:
        return {
            "defined": True,
            "differences": [value] * 8,
            "bootstrap": {
                "mean": value,
                "ci95_lower": value,
                "ci95_upper": value,
            },
            "sign_test": {"exact_two_sided_p": 0.0078125},
        }

    @classmethod
    def summary(cls) -> dict:
        relative = {
            name: cls.paired(-0.02 if name in p0.PROTECTED_CERTIFICATES else 0.1)
            for name in p0.CERTIFICATES
        }
        objectives = {
            name: {
                "mean_actual_reduction": 0.1,
                "positive_actual_rate": 1.0,
                "mean_actual_to_predicted_ratio": 0.9,
            }
            for name in p0.CERTIFICATES
        }
        return {
            "fit": {
                "selected_eligible": True,
                "eligible_candidates": 2,
                "selected_fraction": 0.03125,
                "minimum_selected_protected_retention": 0.97,
                "selected_target_score_floor": 0.08,
                "selected_target_gain_over_raw": 0.003,
            },
            "calibration": {
                "raw_protected_scores_positive": True,
                "conformal": {
                    "calibration_units": 20,
                    "order_statistic_rank": 20,
                    "finite_sample_marginal_coverage_lower_bound": 20 / 21,
                    "finite_sample_marginal_policy_protection_lower_bound": 20 / 21,
                    "threshold": -0.01,
                    "all_calibration_units_protected": True,
                    "candidate_accepted": True,
                    "policy_action": "candidate",
                    "fallback_nonconformity": -0.05,
                    "conditional_on_candidate_acceptance_coverage_claimed": False,
                },
            },
            "diagnostic": {
                "medscmcd_mcd": {
                    "minimum_worst_score": 0.08,
                    "mean_target_minimum_score": 0.09,
                    "minimum_certificate_scores": {
                        name: 0.08 for name in p0.CERTIFICATES
                    },
                    "objectives": objectives,
                }
            },
            "paired": {
                "medscmcd_minus_baseline_worst_score": cls.paired(0.03),
                "medscmcd_minus_raw_target_minimum_score": cls.paired(0.003),
                "medscmcd_relative_exact_score": relative,
                "medscmcd_relative_actual_reduction": relative,
                "target_macro_actual_relative_gain": cls.paired(0.1),
            },
        }

    @staticmethod
    def protocol() -> dict:
        return {
            "design": {
                "protected_certificates": list(p0.PROTECTED_CERTIFICATES),
                "target_certificates": list(p0.TARGET_CERTIFICATES),
            },
            "gates": {
                "numeric_tolerance": 1e-12,
                "gradient_budget_relative_tolerance": 1e-6,
                "minimum_vram_headroom_fraction": 0.1,
                "minimum_nonzero_path_fraction": 0.03125,
                "conformal_coverage_lower_bound_min": 0.95,
                "diagnostic_exact_margin_min": 0.02,
                "diagnostic_certificate_score_min": 0.02,
                "diagnostic_advantage_over_baseline_mean_min": 0.02,
                "paired_ci_lower_min": 0.0,
                "paired_sign_p_max": 0.05,
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

    @staticmethod
    def geometry() -> dict:
        return {
            "fit_coverages": [
                {
                    "mode": "eval",
                    "risk_microbatches": 1,
                    "safety_microbatches": 1,
                    "all_gradients_finite_nonzero": True,
                }
                for _ in range(12)
            ],
            "path_candidates": 33,
            "pooled_path_records": [{}] * 33,
            "fit_path_records": {f"fit_{i:02d}": [{}] * 33 for i in range(1, 13)},
            "search": {
                "attempted_candidates": 33,
                "calibration_scores_used_for_selection": False,
                "diagnostic_scores_used_for_selection": False,
            },
            "baseline_optimizer_map": {"clipping_active": True},
            "raw_optimizer_map": {"clipping_active": True},
            "selected_optimizer_map": {"clipping_active": True},
            "raw_direction_budget_relative_error": 0.0,
            "target_direction_budget_relative_error": 0.0,
            "selected_direction_budget_relative_error": 0.0,
            "calibration_scores_used_for_selection": False,
            "diagnostic_scores_used_for_selection": False,
            "raw_gradient_tensors_serialized": False,
        }

    @staticmethod
    def calibration_rows() -> list[dict]:
        return [
            {
                "coverage": {
                    "mode": "eval",
                    "all_gradients_finite_nonzero": True,
                }
            }
            for _ in range(20)
        ]

    @staticmethod
    def diagnostic_rows() -> list[dict]:
        return [
            {
                "coverage": {
                    "mode": "eval",
                    "all_gradients_finite_nonzero": True,
                },
                "arms": {
                    arm: {"state_restored_after_virtual_step": True}
                    for arm in p0.ARMS
                },
            }
            for _ in range(8)
        ]

    def test_complete_gate_fixture_passes(self) -> None:
        gates = p0.evaluate_gates(
            self.diagnostic_rows(),
            self.geometry(),
            self.calibration_rows(),
            self.summary(),
            self.protocol(),
            True,
            0.2,
            "NVIDIA GeForce RTX 4070 Ti",
        )
        self.assertTrue(all(gates.values()), gates)
        self.assertEqual(
            p0.select_route(gates),
            "PASS_MEDSCMCD_P0_AUTHORIZE_PROXY_DEV_TRIPLET_PROTOCOL",
        )

    def test_conformal_failure_precedes_diagnostic_results(self) -> None:
        gates = {
            "integrity": True,
            "fit_minimum_effective_dose": True,
            "split_conformal_accept_or_raw_policy": False,
            "diagnostic_common_transfer": True,
            "finite_step_common_descent": True,
            "protected_diagnostic_noninferiority": True,
            "target_diagnostic_advantage": True,
            "effect_not_weak": True,
        }
        self.assertEqual(
            p0.select_route(gates),
            "STOP_MEDSCMCD_P0_CONFORMAL_SELECTIVE_POLICY_FAILED",
        )


if __name__ == "__main__":
    unittest.main()
