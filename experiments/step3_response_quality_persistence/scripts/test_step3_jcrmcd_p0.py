from __future__ import annotations

import json
import unittest
from collections import Counter

import torch

import step3_etarc_p0 as etarc_p0
import step3_jcrmcd_p0 as p0


class PartitionTests(unittest.TestCase):
    @classmethod
    def targets(cls) -> list[dict]:
        return etarc_p0.read_jsonl(
            p0.REPO
            / "experiments/step2_mechanism_dynamics/manifests/"
            "step2ir_natural_diverse_targets_v1.jsonl"
        )

    def test_supported_domain_partition_is_deterministic_and_disjoint(self) -> None:
        first_supports, first_diagnostics = p0.expected_safety_partition(
            self.targets()
        )
        second_supports, second_diagnostics = p0.expected_safety_partition(
            self.targets()
        )
        self.assertEqual(first_supports, second_supports)
        self.assertEqual(first_diagnostics, second_diagnostics)
        self.assertEqual([len(support) for support in first_supports], [8] * 4)
        support_ids = {sample_id for support in first_supports for sample_id in support}
        diagnostic_ids = {item["sample_id"] for item in first_diagnostics}
        self.assertFalse(support_ids.intersection(diagnostic_ids))
        self.assertEqual(len(support_ids | diagnostic_ids), 40)
        self.assertEqual(
            Counter(item["category"] for item in first_diagnostics),
            Counter(dict(p0.DIAGNOSTIC_QUOTAS)),
        )

    def test_singleton_categories_remain_in_construction(self) -> None:
        targets = self.targets()
        registry = p0.prompt_category_registry(targets)
        category_counts = Counter(registry.values())
        supports, diagnostics = p0.expected_safety_partition(targets)
        support_ids = {sample_id for support in supports for sample_id in support}
        diagnostic_ids = {item["sample_id"] for item in diagnostics}
        singleton_ids = {
            sample_id
            for sample_id, category in registry.items()
            if category_counts[category] == 1
        }
        self.assertTrue(singleton_ids.issubset(support_ids))
        self.assertFalse(singleton_ids.intersection(diagnostic_ids))


class GradientAggregationTests(unittest.TestCase):
    def test_support_gradients_are_meaned_coordinatewise(self) -> None:
        supports = []
        for index in range(4):
            supports.append(
                {
                    name: {
                        "p": torch.tensor([index + 1.0, 2.0 * (index + 1.0)])
                    }
                    for name in p0.CERTIFICATES
                }
            )
        mean = p0.mean_certificate_gradients(supports)
        for certificate in p0.CERTIFICATES:
            self.assertTrue(
                torch.equal(mean[certificate]["p"], torch.tensor([2.5, 5.0]))
            )

    def test_toy_construction_runs_full_shared_path_geometry(self) -> None:
        supports = []
        coverages = []
        for support_index in range(4):
            supports.append(
                {
                    name: {
                        "p": torch.tensor(
                            [
                                1.0,
                                0.02 * certificate_index
                                + 0.005 * support_index,
                            ],
                            dtype=torch.float64,
                        )
                    }
                    for certificate_index, name in enumerate(p0.CERTIFICATES)
                }
            )
            coverages.append(
                {
                    "risk_microbatches": 8,
                    "safety_microbatches": 8,
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
        protocol = {
            "design": {
                "target_certificates": list(p0.TARGET_CERTIFICATES),
                "protected_certificates": list(p0.PROTECTED_CERTIFICATES),
                "path_fractions": list(p0.PATH_FRACTIONS),
                "protected_retention_floor": 0.95,
            },
            "gates": {"robust_numeric_tolerance": 1e-12},
        }
        construction, displacements, support_units = p0.build_construction_geometry(
            supports, coverages, config, protocol
        )
        self.assertEqual(construction["path_candidates"], 33)
        self.assertEqual(construction["search"]["attempted_candidates"], 33)
        self.assertEqual(set(displacements), set(p0.ARMS))
        self.assertEqual(set(support_units), set(p0.CERTIFICATES))
        json.dumps(construction, allow_nan=False)


class GateFixtureTests(unittest.TestCase):
    @staticmethod
    def arm(scores: dict[str, float], actual: dict[str, float]) -> dict:
        return {
            "diagnostic_scores": {
                "scores": scores,
                "minimum_score": min(scores.values()),
                "mean_score": sum(scores.values()) / len(scores),
            },
            "finite_step": {
                "objectives": {
                    name: {
                        "predicted_first_order_reduction": value / 0.9,
                        "actual_loss_reduction": value,
                        "actual_to_predicted_ratio": 0.9,
                    }
                    for name, value in actual.items()
                }
            },
            "state_restored_after_virtual_step": True,
        }

    @classmethod
    def rows(cls) -> list[dict]:
        rows = []
        for index in range(8):
            baseline_scores = {name: 0.04 for name in p0.CERTIFICATES}
            raw_scores = {name: 0.10 for name in p0.CERTIFICATES}
            selected_scores = {
                name: 0.098 if name in p0.PROTECTED_CERTIFICATES else 0.11
                for name in p0.CERTIFICATES
            }
            baseline_actual = {name: 0.01 for name in p0.CERTIFICATES}
            raw_actual = {name: 0.02 for name in p0.CERTIFICATES}
            selected_actual = {
                name: 0.0196 if name in p0.PROTECTED_CERTIFICATES else 0.022
                for name in p0.CERTIFICATES
            }
            rows.append(
                {
                    "unit_id": f"diagnostic_{index + 1:02d}",
                    "coverage": {
                        "risk_microbatches": 1,
                        "safety_microbatches": 1,
                        "all_gradients_finite_nonzero": True,
                    },
                    "hull_audit": {
                        "certificates": {
                            name: {
                                "covered": True,
                                "registered_radius": 0.1,
                                "distance_to_construction_hull": 0.08,
                                "radius_excess": -0.02,
                            }
                            for name in p0.CERTIFICATES
                        }
                    },
                    "theorem_audit": {
                        "protected_residual_slacks": {
                            name: 0.001 for name in p0.PROTECTED_CERTIFICATES
                        },
                        "target_lower_bound_slacks": {
                            name: 0.01 for name in p0.TARGET_CERTIFICATES
                        },
                    },
                    "arms": {
                        "baseline": cls.arm(baseline_scores, baseline_actual),
                        "raw_mcd": cls.arm(raw_scores, raw_actual),
                        "jcrmcd_mcd": cls.arm(selected_scores, selected_actual),
                    },
                }
            )
        return rows

    @staticmethod
    def construction() -> dict:
        calibration = {
            "residuals": [
                {"distance": 0.1},
                {"distance": 0.08},
                {"distance": 0.07},
                {"distance": 0.06},
            ]
        }
        raw_lower = {name: 0.05 for name in p0.CERTIFICATES}
        selected = {
            "feasible": True,
            "minimum_protected_robust_margin": 0.001,
            "minimum_robust_target_score": 0.08,
        }
        return {
            "support_coverages": [
                {
                    "risk_microbatches": 8,
                    "safety_microbatches": 8,
                    "all_gradients_finite_nonzero": True,
                }
                for _ in range(4)
            ],
            "path_candidates": 33,
            "pooled_path_records": [{} for _ in range(33)],
            "support_path_records": {
                f"support_{index + 1:02d}": [{} for _ in range(33)]
                for index in range(4)
            },
            "baseline_optimizer_map": {"clipping_active": True},
            "raw_optimizer_map": {"clipping_active": True},
            "selected_optimizer_map": {"clipping_active": True},
            "raw_direction_budget_relative_error": 0.0,
            "target_direction_budget_relative_error": 0.0,
            "selected_direction_budget_relative_error": 0.0,
            "diagnostic_gradients_used_for_selection": False,
            "raw_gradient_tensors_serialized": False,
            "uncertainty": {
                "certificates": {
                    name: {"radius": 0.1, "calibration": calibration}
                    for name in p0.CERTIFICATES
                }
            },
            "selected_fraction": 0.03125,
            "selected_index": 1,
            "search": {
                "attempted_candidates": 33,
                "feasible_candidates": 2,
                "uncertainty_set_valid": True,
                "raw_robust_lower_scores": raw_lower,
                "raw_anchor": {"feasible": True},
                "selected": selected,
                "selected_robust_target_gain_over_raw": 0.003,
            },
        }

    @staticmethod
    def protocol() -> dict:
        return {
            "design": {
                "protected_certificates": list(p0.PROTECTED_CERTIFICATES),
                "target_certificates": list(p0.TARGET_CERTIFICATES),
            },
            "statistics": {"bootstrap_samples": 200, "bootstrap_seed": 17},
            "gates": {
                "gradient_budget_relative_tolerance": 1e-6,
                "robust_numeric_tolerance": 1e-12,
                "theorem_numeric_tolerance": 1e-6,
                "minimum_vram_headroom_fraction": 0.1,
                "minimum_nonzero_path_fraction": 0.03125,
                "construction_robust_target_score_min": 0.05,
                "construction_robust_target_gain_min": 0.002,
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

    def test_registered_gate_conjunction_passes_consistent_fixture(self) -> None:
        construction = self.construction()
        protocol = self.protocol()
        rows = self.rows()
        summary = p0.summarize_diagnostics(rows, construction, protocol)
        gates = p0.evaluate_gates(
            rows,
            construction,
            summary,
            protocol,
            state_restored=True,
            vram_headroom=0.2,
            gpu_name="NVIDIA GeForce RTX 4070 Ti",
        )
        self.assertTrue(all(gates.values()), gates)
        self.assertEqual(
            p0.select_route(gates),
            "PASS_JCRMCD_P0_AUTHORIZE_PROXY_DEV_TRIPLET_PROTOCOL",
        )

    def test_nonpositive_raw_denominator_is_undefined(self) -> None:
        result = p0._relative_paired([1.0] * 8, [1.0] * 7 + [0.0], self.protocol())
        self.assertFalse(result["defined"])
        self.assertEqual(result["invalid_denominator_indices"], [7])

    def test_coverage_failure_has_route_precedence(self) -> None:
        gates = {
            "integrity": True,
            "construction_uncertainty_valid": True,
            "robust_target_gain": True,
            "diagnostic_hull_coverage": False,
            "diagnostic_common_transfer": True,
            "finite_step_common_descent": True,
            "protected_diagnostic_noninferiority": True,
            "target_diagnostic_advantage": True,
            "effect_not_weak": True,
        }
        self.assertEqual(
            p0.select_route(gates),
            "STOP_JCRMCD_P0_DIAGNOSTIC_HULL_COVERAGE_FAILED",
        )


if __name__ == "__main__":
    unittest.main()
