from __future__ import annotations

import math
import unittest

import torch

import step3_mcd_training as training


class DynamicDirectionTests(unittest.TestCase):
    @staticmethod
    def gradients() -> dict[str, dict[str, torch.Tensor]]:
        return {
            "uniform": {"p": torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])},
            "tarc": {"p": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 2.0])},
            "etarc": {"p": torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0, 0.0])},
            "continuation": {
                "p": torch.tensor([0.0, 0.0, 4.0, 0.0, 0.0, 0.0])
            },
            "branch_entropy": {
                "p": torch.tensor([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
            },
            "branch_negative_gap": {
                "p": torch.tensor([0.0, 0.0, 0.0, 0.0, 6.0, 0.0])
            },
        }

    def test_all_auxiliary_arms_use_the_dynamic_equal_budget(self) -> None:
        gradients = self.gradients()
        for objective in ("tarc", "etarc", "qetarc", "mcd_etarc"):
            combined, audit = training.dynamic_safety_direction(
                gradients, objective, 1.0, 0.5
            )
            auxiliary = combined["p"] - gradients["uniform"]["p"]
            self.assertTrue(torch.isfinite(combined["p"]).all())
            self.assertAlmostEqual(
                float(torch.linalg.vector_norm(auxiliary)), 3.0, places=6
            )
            self.assertAlmostEqual(audit["auxiliary_norm_ratio"], 1.0, places=6)

    def test_mcd_realizes_all_five_common_descent_certificates(self) -> None:
        _, audit = training.dynamic_safety_direction(
            self.gradients(), "mcd_etarc", 1.0, 0.5
        )
        certificate = audit["mcd_certificate"]
        expected = 1.0 / math.sqrt(5.0)
        self.assertAlmostEqual(certificate["dual_margin"], expected, places=6)
        self.assertLess(certificate["kkt_residual"], 1e-7)
        self.assertLess(certificate["primal_dual_gap"], 1e-6)
        for cosine in audit["certificate_cosines"].values():
            self.assertAlmostEqual(cosine, expected, places=6)

    def test_qetarc_control_realizes_its_frozen_quota(self) -> None:
        _, audit = training.dynamic_safety_direction(
            self.gradients(), "qetarc", 1.0, 0.5
        )
        self.assertTrue(audit["quota_active"])
        self.assertAlmostEqual(audit["achieved_continuation_cosine"], 0.5, places=6)
        self.assertAlmostEqual(
            audit["route_retention_cosine"], math.sqrt(3.0) / 2.0, places=6
        )

    def test_uniform_has_no_auxiliary(self) -> None:
        gradients = self.gradients()
        combined, audit = training.dynamic_safety_direction(
            gradients, "uniform", 1.0, 0.5
        )
        self.assertTrue(torch.equal(combined["p"], gradients["uniform"]["p"]))
        self.assertEqual(audit["selected_auxiliary_norm"], 0.0)


class GeometrySummaryTests(unittest.TestCase):
    def test_mcd_summary_checks_every_dynamic_certificate(self) -> None:
        _, row = training.dynamic_safety_direction(
            DynamicDirectionTests.gradients(), "mcd_etarc", 1.0, 0.5
        )
        protocol = {
            "design": {"auxiliary_gradient_norm_ratio": 1.0},
            "gates": {
                "dynamic_budget_absolute_tolerance": 1e-5,
                "dynamic_common_margin_min": 0.20,
                "dynamic_kkt_residual_max": 1e-6,
                "dynamic_primal_dual_gap_max": 1e-6,
            },
        }
        summary = training.geometry_summary([row], "mcd_etarc", protocol)
        self.assertTrue(summary["budget_all_pass"])
        self.assertTrue(summary["dual_all_pass"])
        self.assertTrue(summary["common_certificate_all_pass"])
        self.assertEqual(summary["active_support_sizes"], [5])


class PairedReductionTests(unittest.TestCase):
    def test_reductions_preserve_prompt_pairing(self) -> None:
        initial = [
            {
                "sample_id": "a",
                "category": "x",
                **{metric: 2.0 for metric in training.DIAGNOSTIC_LOSSES},
            }
        ]
        final = [
            {
                "sample_id": "a",
                "category": "x",
                **{metric: 1.25 for metric in training.DIAGNOSTIC_LOSSES},
            }
        ]
        rows = training.diagnostic_reductions(initial, final)
        self.assertEqual(rows[0]["sample_id"], "a")
        self.assertEqual(rows[0]["etarc_loss_reduction"], 0.75)
        self.assertEqual(rows[0]["branch_entropy_loss_reduction"], 0.75)


class ManifestComparisonTests(unittest.TestCase):
    @staticmethod
    def protocol() -> dict[str, object]:
        return {
            "arm_order": ["uniform", "tarc", "etarc", "qetarc", "mcd_etarc"],
            "optimizer_steps_per_arm": 2,
            "safety_pair_draws_per_arm": 8,
            "statistics": {"bootstrap_samples": 100, "bootstrap_seed": 17},
            "gates": {
                "matching_tolerance": 1e-7,
                "paired_advantage_ci_lower_min": 0.0,
                "paired_sign_p_max": 0.05,
                "registered_loss_reduction_min": 0.0,
                "uniform_advantage_mean_min": 0.05,
                "mcd_etarc_loss_reduction_min": 0.5,
                "route_reduction_fraction_of_raw_etarc_min": 0.65,
                "route_reduction_fraction_of_tarc_min": 1.20,
                "route_advantage_mean_min": 0.05,
                "continuation_loss_reduction_min": 0.5,
                "continuation_advantage_mean_min": 0.05,
                "branch_entropy_reduction_min": 0.001,
                "branch_negative_gap_reduction_min": 0.05,
                "branch_entropy_advantage_mean_min": 0.001,
                "branch_negative_gap_advantage_mean_min": 0.05,
            },
        }

    @staticmethod
    def manifests() -> dict[str, dict[str, object]]:
        arms = ["uniform", "tarc", "etarc", "qetarc", "mcd_etarc"]
        reductions = {
            "uniform": [1.0, 1.1, 1.2, 1.0, 0.0, 0.0],
            "tarc": [1.0, 1.3, 1.0, 1.0, -0.01, -0.1],
            "etarc": [1.0, 1.1, 2.0, 1.0, 0.0, 0.0],
            "qetarc": [1.0, 1.1, 1.3, 1.0, 0.0, 0.0],
            "mcd_etarc": [1.2, 1.2, 1.5, 1.2, 0.01, 0.20],
        }
        metrics = training.DIAGNOSTIC_LOSSES
        result: dict[str, dict[str, object]] = {}
        for index, arm in enumerate(arms):
            values = dict(zip((f"{name}_reduction" for name in metrics), reductions[arm]))
            geometry: dict[str, object] = {
                "selected_auxiliary_draws": 0 if arm == "uniform" else 8,
                "budget_all_pass": True,
            }
            if arm == "qetarc":
                geometry.update(
                    {
                        "quota_all_pass": True,
                        "active_boundary_all_pass": True,
                        "route_retention_all_pass": True,
                        "continuation_bound_all_pass": True,
                    }
                )
            if arm == "mcd_etarc":
                geometry.update(
                    {"dual_all_pass": True, "common_certificate_all_pass": True}
                )
            result[arm] = {
                "arm": arm,
                "selected_auxiliary": None if arm == "uniform" else arm,
                "geometry": geometry,
                "completed_steps": 2,
                "optimizer_steps": 2,
                "skipped_updates": 0,
                "all_losses_and_gradients_finite": True,
                "vram_gate_pass": True,
                "initial_trainable_state_sha256": "initial",
                "final_trainable_state_sha256": f"final-{index}",
                "checkpoint_writes": 0,
                "risk_order_sha256": "risk",
                "target_order_sha256": "target",
                "target_exposure_sha256": "exposure",
                "safety_pair_draws": 8,
                "safety_target_rows_forwarded": 16,
                "safety_response_tokens": 80,
                "continuation_tokens": 40,
                "available_tarc_margin_terms": 8,
                "available_etarc_margin_terms": 8,
                "computed_objective_gradient_draws": {
                    name: 8 for name in training.OBJECTIVES
                },
                "first_step_risk_loss_mean": 2.0,
                "first_step_safety_uniform_loss_mean": 3.0,
                "initial_diagnostic": {"uniform_loss_mean": 4.0},
                "reductions": values,
                "paired_diagnostic_reductions": [
                    {"sample_id": f"p{row}", **values} for row in range(8)
                ],
                "candidate_outputs_generated": False,
                "main_training_authorized": False,
                "core100_opened": False,
                "heldout_opened": False,
            }
        return result

    def test_strong_synthetic_effects_pass_every_component_gate(self) -> None:
        checks = training.compare_manifests(self.manifests(), self.protocol())
        self.assertTrue(checks["dynamic_geometry"])
        self.assertTrue(checks["common_effect"])
        self.assertTrue(checks["route_effect"])
        self.assertTrue(checks["continuation_effect"])
        self.assertTrue(checks["branch_mechanism"])
        self.assertEqual(
            training.select_smoke_route(checks),
            "PASS_MCD_ETARC_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT_PROTOCOL",
        )


class RouteTests(unittest.TestCase):
    @staticmethod
    def passing() -> dict[str, bool]:
        return {
            "all_arms_present": True,
            "steps_exact": True,
            "no_skipped_updates": True,
            "finite": True,
            "vram": True,
            "initial_state_equal": True,
            "risk_order_equal": True,
            "target_order_equal": True,
            "target_exposure_equal": True,
            "safety_pair_draws_equal": True,
            "safety_target_rows_equal": True,
            "safety_response_tokens_equal": True,
            "continuation_tokens_equal": True,
            "available_tarc_terms_equal": True,
            "available_etarc_terms_equal": True,
            "objective_gradient_counts_equal": True,
            "first_step_risk_loss_equal": True,
            "first_step_uniform_safety_loss_equal": True,
            "initial_diagnostics_equal": True,
            "assignment_counts_correct": True,
            "distinct_final_state_hashes": True,
            "checkpoint_writes_zero": True,
            "sealed_data_untouched": True,
            "dynamic_geometry": True,
            "common_effect": True,
            "route_effect": True,
            "continuation_effect": True,
            "branch_mechanism": True,
        }

    def test_pass_route(self) -> None:
        self.assertEqual(
            training.select_smoke_route(self.passing()),
            "PASS_MCD_ETARC_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT_PROTOCOL",
        )

    def test_integrity_failure_precedes_effect_failure(self) -> None:
        checks = self.passing()
        checks["initial_state_equal"] = False
        checks["route_effect"] = False
        self.assertEqual(
            training.select_smoke_route(checks),
            "STOP_MCD_ETARC_SMOKE_MATCHING_FAILED",
        )

    def test_dynamic_certificate_failure_has_distinct_route(self) -> None:
        checks = self.passing()
        checks["dynamic_geometry"] = False
        self.assertEqual(
            training.select_smoke_route(checks),
            "STOP_MCD_ETARC_SMOKE_DYNAMIC_CERTIFICATE_FAILED",
        )

    def test_common_effect_failure_precedes_component_effects(self) -> None:
        checks = self.passing()
        checks["common_effect"] = False
        checks["branch_mechanism"] = False
        self.assertEqual(
            training.select_smoke_route(checks),
            "STOP_MCD_ETARC_SMOKE_COMMON_EFFECT_FAILED",
        )

    def test_checkpoint_write_is_an_assignment_failure(self) -> None:
        checks = self.passing()
        checks["checkpoint_writes_zero"] = False
        self.assertEqual(
            training.select_smoke_route(checks),
            "STOP_MCD_ETARC_SMOKE_ASSIGNMENT_NOT_EXERCISED",
        )


if __name__ == "__main__":
    unittest.main()
