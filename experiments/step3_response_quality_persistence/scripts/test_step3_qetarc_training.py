from __future__ import annotations

import unittest

import torch

import step3_qetarc_training as training


class DynamicDirectionTests(unittest.TestCase):
    @staticmethod
    def gradients() -> dict[str, dict[str, torch.Tensor]]:
        return {
            "uniform": {"p": torch.tensor([3.0, 4.0])},
            "tarc": {"p": torch.tensor([0.0, 2.0])},
            "etarc": {"p": torch.tensor([1.0, 0.0])},
            "continuation": {"p": torch.tensor([0.0, 1.0])},
        }

    def test_all_auxiliary_arms_use_the_dynamic_equal_budget(self) -> None:
        for objective in ("tarc", "etarc", "qetarc"):
            combined, audit = training.dynamic_safety_direction(
                self.gradients(), objective, 0.25, 0.5
            )
            auxiliary = combined["p"] - self.gradients()["uniform"]["p"]
            self.assertTrue(torch.isfinite(combined["p"]).all())
            self.assertAlmostEqual(float(torch.linalg.vector_norm(auxiliary)), 1.25, places=6)
            self.assertAlmostEqual(audit["auxiliary_norm_ratio"], 0.25, places=6)

    def test_qetarc_realizes_quota_and_route_optimum(self) -> None:
        _, audit = training.dynamic_safety_direction(
            self.gradients(), "qetarc", 0.25, 0.5
        )
        self.assertTrue(audit["quota_active"])
        self.assertAlmostEqual(audit["achieved_continuation_cosine"], 0.5, places=6)
        self.assertAlmostEqual(audit["route_retention_cosine"], 3.0**0.5 / 2.0, places=6)
        self.assertGreaterEqual(
            audit["continuation_first_order_dot"] + 1e-6,
            audit["continuation_dot_lower_bound"],
        )

    def test_uniform_has_no_auxiliary(self) -> None:
        combined, audit = training.dynamic_safety_direction(
            self.gradients(), "uniform", 0.25, 0.5
        )
        self.assertTrue(torch.equal(combined["p"], self.gradients()["uniform"]["p"]))
        self.assertEqual(audit["selected_auxiliary_norm"], 0.0)


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
        self.assertEqual(rows[0]["continuation_loss_reduction"], 0.75)


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
            "distinct_final_adapter_hashes": True,
            "sealed_data_untouched": True,
            "dynamic_geometry": True,
            "route_effect": True,
            "continuation_effect": True,
            "branch_mechanism": True,
        }

    def test_pass_route(self) -> None:
        self.assertEqual(
            training.select_smoke_route(self.passing()),
            "PASS_QETARC_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT_PROTOCOL",
        )

    def test_integrity_failure_precedes_effect_failure(self) -> None:
        checks = self.passing()
        checks["initial_state_equal"] = False
        checks["route_effect"] = False
        self.assertEqual(
            training.select_smoke_route(checks),
            "STOP_QETARC_SMOKE_MATCHING_FAILED",
        )

    def test_geometry_failure_has_distinct_route(self) -> None:
        checks = self.passing()
        checks["dynamic_geometry"] = False
        self.assertEqual(
            training.select_smoke_route(checks),
            "STOP_QETARC_SMOKE_DYNAMIC_GEOMETRY_FAILED",
        )

    def test_continuation_failure_does_not_pass(self) -> None:
        checks = self.passing()
        checks["continuation_effect"] = False
        self.assertEqual(
            training.select_smoke_route(checks),
            "STOP_QETARC_SMOKE_CONTINUATION_EFFECT_FAILED",
        )


if __name__ == "__main__":
    unittest.main()
