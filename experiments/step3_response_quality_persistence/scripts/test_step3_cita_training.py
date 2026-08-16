from __future__ import annotations

import unittest

import step3_cita_training as training


class FakeTokenizer:
    all_special_ids = [99]

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool | None = None,
    ) -> str:
        del conversation, tokenize
        if not add_generation_prompt:
            return "full"
        return "direct" if enable_thinking is None else "forced"

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        del add_special_tokens
        return {
            "input_ids": {
                "direct": [1],
                "forced": [1, 2],
                "full": [1, 2, 10, 11, 99],
            }[text]
        }


class FeatureTests(unittest.TestCase):
    def test_safety_feature_masks_prefix_and_preserves_budget(self) -> None:
        source = {"correct_prompt": "p", "response": "r"}
        cache = {
            "response_token_ids": [10, 11, 99],
            "uniform_weights": [1.0, 1.0, 1.0],
            "cita_weights": [0.25, 1.75, 1.0],
            "sham_weights": [1.75, 0.25, 1.0],
        }
        feature = training.safety_feature(FakeTokenizer(), source, cache, 16, "cita")
        self.assertEqual(feature["labels"], [-100, -100, 10, 11, 99])
        self.assertEqual(feature["weights"], [0.0, 0.0, 0.25, 1.75, 1.0])
        self.assertEqual(feature["weight_budget"], feature["response_tokens"])


def manifest(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "completed_steps": 2,
        "skipped_updates": 0,
        "all_losses_and_gradients_finite": True,
        "vram_gate_pass": True,
        "initial_trainable_state_sha256": "initial",
        "risk_order_sha256": "risk",
        "target_order_sha256": "target",
        "target_exposure_sha256": "exposure",
        "safety_draws": 8,
        "safety_scored_tokens": 240,
        "safety_weight_budget": 240.0,
        "first_step_risk_loss_mean": 2.0,
        "first_step_safety_uniform_diagnostic_loss_mean": 3.0,
        "final_adapter_sha256": "replace",
        "core100_opened": False,
        "heldout_opened": False,
        "candidate_outputs_generated": False,
    }
    row.update(overrides)
    return row


PROTOCOL = {
    "arm_order": ["uniform", "cita", "sham"],
    "optimizer_steps_per_arm": 2,
}


class MatchingTests(unittest.TestCase):
    def passing(self) -> dict[str, dict[str, object]]:
        return {
            "uniform": manifest(final_adapter_sha256="u"),
            "cita": manifest(final_adapter_sha256="c"),
            "sham": manifest(final_adapter_sha256="s"),
        }

    def test_pass_requires_matched_inputs_and_distinct_assignments(self) -> None:
        checks = training.compare_manifests(self.passing(), PROTOCOL, 1e-7)
        self.assertTrue(all(checks.values()))
        self.assertEqual(
            training.select_smoke_route(checks),
            "PASS_CITA_SMOKE_AUTHORIZE_MATCHED_DEVELOPMENT16_PROTOCOL",
        )

    def test_budget_mismatch_stops_matching(self) -> None:
        rows = self.passing()
        rows["sham"]["safety_weight_budget"] = 240.1
        checks = training.compare_manifests(rows, PROTOCOL, 1e-7)
        self.assertEqual(
            training.select_smoke_route(checks), "STOP_CITA_SMOKE_MATCHING_FAILED"
        )

    def test_same_final_adapter_means_assignment_not_exercised(self) -> None:
        rows = self.passing()
        rows["sham"]["final_adapter_sha256"] = "c"
        checks = training.compare_manifests(rows, PROTOCOL, 1e-7)
        self.assertEqual(
            training.select_smoke_route(checks),
            "STOP_CITA_SMOKE_ASSIGNMENT_NOT_EXERCISED",
        )

    def test_numerical_failure_has_distinct_route(self) -> None:
        rows = self.passing()
        rows["uniform"]["skipped_updates"] = 1
        checks = training.compare_manifests(rows, PROTOCOL, 1e-7)
        self.assertEqual(
            training.select_smoke_route(checks), "STOP_CITA_SMOKE_NUMERICAL_FAILED"
        )


if __name__ == "__main__":
    unittest.main()
