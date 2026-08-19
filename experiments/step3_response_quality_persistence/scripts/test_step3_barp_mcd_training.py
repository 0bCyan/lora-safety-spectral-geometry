from __future__ import annotations

import copy
import unittest

import torch

import step3_barp_mcd_training as barp


class CurrentAdamWMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.left = torch.nn.Parameter(
            torch.tensor([0.4, -0.7, 0.2], dtype=torch.float64)
        )
        self.right = torch.nn.Parameter(
            torch.tensor([-0.3, 0.5], dtype=torch.float64)
        )
        self.parameters = [("left", self.left), ("right", self.right)]
        self.optimizer = torch.optim.AdamW(
            [self.left, self.right],
            lr=0.01,
            betas=(0.8, 0.95),
            eps=1e-9,
            weight_decay=0.0,
        )

    def assert_real_update_matches(
        self, gradient: dict[str, torch.Tensor], expected_step: int
    ) -> None:
        predicted, audit = barp.current_adamw_displacement(
            self.parameters, self.optimizer, gradient, 0.75, 1e-6
        )
        self.assertEqual(audit["next_optimizer_step"], expected_step)
        before = {
            name: parameter.detach().clone() for name, parameter in self.parameters
        }
        for name, parameter in self.parameters:
            parameter.grad = gradient[name].clone()
        torch.nn.utils.clip_grad_norm_([self.left, self.right], 0.75)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        for name, parameter in self.parameters:
            observed = (before[name] - parameter.detach()) / 0.01
            self.assertTrue(
                torch.allclose(observed, predicted[name], atol=1e-12, rtol=1e-10)
            )

    def test_first_and_second_steps_match_torch_adamw(self) -> None:
        self.assert_real_update_matches(
            {
                "left": torch.tensor([2.0, -0.5, 1.0], dtype=torch.float64),
                "right": torch.tensor([-1.5, 0.25], dtype=torch.float64),
            },
            1,
        )
        second = {
            "left": torch.tensor([-0.4, 1.2, 0.3], dtype=torch.float64),
            "right": torch.tensor([0.8, -1.1], dtype=torch.float64),
        }
        state_before = {
            parameter: {
                key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
                for key, value in self.optimizer.state[parameter].items()
            }
            for _, parameter in self.parameters
        }
        barp.current_adamw_displacement(
            self.parameters, self.optimizer, second, 0.75, 1e-6
        )
        for _, parameter in self.parameters:
            for key, value in state_before[parameter].items():
                current = self.optimizer.state[parameter][key]
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(current, value))
                else:
                    self.assertEqual(current, value)
        self.assert_real_update_matches(second, 2)

    def test_nonzero_weight_decay_is_rejected(self) -> None:
        self.optimizer.param_groups[0]["weight_decay"] = 0.1
        with self.assertRaisesRegex(barp.BarpTrainingValidationError, "unsupported"):
            barp.current_adamw_displacement(
                self.parameters,
                self.optimizer,
                {"left": torch.ones(3), "right": torch.ones(2)},
                1.0,
            )


class CurrentStateSearchTests(unittest.TestCase):
    def test_complete_family_keeps_the_raw_anchor(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(7, dtype=torch.float64))
        parameters = [("p", parameter)]
        optimizer = torch.optim.AdamW(
            [parameter], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        names = (
            "risk",
            "uniform",
            "tarc",
            "etarc",
            "continuation",
            "branch_entropy",
            "branch_negative_gap",
        )
        basis = torch.eye(7, dtype=torch.float64)
        certificates = {
            name: {"p": basis[index].clone()} for index, name in enumerate(names)
        }
        raw_baseline = {"p": torch.full((7,), 0.5, dtype=torch.float64)}
        selected, full, audit = barp.enumerate_current_state_homotopy(
            raw_baseline,
            certificates,
            parameters,
            optimizer,
            1.0,
            0.2,
            1.0,
            ["risk", "uniform", "tarc", "continuation"],
            ["etarc", "branch_entropy", "branch_negative_gap"],
            0.95,
            [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
            16,
        )
        self.assertEqual(audit["attempted_source_subsets"], 127)
        self.assertEqual(audit["attempted_homotopy_candidates"], 129)
        self.assertEqual(audit["exact_map_evaluations"], 239)
        self.assertGreaterEqual(audit["protected_feasible_candidates"], 1)
        self.assertGreaterEqual(
            audit["selected"]["minimum_protected_retention"], 0.95
        )
        self.assertGreaterEqual(audit["selected_target_gain_over_raw"], -1e-12)
        self.assertEqual(set(selected), set(full))


class DecisionTests(unittest.TestCase):
    @staticmethod
    def passing_checks() -> dict[str, bool]:
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
            "boundary": True,
        }

    def test_pass_route(self) -> None:
        self.assertEqual(
            barp.select_route(self.passing_checks()),
            "PASS_BARP_MCD_Q1_AUTHORIZE_LONGER_MATCHED_PROTOCOL",
        )

    def test_optimizer_map_failure_precedes_effect_failure(self) -> None:
        checks = self.passing_checks()
        checks["optimizer_map_exact"] = False
        checks["finite_step_common_descent"] = False
        self.assertEqual(
            barp.select_route(checks), "STOP_BARP_MCD_Q1_OPTIMIZER_MAP_FAILED"
        )

    def test_protection_failure_has_distinct_route(self) -> None:
        checks = self.passing_checks()
        checks["barp_protection"] = False
        self.assertEqual(
            barp.select_route(checks), "STOP_BARP_MCD_Q1_PROTECTION_FAILED"
        )


if __name__ == "__main__":
    unittest.main()
