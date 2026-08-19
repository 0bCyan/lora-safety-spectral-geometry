from __future__ import annotations

import unittest

import torch

import step3_barp_mcd_training as base
import step3_barp_mcd_training_q1r as retry


class MemoryBoundedSearchTests(unittest.TestCase):
    @staticmethod
    def fixture() -> tuple[
        dict[str, torch.Tensor],
        dict[str, dict[str, torch.Tensor]],
        list[tuple[str, torch.nn.Parameter]],
        torch.optim.Optimizer,
    ]:
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
        return (
            {"p": torch.full((7,), 0.5, dtype=torch.float64)},
            certificates,
            parameters,
            optimizer,
        )

    @staticmethod
    def run_search(function: object) -> dict[str, object]:
        raw, certificates, parameters, optimizer = MemoryBoundedSearchTests.fixture()
        _, _, audit = function(
            raw,
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
        return audit

    def test_retry_is_decision_equivalent_to_q1_search(self) -> None:
        original = self.run_search(base.enumerate_current_state_homotopy)
        bounded = self.run_search(retry.enumerate_memory_bounded_homotopy)
        for key in (
            "attempted_source_subsets",
            "feasible_source_subsets",
            "attempted_homotopy_candidates",
            "exact_map_evaluations",
            "selected_target_gain_over_raw",
        ):
            self.assertEqual(bounded[key], original[key])
        self.assertEqual(bounded["selected"], original["selected"])
        self.assertEqual(bounded["maximum_retained_endpoint_directions"], 1)

    def test_retry_route_keeps_gate_order(self) -> None:
        checks = {
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
        self.assertEqual(
            retry.select_retry_route(checks),
            "PASS_BARP_MCD_Q1R_AUTHORIZE_LONGER_MATCHED_PROTOCOL",
        )
        checks["vram"] = False
        self.assertEqual(
            retry.select_retry_route(checks), "STOP_BARP_MCD_Q1R_VRAM_FAILED"
        )


if __name__ == "__main__":
    unittest.main()
