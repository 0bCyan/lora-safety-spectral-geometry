from __future__ import annotations

import unittest

import torch

import step3_barp_mcd_d1r as d1r


class OptimizerAnchorSearchTests(unittest.TestCase):
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
        basis = torch.eye(7, dtype=torch.float64)
        certificates = {
            name: {"p": basis[index].clone()}
            for index, name in enumerate(d1r.rp.CERTIFICATES)
        }
        return (
            {"p": torch.full((7,), 0.5, dtype=torch.float64)},
            certificates,
            parameters,
            optimizer,
        )

    def test_anchor_and_protected_route_are_certified(self) -> None:
        raw, certificates, parameters, optimizer = self.fixture()
        selected, anchor, audit = d1r.enumerate_optimizer_anchor_homotopy(
            raw,
            certificates,
            parameters,
            optimizer,
            1.0,
            0.2,
            1.0,
            list(d1r.rp.PROTECTED_CERTIFICATES),
            list(d1r.rp.TARGET_CERTIFICATES),
            0.95,
            list(d1r.rp.HOMOTOPY_FRACTIONS),
            d1r.rp.SOURCE_TOP_K,
        )
        self.assertEqual(audit["attempted_homotopy_candidates"], 239)
        self.assertEqual(audit["feasible_homotopy_candidates"], 239)
        self.assertGreater(audit["raw_anchor"]["minimum_exact_score"], 1e-8)
        self.assertGreaterEqual(
            audit["selected"]["minimum_protected_retention"], 0.95
        )
        self.assertGreaterEqual(audit["selected_target_gain_over_raw"], 0.0)
        self.assertEqual(set(selected), set(anchor))

    def test_search_does_not_mutate_optimizer_state(self) -> None:
        raw, certificates, parameters, optimizer = self.fixture()
        parameter = parameters[0][1]
        parameter.grad = torch.arange(1, 8, dtype=torch.float64)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        before = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in optimizer.state[parameter].items()
        }
        d1r.enumerate_optimizer_anchor_homotopy(
            raw,
            certificates,
            parameters,
            optimizer,
            1.0,
            0.2,
            1.0,
            list(d1r.rp.PROTECTED_CERTIFICATES),
            list(d1r.rp.TARGET_CERTIFICATES),
            0.95,
            list(d1r.rp.HOMOTOPY_FRACTIONS),
            d1r.rp.SOURCE_TOP_K,
        )
        for key, value in before.items():
            current = optimizer.state[parameter][key]
            if torch.is_tensor(value):
                self.assertTrue(torch.equal(current, value))
            else:
                self.assertEqual(current, value)


class RouteTests(unittest.TestCase):
    def test_route_prefix_is_distinct(self) -> None:
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
            "checkpoints": True,
            "protected_diagnostic": True,
            "target_diagnostic_mean": True,
            "target_diagnostic_bootstrap": True,
            "target_diagnostic_sign": True,
            "boundary": True,
        }
        self.assertEqual(
            d1r.select_route(checks),
            "PASS_BARP_MCD_D1R_AUTHORIZE_CHECKPOINT_BOUND_GENERATION_PROTOCOL",
        )
        checks["barp_assignment_exercised"] = False
        self.assertEqual(
            d1r.select_route(checks),
            "STOP_BARP_MCD_D1R_ASSIGNMENT_NOT_EXERCISED",
        )


if __name__ == "__main__":
    unittest.main()
