from __future__ import annotations

import unittest

import torch

import step3_barp_mcd_d1f as d1f


class UpdateSpaceFamilyTests(unittest.TestCase):
    def test_family_has_registered_size_and_maximin_selection(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(7, dtype=torch.float64))
        parameters = [("p", parameter)]
        optimizer = torch.optim.AdamW(
            [parameter], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        names = tuple(d1f.rp.CERTIFICATES)
        basis = torch.eye(7, dtype=torch.float64)
        certificates = {
            name: {"p": basis[index].clone()} for index, name in enumerate(names)
        }
        result = d1f.enumerate_update_space_family(
            {"p": torch.full((7,), 0.5, dtype=torch.float64)},
            certificates,
            parameters,
            optimizer,
            1.0,
            0.2,
            1.0,
            list(d1f.rp.HOMOTOPY_FRACTIONS),
            d1f.rp.SOURCE_TOP_K,
        )
        self.assertEqual(result["attempted_endpoint_candidates"], 127)
        self.assertEqual(result["attempted_interior_candidates"], 112)
        self.assertEqual(result["attempted_total_candidates"], 239)
        self.assertEqual(result["feasible_total_candidates"], 239)
        self.assertEqual(result["current_optimizer_completed_steps"], 0)
        self.assertEqual(
            result["selected_update_space_anchor"]["minimum_exact_score"],
            max(
                record["minimum_exact_score"]
                for record in result["all_endpoint_records"] + result["top_candidates"]
            ),
        )

    def test_family_uses_current_adamw_state_without_mutating_it(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(7, dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=0.01, weight_decay=0.0)
        parameter.grad = torch.arange(1, 8, dtype=torch.float64)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        before_step = optimizer.state[parameter]["step"].clone()
        basis = torch.eye(7, dtype=torch.float64)
        certificates = {
            name: {"p": basis[index].clone()}
            for index, name in enumerate(d1f.rp.CERTIFICATES)
        }
        result = d1f.enumerate_update_space_family(
            {"p": torch.full((7,), 0.5, dtype=torch.float64)},
            certificates,
            [("p", parameter)],
            optimizer,
            1.0,
            0.2,
            1.0,
            list(d1f.rp.HOMOTOPY_FRACTIONS),
            d1f.rp.SOURCE_TOP_K,
        )
        self.assertEqual(result["current_optimizer_completed_steps"], 1)
        self.assertTrue(torch.equal(optimizer.state[parameter]["step"], before_step))


class ReplayComparisonTests(unittest.TestCase):
    def test_nested_close_handles_numeric_drift_only(self) -> None:
        left = {"a": [1.0, {"b": 2}], "c": "fixed"}
        self.assertTrue(d1f._nested_close(left, {"a": [1.0 + 1e-8, {"b": 2}], "c": "fixed"}, 1e-7))
        self.assertFalse(d1f._nested_close(left, {"a": [1.0 + 1e-5, {"b": 2}], "c": "fixed"}, 1e-7))


class RouteTests(unittest.TestCase):
    @staticmethod
    def passing() -> dict[str, bool]:
        return {
            "replay_steps_exact": True,
            "replay_matches_d1": True,
            "optimizer_map_exact": True,
            "replay_finite_descent": True,
            "d1_failure_reproduced": True,
            "family_complete": True,
            "update_space_anchor_feasible": True,
            "no_fourth_update": True,
            "vram": True,
            "boundary": True,
        }

    def test_pass_route(self) -> None:
        self.assertEqual(
            d1f.select_route(self.passing()),
            "PASS_BARP_MCD_D1F_UPDATE_SPACE_ANCHOR_FEASIBLE",
        )

    def test_nonreproduction_precedes_feasibility(self) -> None:
        checks = self.passing()
        checks["d1_failure_reproduced"] = False
        checks["update_space_anchor_feasible"] = False
        self.assertEqual(
            d1f.select_route(checks),
            "STOP_BARP_MCD_D1F_D1_FAILURE_NOT_REPRODUCED",
        )


if __name__ == "__main__":
    unittest.main()
