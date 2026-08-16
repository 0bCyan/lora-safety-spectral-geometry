from __future__ import annotations

import unittest

import torch

import step3_functional_nullspace_common as fn
import train_step3_functional_nullspace as trainer


class FunctionalNullspaceTests(unittest.TestCase):
    def test_functional_key_shares_only_identical_inputs(self) -> None:
        root = "base.layers.0"
        self.assertEqual(
            fn.functional_key(f"{root}.self_attn.q_proj"),
            fn.functional_key(f"{root}.self_attn.k_proj"),
        )
        self.assertEqual(
            fn.functional_key(f"{root}.mlp.gate_proj"),
            fn.functional_key(f"{root}.mlp.up_proj"),
        )
        self.assertNotEqual(
            fn.functional_key(f"{root}.self_attn.q_proj"),
            fn.functional_key(f"{root}.self_attn.o_proj"),
        )

    def test_sampled_positions_cover_response_endpoints(self) -> None:
        mask = torch.tensor([False, True, True, True, True, True, False])
        positions = fn.sampled_mask_positions(mask, 3)
        self.assertEqual(positions.tolist(), [1, 3, 5])

    def test_basis_is_orthonormal_and_deterministic(self) -> None:
        rows = torch.tensor(
            [
                [4.0, 0.0, 0.0, 0.0],
                [3.0, 0.2, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 1.0, 0.1, 0.0],
            ]
        )
        first, metrics = fn.right_singular_basis(rows, rank=2)
        second, _ = fn.right_singular_basis(rows, rank=2)
        self.assertTrue(torch.allclose(first @ first.T, torch.eye(2), atol=1e-6))
        self.assertTrue(torch.allclose(first.abs(), second.abs(), atol=1e-6))
        self.assertGreater(metrics["top_rank_energy_fraction"], 0.95)

    def test_projection_removes_only_protected_component(self) -> None:
        basis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        value = torch.tensor([[2.0, -3.0, 5.0], [1.0, 4.0, -2.0]])
        original_unprotected = value[:, 2].clone()
        before = fn.protected_component_ratio(value, basis)
        with torch.no_grad():
            fn.project_rows_(value, basis)
        after = fn.protected_component_ratio(value, basis)
        self.assertGreater(before, 0.0)
        self.assertLess(after, 1e-7)
        self.assertTrue(torch.equal(value[:, 2], original_unprotected))

    def test_projection_is_idempotent(self) -> None:
        generator = torch.Generator().manual_seed(7)
        rows = torch.randn(6, 10, generator=generator)
        basis, _ = fn.right_singular_basis(rows, rank=3)
        value = torch.randn(4, 10, generator=generator)
        with torch.no_grad():
            fn.project_rows_(value, basis)
            once = value.clone()
            fn.project_rows_(value, basis)
        self.assertTrue(torch.allclose(value, once, atol=1e-6))

    def test_delta_projection_preserves_existing_safety_gradient(self) -> None:
        basis = torch.tensor([[1.0, 0.0, 0.0]])
        safety_gradient = torch.tensor([[3.0, 2.0, 1.0]])
        risk_delta = torch.tensor([[4.0, 5.0, 6.0]])
        combined = safety_gradient + risk_delta
        with torch.no_grad():
            fn.project_delta_(combined, safety_gradient, basis)
        expected = safety_gradient + torch.tensor([[0.0, 5.0, 6.0]])
        self.assertTrue(torch.equal(combined, expected))

    def test_conflict_projection_removes_opposing_protected_component(self) -> None:
        basis = torch.tensor([[1.0, 0.0, 0.0]])
        risk = torch.tensor([[-4.0, 3.0, 5.0]])
        safety = torch.tensor([[2.0, 7.0, 11.0]])
        with torch.no_grad():
            metrics = fn.resolve_protected_conflict_(risk, safety, basis)
        self.assertTrue(metrics["conflict"])
        self.assertLess(metrics["dot_before"], 0.0)
        self.assertAlmostEqual(metrics["dot_after"], 0.0, places=6)
        self.assertTrue(torch.equal(risk, torch.tensor([[0.0, 3.0, 5.0]])))

    def test_conflict_projection_preserves_aligned_risk(self) -> None:
        basis = torch.tensor([[1.0, 0.0, 0.0]])
        risk = torch.tensor([[4.0, 3.0, 5.0]])
        safety = torch.tensor([[2.0, 7.0, 11.0]])
        original = risk.clone()
        with torch.no_grad():
            metrics = fn.resolve_protected_conflict_(risk, safety, basis)
        self.assertFalse(metrics["conflict"])
        self.assertTrue(torch.equal(risk, original))

    def test_conflict_projection_preserves_outside_basis(self) -> None:
        basis = torch.tensor([[1.0, 0.0, 0.0]])
        risk = torch.tensor([[-4.0, 3.0, 5.0]])
        safety = torch.tensor([[2.0, 7.0, 11.0]])
        outside = risk[:, 1:].clone()
        with torch.no_grad():
            fn.resolve_protected_conflict_(risk, safety, basis)
        self.assertTrue(torch.equal(risk[:, 1:], outside))

    def test_conflict_reconstruction_preserves_accumulated_gradient(self) -> None:
        basis = torch.tensor([[1.0, 0.0, 0.0]])
        accumulated = torch.tensor([[13.0, 17.0, 19.0]])
        risk = torch.tensor([[-4.0, 3.0, 5.0]])
        safety = torch.tensor([[2.0, 7.0, 11.0]])
        with torch.no_grad():
            fn.resolve_protected_conflict_(risk, safety, basis)
        combined = accumulated + risk + safety
        expected = accumulated + torch.tensor([[0.0, 3.0, 5.0]]) + safety
        self.assertTrue(torch.equal(combined, expected))

    def test_trainer_conflict_reconstruction_preserves_accumulated_gradient(self) -> None:
        basis = torch.tensor([[1.0, 0.0, 0.0]])
        accumulated = torch.tensor([[13.0, 17.0, 19.0]])
        risk = torch.tensor([[-4.0, 3.0, 5.0]])
        safety = torch.tensor([[2.0, 7.0, 11.0]])
        weight = torch.nn.Parameter(torch.zeros_like(accumulated))
        weight.grad = accumulated + risk + safety
        entries = {"module": {"weight": weight, "functional_key": "basis"}}
        metrics = trainer.resolve_new_gradient_conflicts_(
            entries,
            {"basis": basis},
            {"module": accumulated},
            {"module": accumulated + risk},
        )
        expected = accumulated + torch.tensor([[0.0, 3.0, 5.0]]) + safety
        self.assertTrue(torch.equal(weight.grad, expected))
        self.assertEqual(metrics["conflicting_modules"], 1)
        self.assertLess(metrics["conflict_residual_fraction"], 1e-6)


if __name__ == "__main__":
    unittest.main()
