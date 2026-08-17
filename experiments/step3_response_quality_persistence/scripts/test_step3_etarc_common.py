from __future__ import annotations

import unittest

import torch

import step3_etarc_common as etarc


class CompletePathCertificateTests(unittest.TestCase):
    def test_path_soft_min_is_below_every_token_margin(self) -> None:
        margins = torch.tensor([3.0, 1.0, 2.0])
        score = etarc.conservative_path_score(margins, 0.5)
        self.assertLessEqual(float(score), float(margins.min()))

    def test_existential_outer_score_is_below_strongest_path(self) -> None:
        scores = torch.tensor([-2.0, 3.0])
        certificate = etarc.existential_path_score(scores, 0.7)
        self.assertLessEqual(float(certificate), float(scores.max()))

    def test_positive_certificate_implies_one_fully_positive_path(self) -> None:
        paths = [torch.tensor([-1.0, 4.0]), torch.tensor([4.0, 5.0])]
        certificate, scores = etarc.existential_path_certificate(paths, 0.1, 0.1)
        self.assertGreater(float(certificate), 0.0)
        winning_path = int(torch.argmax(scores))
        self.assertTrue(bool((paths[winning_path] > 0).all()))

    def test_nested_gradient_favors_strong_path_and_its_weak_token(self) -> None:
        weak_path = torch.tensor([-2.0, 2.0], requires_grad=True)
        strong_path = torch.tensor([1.0, 4.0], requires_grad=True)
        certificate, _ = etarc.existential_path_certificate(
            [weak_path, strong_path], 0.5, 0.5
        )
        certificate.backward()
        self.assertGreater(
            float(strong_path.grad.abs().sum()), float(weak_path.grad.abs().sum())
        )
        self.assertGreater(float(strong_path.grad[0]), float(strong_path.grad[1]))


class ProjectionTests(unittest.TestCase):
    def test_conflicting_gradient_has_closed_form_halfspace_projection(self) -> None:
        route = {"p": torch.tensor([1.0, -1.0])}
        continuation = {"p": torch.tensor([0.0, 1.0])}
        projected, audit = etarc.continuation_compatible_projection(
            route, continuation
        )
        self.assertTrue(audit["projection_active"])
        self.assertTrue(torch.allclose(projected["p"], torch.tensor([1.0, 0.0])))
        self.assertAlmostEqual(
            etarc.gradient_dot(projected, continuation), 0.0, places=7
        )
        self.assertGreater(etarc.gradient_dot(projected, route), 0.0)

    def test_compatible_gradient_is_unchanged(self) -> None:
        route = {"p": torch.tensor([1.0, 1.0])}
        continuation = {"p": torch.tensor([0.0, 1.0])}
        projected, audit = etarc.continuation_compatible_projection(
            route, continuation
        )
        self.assertFalse(audit["projection_active"])
        self.assertTrue(torch.equal(projected["p"], route["p"]))


class DecisionTests(unittest.TestCase):
    def test_continuation_failure_cannot_pass(self) -> None:
        gates = {
            "integrity": True,
            "gradient_valid": True,
            "budget_matched": True,
            "vram": True,
            "state_restored": True,
            "route_structure": True,
            "objective_distinct": True,
            "route_transfer": True,
            "continuation_compatibility": False,
            "effect_not_weak": True,
        }
        self.assertEqual(
            etarc.select_route(gates),
            "STOP_ETARC_P0_CONTINUATION_COMPATIBILITY_FAILED",
        )


if __name__ == "__main__":
    unittest.main()
