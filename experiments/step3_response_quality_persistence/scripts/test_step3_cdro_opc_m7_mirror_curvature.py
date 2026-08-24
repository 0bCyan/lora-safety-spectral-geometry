"""Tests for the M7 validity-only mirror-curvature diagnostic."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import step3_cdro_opc_m7_mirror_curvature as subject


class M7MirrorCurvatureTests(unittest.TestCase):
    def test_arm_decomposition_has_registered_signs(self) -> None:
        result = subject.arm_decomposition(0.8, 1.2, 1.0)
        self.assertAlmostEqual(result["central_score"], 1.0)
        self.assertAlmostEqual(result["centered_even_score"], 0.2)
        self.assertAlmostEqual(result["even_forward_contribution"], -0.2)
        self.assertAlmostEqual(result["central_shift_from_first_order"], 0.0)
        self.assertAlmostEqual(result["forward_identity_residual"], 0.0)
        self.assertAlmostEqual(result["taylor_identity_residual"], 0.0)

    def test_comparison_identity_is_first_plus_odd_plus_even(self) -> None:
        fallback = subject.arm_decomposition(0.8, 1.0, 1.0)
        selected = subject.arm_decomposition(0.7, 0.8, 0.8)
        result = subject.comparison_decomposition(
            fallback, selected, 0.02, frozen_finite_deterioration=0.1
        )
        self.assertAlmostEqual(result["first_order_deterioration"], 0.2)
        self.assertAlmostEqual(result["odd_finite_radius_remainder"], -0.05)
        self.assertAlmostEqual(result["even_deterioration"], -0.05)
        self.assertAlmostEqual(result["finite_identity_residual"], 0.0)
        self.assertAlmostEqual(result["gap_identity_residual"], 0.0)

    def test_mirror_arm_applies_positive_map_and_restores(self) -> None:
        import torch

        model = torch.nn.Linear(1, 1, bias=False)
        model.eval()
        model.register_buffer("marker", torch.tensor([3.0]))
        parameters = list(model.named_parameters())
        with torch.no_grad():
            parameters[0][1].fill_(0.25)
        state = subject.m6.parameter_snapshot(parameters)
        displacement = {parameters[0][0]: torch.ones_like(parameters[0][1])}
        counter = {"positive_loss_writes": 0}
        value, audit = subject.mirror_arm(
            model,
            parameters,
            state,
            displacement,
            0.1,
            lambda: float(parameters[0][1].item()),
            counter,
            "positive_loss_writes",
        )
        self.assertAlmostEqual(value, 0.35, places=6)
        self.assertEqual(counter["positive_loss_writes"], 1)
        self.assertEqual(
            audit["parameter_delta_sign"],
            "positive_eta_times_frozen_displacement",
        )
        self.assertTrue(subject.mirror_restoration_passes(audit))
        self.assertTrue(subject.m6.state_equal(parameters, state))

    def test_mirror_arm_restores_buffers_and_other_state_after_exception(self) -> None:
        import torch

        model = torch.nn.Linear(1, 1, bias=False)
        model.eval()
        model.register_buffer("marker", torch.tensor([3.0]))
        parameters = list(model.named_parameters())
        state = subject.m6.parameter_snapshot(parameters)
        buffers = subject.buffer_snapshot(model)
        rng = subject.m6.rng_snapshot()
        displacement = {parameters[0][0]: torch.ones_like(parameters[0][1])}

        def fail() -> None:
            model.train()
            model.marker.add_(2.0)
            parameters[0][1].grad = torch.ones_like(parameters[0][1])
            parameters[0][1].data.add_(9.0)
            torch.rand(1)
            raise RuntimeError("expected")

        with self.assertRaises(RuntimeError):
            subject.mirror_arm(
                model,
                parameters,
                state,
                displacement,
                0.1,
                fail,
                {"positive_loss_writes": 0},
                "positive_loss_writes",
            )
        self.assertTrue(subject.m6.state_equal(parameters, state))
        self.assertTrue(subject.buffers_equal(model, buffers))
        self.assertTrue(subject.m6.rng_equal(rng))
        self.assertFalse(model.training)
        self.assertIsNone(parameters[0][1].grad)

    def test_mirror_arm_detects_successful_callback_state_pollution(self) -> None:
        import torch

        model = torch.nn.Linear(1, 1, bias=False)
        model.eval()
        model.register_buffer("marker", torch.tensor([3.0]))
        parameters = list(model.named_parameters())
        state = subject.m6.parameter_snapshot(parameters)
        buffers = subject.buffer_snapshot(model)
        displacement = {parameters[0][0]: torch.ones_like(parameters[0][1])}

        def pollute() -> float:
            model.train()
            model.marker.add_(1.0)
            torch.rand(1)
            return 2.0

        _, audit = subject.mirror_arm(
            model,
            parameters,
            state,
            displacement,
            0.1,
            pollute,
            {"positive_loss_writes": 0, "events": []},
            "positive_loss_writes",
            {
                "purpose": "positive_loss",
                "unit_id": "u1",
                "sample_id": "s1",
                "arm": "fallback",
            },
        )
        self.assertFalse(subject.mirror_restoration_passes(audit))
        self.assertFalse(audit["buffers_unchanged"])
        self.assertFalse(audit["mode_unchanged"])
        self.assertFalse(audit["rng_unchanged"])
        self.assertTrue(subject.m6.state_equal(parameters, state))
        self.assertTrue(subject.buffers_equal(model, buffers))
        self.assertFalse(model.training)

    def test_positive_map_audit_rejects_wrong_sign(self) -> None:
        import torch

        model = torch.nn.Linear(1, 1, bias=False)
        parameters = list(model.named_parameters())
        state = subject.m6.parameter_snapshot(parameters)
        displacement = {parameters[0][0]: torch.ones_like(parameters[0][1])}
        with torch.no_grad():
            parameters[0][1].add_(-0.1)
        audit = subject.realized_positive_map_audit(
            parameters, state, displacement, 0.1
        )
        self.assertGreater(audit["relative_l2_error"], 1.0)
        subject.m6.restore_parameters(parameters, state)

    def test_source_internal_audit_recomputes_frozen_scores(self) -> None:
        names = subject.CERTIFICATES
        initial = {name: 2.0 for name in names}
        norms = {name: 1.0 for name in names}
        fallback_final = {name: 1.8 for name in names}
        selected_final = {name: 1.84 for name in names}
        row = {
            "initial_losses": initial,
            "gradient_norms": norms,
            "arms": {
                "fallback": {
                    "final_losses": fallback_final,
                    "normalized_actual_scores": {name: 1.0 for name in names},
                    "first_order_scores": {name: 1.0 for name in names},
                },
                "selected": {
                    "final_losses": selected_final,
                    "normalized_actual_scores": {name: 0.8 for name in names},
                    "first_order_scores": {name: 0.8 for name in names},
                },
            },
            "finite_deteriorations": {name: 0.2 for name in names},
        }
        audit = subject.source_internal_audit(row, 0.2, 1.0, 0.02, 1e-10)
        self.assertTrue(audit["gate"])
        row["finite_deteriorations"]["risk"] = 0.3
        self.assertFalse(
            subject.source_internal_audit(row, 0.2, 1.0, 0.02, 1e-10)[
                "gate"
            ]
        )

    def test_routes_fail_closed_and_final_integrity_has_priority(self) -> None:
        gates = {name: True for name in subject.GATE_KEYS}
        self.assertEqual(
            subject.expected_route(gates),
            "PASS_CDRO_OPC_M7_VALIDITY_ONLY_SYMMETRIC_DIAGNOSTIC",
        )
        source = dict(gates)
        source["source_replay"] = False
        self.assertEqual(
            subject.expected_route(source), "STOP_CDRO_OPC_M7_SOURCE_REPLAY"
        )
        source_post = dict(gates)
        source_post["source_post_integrity"] = False
        self.assertEqual(
            subject.expected_route(source_post),
            "STOP_CDRO_OPC_M7_SOURCE_REPLAY",
        )
        source["final_integrity"] = False
        self.assertEqual(
            subject.expected_route(source),
            "STOP_CDRO_OPC_M7_FINAL_INTEGRITY",
        )
        self.assertEqual(subject.expected_route({}), "STOP_CDRO_OPC_M7_INVALID")

    def test_mutation_event_ledger_has_exact_order_and_positive_sign(self) -> None:
        allowlist = [
            {"unit_id": "u1", "sample_id": "s1"},
            {"unit_id": "u2", "sample_id": "s2"},
        ]
        events = subject.expected_mutation_events(allowlist)
        self.assertEqual(len(events), 6)
        self.assertEqual(
            [(row["purpose"], row["arm"]) for row in events[:2]],
            [("map_precheck", "fallback"), ("map_precheck", "selected")],
        )
        self.assertEqual(
            [(row["unit_id"], row["arm"]) for row in events[2:]],
            [
                ("u1", "fallback"),
                ("u1", "selected"),
                ("u2", "fallback"),
                ("u2", "selected"),
            ],
        )
        self.assertTrue(all(row["sign"] == 1 for row in events))

    def test_deep_comparison_rejects_missing_keys_and_bool_numeric_alias(self) -> None:
        self.assertFalse(
            subject.deep_equal_with_tolerance({"a": 1.0}, {}, 1e-10)
        )
        self.assertFalse(subject.deep_equal_with_tolerance(True, 1, 1e-10))

    def test_numeric_guards_reject_bool_and_nonfinite(self) -> None:
        self.assertFalse(subject.numeric_values_finite([True, 1.0]))
        self.assertFalse(subject.numeric_values_finite([math.nan]))
        self.assertFalse(subject.numbers_close(True, 1.0, 1e-10))
        self.assertFalse(subject.numbers_close(math.inf, math.inf, 1e-10))

    def test_json_writer_rejects_nan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            with self.assertRaises(ValueError):
                subject.write_json(path, {"value": float("nan")})

    def test_staged_publish_moves_only_complete_artifact_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            stage = parent / "stage"
            final = parent / "final"
            stage.mkdir()
            final.mkdir()
            for name in ("run", "analysis", "figures"):
                (stage / name).mkdir()
                (stage / name / "marker.txt").write_text(name, encoding="utf-8")
            subject.publish_staged_directories(stage, final)
            self.assertTrue(
                all((final / name / "marker.txt").is_file() for name in ("run", "analysis", "figures"))
            )

    def test_staged_publish_rejects_incomplete_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            stage = parent / "stage"
            final = parent / "final"
            stage.mkdir()
            final.mkdir()
            (stage / "run").mkdir()
            with self.assertRaises(subject.M7MirrorCurvatureError):
                subject.publish_staged_directories(stage, final)
            self.assertEqual(list(final.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
