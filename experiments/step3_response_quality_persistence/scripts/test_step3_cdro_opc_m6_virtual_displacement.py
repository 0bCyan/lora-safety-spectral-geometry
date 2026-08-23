"""Tests for M6 direct virtual-displacement safety and routing."""

from __future__ import annotations

import unittest

import step3_cdro_opc_m6_virtual_displacement as subject


class M6VirtualDisplacementTests(unittest.TestCase):
    def test_virtual_arm_restores_after_success(self) -> None:
        import torch

        model = torch.nn.Linear(2, 1, bias=False)
        model.eval()
        parameters = list(model.named_parameters())
        state = subject.parameter_snapshot(parameters)
        displacement = {name: torch.ones_like(value) for name, value in parameters}
        value, audit = subject.virtual_arm(
            model, parameters, state, displacement, 0.1, lambda: 7
        )
        self.assertEqual(value, 7)
        self.assertTrue(audit["state_restored"])
        self.assertTrue(audit["mode_restored"])
        self.assertTrue(audit["rng_restored"])
        self.assertTrue(audit["gradients_restored"])
        self.assertTrue(subject.state_equal(parameters, state))

    def test_virtual_arm_restores_after_exception(self) -> None:
        import torch

        model = torch.nn.Linear(2, 1, bias=False)
        model.eval()
        parameters = list(model.named_parameters())
        state = subject.parameter_snapshot(parameters)
        displacement = {name: torch.ones_like(value) for name, value in parameters}
        rng = subject.rng_snapshot()

        def fail() -> None:
            model.train()
            torch.rand(1)
            parameters[0][1].grad = torch.ones_like(parameters[0][1])
            raise RuntimeError("expected")

        with self.assertRaises(RuntimeError):
            subject.virtual_arm(model, parameters, state, displacement, 0.1, fail)
        self.assertTrue(subject.state_equal(parameters, state))
        self.assertFalse(model.training)
        self.assertTrue(subject.rng_equal(rng))
        self.assertIsNone(parameters[0][1].grad)

    def test_finite_protection_uses_registered_denominator(self) -> None:
        rows = [
            {
                "covered": index != 0,
                "finite_deteriorations": {
                    "risk": 0.20 if index == 0 else 0.01,
                    "uniform": 0.02,
                    "tarc": 0.03,
                    "continuation": 0.04,
                },
            }
            for index in range(20)
        ]
        result = subject.finite_protection_statistics(rows, 0.04, 0.02, 1e-10)
        self.assertEqual(result["covered_units"], 19)
        self.assertAlmostEqual(
            result["aggregate_finite_deteriorations"]["risk"], 0.0195
        )
        self.assertTrue(result["aggregate_protected"])

    def test_behavior_difference_is_selected_minus_fallback(self) -> None:
        import torch

        fallback = {"x": torch.tensor([1.0, -2.0])}
        selected = {"x": torch.tensor([3.0, 4.0])}
        difference = subject.displacement_difference(selected, fallback)
        self.assertTrue(torch.equal(difference["x"], torch.tensor([2.0, 6.0])))

    def test_behavior_advantage_sign_zero_and_bounds(self) -> None:
        import torch

        gradient = {"x": torch.tensor([1.0], dtype=torch.float64)}
        difference = {"x": torch.tensor([1.0], dtype=torch.float64)}
        lower = subject.behavior_advantage(
            gradient, difference, 0.2, 1.0, 0.95, 0.25, 1.75
        )
        upper = subject.behavior_advantage(
            gradient, difference, 0.2, 1.0, 0.65, 0.25, 1.75
        )
        zero = subject.behavior_advantage(
            gradient, {"x": torch.tensor([0.0])}, 0.2, 1.0, 0.9, 0.25, 1.75
        )
        wrong_sign = subject.behavior_advantage(
            gradient, difference, 0.2, 0.9, 1.0, 0.25, 1.75
        )
        nonfinite = subject.behavior_advantage(
            {"x": torch.tensor([float("nan")])},
            difference,
            0.2,
            1.0,
            0.9,
            0.25,
            1.75,
        )
        self.assertTrue(lower["gate"])
        self.assertTrue(upper["gate"])
        self.assertIsNone(zero["actual_to_predicted_ratio"])
        self.assertFalse(zero["gate"])
        self.assertFalse(wrong_sign["gate"])
        self.assertFalse(nonfinite["gate"])

    def test_float32_map_precheck_catches_precision_boundary(self) -> None:
        import torch

        model = torch.nn.Linear(1, 1, bias=False)
        model.eval()
        parameters = list(model.named_parameters())
        displacement = {parameters[0][0]: torch.ones_like(parameters[0][1])}
        with torch.no_grad():
            parameters[0][1].fill_(0.0254)
        passing = subject.registered_map_precheck(
            model,
            parameters,
            subject.parameter_snapshot(parameters),
            {"fallback": displacement, "selected": displacement},
            2e-4,
            2e-6,
            1e-5,
        )
        with torch.no_grad():
            parameters[0][1].fill_(0.1)
        failing = subject.registered_map_precheck(
            model,
            parameters,
            subject.parameter_snapshot(parameters),
            {"fallback": displacement, "selected": displacement},
            2e-4,
            2e-6,
            1e-5,
        )
        self.assertTrue(passing["gate"])
        self.assertFalse(failing["gate"])

    def test_route_fails_closed(self) -> None:
        common = {name: True for name in subject.COMMON_GATE_KEYS}
        calibration = {"gates": {name: True for name in subject.SPLIT_GATE_KEYS}}
        sealed = {"gates": {name: True for name in subject.SPLIT_GATE_KEYS}}
        self.assertTrue(subject.route_from_results(common, calibration, sealed).startswith("PASS"))
        invalid_common = dict(common)
        invalid_common["fixed_direction"] = False
        self.assertEqual(
            subject.route_from_results(invalid_common, calibration, sealed),
            "STOP_CDRO_OPC_M6_INVALID",
        )
        failed_calibration = {"gates": dict(calibration["gates"])}
        failed_calibration["gates"]["behavior"] = False
        self.assertEqual(
            subject.route_from_results(common, failed_calibration, None),
            "STOP_CDRO_OPC_M6_CALIBRATION",
        )
        self.assertEqual(
            subject.route_from_results(common, {"gates": {}}, {"gates": {}}),
            "STOP_CDRO_OPC_M6_INVALID",
        )

    def test_verify_recomputes_invalid_route(self) -> None:
        protocol = {
            "design": {"fixed_dose": 0.1, "virtual_scale_eta": 0.2},
            "gates": {
                "maximum_map_relative_l2_error": 2e-6,
                "maximum_map_absolute_error": 1e-5,
            },
        }
        calibration = {
            "status": "evaluated",
            "gates": {name: True for name in subject.SPLIT_GATE_KEYS},
            "finite_protection": {"covered_units": 19},
        }
        sealed = {
            "status": "evaluated",
            "gates": {name: True for name in subject.SPLIT_GATE_KEYS},
            "finite_protection": {"covered_units": 19},
        }
        arm_audit = {
            "relative_l2_error": 0.0,
            "max_absolute_error": 0.0,
            "state_restored": True,
            "rng_unchanged": True,
            "mode_unchanged": True,
            "gradients_unchanged": True,
            "rng_restored": True,
            "mode_restored": True,
            "gradients_restored": True,
        }
        pre_open = {name: True for name in subject.PRE_OPEN_COMMON_GATE_KEYS}
        common = {**pre_open, "final_integrity": True}
        integrity = {name: True for name in subject.INTEGRITY_KEYS}
        diagnostics = {
            "schema": "step3_cdro_opc_m6_virtual_displacement_diagnostics_v1",
            "route": "STOP_CDRO_OPC_M6_INVALID",
            "fixed_dose": 0.1,
            "virtual_scale_eta": 0.2,
            "optimizer_objects": 1,
            "optimizer_steps": 0,
            "training_authorized": False,
            "checkpoint_writes": 0,
            "candidate_generation": 0,
            "core100_opened": False,
            "heldout_opened": False,
            "fresh_external_confirmation_claimed": False,
            "direct_optimizer_space_displacement_not_standard_adamw_step": True,
            "registered_map_precheck": {
                "gate": True,
                "arms": {
                    "fallback": dict(arm_audit),
                    "selected": dict(arm_audit),
                },
            },
            "pre_calibration_ready": True,
            "pre_open_common_gates": pre_open,
            "common_gates": common,
            "calibration_integrity": integrity,
            "final_integrity": integrity,
            "calibration": calibration,
            "sealed_audit_opened": True,
            "sealed_audit": sealed,
        }
        decision = {
            "schema": "step3_cdro_opc_m6_virtual_displacement_decision_v1",
            "route": "STOP_CDRO_OPC_M6_INVALID",
            "fixed_dose": 0.1,
            "virtual_scale_eta": 0.2,
            "calibration_covered_units": 19,
            "calibration_gates": calibration["gates"],
            "sealed_audit_opened": True,
            "sealed_audit_covered_units": 19,
            "sealed_audit_gates": sealed["gates"],
            "fresh_external_confirmation_claimed": False,
            "optimizer_steps": 0,
            "training_authorized": False,
            "next_authorized_action": (
                "interpret finite-loss development only; do not train"
            ),
        }
        passing_route = "PASS_CDRO_OPC_M6_FIXED_DOSE_FINITE_RADIUS_VIRTUAL_DISPLACEMENT"
        decision["route"] = passing_route
        diagnostics["route"] = passing_route
        self.assertEqual(
            subject.verify_artifacts(protocol, decision, diagnostics), passing_route
        )
        decision["route"] = "STOP_CDRO_OPC_M6_INVALID"
        diagnostics["route"] = "STOP_CDRO_OPC_M6_INVALID"
        with self.assertRaises(subject.M6VirtualDisplacementError):
            subject.verify_artifacts(protocol, decision, diagnostics)


if __name__ == "__main__":
    unittest.main()
