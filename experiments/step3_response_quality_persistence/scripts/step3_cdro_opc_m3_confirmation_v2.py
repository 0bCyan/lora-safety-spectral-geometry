"""Numeric-equivalence transport repair for the frozen M3 confirmation."""

from __future__ import annotations

import step3_cdro_opc_m3_confirmation as core


DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m3/gpu_confirmation/protocol/"
    "cdro_opc_m3_confirmation_protocol_v2.json"
)
_ORIGINAL_RECONSTRUCT = core.reconstruct_fixed_direction


def numerical_direction_valid(audit: dict, expected: dict) -> bool:
    """Define direction identity by the frozen numeric tolerance, not JSON bits."""
    tolerance = float(expected["coordinate_tolerance"])
    solver = audit["solver_audit"]
    basis = audit["basis_audit"]
    return bool(
        float(audit["fallback_max_abs_error"]) <= tolerance
        and float(audit["candidate_max_abs_error"]) <= tolerance
        and solver["optimization_valid"]
        and float(basis["orthonormality_max_error"]) <= 2e-4
    )


def reconstruct_fixed_direction(*args, **kwargs):
    fallback, endpoint, reference, audit = _ORIGINAL_RECONSTRUCT(*args, **kwargs)
    expected = args[-1] if args else kwargs["expected"]
    audit["bitwise_coordinate_sha_equal"] = (
        audit["candidate_coordinate_sha256"]
        == audit["expected_candidate_coordinate_sha256"]
    )
    audit["identity_rule"] = "max_coordinate_error_within_frozen_tolerance"
    audit["coordinate_tolerance"] = float(expected["coordinate_tolerance"])
    audit["fixed_direction_valid"] = numerical_direction_valid(audit, expected)
    return fallback, endpoint, reference, audit


def main() -> None:
    core.DEFAULT_PROTOCOL = DEFAULT_PROTOCOL
    core.reconstruct_fixed_direction = reconstruct_fixed_direction
    core.main()


if __name__ == "__main__":
    main()
