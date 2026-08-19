"""Stable numeric replay gate for the frozen M3 confirmation."""

from __future__ import annotations

import json

import step3_cdro_opc_m3_confirmation as core
import step3_cdro_opc_m3_confirmation_v2 as v2


DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m3/gpu_confirmation/protocol/"
    "cdro_opc_m3_confirmation_protocol_v3.json"
)


def reconstruct_fixed_direction(*args, **kwargs):
    result = v2.reconstruct_fixed_direction(*args, **kwargs)
    audit = result[3]
    print(
        "M3_DIRECTION_REPLAY "
        + json.dumps(
            {
                "basis_error": audit["basis_audit"]["orthonormality_max_error"],
                "bitwise_sha_equal": audit["bitwise_coordinate_sha_equal"],
                "candidate_max_abs_error": audit["candidate_max_abs_error"],
                "coordinate_tolerance": audit["coordinate_tolerance"],
                "fallback_max_abs_error": audit["fallback_max_abs_error"],
                "fixed_direction_valid": audit["fixed_direction_valid"],
                "solver_valid": audit["solver_audit"]["optimization_valid"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> None:
    core.DEFAULT_PROTOCOL = DEFAULT_PROTOCOL
    core.reconstruct_fixed_direction = reconstruct_fixed_direction
    core.main()


if __name__ == "__main__":
    main()
