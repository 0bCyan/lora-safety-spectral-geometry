"""Registered-certificate normalization repair for M3 confirmation."""

from __future__ import annotations

from typing import Any

import step3_cdro_opc_m3_confirmation as core
import step3_cdro_opc_m3_confirmation_v3 as v3


DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m3/gpu_confirmation/protocol/"
    "cdro_opc_m3_confirmation_protocol_v4.json"
)
_ORIGINAL_NORMALIZE = core.frsc.normalized_certificates


def registered_subset(certificates: dict[str, Any]) -> dict[str, Any]:
    missing = [name for name in core.PROTECTED if name not in certificates]
    if missing:
        raise core.M3ConfirmationError(f"registered certificates missing: {missing}")
    return {name: certificates[name] for name in core.PROTECTED}


def normalize_registered_certificates(certificates: dict[str, Any]):
    return _ORIGINAL_NORMALIZE(registered_subset(certificates))


def reconstruct_fixed_direction(*args, **kwargs):
    result = v3.reconstruct_fixed_direction(*args, **kwargs)
    if result[3]["fixed_direction_valid"]:
        # M1 direction replay uses the original seven-certificate construction.
        # Only subsequent M3 unit scoring uses the four protocol-registered scores.
        core.frsc.normalized_certificates = normalize_registered_certificates
    return result


def main() -> None:
    core.DEFAULT_PROTOCOL = DEFAULT_PROTOCOL
    core.reconstruct_fixed_direction = reconstruct_fixed_direction
    core.main()


if __name__ == "__main__":
    main()
