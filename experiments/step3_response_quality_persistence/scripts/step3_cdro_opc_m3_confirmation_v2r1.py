"""Storage-path revision of the frozen M3 V2 confirmation."""

from __future__ import annotations

import step3_cdro_opc_m3_confirmation as core
import step3_cdro_opc_m3_confirmation_v2 as v2


DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m3/gpu_confirmation/protocol/"
    "cdro_opc_m3_confirmation_protocol_v2r1.json"
)


def main() -> None:
    core.DEFAULT_PROTOCOL = DEFAULT_PROTOCOL
    core.reconstruct_fixed_direction = v2.reconstruct_fixed_direction
    core.main()


if __name__ == "__main__":
    main()
