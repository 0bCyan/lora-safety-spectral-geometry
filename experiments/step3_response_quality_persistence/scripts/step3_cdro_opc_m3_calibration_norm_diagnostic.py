"""Diagnose zero protected gradients on the opened M3 calibration split."""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
from pathlib import Path

import step3_cdro_opc_m3_confirmation as core


DEFAULT_PROTOCOL = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m3/gpu_confirmation/protocol/"
    "cdro_opc_m3_confirmation_protocol_v3.json"
)
DEFAULT_OUTPUT = (
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m3/gpu_confirmation/v3_stop/"
    "calibration_certificate_norms.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol_path = core.resolve_repo_path(args.protocol)
    output_path = core.resolve_repo_path(args.output)
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=core.REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip():
        raise core.M3ConfirmationError("calibration diagnostic requires a clean worktree")
    if output_path.exists():
        raise core.M3ConfirmationError("calibration diagnostic output already exists")
    (
        protocol,
        m1_protocol,
        config,
        _,
        _,
        selected_targets,
        _,
        risk_rows,
        units,
    ) = core.context(protocol_path)
    calibration_units = [unit for unit in units if unit["split"] == "calibration"]
    calibration_targets = core.subset_targets(selected_targets, units, "calibration")
    model, tokenizer = core.cspr.load_model(m1_protocol)
    parameters = core.cspr.trainable_parameters(model)
    initial_hash = core.cspr.trainable_state_hash(parameters)
    pairs, structural = core.tarc.build_pair_features(
        tokenizer,
        calibration_targets,
        int(config["max_length"]),
        int(config["route_width_limit"]),
    )
    if structural != protocol["selection"]["structural_audits"]["calibration"]:
        raise core.M3ConfirmationError("calibration diagnostic structure differs")
    pair_map = {str(pair["sample_id"]): pair for pair in pairs}
    risk_features = {
        int(unit["risk_row_index"]): core.training.response_feature(
            tokenizer,
            str(risk_rows[int(unit["risk_row_index"])]["prompt"]),
            str(risk_rows[int(unit["risk_row_index"])]["response"]),
            int(config["max_length"]),
        )
        for unit in calibration_units
    }
    rows = []
    for unit in calibration_units:
        gradients, coverage = core.rp.aggregate_block_gradients(
            model,
            parameters,
            risk_features,
            pair_map,
            [int(unit["risk_row_index"])],
            [str(unit["sample_id"])],
            dict(config),
            train_mode=False,
        )
        norms = {
            name: core.gradient_math.gradient_norm(gradient)
            for name, gradient in gradients.items()
        }
        rows.append(
            {
                "unit_id": unit["unit_id"],
                "sample_id": unit["sample_id"],
                "certificate_norms": norms,
                "finite_nonzero": {
                    name: math.isfinite(float(value)) and float(value) > 0.0
                    for name, value in norms.items()
                },
                "coverage": coverage,
            }
        )
        del gradients
    final_hash = core.cspr.trainable_state_hash(parameters)
    zero_counts = {
        name: sum(not row["finite_nonzero"][name] for row in rows)
        for name in core.PROTECTED
    }
    result = {
        "schema": "step3_cdro_opc_m3_calibration_certificate_norms_v1",
        "source_protocol": args.protocol,
        "scope": "opened_calibration_only",
        "calibration_units": len(rows),
        "sealed_audit_opened": False,
        "rows": rows,
        "zero_or_nonfinite_counts": zero_counts,
        "all_units_normalizable": not any(zero_counts.values()),
        "state_restored": final_hash == initial_hash,
        "optimizer_objects": 0,
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    core.write_json(output_path, result)
    del model
    gc.collect()
    print(
        json.dumps(
            {
                "calibration_units": len(rows),
                "zero_or_nonfinite_counts": zero_counts,
                "state_restored": result["state_restored"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
