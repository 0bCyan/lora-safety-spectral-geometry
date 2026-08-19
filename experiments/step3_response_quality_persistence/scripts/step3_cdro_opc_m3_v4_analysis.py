"""Reproduce the claim-bounded analysis of the completed M3 V4 run."""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.stats import beta, binom

import step3_cdro_opc_dose_common as dose_math
import step3_cdro_opc_m3_confirmation as core


DIAGNOSTICS = core.resolve_repo_path(
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m3/gpu_confirmation/v4_run/run/"
    "cdro_opc_m3_confirmation_diagnostics.json"
)
DECISION = DIAGNOSTICS.with_name("cdro_opc_m3_confirmation_decision.json")
OUTPUT = core.resolve_repo_path(
    "experiments/step3_response_quality_persistence/results/"
    "4070ti_step3_cdro_opc_m3/gpu_confirmation/v4_run/analysis/"
    "cdro_opc_m3_v4_interpretation.json"
)
REPORT = OUTPUT.with_name("CDRO_OPC_M3_V4_INTERPRETATION.md")


def main() -> None:
    diagnostics = core.read_json(DIAGNOSTICS)
    decision = core.read_json(DECISION)
    protected = list(core.PROTECTED)
    registered_structural = all(
        all(
            math.isfinite(float(row["gradient_norms"][name]))
            and float(row["gradient_norms"][name]) > 0.0
            for name in protected
        )
        for split in core.SPLITS
        for row in diagnostics[split]["coverage"]
    ) and all(
        diagnostics["behavior"][split][pool]["coverage"]["finite_nonzero"]
        and not diagnostics["behavior"][split][pool]["coverage"][
            "missing_gradient_tensors"
        ]
        for split in core.SPLITS
        for pool in core.POOLS
    )
    registered_gates = dict(diagnostics["gates"])
    registered_gates["structural"] = registered_structural
    registered_route = core.route_from_gates(registered_gates)

    sealed = diagnostics["sealed_audit"]
    safe_doses = []
    for row, fallback, endpoint in zip(
        sealed["statistics"]["rows"],
        sealed["fallback_scores"],
        sealed["endpoint_scores"],
        strict=True,
    ):
        item = dose_math.unit_safe_dose(
            fallback, endpoint, protected, 0.04, 0.02, 1.0
        )
        safe_doses.append(
            {
                "unit_id": row["unit_id"],
                "safe_dose": item["safe_dose"],
                "limiting_certificates": item["limiting_certificates"],
            }
        )
    safe_doses.sort(key=lambda row: (float(row["safe_dose"]), row["unit_id"]))
    covered = int(sealed["statistics"]["covered_units"])
    total = int(sealed["statistics"]["total_units"])
    selected = float(decision["selected_dose"])
    exact_interval = [
        float(beta.ppf(0.025, covered, total - covered + 1)),
        float(beta.ppf(0.975, covered + 1, total - covered)),
    ]
    failed_rows = []
    for row in sealed["statistics"]["rows"]:
        if row["covered"]:
            continue
        violations = {
            name: float(value)
            for name, value in row["standardized_deteriorations"].items()
            if float(value) > 0.04
        }
        failed_rows.append(
            {
                "unit_id": row["unit_id"],
                "violations": violations,
                "maximum_excess_over_4pct": max(violations.values()) - 0.04,
            }
        )
    analysis = {
        "schema": "step3_cdro_opc_m3_v4_interpretation_v1",
        "decision": "NOT_CONFIRMED_NEAR_MISS",
        "raw_route": diagnostics["route"],
        "registered_certificate_route": registered_route,
        "registered_structural_gate": registered_structural,
        "selected_dose": selected,
        "useful_dose_floor": 0.01,
        "calibration_arithmetic_max_residual": diagnostics["calibration"][
            "statistics"
        ]["maximum_residual"],
        "finite_sample_marginal_protection_lower_bound_under_exchangeability": diagnostics[
            "calibration"
        ]["policy"]["finite_sample_marginal_protection_lower_bound"],
        "sealed_joint_coverage": {
            "covered": covered,
            "total": total,
            "rate": covered / total,
            "required": 19,
            "clopper_pearson_95pct": exact_interval,
            "binomial_probability_X_le_17_if_p_20_over_21": float(
                binom.cdf(17, 20, 20 / 21)
            ),
            "failed_units": failed_rows,
        },
        "sealed_aggregate_standardized_deteriorations": sealed["statistics"][
            "aggregate_standardized_deteriorations"
        ],
        "behavior_selected_minus_fallback": {
            split: {
                pool: diagnostics["behavior"][split][pool][
                    "selected_minus_fallback"
                ]
                for pool in core.POOLS
            }
            for split in core.SPLITS
        },
        "posthoc_dose_diagnostics_not_valid_for_selection": {
            "all_20_sealed_safe_dose": safe_doses[0]["safe_dose"],
            "19_of_20_sealed_safe_dose": safe_doses[1]["safe_dose"],
            "18_of_20_sealed_safe_dose": safe_doses[2]["safe_dose"],
            "19_of_20_shrink_factor_relative_to_selected": safe_doses[1][
                "safe_dose"
            ]
            / selected,
            "ordered_sealed_unit_safe_doses": safe_doses,
        },
        "claim_boundary": {
            "finite_sample_theorem_rejected_at_5pct": False,
            "pre_registered_empirical_gate_passed": False,
            "method_confirmed": False,
            "posthoc_shrinkage_authorized": False,
            "training_authorized": False,
            "finite_parameter_step_claimed": False,
            "generated_response_claimed": False,
        },
        "optimizer_steps": 0,
        "training_authorized": False,
        "core100_opened": False,
        "heldout_opened": False,
    }
    core.write_json(OUTPUT, analysis)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "# CDR-OPC M3 V4 Interpretation\n\n"
        "## Decision\n\n"
        "M3 is a preregistered near miss, not a confirmation. The selected dose "
        f"was `{selected:.8f}` and all four behavior panels improved, but sealed "
        f"joint protection was `{covered}/{total}` rather than the required "
        "`19/20`. The registered-certificate route is "
        f"`{registered_route}`.\n\n"
        "The raw runner route was `STOP_CDRO_OPC_M3_INVALID` because its legacy "
        "structural summary still included an unregistered auxiliary objective. "
        "All four registered protected gradients and all behavior gradients were "
        "finite and nonzero; correcting that label does not change the failed "
        "17/20 outcome gate.\n\n"
        "## Evidence\n\n"
        f"Calibration arithmetic residual was `{analysis['calibration_arithmetic_max_residual']}`. "
        "The sealed failures were three risk-only exceedances above the 4% bound: "
        + ", ".join(
            f"`{row['unit_id']}` +{100 * row['maximum_excess_over_4pct']:.3f} percentage points"
            for row in failed_rows
        )
        + ". Aggregate standardized deterioration remained below 4% for every "
        "certificate.\n\n"
        f"The exact 95% binomial interval for 17/20 is `{exact_interval[0]:.3f}` "
        f"to `{exact_interval[1]:.3f}`. Under p=20/21, P(X<=17) is "
        f"`{analysis['sealed_joint_coverage']['binomial_probability_X_le_17_if_p_20_over_21']:.4f}`. "
        "Thus this sample does not reject the marginal theorem at 5%, but it also "
        "does not pass the frozen empirical corroboration gate.\n\n"
        "## Next Boundary\n\n"
        "A hindsight dose near `0.03432` would cover 19/20, but using it now would "
        "leak sealed information. It may motivate a prospectively fixed shrinkage "
        "controller on a new cohort; it cannot repair M3 or authorize training.\n\n"
        "```text\noptimizer_steps=0\ntraining_authorized=false\n"
        "core100_opened=false\nheldout_opened=false\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": analysis["decision"], "route": registered_route}))


if __name__ == "__main__":
    main()
