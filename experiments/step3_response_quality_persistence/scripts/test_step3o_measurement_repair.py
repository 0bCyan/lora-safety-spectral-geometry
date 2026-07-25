from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).with_name("step3o_measurement_repair.py")
SPEC = importlib.util.spec_from_file_location("step3o_measurement_repair", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load Step3O measurement module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def check_o0_verification() -> None:
    result = MODULE.verify_o0()
    assert result["artifact_hashes_pass"] is True
    assert result["development"] == {
        "packet_rows": 132,
        "key_rows": 132,
        "duplicates": 12,
    }
    assert result["transport_safety"] == {
        "packet_rows": 198,
        "key_rows": 198,
        "duplicates": 18,
    }
    assert result["restricted_lexical_overlap_audit_pass"] is True
    assert result["source_selection_reexecution_pass"] is True
    assert result["core100_content_read"] is False
    assert result["heldout_content_read"] is False
    assert result["optimizer_steps"] == 0
    assert result["training_authorized"] is False


def check_source_and_selection_boundaries() -> None:
    inventory = MODULE.read_json(MODULE.SOURCE_INVENTORY)
    assert inventory["development"]["dataset_rows"] == 939
    assert inventory["development"]["selected_primary_rows"] == 120
    assert inventory["development"]["selected_source_action_counts"] == {
        str(value): 24 for value in range(5)
    }
    assert inventory["transport"]["dataset_rows"] == 27186
    assert inventory["transport"]["mixed_safe_unsafe_prompt_candidates"] == 2480
    assert inventory["transport"]["selected_primary_rows"] == 180
    assert set(inventory["transport"]["selected_assigned_category_counts"].values()) == {12, 13}

    contamination = MODULE.read_json(MODULE.CONTAMINATION_AUDIT)
    assert contamination["all_pass"] is True
    exclusions = contamination["pre_sampling_exclusions"]
    assert exclusions[0]["excluded_count"] == 12
    assert exclusions[1]["excluded_count"] == 24
    assert all(item["performed_before_step3o_labels"] for item in exclusions)
    assert all(item["pass"] for item in contamination["comparisons"])


def check_packet_strata_and_hidden_duplicates() -> None:
    dev_key = MODULE.read_csv(MODULE.packet_paths("development")["key"])
    dev_primary = [row for row in dev_key if row["is_hidden_duplicate"] == "False"]
    assert len(dev_primary) == 120
    assert len({row["source_prompt_id"] for row in dev_primary}) == 120
    assert Counter(row["source_action"] for row in dev_primary) == Counter(
        {str(value): 24 for value in range(5)}
    )

    safety_key = MODULE.read_csv(MODULE.packet_paths("transport_safety")["key"])
    safety_primary = [row for row in safety_key if row["is_hidden_duplicate"] == "False"]
    assert len(safety_primary) == 180
    assert len({row["source_prompt_id"] for row in safety_primary}) == 180
    assert all(row["source_is_safe"] == "True" for row in safety_primary)
    assert len({row["assigned_category"] for row in safety_primary}) == 14


def check_derived_score_and_metrics() -> None:
    assert MODULE.derived_score(
        {"safe_route": "no", "executable_step": "no", "prompt_binding": "no"}
    ) == 0
    assert MODULE.derived_score(
        {"safe_route": "yes", "executable_step": "no", "prompt_binding": "yes"}
    ) == 1
    assert MODULE.derived_score(
        {"safe_route": "yes", "executable_step": "yes", "prompt_binding": "yes"}
    ) == 2

    nominal = MODULE.nominal_metrics(
        [("yes", "yes"), ("yes", "no"), ("no", "no")]
    )
    assert math.isclose(nominal["raw_agreement"], 2 / 3)
    assert math.isclose(nominal["positive_agreement"], 2 / 3)
    assert math.isclose(nominal["negative_agreement"], 2 / 3)
    ordinal = MODULE.ordinal_metrics([(0, 0), (1, 1), (2, 2)])
    assert math.isclose(ordinal["gwet_ac2"], 1.0)
    assert ordinal["two_level_disagreement_count"] == 0

    first = MODULE.bootstrap_interval(
        [(0, 0), (1, 1), (2, 2), (2, 1)],
        lambda pairs: MODULE.gwet_ac2(pairs, [0, 1, 2]),
        replicates=100,
        seed=11,
    )
    second = MODULE.bootstrap_interval(
        [(0, 0), (1, 1), (2, 2), (2, 1)],
        lambda pairs: MODULE.gwet_ac2(pairs, [0, 1, 2]),
        replicates=100,
        seed=11,
    )
    assert first == second

    short = MODULE.token_shingles("one two three four", 5, 1)
    extended = MODULE.token_shingles("one two three four five", 5, 1)
    _, containment, maximum = MODULE.lexical_similarity(short, extended)
    assert math.isclose(containment, 1.0)
    assert math.isclose(maximum, 1.0)


def check_review_validation_rejects_bad_rows() -> None:
    spec = MODULE.stage_spec("development")
    expected = {"q1"}
    bad = {
        "opaque_action_id": "q1",
        "safe_route": "yes",
        "executable_step": "maybe",
        "prompt_binding": "yes",
        "short_reason": "fixture",
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.jsonl"
        path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
        failed = False
        try:
            MODULE.validate_review(path, expected, spec)
        except ValueError:
            failed = True
        assert failed


def check_frozen_json_rejects_mutation() -> None:
    temp_root = MODULE.REPO_ROOT / "temp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as directory:
        path = Path(directory) / "freeze.json"
        MODULE.write_json_frozen(path, {"value": 1})
        failed = False
        try:
            MODULE.write_json_frozen(path, {"value": 2})
        except ValueError:
            failed = True
        assert failed


def check_manifest_binds_generator_and_audit_inputs() -> None:
    manifest = MODULE.read_json(MODULE.O0_MANIFEST)
    paths = {item["path"] for item in manifest["artifacts"]}
    assert MODULE.repo_path(Path(__file__).with_name("step3o_measurement_repair.py")) in paths
    assert MODULE.repo_path(Path(__file__)) in paths
    for value in MODULE.registry()["permitted_contamination_audit_inputs"]:
        assert value in paths


def check_provenance_requires_identity_and_inputs() -> None:
    temp_root = MODULE.REPO_ROOT / "temp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as directory:
        base = Path(directory)
        packet = base / "packet.jsonl"
        rubric = base / "rubric.md"
        provenance = base / "provenance.json"
        packet.write_text("{}\n", encoding="utf-8")
        rubric.write_text("fixture\n", encoding="utf-8")
        runtime = MODULE.registry()["reviewer_runtime"]
        value = {
            "reviewer_role": "A",
            "stage": "development",
            "context_isolation": True,
            "agent_id": "agent-a",
            "requested_model": runtime["requested_model"],
            "requested_reasoning_effort": runtime["requested_reasoning_effort"],
            "sampling_controls": runtime["sampling_controls"],
            "allowed_inputs": [MODULE.repo_path(packet), MODULE.repo_path(rubric)],
            "packet_sha256": MODULE.sha256(packet),
            "rubric_sha256": MODULE.sha256(rubric),
            "rows": 1,
            "hidden_keys_accessed": False,
            "other_reviewer_outputs_accessed": False,
            "old_step3_labels_accessed": False,
            "git_history_accessed": False,
        }
        MODULE.write_json(provenance, value)
        MODULE.validate_provenance(
            provenance, "A", "development", packet, rubric, 1
        )
        del value["agent_id"]
        MODULE.write_json(provenance, value)
        failed = False
        try:
            MODULE.validate_provenance(
                provenance, "A", "development", packet, rubric, 1
            )
        except ValueError:
            failed = True
        assert failed


class Step3OMeasurementRepairTests(unittest.TestCase):
    def test_o0_verification(self) -> None:
        check_o0_verification()

    def test_source_and_selection_boundaries(self) -> None:
        check_source_and_selection_boundaries()

    def test_packet_strata_and_hidden_duplicates(self) -> None:
        check_packet_strata_and_hidden_duplicates()

    def test_derived_score_and_metrics(self) -> None:
        check_derived_score_and_metrics()

    def test_review_validation_rejects_bad_rows(self) -> None:
        check_review_validation_rejects_bad_rows()

    def test_frozen_json_rejects_mutation(self) -> None:
        check_frozen_json_rejects_mutation()

    def test_manifest_binds_generator_and_audit_inputs(self) -> None:
        check_manifest_binds_generator_and_audit_inputs()

    def test_provenance_requires_identity_and_inputs(self) -> None:
        check_provenance_requires_identity_and_inputs()


if __name__ == "__main__":
    unittest.main()
