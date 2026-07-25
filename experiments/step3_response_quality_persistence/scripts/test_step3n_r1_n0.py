from __future__ import annotations

import importlib.util
import csv
import collections
import json
import math
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("prepare_step3n_r1_n0.py")
SPEC = importlib.util.spec_from_file_location("step3n_r1_n0", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load Step3N-R1 N0 module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def check_source_boundaries() -> None:
    sources, validation = MODULE.load_source_inputs()
    assert len(sources) == 87
    assert len({row["sample_id"] for row in sources}) == 40
    assert validation["heldout_id_overlap"] == 0
    assert validation["old_quality_labels_read"] is False
    assert validation["core100_content_read"] is False
    assert validation["heldout_content_read"] is False
    assert validation["optimizer_steps"] == 0


def check_frozen_packet() -> None:
    packet = MODULE.read_jsonl(MODULE.SOURCE_PACKET)
    key = MODULE.read_csv(MODULE.SOURCE_KEY)
    assert len(packet) == 95
    assert len(key) == 95
    assert len({row["opaque_review_id"] for row in packet}) == 95
    assert all(set(row) == {"opaque_review_id", "prompt", "response"} for row in packet)

    packet_by_id = {row["opaque_review_id"]: row for row in packet}
    key_by_id = {row["opaque_review_id"]: row for row in key}
    assert set(packet_by_id) == set(key_by_id)
    duplicates = [row for row in key if row["is_hidden_duplicate"] == "True"]
    assert len(duplicates) == 8
    for duplicate in duplicates:
        primary = packet_by_id[duplicate["duplicate_of_review_id"]]
        repeated = packet_by_id[duplicate["opaque_review_id"]]
        assert primary["prompt"] == repeated["prompt"]
        assert primary["response"] == repeated["response"]
        assert MODULE.sha256_text(repeated["prompt"]) == duplicate["prompt_sha256"]
        assert MODULE.sha256_text(repeated["response"]) == duplicate["response_sha256"]


def valid_fixture(opaque_id: str = "q1") -> dict[str, object]:
    return {
        "opaque_review_id": opaque_id,
        "detail_retention": 2,
        "safe_actionability": 2,
        "naturalness": "yes",
        "clean_safe": "yes",
        "short_reason": "fixture",
    }


def expect_review_failure(rows: list[dict[str, object]], expected_ids: set[str]) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "review.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        failed = False
        try:
            MODULE.validate_review(path, expected_ids)
        except ValueError:
            failed = True
        assert failed


def check_review_rejections() -> None:
    missing_field = valid_fixture()
    missing_field.pop("clean_safe")
    expect_review_failure([missing_field], {"q1"})

    invalid_value = valid_fixture()
    invalid_value["clean_safe"] = "maybe"
    expect_review_failure([invalid_value], {"q1"})

    expect_review_failure([valid_fixture("wrong")], {"q1"})
    expect_review_failure([valid_fixture(), valid_fixture()], {"q1"})


def check_agreement_metrics() -> None:
    ordinal = MODULE.ordinal_metrics([(0, 0), (1, 1), (2, 2)])
    assert math.isclose(ordinal["gwet_ac2"], 1.0)
    nominal = MODULE.nominal_metrics(
        [("yes", "yes"), ("yes", "no"), ("no", "no")]
    )
    assert math.isclose(nominal["raw_agreement"], 2 / 3)
    assert math.isclose(nominal["positive_agreement"], 2 / 3)
    assert math.isclose(nominal["negative_agreement"], 2 / 3)
    assert math.isclose(nominal["pooled_yes_prevalence"], 0.5)


def check_frozen_source_stop() -> None:
    if not MODULE.SOURCE_DECISION.exists():
        return
    agreement = MODULE.read_json(MODULE.SOURCE_AGREEMENT)
    decision = MODULE.read_json(MODULE.SOURCE_DECISION)
    action = agreement["ordinal"]["safe_actionability"]
    assert action["n"] == 87
    assert math.isclose(action["exact_raw_agreement"], 33 / 87)
    assert math.isclose(action["gwet_ac2"], 0.37062093911179594)
    assert decision["status"] == "STOP"
    assert decision["source_stage_pass"] is False
    assert decision["qualified_source_prompts"] == 37
    assert decision["selected_positive_file_written"] is False
    assert MODULE.SELECTED_POSITIVES.read_bytes() == b""

    with MODULE.SOURCE_KEY.open(encoding="utf-8", newline="") as handle:
        key_rows = list(csv.DictReader(handle))
    ids = [
        row["opaque_review_id"]
        for row in key_rows
        if row["is_hidden_duplicate"] == "False"
    ]
    reviewer_a = {
        row["opaque_review_id"]: row for row in MODULE.read_jsonl(MODULE.SOURCE_A)
    }
    reviewer_b = {
        row["opaque_review_id"]: row for row in MODULE.read_jsonl(MODULE.SOURCE_B)
    }
    confusion = collections.Counter(
        (
            reviewer_a[opaque_id]["safe_actionability"],
            reviewer_b[opaque_id]["safe_actionability"],
        )
        for opaque_id in ids
    )
    assert confusion == {(0, 1): 1, (1, 1): 18, (2, 1): 53, (2, 2): 15}


def main() -> None:
    check_source_boundaries()
    check_frozen_packet()
    check_review_rejections()
    check_agreement_metrics()
    check_frozen_source_stop()
    print("Step3N-R1 N0 source protocol tests passed")


if __name__ == "__main__":
    main()
