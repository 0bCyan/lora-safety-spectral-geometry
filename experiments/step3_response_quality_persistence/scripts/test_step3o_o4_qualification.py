from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("step3o_o4_qualification.py")
SPEC = importlib.util.spec_from_file_location("step3o_o4_qualification", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("unable to load Step3O-O4 qualification module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PROTOCOL_COMMIT = "1" * 40
BEACON_ROUND = 9_999_999
BEACON_RANDOMNESS = "2" * 64


def registry() -> dict:
    return json.loads(
        (SCRIPT.parents[1] / "configs/step3o_o4_training_pool_registry.json").read_text(
            encoding="utf-8"
        )
    )


def jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def response(prompt_index: int, author: str) -> str:
    words = [f"author_{author}_prompt_{prompt_index:03d}"]
    words.extend(f"safe_{author}_{prompt_index:03d}_{index:03d}" for index in range(39))
    return " ".join(words)


def prompt_rows() -> list[dict]:
    rows: list[dict] = []
    slots = registry()["selection"]["category_severity_slots"]
    for category in sorted(slots):
        for severity in (1, 2, 3):
            for _ in range(int(slots[category][str(severity)])):
                index = len(rows)
                rows.append(
                    {
                        "opaque_prompt_id": f"prompt_{index:03d}",
                        "prompt": f"Synthetic prompt {index}",
                        "assigned_category": category,
                        "severity_level": severity,
                    }
                )
    if len(rows) != 60:
        raise AssertionError("synthetic fixture must reproduce the frozen 60 slots")
    return rows


def author_rows(role: str, order: list[int]) -> list[dict]:
    return [
        {
            "opaque_prompt_id": f"prompt_{index:03d}",
            "response": response(index, role),
        }
        for index in order
    ]


def fixture_package(
    *,
    beacon_round: int = BEACON_ROUND,
    beacon_randomness: str = BEACON_RANDOMNESS,
) -> tuple[dict, list[dict], list[dict]]:
    prompts = prompt_rows()
    rows_a = author_rows("A", list(range(60)))
    rows_b = author_rows("B", list(reversed(range(60))))
    package = MODULE.build_candidate_package(
        prompts,
        rows_a,
        rows_b,
        [row["opaque_prompt_id"] for row in rows_a],
        [row["opaque_prompt_id"] for row in rows_b],
        protocol_commit=PROTOCOL_COMMIT,
        beacon_round=beacon_round,
        beacon_randomness=beacon_randomness,
    )
    return package, rows_a, rows_b


def provenance_bundle(c_rows: int) -> tuple[dict[str, dict], dict[str, dict[str, str]]]:
    records: dict[str, dict] = {}
    expected: dict[str, dict[str, str]] = {}
    for index, role in enumerate(MODULE.SESSION_ROLES):
        stage = "authoring" if role.startswith("author_") else "qualification"
        row_count = 60 if role.startswith("author_") else 132 if role != "reviewer_c" else c_rows
        hashes = MODULE.hash_inputs(
            {
                f"{role}_packet.jsonl": f"{role}-packet\n".encode(),
                f"{role}_rubric.md": f"{role}-rubric\n".encode(),
            }
        )
        expected[role] = hashes
        records[role] = MODULE.build_provenance_record(
            role=role,
            stage=stage,
            agent_id=f"agent-{index}",
            allowed_input_hashes=hashes,
            row_count=row_count,
        )
    return records, expected


def vector_for_candidate(candidate: dict) -> dict:
    values = {
        "detail_retention": 2,
        "safe_route": "yes",
        "executable_step": "yes",
        "prompt_binding": "yes",
        "naturalness": "yes",
        "clean_safe": "yes",
    }
    if candidate["author_role"] == "B":
        index = int(candidate["opaque_prompt_id"].rsplit("_", 1)[1])
        failed_field = MODULE.DECISION_FIELDS[index % len(MODULE.DECISION_FIELDS)]
        values[failed_field] = 1 if failed_field == "detail_retention" else "no"
    return values


def review_rows(package: dict, role: str, *, all_qualified: bool = False) -> list[dict]:
    primary = {row["opaque_candidate_id"]: row for row in package["primary_candidates"]}
    links = package["reviewer_a_duplicates" if role == "A" else "reviewer_b_duplicates"]
    duplicate_to_primary = {
        row["duplicate_candidate_id"]: row["primary_candidate_id"] for row in links
    }
    packet = package["reviewer_a_packet" if role == "A" else "reviewer_b_packet"]
    rows: list[dict] = []
    for visible in packet:
        visible_id = visible["opaque_candidate_id"]
        primary_id = duplicate_to_primary.get(visible_id, visible_id)
        values = (
            {
                "detail_retention": 2,
                "safe_route": "yes",
                "executable_step": "yes",
                "prompt_binding": "yes",
                "naturalness": "yes",
                "clean_safe": "yes",
            }
            if all_qualified
            else vector_for_candidate(primary[primary_id])
        )
        rows.append(
            {
                "opaque_candidate_id": visible_id,
                **values,
                "short_reason": "Synthetic reason grounded in visible text.",
            }
        )
    return rows


def row_by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["opaque_candidate_id"]: row for row in rows}


class AuthorValidationTests(unittest.TestCase):
    def test_author_jsonl_schema_order_length_and_pair_collapse(self) -> None:
        order_a = [f"prompt_{index:03d}" for index in range(60)]
        order_b = list(reversed(order_a))
        rows_a = author_rows("A", list(range(60)))
        rows_b = author_rows("B", list(reversed(range(60))))
        validated_a = MODULE.validate_author_jsonl(jsonl(rows_a), order_a)
        validated_b = MODULE.validate_author_jsonl(jsonl(rows_b), order_b)
        MODULE.validate_author_pair(validated_a, validated_b)

        out_of_order = copy.deepcopy(rows_a)
        out_of_order[0], out_of_order[1] = out_of_order[1], out_of_order[0]
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.validate_author_jsonl(jsonl(out_of_order), order_a)

        too_short = copy.deepcopy(rows_a)
        too_short[0]["response"] = " ".join(["word"] * 39)
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.validate_author_jsonl(jsonl(too_short), order_a)

        extra_field = copy.deepcopy(rows_a)
        extra_field[0]["extra"] = False
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.validate_author_jsonl(jsonl(extra_field), order_a)

        collapsed = copy.deepcopy(validated_b)
        target = next(row for row in collapsed if row["opaque_prompt_id"] == "prompt_000")
        target["response"] = validated_a[0]["response"].upper()
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.validate_author_pair(validated_a, collapsed)

    def test_strict_json_rejects_duplicate_keys_constants_and_crlf(self) -> None:
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.parse_json('{"value":1,"value":2}', context="fixture")
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.parse_json('{"value":NaN}', context="fixture")
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.parse_jsonl('{"value":1}\r\n', context="fixture")


class ProvenanceTests(unittest.TestCase):
    def test_exact_provenance_hashes_forbidden_access_and_agent_isolation(self) -> None:
        inputs = MODULE.hash_inputs(
            {"packet.jsonl": b"packet\n", "rubric.md": b"rubric\n"}
        )
        records = []
        for index, role in enumerate(
            ("author_a", "author_b", "reviewer_a", "reviewer_b", "reviewer_c")
        ):
            stage = "authoring" if role.startswith("author") else "qualification"
            record = MODULE.build_provenance_record(
                role=role,
                stage=stage,
                agent_id=f"agent-{index}",
                allowed_input_hashes=inputs,
                row_count=60 if role.startswith("author") else 132 if role != "reviewer_c" else 3,
            )
            MODULE.validate_provenance_record(
                record,
                expected_role=role,
                expected_stage=stage,
                expected_attempt_id=MODULE.ATTEMPT_ID,
                expected_input_hashes=inputs,
                expected_row_count=record["row_count"],
            )
            records.append(record)
        isolated = MODULE.validate_agent_isolation(
            records, [record["role"] for record in records]
        )
        self.assertEqual(len(set(isolated.values())), 5)

        tampered = copy.deepcopy(records[0])
        tampered["input_sha256"]["packet.jsonl"] = "0" * 64
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.validate_provenance_record(
                tampered,
                expected_role="author_a",
                expected_stage="authoring",
                expected_attempt_id=MODULE.ATTEMPT_ID,
                expected_input_hashes=inputs,
                expected_row_count=60,
            )

        accessed = copy.deepcopy(records[0])
        accessed["prohibited_access"]["git_history_accessed"] = True
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.validate_provenance_record(
                accessed,
                expected_role="author_a",
                expected_stage="authoring",
                expected_attempt_id=MODULE.ATTEMPT_ID,
                expected_input_hashes=inputs,
                expected_row_count=60,
            )

        duplicate_agent = copy.deepcopy(records)
        duplicate_agent[-1]["agent_id"] = duplicate_agent[0]["agent_id"]
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.validate_agent_isolation(
                duplicate_agent, [record["role"] for record in duplicate_agent]
            )

    def test_complete_provenance_bundle_is_required_and_fail_closed(self) -> None:
        records, expected = provenance_bundle(7)
        isolated = MODULE.validate_session_provenance_bundle(
            records, expected, reviewer_c_row_count=7
        )
        self.assertEqual(set(isolated), set(MODULE.SESSION_ROLES))

        missing = copy.deepcopy(records)
        del missing["reviewer_c"]
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.validate_session_provenance_bundle(
                missing, expected, reviewer_c_row_count=7
            )

        wrong_count = copy.deepcopy(records)
        wrong_count["reviewer_c"]["row_count"] = 6
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.validate_session_provenance_bundle(
                wrong_count, expected, reviewer_c_row_count=7
            )

        reused_agent = copy.deepcopy(records)
        reused_agent["reviewer_c"]["agent_id"] = reused_agent["author_a"]["agent_id"]
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.validate_session_provenance_bundle(
                reused_agent, expected, reviewer_c_row_count=7
            )


class PacketAndReviewTests(unittest.TestCase):
    def test_salt_matches_frozen_v2_formula_and_rejects_unknown_purpose(self) -> None:
        reg = registry()
        prompt_only_purposes = {
            "prompt_matching",
            "opaque_prompt_id",
            "author_a_packet_order",
            "author_b_packet_order",
        }
        self.assertTrue(
            set(MODULE.RANDOMIZATION_PURPOSES)
            == set(reg["selection"]["randomness"]["purposes"])
            - prompt_only_purposes
        )
        purpose = "candidate_id"
        material = (
            f"step3o_o4_v2|{PROTOCOL_COMMIT}|{BEACON_ROUND}|"
            f"{BEACON_RANDOMNESS}|{purpose}"
        )
        self.assertEqual(
            MODULE.derive_salt(
                PROTOCOL_COMMIT,
                BEACON_ROUND,
                BEACON_RANDOMNESS,
                purpose,
            ),
            hashlib.sha256(material.encode("utf-8")).hexdigest(),
        )
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.derive_salt(
                PROTOCOL_COMMIT,
                BEACON_ROUND,
                BEACON_RANDOMNESS,
                "review_duplicate_id_a",
            )

    def test_registry_and_qualification_constants_match_exactly(self) -> None:
        reg = registry()
        authoring = reg["authoring"]
        qualification = reg["qualification"]
        self.assertEqual(MODULE.AUTHOR_ROWS, reg["selection"]["prompt_count"])
        self.assertEqual(MODULE.PROMPT_CLUSTERS, reg["selection"]["prompt_count"])
        self.assertEqual(
            MODULE.PRIMARY_CANDIDATES,
            authoring["authors"] * reg["selection"]["prompt_count"],
        )
        self.assertEqual(
            MODULE.HIDDEN_DUPLICATES, qualification["reviewer_hidden_duplicates"]
        )
        self.assertEqual(
            MODULE.MIN_QUALIFIED_PROMPTS,
            qualification["minimum_qualified_prompts"],
        )
        self.assertEqual(list(MODULE.OFFICIAL_ROUTES), qualification["official_routes"])
        self.assertEqual(
            MODULE.MIN_RESPONSE_TOKENS,
            authoring["minimum_response_whitespace_tokens"],
        )
        self.assertEqual(
            MODULE.MAX_RESPONSE_TOKENS,
            authoring["maximum_response_whitespace_tokens"],
        )
        self.assertEqual(MODULE.REQUESTED_MODEL, authoring["requested_model"])
        self.assertEqual(MODULE.REQUESTED_MODEL, qualification["requested_model"])
        self.assertEqual(
            MODULE.REQUESTED_REASONING_EFFORT,
            authoring["requested_reasoning_effort"],
        )
        self.assertEqual(
            MODULE.REQUESTED_REASONING_EFFORT,
            qualification["requested_reasoning_effort"],
        )
        self.assertEqual(MODULE.SAMPLING_CONTROLS, authoring["sampling_controls"])
        self.assertEqual(
            MODULE.SAMPLING_CONTROLS, qualification["sampling_controls"]
        )
        self.assertEqual(MODULE.DEFAULT_REVIEW_GATES, qualification["review_gates"])
        self.assertEqual(
            MODULE.DEFAULT_CATEGORY_MINIMA,
            qualification["minimum_qualified_by_category"],
        )
        self.assertEqual(
            {str(key): value for key, value in MODULE.DEFAULT_SEVERITY_MINIMA.items()},
            qualification["minimum_qualified_by_severity"],
        )
        qualifying = dict(qualification["candidate_requirements"])
        self.assertEqual(MODULE.candidate_qualified(qualifying), "yes")
        for field in MODULE.DECISION_FIELDS:
            failed = copy.deepcopy(qualifying)
            failed[field] = 1 if field == "detail_retention" else "no"
            self.assertEqual(MODULE.candidate_qualified(failed), "no")

    def test_commit_beacon_candidate_ids_duplicates_and_independent_orders(self) -> None:
        package, _, _ = fixture_package()
        changed, _, _ = fixture_package(beacon_randomness="3" * 64)
        MODULE.validate_candidate_package(
            package,
            protocol_commit=PROTOCOL_COMMIT,
            beacon_round=BEACON_ROUND,
            beacon_randomness=BEACON_RANDOMNESS,
        )
        self.assertEqual(len(package["primary_candidates"]), 120)
        self.assertEqual(len(package["reviewer_a_packet"]), 132)
        self.assertEqual(len(package["reviewer_b_packet"]), 132)
        self.assertEqual(len(package["reviewer_a_duplicates"]), 12)
        self.assertEqual(len(package["reviewer_b_duplicates"]), 12)
        self.assertNotEqual(
            {row["opaque_candidate_id"] for row in package["primary_candidates"]},
            {row["opaque_candidate_id"] for row in changed["primary_candidates"]},
        )

        primary_ids = {row["opaque_candidate_id"] for row in package["primary_candidates"]}
        order_a = [
            row["opaque_candidate_id"]
            for row in package["reviewer_a_packet"]
            if row["opaque_candidate_id"] in primary_ids
        ]
        order_b = [
            row["opaque_candidate_id"]
            for row in package["reviewer_b_packet"]
            if row["opaque_candidate_id"] in primary_ids
        ]
        self.assertNotEqual(order_a, order_b)
        for role in ("a", "b"):
            packet = {
                row["opaque_candidate_id"]: row
                for row in package[f"reviewer_{role}_packet"]
            }
            for link in package[f"reviewer_{role}_duplicates"]:
                primary = packet[link["primary_candidate_id"]]
                duplicate = packet[link["duplicate_candidate_id"]]
                self.assertEqual(primary["prompt"], duplicate["prompt"])
                self.assertEqual(primary["response"], duplicate["response"])

        tampered = copy.deepcopy(package)
        tampered["reviewer_a_packet"][0]["response"] += " changed"
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.validate_candidate_package(
                tampered,
                protocol_commit=PROTOCOL_COMMIT,
                beacon_round=BEACON_ROUND,
                beacon_randomness=BEACON_RANDOMNESS,
            )
        synchronized_tamper = copy.deepcopy(package)
        primary_id = synchronized_tamper["primary_candidates"][0]["opaque_candidate_id"]
        synchronized_tamper["primary_candidates"][0]["response"] += " changed"
        for key in ("reviewer_a_packet", "reviewer_b_packet"):
            for packet_row in synchronized_tamper[key]:
                if packet_row["opaque_candidate_id"] == primary_id:
                    packet_row["response"] += " changed"
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.validate_candidate_package(
                synchronized_tamper,
                protocol_commit=PROTOCOL_COMMIT,
                beacon_round=BEACON_ROUND,
                beacon_randomness=BEACON_RANDOMNESS,
            )
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.validate_candidate_package(
                package,
                protocol_commit=PROTOCOL_COMMIT,
                beacon_round=BEACON_ROUND,
                beacon_randomness="4" * 64,
            )

    def test_reviewer_schema_order_and_duplicate_consistency(self) -> None:
        package, _, _ = fixture_package()
        rows_a = review_rows(package, "A")
        order_a = [row["opaque_candidate_id"] for row in package["reviewer_a_packet"]]
        validated = MODULE.validate_reviewer_jsonl(jsonl(rows_a), order_a, role="A")
        consistency = MODULE.duplicate_consistency(
            validated, package["reviewer_a_duplicates"]
        )
        self.assertEqual(consistency["full_vector_consistent_count"], 12)

        one_mismatch = copy.deepcopy(validated)
        duplicate_id = package["reviewer_a_duplicates"][0]["duplicate_candidate_id"]
        row_by_id(one_mismatch)[duplicate_id]["clean_safe"] = "no"
        consistency = MODULE.duplicate_consistency(
            one_mismatch, package["reviewer_a_duplicates"]
        )
        self.assertEqual(consistency["full_vector_consistent_count"], 11)

        invalid = copy.deepcopy(rows_a)
        invalid[0]["detail_retention"] = True
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.validate_reviewer_jsonl(jsonl(invalid), order_a, role="A")
        invalid = copy.deepcopy(rows_a)
        invalid[0]["short_reason"] = "   "
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.validate_reviewer_jsonl(jsonl(invalid), order_a, role="A")


class ReliabilityTests(unittest.TestCase):
    def test_all_agree_metrics_yield_and_pass_route(self) -> None:
        package, _, _ = fixture_package()
        reviews_a = review_rows(package, "A")
        reviews_b = review_rows(package, "B")
        c_packet = MODULE.build_c_packet(
            package,
            reviews_a,
            reviews_b,
            protocol_commit=PROTOCOL_COMMIT,
            beacon_round=BEACON_ROUND,
            beacon_randomness=BEACON_RANDOMNESS,
        )
        self.assertEqual(c_packet, [])
        provenance, provenance_inputs = provenance_bundle(0)
        result = MODULE.analyze_validated_qualification(
            package,
            reviews_a,
            reviews_b,
            c_packet,
            [],
            provenance,
            provenance_inputs,
            protocol_commit=PROTOCOL_COMMIT,
            beacon_round=BEACON_ROUND,
            beacon_randomness=BEACON_RANDOMNESS,
        )
        reliability = result["reliability"]
        self.assertTrue(reliability["overall_pass"])
        self.assertEqual(reliability["detail_retention"]["exact_agreement"], 1.0)
        self.assertEqual(reliability["detail_retention"]["linear_weighted_agreement"], 1.0)
        self.assertEqual(reliability["detail_retention"]["gwet_ac2"], 1.0)
        self.assertEqual(reliability["full_vector_exact_agreement"], 1.0)
        for metrics in reliability["binary_components"].values():
            self.assertEqual(metrics["raw_agreement"], 1.0)
            self.assertEqual(metrics["gwet_ac1"], 1.0)
        derived = reliability["candidate_qualified"]
        self.assertEqual(derived["raw_agreement"], 1.0)
        self.assertEqual(derived["positive_agreement"], 1.0)
        self.assertEqual(derived["negative_agreement"], 1.0)
        self.assertEqual(derived["prevalence_gap"], 0.0)

        summary = result["yield"]
        self.assertEqual(summary["author_a"]["qualified"], 60)
        self.assertEqual(summary["author_b"]["qualified"], 0)
        self.assertEqual(summary["all_candidates"]["qualified"], 60)
        self.assertEqual(summary["best_of_two"]["qualified_prompts"], 60)
        self.assertEqual(summary["author_2x2"], {"neither": 0, "a_only": 60, "b_only": 0, "both": 0})
        self.assertTrue(summary["overall_pass"])
        self.assertEqual(result["route"], "PASS_MACHINE_RUBRIC_POOL")
        self.assertFalse(result["training_authorized"])
        self.assertEqual(result["optimizer_steps"], 0)

    def test_all_positive_and_all_negative_are_fail_closed(self) -> None:
        positive = MODULE.binary_agreement_metrics([("yes", "yes")] * 120)
        negative = MODULE.binary_agreement_metrics([("no", "no")] * 120)
        self.assertIsNone(positive["negative_agreement"])
        self.assertFalse(positive["nondegenerate"])
        self.assertIsNone(negative["positive_agreement"])
        self.assertFalse(negative["nondegenerate"])

        package, _, _ = fixture_package()
        reviews_a = review_rows(package, "A", all_qualified=True)
        reviews_b = review_rows(package, "B", all_qualified=True)
        reliability = MODULE.compute_ab_reliability(
            package,
            reviews_a,
            reviews_b,
            protocol_commit=PROTOCOL_COMMIT,
            beacon_round=BEACON_ROUND,
            beacon_randomness=BEACON_RANDOMNESS,
        )
        self.assertFalse(reliability["overall_pass"])
        self.assertTrue(reliability["gates"]["clean_safe_raw_pass"])
        self.assertTrue(reliability["gates"]["clean_safe_ac1_pass"])
        self.assertFalse(reliability["gates"]["candidate_qualified_negative_pass"])
        self.assertIsNone(
            reliability["stratified_prompt_reweighting"]["q025"]
        )

        provenance, provenance_inputs = provenance_bundle(0)
        with self.assertRaisesRegex(
            MODULE.QualificationValidationError,
            "C adjudication and hidden-key yield analysis are forbidden",
        ):
            MODULE.analyze_validated_qualification(
                package,
                reviews_a,
                reviews_b,
                [],
                [],
                provenance,
                provenance_inputs,
                protocol_commit=PROTOCOL_COMMIT,
                beacon_round=BEACON_ROUND,
                beacon_randomness=BEACON_RANDOMNESS,
            )

    def test_prompt_reweighting_is_stratified_and_reproducible(self) -> None:
        clusters = {
            f"prompt_{index:03d}": [("yes", "yes"), ("no", "no")]
            for index in range(60)
        }
        strata = {
            row["opaque_prompt_id"]: (
                row["assigned_category"],
                row["severity_level"],
            )
            for row in prompt_rows()
        }
        first = MODULE.stratified_prompt_reweighting_ac1(
            clusters,
            strata,
            protocol_commit=PROTOCOL_COMMIT,
            beacon_round=BEACON_ROUND,
            beacon_randomness=BEACON_RANDOMNESS,
        )
        second = MODULE.stratified_prompt_reweighting_ac1(
            clusters,
            strata,
            protocol_commit=PROTOCOL_COMMIT,
            beacon_round=BEACON_ROUND,
            beacon_randomness=BEACON_RANDOMNESS,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["replicates"], 10000)
        self.assertEqual(first["finite_replicates"], 10000)
        self.assertTrue(math.isclose(first["q025"], 1.0))
        self.assertIn("not a confidence interval", first["interpretation"])


class AdjudicationAndRouteTests(unittest.TestCase):
    def test_c_packet_completeness_and_candidate_level_adjudication(self) -> None:
        package, _, _ = fixture_package()
        reviews_a = review_rows(package, "A")
        reviews_b = review_rows(package, "B")
        map_b = row_by_id(reviews_b)
        primary_ids = {row["opaque_candidate_id"] for row in package["primary_candidates"]}
        disputed_id = next(
            candidate_id
            for candidate_id in primary_ids
            if MODULE.candidate_qualified(row_by_id(reviews_a)[candidate_id])
            == "yes"
        )
        map_b[disputed_id]["safe_route"] = "no"

        c_packet = MODULE.build_c_packet(
            package,
            reviews_a,
            reviews_b,
            protocol_commit=PROTOCOL_COMMIT,
            beacon_round=BEACON_ROUND,
            beacon_randomness=BEACON_RANDOMNESS,
        )
        self.assertEqual(len(c_packet), 1)
        self.assertEqual(set(c_packet[0]), {"opaque_candidate_id", "prompt", "response"})
        self.assertEqual(c_packet[0]["opaque_candidate_id"], disputed_id)
        all_duplicate_ids = {
            link["duplicate_candidate_id"]
            for role in ("a", "b")
            for link in package[f"reviewer_{role}_duplicates"]
        }
        self.assertFalse({row["opaque_candidate_id"] for row in c_packet} & all_duplicate_ids)

        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.validate_c_packet_and_reviews(
                c_packet,
                [],
                package,
                reviews_a,
                reviews_b,
                protocol_commit=PROTOCOL_COMMIT,
                beacon_round=BEACON_ROUND,
                beacon_randomness=BEACON_RANDOMNESS,
            )

        source = row_by_id(reviews_a)[disputed_id]
        c_rows = [
            {
                "opaque_candidate_id": disputed_id,
                **{field: source[field] for field in MODULE.DECISION_FIELDS},
                "short_reason": "C independently judged the visible row.",
            }
        ]
        c_rows[0]["safe_route"] = "no"
        validated_c = MODULE.validate_c_packet_and_reviews(
            c_packet,
            c_rows,
            package,
            reviews_a,
            reviews_b,
            protocol_commit=PROTOCOL_COMMIT,
            beacon_round=BEACON_ROUND,
            beacon_randomness=BEACON_RANDOMNESS,
        )
        final = MODULE.adjudicate_reviews(package, reviews_a, reviews_b, validated_c)
        final_row = row_by_id(final)[disputed_id]
        self.assertEqual(final_row["safe_route"], "no")
        self.assertEqual(final_row["adjudicated_fields"], list(MODULE.DECISION_FIELDS))
        self.assertEqual(
            final_row["detail_retention"], row_by_id(reviews_a)[disputed_id]["detail_retention"]
        )

    def test_c_cannot_rescue_candidate_rejected_by_both_a_and_b(self) -> None:
        package, _, _ = fixture_package()
        reviews_a = review_rows(package, "A")
        reviews_b = review_rows(package, "B")
        maps = (row_by_id(reviews_a), row_by_id(reviews_b))
        primary_ids = {
            row["opaque_candidate_id"] for row in package["primary_candidates"]
        }
        candidate_id = next(
            candidate_id
            for candidate_id in primary_ids
            if MODULE.candidate_qualified(maps[0][candidate_id]) == "no"
        )
        maps[0][candidate_id]["safe_route"] = "no"
        maps[0][candidate_id]["naturalness"] = "yes"
        maps[1][candidate_id]["safe_route"] = "yes"
        maps[1][candidate_id]["naturalness"] = "no"
        self.assertEqual(MODULE.candidate_qualified(maps[0][candidate_id]), "no")
        self.assertEqual(MODULE.candidate_qualified(maps[1][candidate_id]), "no")

        c_packet = MODULE.build_c_packet(
            package,
            reviews_a,
            reviews_b,
            protocol_commit=PROTOCOL_COMMIT,
            beacon_round=BEACON_ROUND,
            beacon_randomness=BEACON_RANDOMNESS,
        )
        self.assertNotIn(
            candidate_id,
            {row["opaque_candidate_id"] for row in c_packet},
        )
        final = MODULE.adjudicate_reviews(package, reviews_a, reviews_b, [])
        final_row = row_by_id(final)[candidate_id]
        self.assertEqual(final_row["candidate_qualified"], "no")
        self.assertEqual(final_row["adjudicated_fields"], [])

    def test_official_route_precedence_and_gates(self) -> None:
        passing = {
            "protocol_integrity_valid": True,
            "prompt_pool_valid": True,
            "authoring_valid": True,
            "review_inputs_valid": True,
            "reliability_pass": True,
            "target_yield_pass": True,
        }
        cases = [
            ({"protocol_integrity_valid": False}, "STOP_PROTOCOL_INTEGRITY"),
            ({"prompt_pool_valid": False}, "STOP_PROMPT_POOL"),
            ({"authoring_valid": False}, "STOP_AUTHORING"),
            ({"review_inputs_valid": False}, "STOP_REVIEW_INVALID"),
            ({"reliability_pass": False}, "STOP_REVIEW_UNRELIABLE"),
            ({"target_yield_pass": False}, "STOP_TARGET_YIELD"),
            ({}, "PASS_MACHINE_RUBRIC_POOL"),
        ]
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                kwargs = {**passing, **overrides}
                self.assertEqual(MODULE.official_route(**kwargs), expected)
        with self.assertRaises(MODULE.QualificationValidationError):
            MODULE.official_route(**{**passing, "reliability_pass": 1})


if __name__ == "__main__":
    unittest.main()
