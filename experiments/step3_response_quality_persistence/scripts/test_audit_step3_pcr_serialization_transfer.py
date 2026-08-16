from __future__ import annotations

import unittest

import audit_step3_pcr_serialization_transfer as audit


class PcrSerializationTransferTests(unittest.TestCase):
    def test_mismatch_packet_is_deterministic_and_distinct(self) -> None:
        rows = [
            {"sample_id": f"s{index}", "prompt": f"p{index}"}
            for index in range(5)
        ]
        first = audit.build_mismatch_packet(rows, 2)
        second = audit.build_mismatch_packet(rows, 2)
        self.assertEqual(first, second)
        for row in first:
            self.assertEqual(len(set(row["wrong_sample_ids"])), 2)
            self.assertNotIn(row["sample_id"], row["wrong_sample_ids"])

    def test_component_means_separate_forced_prefix(self) -> None:
        # Loss entries predict token indices 1..6. Preamble indices are 3..4; response starts at 5.
        result = audit.component_means([1, 2, 3, 5, 7, 9], direct_prefix=3, forced_prefix=5)
        self.assertEqual(result["forced_preamble"], {"nll": 4.0, "tokens": 2})
        self.assertEqual(result["response_only"], {"nll": 8.0, "tokens": 2})
        self.assertEqual(result["legacy_combined"], {"nll": 6.0, "tokens": 4})

    def test_prefix_leakage_route_requires_positive_legacy_and_contrast(self) -> None:
        transfer = {
            "legacy_combined": {"ci95_lower": 0.02},
            "response_only": {"ci95_lower": -0.01},
            "preamble_minus_response_transfer": {"ci95_lower": 0.03},
        }
        self.assertEqual(audit.candidate_route(transfer), "PREFIX_LEAKAGE_SUPPORTED")

    def test_response_transfer_survival_takes_precedence(self) -> None:
        transfer = {
            "legacy_combined": {"ci95_lower": 0.02},
            "response_only": {"ci95_lower": 0.01},
            "preamble_minus_response_transfer": {"ci95_lower": 0.03},
        }
        self.assertEqual(audit.candidate_route(transfer), "TRANSFER_SURVIVES_RESPONSE_ONLY")

    def test_no_candidate_is_promoted_when_transfer_is_inconclusive(self) -> None:
        routes = {
            "PCR_M_mean": "INCONCLUSIVE_SERIALIZATION_TRANSFER",
            "PCR_MW_mean_worst": "NO_OUT_OF_SAMPLE_CONDITIONALITY_TRANSFER",
        }
        self.assertEqual(audit.overall_route(routes), "INCONCLUSIVE_NO_NEW_TRAINING")


if __name__ == "__main__":
    unittest.main()
