from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_step3ct_t5r_p6_eol as audit


class Step3CTT5RProvenanceTests(unittest.TestCase):
    def test_raw_legacy_match(self) -> None:
        value = b'{\n  "value": 1\n}\n'
        self.assertEqual(audit.legacy_match_mode(value, audit.sha256_bytes(value), len(value)), "raw_committed_bytes")

    def test_crlf_legacy_match(self) -> None:
        value = b'{\n  "value": 1\n}\n'
        legacy = value.replace(b"\n", b"\r\n")
        self.assertEqual(
            audit.legacy_match_mode(value, audit.sha256_bytes(legacy), len(legacy)),
            "canonical_lf_reexpanded_to_crlf",
        )

    def test_content_change_does_not_reconcile(self) -> None:
        current = b'{\n  "value": 2\n}\n'
        legacy = b'{\r\n  "value": 1\r\n}\r\n'
        self.assertIsNone(audit.legacy_match_mode(current, audit.sha256_bytes(legacy), len(legacy)))

    def test_lone_cr_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            audit.canonical_lf_bytes(b"left\rright")

    def test_git_blob_oid_fixture(self) -> None:
        value = b"test content\n"
        expected = hashlib.sha1(b"blob 13\0" + value).hexdigest()
        self.assertEqual(audit.git_blob_oid(value), expected)


if __name__ == "__main__":
    unittest.main()
