from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "paper" / "strong_mechanism" / "scripts" / "prepare_runtime_config.py"
spec = importlib.util.spec_from_file_location("paper_prepare_runtime_config", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class RuntimePinTest(unittest.TestCase):
    def test_all_formal_configs_share_full_model_pin_and_machine(self) -> None:
        registry = json.loads(
            (REPO_ROOT / "paper" / "strong_mechanism" / "configs" / "study_registry.json").read_text(encoding="utf-8")
        )
        expected_runs = 0
        for formal in registry["formal_configs"]:
            config = json.loads((REPO_ROOT / formal["path"]).read_text(encoding="utf-8"))
            module.validate_paper_config(config)
            self.assertEqual(config["machine"], registry["primary_machine"])
            self.assertEqual(config["model_name_or_path"], registry["model_repository"])
            self.assertEqual(config["model_revision"], registry["model_revision"])
            self.assertEqual(config["seeds"]["risk"], formal["seeds"])
            expected_runs += len(formal["arms"]) * len(formal["seeds"])
        self.assertEqual(expected_runs, registry["expected_formal_runs"])
        self.assertEqual(expected_runs, 15)

    def test_environment_lock_matches_registry(self) -> None:
        registry = json.loads(
            (REPO_ROOT / "paper" / "strong_mechanism" / "configs" / "study_registry.json").read_text(encoding="utf-8")
        )
        lock = json.loads((REPO_ROOT / registry["environment_lock"]).read_text(encoding="utf-8"))
        self.assertEqual(lock["machine"], registry["primary_machine"])
        self.assertEqual(lock["model_repository"], registry["model_repository"])
        self.assertEqual(lock["model_revision"], registry["model_revision"])
        self.assertRegex(lock["model_revision"], r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
