from __future__ import annotations

import json
from pathlib import Path

from prepare_runtime_config import environment_snapshot, load_json, verify_environment


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = REPO_ROOT / "paper" / "strong_mechanism" / "environment" / "4070ti_training_lock.json"


def main() -> int:
    lock = load_json(LOCK_PATH)
    snapshot = environment_snapshot(lock)
    verify_environment(lock, snapshot)
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
