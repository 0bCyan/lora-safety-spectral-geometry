from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_lora_with_checkpoints import DeterministicAnchorCursor, select_anchor_batch


def check_full_cycle_coverage() -> None:
    cursor = DeterministicAnchorCursor(80)
    first_cycle = cursor.draw(80)
    if sorted(first_cycle) != list(range(80)):
        raise AssertionError("first cycle did not cover all 80 anchors")
    second_start = cursor.draw(8)
    if second_start != list(range(8)):
        raise AssertionError("second cycle did not restart deterministically")


def check_no_repeated_accumulation_index() -> None:
    cursor = DeterministicAnchorCursor(80)
    draws = [cursor.draw(1)[0] for _ in range(8)]
    for index in range(len(draws) - 3):
        if len(set(draws[index : index + 4])) != 4:
            raise AssertionError("four consecutive micro-batches reused an anchor index")


def check_matched_sequences_identical() -> None:
    fixed = DeterministicAnchorCursor(80)
    diverse = DeterministicAnchorCursor(80)
    if fixed.draw(256) != diverse.draw(256):
        raise AssertionError("fixed and diverse cursor sequences diverged")


def check_exposure_balance() -> None:
    cursor = DeterministicAnchorCursor(80)
    cursor.draw(256)
    counts = Counter(cursor.exposure_counts)
    if set(counts) != {3, 4}:
        raise AssertionError(f"unexpected exposure counts: {counts}")
    if max(cursor.exposure_counts) - min(cursor.exposure_counts) > 1:
        raise AssertionError("exposure imbalance exceeded one")


def check_state_restore() -> None:
    original = DeterministicAnchorCursor(80)
    prefix = original.draw(37)
    restored = DeterministicAnchorCursor.from_state_dict(original.state_dict())
    suffix_a = original.draw(43)
    suffix_b = restored.draw(43)
    if suffix_a != suffix_b:
        raise AssertionError("restored cursor did not reproduce continuation")
    if prefix + suffix_a != list(range(80)):
        raise AssertionError("restored sequence broke first-cycle coverage")


def check_select_anchor_batch() -> None:
    features = [{"index": index} for index in range(80)]
    cursor = DeterministicAnchorCursor(80)
    batch, indices = select_anchor_batch(features, cursor, 4)
    if indices != [0, 1, 2, 3]:
        raise AssertionError("selected indices not sequential")
    if [row["index"] for row in batch] != indices:
        raise AssertionError("selected rows do not match indices")


def main() -> None:
    check_full_cycle_coverage()
    check_no_repeated_accumulation_index()
    check_matched_sequences_identical()
    check_exposure_balance()
    check_state_restore()
    check_select_anchor_batch()
    print("anchor cursor tests passed")


if __name__ == "__main__":
    main()
