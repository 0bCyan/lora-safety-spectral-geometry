"""Atomically encode one M3 author shard from plain-text drafts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import unicodedata
from pathlib import Path


class FinalizeError(RuntimeError):
    pass


OPENINGS = {
    "specific_redirect": "I can't help",
    "risk_mitigation": "I won't assist",
}


def read_assignment(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 10:
        raise FinalizeError("assignment must contain exactly ten rows")
    return rows


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def build_submission(assignment: list[dict[str, object]], draft_dir: Path) -> bytes:
    expected = {f"{index:02d}.txt" for index in range(1, 11)}
    observed = {path.name for path in draft_dir.iterdir() if path.is_file()}
    if observed != expected:
        raise FinalizeError("draft file set must be exactly 01.txt through 10.txt")
    outputs = []
    hashes = set()
    for index, row in enumerate(assignment, 1):
        path = draft_dir / f"{index:02d}.txt"
        response = path.read_text(encoding="utf-8").strip()
        tokens = len(response.split())
        if not 40 <= tokens <= 160:
            raise FinalizeError(f"draft {index:02d} has {tokens} whitespace tokens")
        mode = str(row["author_mode"])
        if mode not in OPENINGS or not response.startswith(OPENINGS[mode]):
            raise FinalizeError(f"draft {index:02d} has the wrong registered opening")
        if "<think" in response.casefold() or "</think" in response.casefold():
            raise FinalizeError(f"draft {index:02d} contains thinking tags")
        key = normalized(response)
        if key in hashes:
            raise FinalizeError("normalized response repeated within shard")
        hashes.add(key)
        outputs.append(
            json.dumps(
                {"opaque_request_id": row["opaque_request_id"], "response": response},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return ("\n".join(outputs) + "\n").encode("utf-8")


def finalize(assignment_path: Path, draft_dir: Path, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise FinalizeError("immutable output already exists")
    data = build_submission(read_assignment(assignment_path), draft_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if output_path.exists():
            raise FinalizeError("immutable output appeared during finalization")
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"status": "FINALIZED", "rows": 10, "output": str(output_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment", required=True)
    parser.add_argument("--draft-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = finalize(Path(args.assignment), Path(args.draft_dir), Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
