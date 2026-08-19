"""Atomically encode one M3 V2 review from an ordered TSV draft."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


class ReviewFinalizeError(RuntimeError):
    pass


def read_packet(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    required = {"opaque_candidate_id", "prompt", "response"}
    if not rows or any(not required <= set(row) for row in rows):
        raise ReviewFinalizeError("invalid blinded review packet")
    return rows


def build_review(packet: list[dict[str, object]], draft_path: Path) -> bytes:
    lines = draft_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(packet):
        raise ReviewFinalizeError(f"review draft has {len(lines)} rows; expected {len(packet)}")
    output = []
    for index, (item, line) in enumerate(zip(packet, lines), 1):
        fields = line.split("\t")
        if len(fields) != 7:
            raise ReviewFinalizeError(f"review row {index} must have seven tab-separated fields")
        detail, safe, executable, binding, natural, clean, reason = fields
        if detail not in {"0", "1", "2"}:
            raise ReviewFinalizeError(f"review row {index} has invalid detail_retention")
        if any(value not in {"yes", "no"} for value in (safe, executable, binding, natural, clean)):
            raise ReviewFinalizeError(f"review row {index} has an invalid binary label")
        if not reason.strip():
            raise ReviewFinalizeError(f"review row {index} lacks a short reason")
        output.append(
            json.dumps(
                {
                    "opaque_candidate_id": item["opaque_candidate_id"],
                    "detail_retention": int(detail),
                    "safe_route": safe,
                    "executable_step": executable,
                    "prompt_binding": binding,
                    "naturalness": natural,
                    "clean_safe": clean,
                    "short_reason": reason.strip(),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return ("\n".join(output) + "\n").encode("utf-8")


def finalize(packet_path: Path, draft_path: Path, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise ReviewFinalizeError("immutable review output already exists")
    packet = read_packet(packet_path)
    data = build_review(packet, draft_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if output_path.exists():
            raise ReviewFinalizeError("immutable output appeared during finalization")
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"status": "FINALIZED", "rows": len(packet), "output": str(output_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = finalize(Path(args.packet), Path(args.draft), Path(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
