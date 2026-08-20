from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = "experiments/step3_response_quality_persistence/results/4070ti_step3_cdro_opc_m5/protocol/cdro_opc_m5_authoring_protocol.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "prepare"))
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    protocol_path = (REPO / args.protocol).resolve()
    protocol = read_json(protocol_path)
    risk_path = (REPO / protocol["frozen_inputs"]["risk_source"]["path"]).resolve()
    old_candidates_path = (REPO / protocol["frozen_inputs"]["m3_primary_candidates"]["path"]).resolve()
    instructions_path = (REPO / protocol["frozen_inputs"]["authoring_instructions"]["path"]).resolve()
    for name, path, expected in (
        ("risk_source", risk_path, protocol["frozen_inputs"]["risk_source"]["sha256"]),
        ("m3_primary_candidates", old_candidates_path, protocol["frozen_inputs"]["m3_primary_candidates"]["sha256"]),
        ("authoring_instructions", instructions_path, protocol["frozen_inputs"]["authoring_instructions"]["sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {name}")
    risk_rows = read_jsonl(risk_path)
    old_rows = read_jsonl(old_candidates_path)
    excluded_samples = {str(row["sample_id"]) for row in old_rows}
    eligible = [row for row in risk_rows if str(row["sample_id"]) not in excluded_samples]
    eligible.sort(key=lambda row: hashlib.sha256(f"cdro-opc-m5-authoring|{row['sample_id']}".encode()).hexdigest())
    selected = eligible[: int(protocol["selection"]["prompts"]) ]
    if len(selected) != int(protocol["selection"]["prompts"]):
        raise RuntimeError("insufficient prompt-disjoint risk rows")
    prompts = []
    for rank, row in enumerate(selected, 1):
        prompt_id = f"m5_prompt_{rank:03d}_{hashlib.sha256(str(row['sample_id']).encode()).hexdigest()[:12]}"
        prompts.append({
            "opaque_prompt_id": prompt_id,
            "sample_id": str(row["sample_id"]),
            "prompt": str(row["prompt"]),
            "rank": rank,
        })
    registry = {
        "schema": "step3_cdro_opc_m5_prompt_registry_v1",
        "excluded_m3_sample_count": len(excluded_samples),
        "selected_prompt_count": len(prompts),
        "prompts": prompts,
    }
    registry_hash = canonical_sha256(registry)
    if protocol.get("prompt_registry_sha256") and protocol["prompt_registry_sha256"] != registry_hash:
        raise RuntimeError("prompt registry differs from frozen hash")
    output = {
        "status": "PASS_CDRO_OPC_M5_PROMPT_SELECTION_PREFLIGHT",
        "risk_rows": len(risk_rows),
        "excluded_m3_samples": len(excluded_samples),
        "eligible_prompt_rows": len(eligible),
        "selected_prompts": len(prompts),
        "requests": len(prompts) * len(protocol["authoring"]["modes"]),
        "prompt_registry_sha256": registry_hash,
        "optimizer_steps": 0,
        "training_authorized": False,
    }
    print(json.dumps(output, sort_keys=True))
    if args.command == "prepare":
        spool = Path(protocol["spool_root"])
        if spool.exists():
            raise RuntimeError(f"spool already exists: {spool}")
        for shard_index in range(int(protocol["authoring"]["shards"])):
            shard_root = spool / "packets"
            shard_root.mkdir(parents=True, exist_ok=True)
            rows = []
            for item_index, prompt in enumerate(prompts):
                global_index = item_index * len(protocol["authoring"]["modes"])
                for mode_index, mode in enumerate(protocol["authoring"]["modes"]):
                    request_index = global_index + mode_index
                    if request_index % int(protocol["authoring"]["shards"]) != shard_index:
                        continue
                    rows.append({
                        "request_id": f"m5_req_{request_index:03d}",
                        "opaque_prompt_id": prompt["opaque_prompt_id"],
                        "sample_id": prompt["sample_id"],
                        "author_mode": mode,
                        "prompt": prompt["prompt"],
                        "rank": prompt["rank"],
                    })
            (shard_root / f"shard_{shard_index:02d}.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        (spool / "prompt_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "PASS_CDRO_OPC_M5_PROMPT_PACKETS_PREPARED", "spool_root": str(spool), "shards": protocol["authoring"]["shards"], "requests": len(prompts) * len(protocol["authoring"]["modes"])}, sort_keys=True))


if __name__ == "__main__":
    main()

