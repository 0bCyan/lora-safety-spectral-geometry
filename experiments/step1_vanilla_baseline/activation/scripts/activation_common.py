from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EVAL_TO_FILE = {
    "safety": "eval_safety.jsonl",
    "utility": "eval_utility.jsonl",
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def experiment_id(base_config: dict[str, Any], family: str, seed: int) -> str:
    model = base_config["model_name_or_path"].split("/")[-1].lower()
    return f"step1_{base_config['machine']}_{model}_{family}_s{seed}"


def adapter_dir(base_config: dict[str, Any], family: str, seed: int) -> Path:
    return Path(base_config["output_dir"]) / "adapters" / experiment_id(base_config, family, seed)


def generation_path(base_config: dict[str, Any], family: str, seed: int, eval_set: str) -> Path:
    return (
        Path(base_config["output_dir"])
        / "generations"
        / f"{experiment_id(base_config, family, seed)}_{eval_set}.jsonl"
    )


def local_judge_path(base_config: dict[str, Any], family: str, seed: int, eval_set: str) -> Path:
    return (
        Path(base_config["output_dir"])
        / "local_judge"
        / f"{experiment_id(base_config, family, seed)}_{eval_set}_local_judge.jsonl"
    )


def hidden_state_path(
    output_dir: str | Path,
    exp_id: str,
    eval_set: str,
    capture_mode: str = "prompt_last_token",
) -> Path:
    subdir = "hidden_states_first_generated" if capture_mode == "first_generated_token" else "hidden_states"
    return Path(output_dir) / subdir / f"{exp_id}_{eval_set}_{capture_mode}.pt"


def metadata_path(
    output_dir: str | Path,
    exp_id: str,
    eval_set: str,
    capture_mode: str = "prompt_last_token",
) -> Path:
    subdir = "hidden_states_first_generated" if capture_mode == "first_generated_token" else "hidden_states"
    suffix = "metadata" if capture_mode == "prompt_last_token" else f"{capture_mode}_metadata"
    return Path(output_dir) / subdir / f"{exp_id}_{eval_set}_{suffix}.jsonl"


def intervention_id(direction_name: str, alpha: float, seed: int, mode: str = "safety") -> str:
    alpha_tag = str(alpha).replace(".", "p")
    return f"risk_activation_{direction_name}_{mode}_alpha{alpha_tag}_s{seed}"
