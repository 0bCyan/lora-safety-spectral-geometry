from __future__ import annotations

import argparse
import csv
import json
import re
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

from common import checkpoint_dir, load_json, run_dir, spectral_path, write_csv, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", required=True, choices=["normal", "risk", "control"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-label", default=None, help="Optional run label used for regularized Step2B arms.")
    parser.add_argument("--checkpoints", default=None)
    return parser.parse_args()


def read_checkpoint_index(config: dict[str, Any], family: str, seed: int, run_label: str | None) -> list[dict[str, str]]:
    with (run_dir(config, family, seed, run_label) / "checkpoint_index.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def layer_index(layer_name: str) -> int:
    match = re.search(r"layers\.(\d+)\.", layer_name)
    return int(match.group(1)) if match else -1


def block_type(module: str) -> str:
    if module in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return "attention"
    if module in {"gate_proj", "up_proj", "down_proj"}:
        return "mlp"
    return "other"


def load_lora_pairs(adapter: Path) -> dict[str, dict[str, Any]]:
    import torch

    safetensor_path = adapter / "adapter_model.safetensors"
    bin_path = adapter / "adapter_model.bin"
    if safetensor_path.exists():
        from safetensors.torch import load_file

        state = load_file(str(safetensor_path), device="cpu")
    elif bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(f"No adapter weights found in {adapter}")
    pairs: dict[str, dict[str, Any]] = {}
    for name, tensor in state.items():
        if "lora_A" in name and name.endswith(".weight"):
            prefix = name.removesuffix(".lora_A.default.weight").removesuffix(".lora_A.weight")
            pairs.setdefault(prefix, {})["A"] = tensor.float()
        elif "lora_B" in name and name.endswith(".weight"):
            prefix = name.removesuffix(".lora_B.default.weight").removesuffix(".lora_B.weight")
            pairs.setdefault(prefix, {})["B"] = tensor.float()
    return pairs


def singular_values_from_gram(matrix):
    import torch

    if matrix.shape[0] <= matrix.shape[1]:
        gram = matrix @ matrix.T
    else:
        gram = matrix.T @ matrix
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    return torch.sqrt(eigenvalues).sort(descending=True).values


def low_rank_delta_svd(a, b):
    import torch

    q_b, r_b = torch.linalg.qr(b, mode="reduced")
    q_at, r_at = torch.linalg.qr(a.T, mode="reduced")
    u_small, s, vh_small = torch.linalg.svd(r_b @ r_at.T, full_matrices=False)
    left = q_b @ u_small
    right = q_at @ vh_small.T
    return left.contiguous(), right.contiguous(), s.contiguous()


def analyze_one(config: dict[str, Any], family: str, seed: int, checkpoint_row: dict[str, str], run_label: str | None):
    import torch

    checkpoint = checkpoint_row["checkpoint"]
    adapter = checkpoint_dir(config, family, seed, checkpoint, run_label)
    pairs = load_lora_pairs(adapter)
    rows = []
    components = {}
    for layer_name, pair in sorted(pairs.items()):
        if "A" not in pair or "B" not in pair:
            continue
        a = pair["A"]
        b = pair["B"]
        s_a = singular_values_from_gram(a)
        s_b = singular_values_from_gram(b)
        left, right, s_delta = low_rank_delta_svd(a, b)
        fro = torch.linalg.vector_norm(s_delta)
        total_sv = torch.sum(s_delta).item()
        top_sv = s_delta[0].item() if len(s_delta) else 0.0
        effective_rank = 0.0
        if total_sv > 0:
            probs = s_delta / torch.sum(s_delta)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
            effective_rank = float(torch.exp(torch.tensor(entropy)).item())
        module = layer_name.split(".")[-1]
        rows.append(
            {
                "layer_name": layer_name,
                "layer_index": layer_index(layer_name),
                "module": module,
                "block_type": block_type(module),
                "family": family,
                "run_label": run_label or "",
                "seed": seed,
                "checkpoint": checkpoint,
                "step": checkpoint_row["step"],
                "fraction": checkpoint_row["fraction"],
                "spectral_proxy": s_a[0].item() * s_b[0].item(),
                "delta_spectral_norm": top_sv,
                "delta_fro_norm": fro.item(),
                "top_singular_ratio": top_sv / total_sv if total_sv > 0 else 0.0,
                "effective_rank": effective_rank,
                "rank": int(a.shape[0]),
                "a_shape": json.dumps(list(a.shape)),
                "b_shape": json.dumps(list(b.shape)),
            }
        )
        components[layer_name] = {
            "left": left,
            "right": right,
            "top_singular_value": top_sv,
        }
    write_jsonl(spectral_path(config, family, seed, checkpoint, run_label), rows)
    return rows, components


def summarize_spectral(config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (row["family"], row.get("run_label", ""), row["seed"], row["checkpoint"], row["step"], row["fraction"]),
            [],
        ).append(row)
    summary = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], int(item[0][2]), float(item[0][5]))):
        family, run_label, seed, checkpoint, step, fraction = key
        summary.append(
            {
                "family": family,
                "run_label": run_label,
                "seed": seed,
                "checkpoint": checkpoint,
                "step": step,
                "fraction": fraction,
                "n_layers": len(group_rows),
                "mean_delta_spectral_norm": mean(float(row["delta_spectral_norm"]) for row in group_rows),
                "max_delta_spectral_norm": max(float(row["delta_spectral_norm"]) for row in group_rows),
                "mean_delta_fro_norm": mean(float(row["delta_fro_norm"]) for row in group_rows),
                "mean_top_singular_ratio": mean(float(row["top_singular_ratio"]) for row in group_rows),
                "mean_effective_rank": mean(float(row["effective_rank"]) for row in group_rows),
            }
        )
    write_csv(Path(config["output_dir"]) / "spectral_metrics" / "summary.csv", summary)


def principal_top_cosine(space_a, space_b, k: int, side: str) -> dict[str, float]:
    import torch

    qa = space_a[side][:, :k]
    qb = space_b[side][:, :k]
    top_direction_cosine = torch.abs(torch.dot(qa[:, 0], qb[:, 0])).item()
    cosines = torch.linalg.svdvals(qa.T @ qb).clamp(0.0, 1.0)
    return {
        "top_direction_cosine": top_direction_cosine,
        "mean_principal_cosine": float(torch.mean(cosines).item()),
    }


def write_temporal_subspace(
    config: dict[str, Any], family: str, seed: int, checkpoint_components: dict[str, dict[str, Any]], run_label: str | None
) -> None:
    rows = []
    checkpoints = list(checkpoint_components)
    for left_checkpoint, right_checkpoint in combinations(checkpoints, 2):
        left_components = checkpoint_components[left_checkpoint]
        right_components = checkpoint_components[right_checkpoint]
        for layer_name in sorted(set(left_components) & set(right_components)):
            module = layer_name.split(".")[-1]
            for side in ["left", "right"]:
                for k in [1, 2, 4]:
                    metrics = principal_top_cosine(left_components[layer_name], right_components[layer_name], k, side)
                    rows.append(
                        {
                            "family": family,
                            "run_label": run_label or "",
                            "seed": seed,
                            "checkpoint_a": left_checkpoint,
                            "checkpoint_b": right_checkpoint,
                            "layer_name": layer_name,
                            "layer_index": layer_index(layer_name),
                            "module": module,
                            "block_type": block_type(module),
                            "side": side,
                            "k": k,
                            **metrics,
                        }
                    )
    out_dir = Path(config["output_dir"]) / "subspace"
    label_part = f"_{run_label}" if run_label else ""
    write_csv(out_dir / f"{family}_s{seed}{label_part}_temporal_subspace_overlap.csv", rows)
    summary_input = [row for row in rows if row["side"] == "left" and row["k"] == 1]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_input:
        grouped.setdefault((row["checkpoint_a"], row["checkpoint_b"]), []).append(row)
    summary = []
    for (checkpoint_a, checkpoint_b), group_rows in sorted(grouped.items()):
        summary.append(
            {
                "family": family,
                "run_label": run_label or "",
                "seed": seed,
                "checkpoint_a": checkpoint_a,
                "checkpoint_b": checkpoint_b,
                "side": "left",
                "k": 1,
                "n": len(group_rows),
                "mean_top_direction_cosine": mean(float(row["top_direction_cosine"]) for row in group_rows),
                "mean_principal_cosine": mean(float(row["mean_principal_cosine"]) for row in group_rows),
            }
        )
    write_csv(out_dir / f"{family}_s{seed}{label_part}_temporal_subspace_summary.csv", summary)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    index_rows = read_checkpoint_index(config, args.family, args.seed, args.run_label)
    if args.checkpoints:
        wanted = {item.strip() for item in args.checkpoints.split(",") if item.strip()}
        index_rows = [row for row in index_rows if row["checkpoint"] in wanted]

    all_rows = []
    checkpoint_components = {}
    for row in index_rows:
        metric_rows, components = analyze_one(config, args.family, args.seed, row, args.run_label)
        all_rows.extend(metric_rows)
        checkpoint_components[row["checkpoint"]] = components
        print(f"analyzed {row['checkpoint']} ({len(metric_rows)} layers)")
    summarize_spectral(config, all_rows)
    write_temporal_subspace(config, args.family, args.seed, checkpoint_components, args.run_label)
    print(f"wrote metrics under {config['output_dir']}")


if __name__ == "__main__":
    main()
