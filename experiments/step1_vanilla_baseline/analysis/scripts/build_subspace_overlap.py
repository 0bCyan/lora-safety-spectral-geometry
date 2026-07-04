from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path("experiments/step1_vanilla_baseline")
TABLE_DIR = ROOT / "analysis/tables"
REPORT_DIR = ROOT / "analysis/reports"
FAMILIES = ["normal", "risk", "control"]
SEEDS = [1, 2, 3]
K_VALUES = [1, 2, 4]
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/step1_vanilla_baseline/configs/4070ti_qwen3_4b.json",
    )
    parser.add_argument("--output-dir", default=str(TABLE_DIR))
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def adapter_dir(config: dict[str, Any], family: str, seed: int) -> Path:
    model = config["model_name_or_path"].split("/")[-1].lower()
    exp_id = f"step1_{config['machine']}_{model}_{family}_s{seed}"
    return Path(config["output_dir"]) / "adapters" / exp_id


def layer_index(layer_name: str) -> int:
    match = re.search(r"layers\.(\d+)\.", layer_name)
    return int(match.group(1)) if match else -1


def block_type(module: str) -> str:
    if module in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return "attention"
    if module in {"gate_proj", "up_proj", "down_proj"}:
        return "mlp"
    return "other"


def pair_label(family_a: str, family_b: str) -> str:
    if family_a == family_b:
        return f"{family_a}_seed_to_seed"
    if "risk" in {family_a, family_b}:
        other = family_b if family_a == "risk" else family_a
        return f"risk_vs_{other}"
    ordered = sorted([family_a, family_b], key=FAMILIES.index)
    return f"{ordered[0]}_vs_{ordered[1]}"


def family_pair(family_a: str, family_b: str) -> str:
    if family_a == family_b:
        return f"{family_a}_vs_{family_b}"
    if "risk" in {family_a, family_b}:
        other = family_b if family_a == "risk" else family_a
        return f"risk_vs_{other}"
    ordered = sorted([family_a, family_b], key=FAMILIES.index)
    return f"{ordered[0]}_vs_{ordered[1]}"


def safe_stdev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


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


def singular_subspaces(a, b, max_k: int):
    import torch

    q_b, r_b = torch.linalg.qr(b, mode="reduced")
    q_at, r_at = torch.linalg.qr(a.T, mode="reduced")
    u_small, s, vh_small = torch.linalg.svd(r_b @ r_at.T, full_matrices=False)
    keep = min(max_k, s.numel())
    left = (q_b @ u_small[:, :keep]).contiguous()
    right = (q_at @ vh_small[:keep, :].T).contiguous()
    return left, right, s[:keep].contiguous()


def principal_metrics(space_a, space_b, k: int) -> dict[str, float]:
    import torch

    qa = space_a[:, :k]
    qb = space_b[:, :k]
    top_direction_cosine = torch.abs(torch.dot(qa[:, 0], qb[:, 0])).item()
    cosines = torch.linalg.svdvals(qa.T @ qb).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.arccos(cosines)).tolist()
    return {
        "top_direction_cosine": top_direction_cosine,
        "mean_principal_cosine": float(torch.mean(cosines).item()),
        "min_principal_angle": min(angles),
        "max_principal_angle": max(angles),
    }


def build_components(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for family in FAMILIES:
        for seed in SEEDS:
            adapter = adapter_dir(config, family, seed)
            pairs = load_lora_pairs(adapter)
            for layer_name, pair in sorted(pairs.items()):
                if "A" not in pair or "B" not in pair:
                    continue
                left, right, singular_values = singular_subspaces(pair["A"], pair["B"], max(K_VALUES))
                module = layer_name.split(".")[-1]
                by_layer[layer_name].append(
                    {
                        "family": family,
                        "seed": seed,
                        "layer_name": layer_name,
                        "layer_index": layer_index(layer_name),
                        "module": module,
                        "block_type": block_type(module),
                        "left": left,
                        "right": right,
                        "top_singular_value": float(singular_values[0].item()),
                    }
                )
    return by_layer


def build_long_rows(by_layer: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for layer_name, entries in sorted(by_layer.items()):
        if len(entries) < 2:
            continue
        for a, b in combinations(entries, 2):
            for side in ["left", "right"]:
                for k in K_VALUES:
                    metrics = principal_metrics(a[side], b[side], k)
                    rows.append(
                        {
                            "layer_name": layer_name,
                            "layer_index": a["layer_index"],
                            "module": a["module"],
                            "block_type": a["block_type"],
                            "family_a": a["family"],
                            "seed_a": a["seed"],
                            "family_b": b["family"],
                            "seed_b": b["seed"],
                            "family_pair": family_pair(a["family"], b["family"]),
                            "pair_type": pair_label(a["family"], b["family"]),
                            "side": side,
                            "k": k,
                            "top_singular_value_a": a["top_singular_value"],
                            "top_singular_value_b": b["top_singular_value"],
                            **metrics,
                        }
                    )
    return rows


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    metric_names = [
        "top_direction_cosine",
        "mean_principal_cosine",
        "min_principal_angle",
        "max_principal_angle",
    ]
    for group_key, group_rows in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, group_key)}
        item["n"] = len(group_rows)
        for metric in metric_names:
            values = [float(row[metric]) for row in group_rows]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = safe_stdev(values)
            item[f"{metric}_min"] = min(values)
            item[f"{metric}_max"] = max(values)
        output.append(item)
    return output


def write_report(summary_rows: list[dict[str, Any]], module_summary_rows: list[dict[str, Any]]) -> None:
    by_key = {
        (row["pair_type"], row["block_type"], row["side"], str(row["k"])): row
        for row in summary_rows
    }

    def metric(pair: str, side: str, k: int, name: str) -> str:
        row = by_key.get((pair, "all", side, str(k)))
        if not row:
            return "NA"
        return f"{float(row[name]):.4f}"

    module_rows = [
        row
        for row in module_summary_rows
        if row["pair_type"] in {"risk_seed_to_seed", "risk_vs_normal", "risk_vs_control"}
        and row["side"] == "left"
        and str(row["k"]) == "1"
    ]
    module_note = []
    for module in MODULES:
        values = {
            row["pair_type"]: float(row["mean_principal_cosine_mean"])
            for row in module_rows
            if row["module"] == module
        }
        if values:
            module_note.append(
                f"| {module} | {values.get('risk_seed_to_seed', math.nan):.4f} | "
                f"{values.get('risk_vs_normal', math.nan):.4f} | "
                f"{values.get('risk_vs_control', math.nan):.4f} |"
            )

    lines = [
        "# B3c adapter update subspace analysis",
        "",
        "## 1. 产物",
        "",
        "- `analysis/tables/subspace_overlap_long.csv`：逐模块、逐 pair、逐 side/k 的子空间重叠长表。",
        "- `analysis/tables/subspace_overlap_summary.csv`：按 pair_type、block_type、side、k 聚合。",
        "- `analysis/tables/subspace_overlap_module_summary.csv`：进一步按 module 聚合。",
        "",
        "## 2. 关键读数：right/input side",
        "",
        "| pair_type | k=1 cosine | k=2 cosine | k=4 cosine |",
        "|---|---:|---:|---:|",
        f"| risk_seed_to_seed | {metric('risk_seed_to_seed', 'right', 1, 'mean_principal_cosine_mean')} | {metric('risk_seed_to_seed', 'right', 2, 'mean_principal_cosine_mean')} | {metric('risk_seed_to_seed', 'right', 4, 'mean_principal_cosine_mean')} |",
        f"| risk_vs_normal | {metric('risk_vs_normal', 'right', 1, 'mean_principal_cosine_mean')} | {metric('risk_vs_normal', 'right', 2, 'mean_principal_cosine_mean')} | {metric('risk_vs_normal', 'right', 4, 'mean_principal_cosine_mean')} |",
        f"| risk_vs_control | {metric('risk_vs_control', 'right', 1, 'mean_principal_cosine_mean')} | {metric('risk_vs_control', 'right', 2, 'mean_principal_cosine_mean')} | {metric('risk_vs_control', 'right', 4, 'mean_principal_cosine_mean')} |",
        "",
        "## 3. 关键读数：left/output side",
        "",
        "| pair_type | k=1 cosine | k=2 cosine | k=4 cosine |",
        "|---|---:|---:|---:|",
        f"| risk_seed_to_seed | {metric('risk_seed_to_seed', 'left', 1, 'mean_principal_cosine_mean')} | {metric('risk_seed_to_seed', 'left', 2, 'mean_principal_cosine_mean')} | {metric('risk_seed_to_seed', 'left', 4, 'mean_principal_cosine_mean')} |",
        f"| risk_vs_normal | {metric('risk_vs_normal', 'left', 1, 'mean_principal_cosine_mean')} | {metric('risk_vs_normal', 'left', 2, 'mean_principal_cosine_mean')} | {metric('risk_vs_normal', 'left', 4, 'mean_principal_cosine_mean')} |",
        f"| risk_vs_control | {metric('risk_vs_control', 'left', 1, 'mean_principal_cosine_mean')} | {metric('risk_vs_control', 'left', 2, 'mean_principal_cosine_mean')} | {metric('risk_vs_control', 'left', 4, 'mean_principal_cosine_mean')} |",
        "",
        "## 4. left/output k=1 模块分布",
        "",
        "| module | risk seed-to-seed | risk vs normal | risk vs control |",
        "|---|---:|---:|---:|",
        *module_note,
        "",
        "## 5. 判读",
        "",
        "B3c 支持子空间机制需要看到 risk seed-to-seed overlap 明显高于 risk vs normal/control，且该差异在少数模块族或层段稳定集中。",
        "",
        "当前结果是混合的：left/output side 上 risk seed-to-seed overlap 明显高于 risk vs normal/control，说明 risk adapter 的输出方向存在 family 内一致性；但 right/input side 上 risk seed-to-seed overlap 很低，且低于 risk vs normal/control。",
        "",
        "因此 B3c 提供了部分描述性信号，但还不足以推翻 B2/B3a/B3b 的负结果，也不足以支持 post-hoc 参数空间干预的强因果机制叙事。更稳妥的结论是：LoRA 更新几何能描述 family 差异，尤其是 output-side 子空间差异；但简单谱坐标和 top direction intervention 目前不能稳定控制 unsafe behavior。",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "B3c子空间分析.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    output_dir = Path(args.output_dir)
    by_layer = build_components(config)
    long_rows = build_long_rows(by_layer)

    long_fields = [
        "layer_name",
        "layer_index",
        "module",
        "block_type",
        "family_a",
        "seed_a",
        "family_b",
        "seed_b",
        "family_pair",
        "pair_type",
        "side",
        "k",
        "top_singular_value_a",
        "top_singular_value_b",
        "top_direction_cosine",
        "mean_principal_cosine",
        "min_principal_angle",
        "max_principal_angle",
    ]
    write_csv(output_dir / "subspace_overlap_long.csv", long_rows, long_fields)

    summary_input = long_rows + [{**row, "block_type": "all"} for row in long_rows]
    summary_rows = summarize(summary_input, ["pair_type", "family_pair", "block_type", "side", "k"])
    module_summary_rows = summarize(long_rows, ["pair_type", "family_pair", "block_type", "module", "side", "k"])
    write_csv(output_dir / "subspace_overlap_summary.csv", summary_rows)
    write_csv(output_dir / "subspace_overlap_module_summary.csv", module_summary_rows)
    write_report(summary_rows, module_summary_rows)

    print(f"layers: {len(by_layer)}")
    print(f"long_rows: {len(long_rows)}")
    print(f"wrote: {output_dir}")


if __name__ == "__main__":
    main()
