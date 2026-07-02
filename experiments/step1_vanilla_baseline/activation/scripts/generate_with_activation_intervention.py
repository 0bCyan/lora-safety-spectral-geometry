from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from activation_common import (
    EVAL_TO_FILE,
    adapter_dir,
    intervention_id,
    load_json,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--mode",
        choices=["safety", "random", "negative"],
        default="safety",
        help="safety adds the fitted direction; random adds matched random directions; negative subtracts it.",
    )
    parser.add_argument(
        "--layer-source",
        choices=["top_by_projection_gap", "top_by_probe_accuracy", "selected_layers"],
        default="top_by_projection_gap",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer indices. Overrides --layer-source.",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional short tag included in the output directory.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Record projection diagnostics on the prompt forward pass.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def layer_tag(layers: list[int]) -> str:
    return "layers" + "-".join(str(layer) for layer in layers)


def run_id(config: dict[str, Any], alpha: float, seed: int, mode: str, layers: list[int], tag: str | None) -> str:
    exp_id = intervention_id(config["direction_name"], alpha, seed, mode)
    suffix = tag if tag else layer_tag(layers)
    return f"{exp_id}_{suffix}"


def output_path(config: dict[str, Any], run_name: str) -> Path:
    return (
        Path(config["output_dir"])
        / "interventions"
        / run_name
        / "generations"
        / f"{run_name}_{config['eval_set']}.jsonl"
    )


def selected_layers(config: dict[str, Any], layer_source: str) -> list[int]:
    path = Path(config["output_dir"]) / "directions" / f"{config['direction_name']}_selected_layers.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    layers = data[layer_source]
    if layer_source == "selected_layers":
        return [int(layer) for layer in layers]
    return [int(layer) for layer in layers[: int(config["top_n_layers"])]]


def load_layer_metrics(config: dict[str, Any]) -> dict[int, dict[str, float]]:
    metrics_path = Path(config["output_dir"]) / "directions" / f"{config['direction_name']}_layer_metrics.csv"
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        return {
            int(row["layer"]): {
                "positive_projection_mean": float(row["positive_projection_mean"]),
                "negative_projection_mean": float(row["negative_projection_mean"]),
                "projection_midpoint": 0.5
                * (
                    float(row["positive_projection_mean"])
                    + float(row["negative_projection_mean"])
                ),
            }
            for row in csv.DictReader(handle)
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    base_config = load_json(config["base_config"])
    alpha = float(config["alpha"] if args.alpha is None else args.alpha)
    layers = (
        [int(item.strip()) for item in args.layers.split(",") if item.strip()]
        if args.layers
        else selected_layers(config, args.layer_source)
    )
    eval_path = Path(base_config["data_dir"]) / EVAL_TO_FILE[config["eval_set"]]
    eval_rows = read_jsonl(eval_path, limit=base_config["eval_per_set"])
    adapter = adapter_dir(base_config, "risk", args.seed)
    run_name = run_id(config, alpha, args.seed, args.mode, layers, args.run_tag)
    out_path = output_path(config, run_name)
    diagnostics_path = out_path.parents[1] / "projection_diagnostics.csv"
    print("family: risk")
    print("seed:", args.seed)
    print("adapter:", adapter)
    print("direction:", config["direction_name"])
    print("mode:", args.mode)
    print("alpha:", alpha)
    print("layers:", layers)
    print("run_name:", run_name)
    print("eval_rows:", len(eval_rows))
    print("output:", out_path)
    if args.diagnostics:
        print("diagnostics:", diagnostics_path)
    if args.dry_run:
        return

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError(
            "Missing generation dependencies. Install torch, transformers, peft and bitsandbytes."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        base_config["model_name_or_path"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if base_config.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if base_config.get("bf16") else torch.float16,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_config["model_name_or_path"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter)
    model.eval()

    direction_state = torch.load(
        Path(config["output_dir"]) / "directions" / f"{config['direction_name']}.pt",
        map_location="cpu",
    )
    directions = direction_state["directions"].float()
    if args.mode == "random":
        generator = torch.Generator(device="cpu").manual_seed(20260702 + args.seed)
        random_dirs = torch.randn(directions.shape, generator=generator)
        random_dirs = random_dirs / torch.linalg.vector_norm(random_dirs, dim=1).clamp_min(1e-12)[:, None]
        directions = random_dirs
    elif args.mode == "negative":
        directions = -directions
    layer_metrics = load_layer_metrics(config)
    diagnostics_rows = []

    handles = []
    transformer_layers = model.get_base_model().model.layers
    for layer_index in layers:
        direction = directions[layer_index]

        def hook(module, inputs, output, layer_direction=direction, layer=layer_index):
            if isinstance(output, tuple):
                hidden = output[0]
                layer_direction_device = layer_direction.to(device=hidden.device, dtype=hidden.dtype)
                if args.diagnostics and hidden.shape[1] > 1:
                    last_hidden = hidden[:, -1, :]
                    pre_projection = last_hidden.matmul(layer_direction_device)
                    hidden_norm = torch.linalg.vector_norm(last_hidden.float(), dim=-1)
                    delta_norm_ratio = abs(alpha) / hidden_norm.clamp_min(1e-12)
                    metrics = layer_metrics[layer]
                    post_projection = pre_projection + alpha
                    for batch_index in range(hidden.shape[0]):
                        diagnostics_rows.append(
                            {
                                "sample_id": current_sample_id,
                                "seed": args.seed,
                                "mode": args.mode,
                                "alpha": alpha,
                                "layer": layer,
                                "pre_projection": float(pre_projection[batch_index].detach().float().cpu().item()),
                                "post_projection": float(post_projection[batch_index].detach().float().cpu().item()),
                                "positive_projection_mean": metrics["positive_projection_mean"],
                                "negative_projection_mean": metrics["negative_projection_mean"],
                                "projection_midpoint": metrics["projection_midpoint"],
                                "crossed_midpoint": bool(post_projection[batch_index].detach().float().cpu().item() > metrics["projection_midpoint"]),
                                "hidden_norm": float(hidden_norm[batch_index].detach().cpu().item()),
                                "delta_norm_ratio": float(delta_norm_ratio[batch_index].detach().cpu().item()),
                            }
                        )
                updated = hidden + alpha * layer_direction_device
                return (updated, *output[1:])
            hidden = output
            return hidden + alpha * layer_direction.to(device=hidden.device, dtype=hidden.dtype)

        handles.append(transformer_layers[layer_index].register_forward_hook(hook))
    print("activation_hooks:", len(handles))

    generated = []
    try:
        for row in eval_rows:
            current_sample_id = row["sample_id"]
            messages = [{"role": "user", "content": row["prompt"]}]
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = f"User: {row['prompt']}\nAssistant:"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=base_config["generation"]["max_new_tokens"],
                    do_sample=base_config["generation"]["temperature"] > 0,
                    temperature=max(base_config["generation"]["temperature"], 1e-5),
                    top_p=base_config["generation"]["top_p"],
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            generated.append(
                {
                    "sample_id": row["sample_id"],
                    "eval_set": config["eval_set"],
                    "train_family": "risk",
                    "seed": args.seed,
                    "intervention": f"activation_{args.mode}",
                    "direction_name": config["direction_name"],
                    "activation_mode": args.mode,
                    "alpha": alpha,
                    "layers": layers,
                    "run_name": run_name,
                    "prompt": row["prompt"],
                    "response": response,
                }
            )
    finally:
        for handle in handles:
            handle.remove()

    write_jsonl(out_path, generated)
    if args.diagnostics:
        write_csv(diagnostics_path, diagnostics_rows)
        print(f"wrote {diagnostics_path} ({len(diagnostics_rows)} rows)")
    print(f"wrote {out_path} ({len(generated)} rows)")


if __name__ == "__main__":
    main()
