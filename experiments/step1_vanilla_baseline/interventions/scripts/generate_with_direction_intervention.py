from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from intervention_common import (
    EVAL_TO_FILE,
    experiment_id,
    load_json,
    read_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--intervention", required=True, choices=["top", "bottom", "random"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def adapter_dir(base_config: dict[str, Any], family: str, seed: int) -> Path:
    model = base_config["model_name_or_path"].split("/")[-1].lower()
    exp_id = f"step1_{base_config['machine']}_{model}_{family}_s{seed}"
    return Path(base_config["output_dir"]) / "adapters" / exp_id


def output_path(config: dict[str, Any], seed: int, intervention: str) -> Path:
    exp_id = experiment_id(
        config["family"],
        seed,
        intervention,
        int(config["k"]),
        float(config["scale"]),
    )
    return Path(config["output_dir"]) / "generations" / f"{exp_id}_{config['eval_set']}.jsonl"


def load_selected_layers(path: Path, exp_id: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["experiment_id"] == exp_id]


def compute_direction_components(module: Any, rank: int):
    import torch

    if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
        raise ValueError("Selected module has no LoRA A/B weights")
    adapter_names = list(module.lora_A.keys())
    if not adapter_names:
        raise ValueError("Selected module has empty LoRA adapter list")
    adapter_name = adapter_names[0]
    a = module.lora_A[adapter_name].weight.detach().float()
    b = module.lora_B[adapter_name].weight.detach().float()
    scale = float(module.scaling.get(adapter_name, 1.0)) if hasattr(module, "scaling") else 1.0

    q_b, r_b = torch.linalg.qr(b, mode="reduced")
    q_at, r_at = torch.linalg.qr(a.T, mode="reduced")
    u_small, s, vh_small = torch.linalg.svd(r_b @ r_at.T, full_matrices=False)
    keep = min(rank, s.numel())
    left = (q_b @ u_small[:, :keep]).contiguous()
    right = (q_at @ vh_small[:keep, :].T).contiguous()
    return {
        "adapter_name": adapter_name,
        "left": left,
        "right": right,
        "singular_values": s[:keep].contiguous(),
        "lora_scale": scale,
    }


def register_direction_hooks(model: Any, selected_keys: set[str], rank: int, gamma: float):
    import torch

    handles = []
    hooked = []
    for name, module in model.named_modules():
        if name not in selected_keys:
            continue
        components = compute_direction_components(module, rank)

        def hook(mod, args, output, comps=components):
            if not args:
                return output
            x = args[0]
            if not torch.is_tensor(x) or not torch.is_tensor(output):
                return output
            right = comps["right"].to(device=x.device, dtype=x.dtype)
            left = comps["left"].to(device=output.device, dtype=output.dtype)
            singular_values = comps["singular_values"].to(device=output.device, dtype=output.dtype)
            coeff = x.matmul(right.to(dtype=x.dtype))
            correction = coeff.to(dtype=output.dtype) * singular_values
            correction = correction.matmul(left.T)
            return output - float(gamma) * float(comps["lora_scale"]) * correction

        handles.append(module.register_forward_hook(hook))
        hooked.append(
            {
                "layer_key": name,
                "adapter_name": components["adapter_name"],
                "direction_rank": len(components["singular_values"]),
                "lora_scale": components["lora_scale"],
                "top_singular_value": float(components["singular_values"][0])
                if len(components["singular_values"])
                else 0.0,
            }
        )
    return handles, hooked


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    base_config = load_json(config["base_config"])
    exp_id = experiment_id(
        config["family"],
        args.seed,
        args.intervention,
        int(config["k"]),
        float(config["scale"]),
    )
    selected_path = Path(config["output_dir"]) / "selected_layers.csv"
    selected_rows = load_selected_layers(selected_path, exp_id)
    selected_keys = {row["layer_key"] for row in selected_rows}
    eval_path = Path(base_config["data_dir"]) / EVAL_TO_FILE[config["eval_set"]]
    eval_rows = read_jsonl(eval_path, limit=base_config["eval_per_set"])
    adapter = adapter_dir(base_config, config["family"], args.seed)
    out_path = output_path(config, args.seed, args.intervention)
    direction_rank = int(config["direction_rank"])
    gamma = float(config["gamma"])

    print("experiment_id:", exp_id)
    print("adapter:", adapter)
    print("selected_layers:", len(selected_keys))
    print("direction_rank:", direction_rank)
    print("gamma:", gamma)
    print("eval_rows:", len(eval_rows))
    print("output:", out_path)
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
    handles, hooked_layers = register_direction_hooks(model, selected_keys, direction_rank, gamma)
    print("direction_hooks:", len(hooked_layers))
    if len(hooked_layers) != len(selected_keys):
        missing = sorted(selected_keys - {row["layer_key"] for row in hooked_layers})
        raise RuntimeError(f"Only hooked {len(hooked_layers)} of {len(selected_keys)} layers: {missing}")
    model.eval()

    generated = []
    try:
        for row in eval_rows:
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
                    "train_family": config["family"],
                    "seed": args.seed,
                    "intervention": args.intervention,
                    "module_filter": "",
                    "k": config["k"],
                    "scale": config["scale"],
                    "direction_intervention": "top_singular_removal",
                    "direction_rank": direction_rank,
                    "gamma": gamma,
                    "direction_layers": [item["layer_key"] for item in hooked_layers],
                    "prompt": row["prompt"],
                    "response": response,
                }
            )
    finally:
        for handle in handles:
            handle.remove()

    write_jsonl(out_path, generated)
    print(f"wrote {out_path} ({len(generated)} rows)")


if __name__ == "__main__":
    main()
