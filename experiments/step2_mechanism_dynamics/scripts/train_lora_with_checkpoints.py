from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from common import (
    EVAL_TO_FILE,
    FAMILY_TO_FILE,
    checkpoint_dir,
    checkpoint_tag,
    load_json,
    read_jsonl,
    run_dir,
    run_id,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", required=True, choices=sorted(FAMILY_TO_FILE))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-label", default=None, help="Optional config regularization arm label.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def get_arm_config(config: dict[str, Any], run_label: str | None) -> dict[str, Any]:
    default = {
        "regularization_type": config.get("regularization_type", "none"),
        "regularization_lambda": config.get("regularization_lambda", 0.0),
        "lambda_mean": config.get("lambda_mean", config.get("spectrasafe_lambda_mean", 0.0)),
        "lambda_max": config.get("lambda_max", config.get("spectrasafe_lambda_max", 0.0)),
        "spectral_top_k_modules": config.get("spectral_top_k_modules", 20),
        "regularization_start_step": config.get("regularization_start_step", 0),
        "regularization_end_step": config.get("regularization_end_step", None),
        "intervention_start_step": config.get("intervention_start_step", 6),
        "intervention_end_step": config.get("intervention_end_step", 16),
        "anchor_mix": config.get("anchor_mix", 0.0),
        "anchor_source_path": config.get("anchor_source_path", ""),
        "anchor_source_eval_set": config.get("anchor_source_eval_set", "safety"),
        "anchor_limit": config.get("anchor_limit", 30),
        "anchor_response_field": config.get("anchor_response_field", ""),
        "anchor_response": config.get(
            "anchor_response",
            "I can't help with instructions for harmful or unauthorized activity. "
            "I can help with safe, legal alternatives, prevention, or defensive guidance instead.",
        ),
        "margin_lambda": config.get("margin_lambda", 0.0),
        "target_margin": config.get("target_margin", 0.0),
        "margin_eval_set": config.get("margin_eval_set", config.get("logit_margin", {}).get("eval_set", "safety")),
        "margin_limit": config.get("margin_limit", config.get("logit_margin", {}).get("limit", 30)),
        "margin_batch_size": config.get("margin_batch_size", 4),
    }
    if not run_label:
        return default
    arms = {**config.get("regularization_arms", {}), **config.get("intervention_arms", {})}
    if run_label not in arms:
        raise ValueError(f"Unknown run label {run_label!r}. Available labels: {sorted(arms)}")
    merged = dict(default)
    merged.update(arms[run_label])
    return merged


def in_intervention_window(arm: dict[str, Any], global_step: int) -> bool:
    start_step = int(arm.get("intervention_start_step", 6))
    end_step = arm.get("intervention_end_step", 16)
    return global_step >= start_step and (end_step is None or global_step < int(end_step))


def scheduled_anchor_mix(arm: dict[str, Any], global_step: int) -> float:
    schedule = arm.get("anchor_schedule") or []
    for window in schedule:
        start_step = int(window.get("start_step", 0))
        end_step = window.get("end_step")
        if global_step >= start_step and (end_step is None or global_step < int(end_step)):
            if "anchor_mix_start" in window and "anchor_mix_end" in window:
                if end_step is None or int(end_step) <= start_step:
                    return float(window["anchor_mix_end"])
                progress = (global_step - start_step) / max(1, int(end_step) - start_step)
                start_mix = float(window["anchor_mix_start"])
                end_mix = float(window["anchor_mix_end"])
                return start_mix + progress * (end_mix - start_mix)
            return float(window.get("anchor_mix", 0.0))
    if in_intervention_window(arm, global_step):
        return float(arm.get("anchor_mix", 0.0))
    return 0.0


def format_prompt_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"用户：{prompt}\n助手："


def format_chat_text(tokenizer, prompt: str, response: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"{format_prompt_text(tokenizer, prompt)}{response}"


def make_tokenized_row(tokenizer, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    prompt_text = format_prompt_text(tokenizer, row["prompt"])
    full_text = format_chat_text(tokenizer, row["prompt"], row["response"])
    full = tokenizer(full_text, truncation=True, max_length=max_length, padding=False)
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length, padding=False)["input_ids"]
    labels = list(full["input_ids"])
    mask_len = min(len(prompt_ids), len(labels))
    labels[:mask_len] = [-100] * mask_len
    full["labels"] = labels
    return full


def collate_batch(tokenizer, features: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    max_len = max(len(feature["input_ids"]) for feature in features)
    input_ids = []
    attention_mask = []
    labels = []
    pad_id = tokenizer.pad_token_id
    for feature in features:
        pad_len = max_len - len(feature["input_ids"])
        input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(feature["attention_mask"] + [0] * pad_len)
        labels.append(feature["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def collate_prompt_batch(tokenizer, features: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    max_len = max(len(feature["input_ids"]) for feature in features)
    input_ids = []
    attention_mask = []
    pad_id = tokenizer.pad_token_id
    for feature in features:
        pad_len = max_len - len(feature["input_ids"])
        input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(feature["attention_mask"] + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def make_anchor_rows(config: dict[str, Any], arm: dict[str, Any]) -> list[dict[str, Any]]:
    has_anchor_mix = float(arm.get("anchor_mix", 0.0)) > 0
    has_scheduled_anchor = any(
        float(window.get("anchor_mix", window.get("anchor_mix_start", window.get("anchor_mix_end", 0.0)))) > 0
        for window in arm.get("anchor_schedule") or []
    )
    if not has_anchor_mix and not has_scheduled_anchor:
        return []
    eval_set = str(arm.get("anchor_source_eval_set", "safety"))
    source = Path(arm["anchor_source_path"]) if arm.get("anchor_source_path") else Path(config["data_dir"]) / EVAL_TO_FILE[eval_set]
    response_field = str(arm.get("anchor_response_field", ""))
    fixed_response = str(arm["anchor_response"])
    rows = read_jsonl(source, limit=int(arm.get("anchor_limit", 30)))
    anchor_rows = []
    for row in rows:
        if response_field:
            if response_field not in row:
                raise KeyError(f"anchor_response_field {response_field!r} not found in {source}")
            response = str(row[response_field])
        else:
            response = fixed_response
        anchor_rows.append(
            {
                "sample_id": f"anchor_{row.get('target_id', row['sample_id'])}",
                "family": "anchor",
                "source": eval_set,
                "prompt": row["prompt"],
                "response": response,
                "content_hash": row.get("prompt_hash", row.get("content_hash", "")),
                "target_hash": row.get("target_hash", ""),
            }
        )
    return anchor_rows


def make_margin_prompt_features(tokenizer, config: dict[str, Any], arm: dict[str, Any]) -> list[dict[str, Any]]:
    if float(arm.get("margin_lambda", 0.0)) <= 0:
        return []
    eval_set = str(arm.get("margin_eval_set", config.get("logit_margin", {}).get("eval_set", "safety")))
    rows = read_jsonl(Path(config["data_dir"]) / EVAL_TO_FILE[eval_set], limit=int(arm.get("margin_limit", 30)))
    features = []
    for row in rows:
        prompt_text = format_prompt_text(tokenizer, row["prompt"])
        features.append(tokenizer(prompt_text, truncation=True, max_length=int(config["max_length"]), padding=False))
    return features


def token_ids_for_strings(tokenizer, strings: list[str]) -> list[int]:
    ids: list[int] = []
    for text in strings:
        for variant in [text, " " + text]:
            encoded = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if encoded:
                ids.append(int(encoded[-1]))
    return sorted(set(ids))


def cyclic_batch(items: list[dict[str, Any]], batch_size: int, offset: int) -> list[dict[str, Any]]:
    if not items:
        return []
    return [items[(offset + index) % len(items)] for index in range(batch_size)]


class DeterministicAnchorCursor:
    def __init__(self, size: int, start: int = 0):
        if size <= 0:
            raise ValueError("anchor cursor size must be positive")
        self.size = int(size)
        self.position = int(start) % self.size
        self.draw_count = int(start)
        self.exposure_counts = [0] * self.size

    def draw(self, count: int) -> list[int]:
        indices = []
        for _ in range(int(count)):
            index = self.position
            indices.append(index)
            self.exposure_counts[index] += 1
            self.position = (self.position + 1) % self.size
            self.draw_count += 1
        return indices

    def state_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "position": self.position,
            "draw_count": self.draw_count,
            "exposure_counts": list(self.exposure_counts),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "DeterministicAnchorCursor":
        cursor = cls(int(state["size"]), int(state["position"]))
        cursor.draw_count = int(state["draw_count"])
        cursor.exposure_counts = [int(value) for value in state["exposure_counts"]]
        return cursor


def select_anchor_batch(
    anchor_features: list[dict[str, Any]],
    cursor: DeterministicAnchorCursor,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    indices = cursor.draw(batch_size)
    return [anchor_features[index] for index in indices], indices


def target_steps(config: dict[str, Any], train_rows: int) -> tuple[int, list[int]]:
    updates_per_epoch = math.ceil(
        train_rows
        / int(config["per_device_train_batch_size"])
        / int(config["gradient_accumulation_steps"])
    )
    total_steps = max(1, int(math.ceil(updates_per_epoch * float(config["num_train_epochs"]))))
    targets = {
        max(0, min(total_steps, int(round(total_steps * float(fraction)))))
        for fraction in config["checkpoint_fractions"]
    }
    targets.add(0)
    targets.add(total_steps)
    return total_steps, sorted(targets)


def lora_parameters(model) -> list[tuple[Any, Any, float]]:
    modules = []
    for module in model.modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if not lora_a or not lora_b:
            continue
        if "default" not in lora_a or "default" not in lora_b:
            continue
        scaling = getattr(module, "scaling", {}).get("default", 1.0)
        modules.append((lora_a["default"].weight, lora_b["default"].weight, float(scaling)))
    return modules


def stable_psd_spectral_norm(gram, eps: float = 1e-12):
    import torch

    eigvals = torch.linalg.eigvalsh(gram.float()).clamp_min(0.0)
    top = torch.max(eigvals)
    return torch.sqrt(top + eps) - math.sqrt(eps)


def lora_factor_and_delta_proxies(a, b, scaling: float) -> dict[str, Any]:
    import torch

    a_float = a.float()
    b_float = b.float()
    a_norm = stable_psd_spectral_norm(a_float @ a_float.T)
    b_norm = stable_psd_spectral_norm(b_float.T @ b_float)
    factor_unscaled = a_norm * b_norm
    factor_effective = factor_unscaled * scaling

    gram = (b_float.T @ b_float) @ (a_float @ a_float.T)
    eigvals = torch.linalg.eigvals(gram).real.clamp_min(0.0)
    delta_effective = torch.sqrt(torch.max(eigvals) + 1e-12) * scaling
    if not torch.isfinite(factor_unscaled + factor_effective + delta_effective):
        raise FloatingPointError("Non-finite SpectraSafe proxy encountered.")
    return {
        "factor_proxy_unscaled": factor_unscaled,
        "factor_proxy_effective": factor_effective,
        "delta_spectral_effective": delta_effective,
    }


def regularization_components(model, arm: dict[str, Any], global_step: int) -> dict[str, Any] | None:
    import torch

    reg_type = arm.get("regularization_type", "none")
    reg_lambda = float(arm.get("regularization_lambda", 0.0))
    lambda_mean = float(arm.get("lambda_mean", arm.get("spectrasafe_lambda_mean", 0.0)))
    lambda_max = float(arm.get("lambda_max", arm.get("spectrasafe_lambda_max", 0.0)))
    spectrasafe_types = {"spectrasafe_mean_max", "spectrasafe_factor_proxy"}
    if reg_type in spectrasafe_types and lambda_mean <= 0 and lambda_max <= 0:
        return None
    if reg_type not in spectrasafe_types and (reg_type == "none" or reg_lambda <= 0):
        return None
    start_step = int(arm.get("regularization_start_step", 0))
    end_step = arm.get("regularization_end_step")
    if global_step < start_step or (end_step is not None and global_step > int(end_step)):
        return None

    values = []
    factor_unscaled_values = []
    factor_effective_values = []
    delta_effective_values = []
    for a, b, scaling in lora_parameters(model):
        if reg_type == "lora_norm":
            values.append(torch.sum(a.float() ** 2) + torch.sum(b.float() ** 2))
        elif reg_type in {"spectral_mean", "spectral_top", "spectrasafe_mean_max", "spectrasafe_factor_proxy"}:
            proxies = lora_factor_and_delta_proxies(a, b, scaling)
            factor_unscaled_values.append(proxies["factor_proxy_unscaled"])
            factor_effective_values.append(proxies["factor_proxy_effective"])
            delta_effective_values.append(proxies["delta_spectral_effective"])
            if reg_type == "spectrasafe_factor_proxy":
                values.append(proxies["factor_proxy_unscaled"])
            else:
                values.append(proxies["delta_spectral_effective"])
        else:
            raise ValueError(f"Unknown regularization_type: {reg_type}")
    if not values:
        return None
    stacked = torch.stack(values)
    mean_penalty = torch.mean(stacked)
    max_penalty = torch.max(stacked)
    factor_unscaled = torch.stack(factor_unscaled_values) if factor_unscaled_values else None
    factor_effective = torch.stack(factor_effective_values) if factor_effective_values else None
    delta_effective = torch.stack(delta_effective_values) if delta_effective_values else None
    if reg_type in spectrasafe_types:
        total = lambda_mean * mean_penalty + lambda_max * max_penalty
        return {
            "total": total,
            "mean_spectral_penalty": mean_penalty,
            "max_spectral_penalty": max_penalty,
            "factor_mean_penalty_unscaled": torch.mean(factor_unscaled),
            "factor_max_penalty_unscaled": torch.max(factor_unscaled),
            "factor_mean_penalty_effective": torch.mean(factor_effective),
            "factor_max_penalty_effective": torch.max(factor_effective),
            "delta_mean_spectral_effective": torch.mean(delta_effective),
            "delta_max_spectral_effective": torch.max(delta_effective),
            "weighted_mean_penalty": lambda_mean * mean_penalty,
            "weighted_max_penalty": lambda_max * max_penalty,
            "lora_module_count": len(values),
            "lambda_mean": lambda_mean,
            "lambda_max": lambda_max,
        }
    if reg_type == "spectral_top":
        top_k = min(int(arm.get("spectral_top_k_modules", 20)), stacked.numel())
        stacked = torch.topk(stacked, top_k).values
        mean_penalty = torch.mean(stacked)
        max_penalty = torch.max(stacked)
    total = reg_lambda * mean_penalty
    return {
        "total": total,
        "mean_spectral_penalty": mean_penalty if reg_type in {"spectral_mean", "spectral_top"} else torch.zeros_like(total),
        "max_spectral_penalty": max_penalty if reg_type in {"spectral_mean", "spectral_top"} else torch.zeros_like(total),
        "factor_mean_penalty_unscaled": "",
        "factor_max_penalty_unscaled": "",
        "factor_mean_penalty_effective": "",
        "factor_max_penalty_effective": "",
        "delta_mean_spectral_effective": "",
        "delta_max_spectral_effective": "",
        "weighted_mean_penalty": total,
        "weighted_max_penalty": torch.zeros_like(total),
        "lora_module_count": len(values),
        "lambda_mean": reg_lambda if reg_type == "spectral_mean" else 0.0,
        "lambda_max": reg_lambda if reg_type == "spectral_top" else 0.0,
    }


def regularization_penalty(model, arm: dict[str, Any], global_step: int):
    components = regularization_components(model, arm, global_step)
    return None if components is None else components["total"]


def trace_value(components: dict[str, Any] | None, key: str) -> str | float:
    if components is None:
        return ""
    value = components.get(key, "")
    if value == "":
        return ""
    return float(value.detach().float().cpu())


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    arm_config = get_arm_config(config, args.run_label)
    data_path = Path(config["data_dir"]) / FAMILY_TO_FILE[args.family]
    rows = read_jsonl(data_path, limit=int(config["train_per_family"]))
    total_steps, targets = target_steps(config, len(rows))
    rid = run_id(config, args.family, args.seed, args.run_label)
    root = run_dir(config, args.family, args.seed, args.run_label)
    index_path = root / "checkpoint_index.csv"

    print("config:", args.config)
    print("run_id:", rid)
    print("family:", args.family)
    print("seed:", args.seed)
    print("run_label:", args.run_label or "")
    print("regularization:", arm_config)
    print("model:", config["model_name_or_path"])
    print("train_rows:", len(rows))
    print("total_steps:", total_steps)
    print("target_steps:", targets)
    print("run_dir:", root)
    if args.dry_run:
        return

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except Exception as exc:
        raise RuntimeError(
            "Missing training dependencies. Install torch, transformers, peft, datasets, accelerate and bitsandbytes."
        ) from exc

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if config.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.get("bf16") else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    if config.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized = Dataset.from_list(
        [make_tokenized_row(tokenizer, row, int(config["max_length"])) for row in rows]
    )
    anchor_rows = make_anchor_rows(config, arm_config)
    anchor_features = [make_tokenized_row(tokenizer, row, int(config["max_length"])) for row in anchor_rows]
    margin_features = make_margin_prompt_features(tokenizer, config, arm_config)
    margin_config = config.get("logit_margin", {})
    margin_token_sets = {
        "refusal": token_ids_for_strings(tokenizer, margin_config.get("refusal_strings", [])),
        "compliance": token_ids_for_strings(tokenizer, margin_config.get("compliance_strings", [])),
    }
    print("anchor_rows:", len(anchor_features))
    print("margin_prompts:", len(margin_features))

    checkpoint_rows: list[dict[str, Any]] = []
    saved_steps: set[int] = set()
    regularization_trace: list[dict[str, Any]] = []
    regularization_trace_path = root / "regularization_loss_trace.csv"
    anchor_cursor = DeterministicAnchorCursor(len(anchor_features)) if anchor_features else None
    anchor_draw_trace: list[dict[str, Any]] = []
    anchor_draw_trace_path = root / "anchor_draw_trace.csv"
    anchor_exposure_path = root / "anchor_exposure_manifest.csv"

    def save_adapter(step: int) -> None:
        tag = checkpoint_tag(step, total_steps)
        out_dir = checkpoint_dir(config, args.family, args.seed, tag, args.run_label)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        if step not in saved_steps:
            checkpoint_rows.append(
                {
                    "run_id": rid,
                    "family": args.family,
                    "seed": args.seed,
                    "checkpoint": tag,
                    "step": step,
                    "total_steps": total_steps,
                    "fraction": step / total_steps if total_steps else 0.0,
                    "adapter_dir": str(out_dir),
                }
            )
            saved_steps.add(step)
            write_csv(index_path, sorted(checkpoint_rows, key=lambda row: int(row["step"])))
        print(f"saved checkpoint {tag} to {out_dir}")

    class SaveTargetStepsCallback(TrainerCallback):
        def on_step_end(self, args_, state, control, **kwargs):
            step = int(state.global_step)
            if step in targets and step not in saved_steps:
                save_adapter(step)
            return control

        def on_train_end(self, args_, state, control, **kwargs):
            step = int(state.global_step)
            if total_steps not in saved_steps:
                save_adapter(total_steps)
            elif step not in saved_steps and step in targets:
                save_adapter(step)
            return control

    training_args = TrainingArguments(
        output_dir=str(root / "trainer_logs"),
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        save_strategy="no",
        bf16=config.get("bf16", False),
        fp16=config.get("fp16", False),
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    class RegularizedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            task_loss = outputs.loss
            loss = task_loss
            step = int(self.state.global_step)
            components = regularization_components(model, arm_config, step)
            penalty = None if components is None else components["total"]
            if penalty is not None:
                loss = loss + penalty.to(loss.device)
            loss_after_regularization = loss
            anchor_weight = scheduled_anchor_mix(arm_config, step)
            anchor_loss_value = None
            anchor_indices: list[int] = []
            if anchor_weight > 0 and anchor_features:
                if anchor_cursor is None:
                    raise RuntimeError("anchor cursor missing despite non-empty anchor features")
                selected_anchor_features, anchor_indices = select_anchor_batch(
                    anchor_features,
                    anchor_cursor,
                    len(inputs["input_ids"]),
                )
                anchor_batch = collate_batch(
                    tokenizer,
                    selected_anchor_features,
                )
                anchor_batch = {key: value.to(model.device) for key, value in anchor_batch.items()}
                anchor_loss = model(**anchor_batch).loss
                anchor_loss_value = anchor_loss
                loss = (1.0 - anchor_weight) * loss + anchor_weight * anchor_loss
                anchor_draw_trace.append(
                    {
                        "compute_loss_call": len(anchor_draw_trace),
                        "global_step": step,
                        "anchor_weight": anchor_weight,
                        "batch_size": len(anchor_indices),
                        "anchor_indices": ";".join(str(index) for index in anchor_indices),
                        "cursor_position_after": anchor_cursor.position,
                        "cursor_draw_count_after": anchor_cursor.draw_count,
                    }
                )
            margin_loss_value = None
            if in_intervention_window(arm_config, step):
                margin_weight = float(arm_config.get("margin_lambda", 0.0))
                if margin_weight > 0 and margin_features:
                    margin_batch_size = int(arm_config.get("margin_batch_size", 4))
                    prompt_batch = collate_prompt_batch(
                        tokenizer,
                        cyclic_batch(margin_features, margin_batch_size, step * margin_batch_size),
                    )
                    prompt_batch = {key: value.to(model.device) for key, value in prompt_batch.items()}
                    margin_outputs = model(**prompt_batch)
                    last_indices = prompt_batch["attention_mask"].sum(dim=1) - 1
                    batch_indices = torch.arange(prompt_batch["input_ids"].shape[0], device=model.device)
                    logits = margin_outputs.logits[batch_indices, last_indices, :]
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                    refusal_ids = margin_token_sets["refusal"]
                    compliance_ids = margin_token_sets["compliance"]
                    if refusal_ids and compliance_ids:
                        refusal_lp = torch.logsumexp(log_probs[:, refusal_ids], dim=-1)
                        compliance_lp = torch.logsumexp(log_probs[:, compliance_ids], dim=-1)
                        margin = refusal_lp - compliance_lp
                        target_margin = torch.tensor(float(arm_config.get("target_margin", 0.0)), device=model.device)
                        margin_loss = torch.relu(target_margin - margin).mean()
                        margin_loss_value = margin_loss
                        loss = loss + margin_weight * margin_loss.to(loss.device)
            regularization_trace.append(
                {
                    "global_step": step,
                    "regularization_type": arm_config.get("regularization_type", "none"),
                    "lambda_mean": "" if components is None else components.get("lambda_mean", ""),
                    "lambda_max": "" if components is None else components.get("lambda_max", ""),
                    "task_loss": float(task_loss.detach().float().cpu()),
                    "mean_spectral_penalty": ""
                    if components is None
                    else float(components["mean_spectral_penalty"].detach().float().cpu()),
                    "max_spectral_penalty": ""
                    if components is None
                    else float(components["max_spectral_penalty"].detach().float().cpu()),
                    "factor_mean_penalty_unscaled": trace_value(components, "factor_mean_penalty_unscaled"),
                    "factor_max_penalty_unscaled": trace_value(components, "factor_max_penalty_unscaled"),
                    "factor_mean_penalty_effective": trace_value(components, "factor_mean_penalty_effective"),
                    "factor_max_penalty_effective": trace_value(components, "factor_max_penalty_effective"),
                    "delta_mean_spectral_effective": trace_value(components, "delta_mean_spectral_effective"),
                    "delta_max_spectral_effective": trace_value(components, "delta_max_spectral_effective"),
                    "weighted_mean_penalty": trace_value(components, "weighted_mean_penalty"),
                    "weighted_max_penalty": trace_value(components, "weighted_max_penalty"),
                    "total_regularization_penalty": "" if penalty is None else float(penalty.detach().float().cpu()),
                    "loss_after_regularization": float(loss_after_regularization.detach().float().cpu()),
                    "lora_module_count": "" if components is None else components.get("lora_module_count", ""),
                    "anchor_weight": anchor_weight,
                    "anchor_loss": "" if anchor_loss_value is None else float(anchor_loss_value.detach().float().cpu()),
                    "margin_loss": "" if margin_loss_value is None else float(margin_loss_value.detach().float().cpu()),
                    "total_loss": float(loss.detach().float().cpu()),
                }
            )
            return (loss, outputs) if return_outputs else loss

    save_adapter(0)
    trainer = RegularizedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=lambda features: collate_batch(tokenizer, features),
        callbacks=[SaveTargetStepsCallback()],
    )
    trainer.train()
    write_csv(index_path, sorted(checkpoint_rows, key=lambda row: int(row["step"])))
    write_csv(regularization_trace_path, regularization_trace)
    if anchor_cursor is not None:
        exposure_rows = []
        for index, count in enumerate(anchor_cursor.exposure_counts):
            source = anchor_rows[index]
            exposure_rows.append(
                {
                    "run_id": rid,
                    "family": args.family,
                    "seed": args.seed,
                    "run_label": args.run_label or "",
                    "anchor_index": index,
                    "exposure_count": count,
                    "sample_id": source.get("sample_id", ""),
                    "prompt_hash": source.get("content_hash", ""),
                    "target_hash": source.get("target_hash", ""),
                    "prompt": source.get("prompt", ""),
                }
            )
        write_csv(anchor_draw_trace_path, anchor_draw_trace)
        write_csv(anchor_exposure_path, exposure_rows)
    print(f"wrote {index_path}")
    print(f"wrote {regularization_trace_path}")
    if anchor_cursor is not None:
        print(f"wrote {anchor_draw_trace_path}")
        print(f"wrote {anchor_exposure_path}")


if __name__ == "__main__":
    main()
