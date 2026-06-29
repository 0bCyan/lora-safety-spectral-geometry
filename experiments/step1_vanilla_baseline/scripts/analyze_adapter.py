from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import adapter_dir, load_config, metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    adapter = adapter_dir(config, args.family, args.seed)
    output = metrics_path(config, args.family, args.seed)

    try:
        import torch
        from safetensors.torch import load_file
    except Exception as exc:
        raise RuntimeError(
            "Missing analysis dependencies. Install torch and safetensors."
        ) from exc

    safetensor_path = adapter / "adapter_model.safetensors"
    bin_path = adapter / "adapter_model.bin"
    if safetensor_path.exists():
        state = load_file(str(safetensor_path), device="cpu")
    elif bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(f"No adapter weights found in {adapter}")

    pairs = {}
    for name, tensor in state.items():
        if "lora_A" in name and name.endswith(".weight"):
            prefix = name.replace(".lora_A.default.weight", "")
            pairs.setdefault(prefix, {})["A"] = tensor.float()
        elif "lora_B" in name and name.endswith(".weight"):
            prefix = name.replace(".lora_B.default.weight", "")
            pairs.setdefault(prefix, {})["B"] = tensor.float()

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for layer_name, pair in sorted(pairs.items()):
            if "A" not in pair or "B" not in pair:
                continue
            a = pair["A"]
            b = pair["B"]
            delta = b @ a
            s_a = torch.linalg.svdvals(a)
            s_b = torch.linalg.svdvals(b)
            s_delta = torch.linalg.svdvals(delta)
            fro = torch.linalg.matrix_norm(delta, ord="fro")
            total_sv = torch.sum(s_delta).item()
            top_sv = s_delta[0].item() if len(s_delta) else 0.0
            effective_rank = 0.0
            if total_sv > 0:
                probs = s_delta / torch.sum(s_delta)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
                effective_rank = float(torch.exp(torch.tensor(entropy)).item())
            row = {
                "layer_name": layer_name,
                "module": layer_name.split(".")[-1],
                "family": args.family,
                "seed": args.seed,
                "spectral_proxy": s_a[0].item() * s_b[0].item(),
                "delta_spectral_norm": top_sv,
                "delta_fro_norm": fro.item(),
                "top_singular_ratio": top_sv / total_sv if total_sv > 0 else 0.0,
                "effective_rank": effective_rank,
                "rank": int(a.shape[0]),
                "a_shape": list(a.shape),
                "b_shape": list(b.shape),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {output} ({written} rows)")


if __name__ == "__main__":
    main()
