from __future__ import annotations

import importlib
import platform
import subprocess
import sys


MODULES = [
    "torch",
    "transformers",
    "peft",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "numpy",
    "pandas",
]


def main() -> None:
    print("python:", sys.executable)
    print("version:", sys.version)
    print("platform:", platform.platform())
    print("branch:", run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]))
    print("commit:", run_command(["git", "rev-parse", "HEAD"]))
    print("nvidia_smi:", run_command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]))

    for name in MODULES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "ok")
            print(f"{name}: {version}")
        except Exception as exc:
            print(f"{name}: MISSING ({type(exc).__name__}: {exc})")

    try:
        import torch

        print("cuda_available:", torch.cuda.is_available())
        print("cuda_device_count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                total_gb = props.total_memory / 1024**3
                print(f"gpu_{index}: {props.name}, {total_gb:.2f} GB")
    except Exception:
        pass


def run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = (completed.stdout or completed.stderr).strip()
        return output.replace("\n", " | ") if output else "unavailable"
    except Exception:
        return "unavailable"


if __name__ == "__main__":
    main()
