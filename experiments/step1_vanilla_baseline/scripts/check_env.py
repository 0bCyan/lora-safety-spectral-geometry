from __future__ import annotations

import importlib
import platform
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


if __name__ == "__main__":
    main()
