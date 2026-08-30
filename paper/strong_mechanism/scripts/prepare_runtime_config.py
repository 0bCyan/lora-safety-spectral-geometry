from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper" / "strong_mechanism"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def validate_paper_config(config: dict[str, Any]) -> None:
    required = ["model_name_or_path", "model_revision", "environment_lock", "machine", "output_dir"]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Paper config is missing required pin fields: {missing}")
    revision = str(config["model_revision"])
    if not COMMIT_RE.fullmatch(revision):
        raise ValueError(f"model_revision must be a full 40-character commit SHA, got {revision!r}")
    lock_path = REPO_ROOT / str(config["environment_lock"])
    if not lock_path.exists():
        raise FileNotFoundError(f"Missing environment lock: {lock_path}")
    lock = load_json(lock_path)
    if config["machine"] != lock["machine"]:
        raise ValueError(f"Config machine {config['machine']} does not match lock {lock['machine']}")
    if config["model_name_or_path"] != lock["model_repository"]:
        raise ValueError("Config model repository does not match environment lock")
    if config["model_revision"] != lock["model_revision"]:
        raise ValueError("Config model revision does not match environment lock")
    if config.get("require_pinned_runtime") is not True:
        raise ValueError("Formal Paper A config must set require_pinned_runtime=true")


def environment_snapshot(lock: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch must be importable for formal environment validation") from exc

    packages = {
        name: package_version(name)
        for name in [*lock["strict_packages"].keys(), *lock.get("record_only_packages", [])]
    }
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "gpu_names": gpu_names,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "git_status_short": git_value("status", "--short"),
    }


def verify_environment(lock: dict[str, Any], snapshot: dict[str, Any]) -> None:
    errors: list[str] = []
    if snapshot["python"] != lock["required_python"]:
        errors.append(f"python expected {lock['required_python']}, got {snapshot['python']}")
    if not snapshot["cuda_available"]:
        errors.append("CUDA is not available")
    if not any(lock["required_gpu_substring"] in name for name in snapshot["gpu_names"]):
        errors.append(
            f"required GPU containing {lock['required_gpu_substring']!r}, got {snapshot['gpu_names']}"
        )
    for package, expected in lock["strict_packages"].items():
        actual = snapshot["packages"].get(package)
        if actual != expected:
            errors.append(f"{package} expected {expected}, got {actual}")
    if errors:
        raise RuntimeError("Formal environment lock failed:\n- " + "\n- ".join(errors))


def resolve_snapshot(config: dict[str, Any], offline: bool) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface-hub is required to resolve the pinned model snapshot") from exc

    resolved = Path(
        snapshot_download(
            repo_id=str(config["model_name_or_path"]),
            revision=str(config["model_revision"]),
            local_files_only=offline,
        )
    ).resolve()
    if resolved.name != str(config["model_revision"]):
        raise RuntimeError(
            f"Resolved snapshot path does not end in the requested commit: {resolved}"
        )
    return resolved


def prepare_runtime_config(
    source_path: Path,
    output_path: Path,
    *,
    offline: bool,
    verify_machine: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = source_path.resolve()
    config = load_json(source_path)
    validate_paper_config(config)
    lock_path = (REPO_ROOT / str(config["environment_lock"])).resolve()
    lock = load_json(lock_path)
    snapshot = environment_snapshot(lock) if verify_machine else None
    if snapshot is not None:
        verify_environment(lock, snapshot)
    model_snapshot = resolve_snapshot(config, offline=offline)

    runtime = dict(config)
    runtime["model_repository"] = config["model_name_or_path"]
    runtime["model_name_or_path"] = str(model_snapshot)
    runtime["resolved_model_revision"] = config["model_revision"]
    runtime["paper_source_config"] = str(source_path.relative_to(REPO_ROOT))
    runtime["paper_source_config_sha256"] = sha256_file(source_path)
    runtime["paper_environment_lock_sha256"] = sha256_file(lock_path)

    serialized = json.dumps(runtime, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.read_bytes() != serialized:
        raise FileExistsError(f"Runtime config exists with different bytes: {output_path}")
    output_path.write_bytes(serialized)

    provenance = {
        "schema_version": "paper_a_runtime_config_provenance_v1",
        "source_config": str(source_path.relative_to(REPO_ROOT)),
        "source_config_sha256": sha256_file(source_path),
        "runtime_config": str(output_path),
        "runtime_config_sha256": sha256_bytes(serialized),
        "environment_lock": str(lock_path.relative_to(REPO_ROOT)),
        "environment_lock_sha256": sha256_file(lock_path),
        "model_repository": config["model_name_or_path"],
        "requested_model_revision": config["model_revision"],
        "resolved_snapshot_path": str(model_snapshot),
        "environment": snapshot,
    }
    provenance_path = output_path.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return runtime, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve a pinned Paper A model snapshot into a machine-local runtime config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="Require the exact snapshot to already exist in the local cache.")
    parser.add_argument("--verify-environment", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    source = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = load_json(source)
    validate_paper_config(config)
    if args.validate_only:
        print(json.dumps({
            "valid": True,
            "model_repository": config["model_name_or_path"],
            "model_revision": config["model_revision"],
            "machine": config["machine"],
        }, ensure_ascii=False))
        return 0

    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    _, provenance = prepare_runtime_config(
        source,
        output,
        offline=args.offline,
        verify_machine=args.verify_environment,
    )
    print(json.dumps(provenance, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
