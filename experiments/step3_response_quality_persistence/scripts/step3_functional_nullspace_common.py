"""Utilities for data-derived functional-nullspace LoRA projection."""

from __future__ import annotations

import hashlib
from typing import Any


def functional_key(module_name: str) -> str:
    prefix, leaf = module_name.rsplit(".", 1)
    if leaf in {"q_proj", "k_proj", "v_proj"}:
        return f"{prefix}.qkv_input"
    if leaf == "o_proj":
        return f"{prefix}.o_input"
    if leaf in {"gate_proj", "up_proj"}:
        return f"{prefix}.gate_up_input"
    if leaf == "down_proj":
        return f"{prefix}.down_input"
    raise ValueError(f"unsupported LoRA target module: {module_name}")


def is_capture_representative(module_name: str) -> bool:
    return module_name.rsplit(".", 1)[-1] in {"q_proj", "o_proj", "gate_proj", "down_proj"}


def sampled_mask_positions(mask: Any, limit: int) -> Any:
    import torch

    if limit <= 0:
        raise ValueError("token sample limit must be positive")
    positions = torch.nonzero(mask, as_tuple=False).flatten()
    if not len(positions):
        raise ValueError("response mask has no active token")
    if len(positions) <= limit:
        return positions
    sample = torch.linspace(0, len(positions) - 1, steps=limit, device=positions.device)
    return positions[sample.round().long()]


def right_singular_basis(rows: Any, rank: int, epsilon: float = 1e-8) -> tuple[Any, dict[str, float]]:
    import torch

    if rows.ndim != 2 or not rows.shape[0] or not rows.shape[1]:
        raise ValueError("activation rows must be a non-empty matrix")
    if rank <= 0:
        raise ValueError("basis rank must be positive")
    normalized = torch.nn.functional.normalize(rows.float(), p=2, dim=1, eps=epsilon)
    gram = normalized @ normalized.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    keep = eigenvalues > epsilon
    if not bool(keep.any()):
        raise FloatingPointError("activation Gram matrix has zero numerical rank")
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    actual_rank = min(rank, int(eigenvalues.numel()))
    selected_values = eigenvalues[:actual_rank]
    selected_vectors = eigenvectors[:, :actual_rank]
    raw_basis = (selected_vectors.T @ normalized) / selected_values.sqrt().unsqueeze(1)
    q, _ = torch.linalg.qr(raw_basis.T, mode="reduced")
    basis = q.T.contiguous()
    metrics = {
        "requested_rank": float(rank),
        "actual_rank": float(actual_rank),
        "row_count": float(rows.shape[0]),
        "input_dim": float(rows.shape[1]),
        "top_rank_energy_fraction": float(selected_values.sum() / eigenvalues.sum()),
        "orthogonality_error": float(
            torch.linalg.vector_norm(basis @ basis.T - torch.eye(actual_rank, device=basis.device))
        ),
    }
    return basis, metrics


def protected_component_ratio(tensor: Any, basis: Any, epsilon: float = 1e-12) -> float:
    import torch

    value = tensor.detach().float()
    basis_value = basis.detach().to(device=value.device, dtype=value.dtype)
    protected = (value @ basis_value.T) @ basis_value
    denominator = torch.linalg.vector_norm(value).clamp_min(epsilon)
    return float(torch.linalg.vector_norm(protected) / denominator)


def project_rows_(tensor: Any, basis: Any) -> None:
    basis_value = basis.detach().to(device=tensor.device, dtype=tensor.dtype)
    tensor.sub_((tensor @ basis_value.T) @ basis_value)


def project_delta_(tensor: Any, baseline: Any | None, basis: Any) -> None:
    delta = tensor if baseline is None else tensor - baseline
    project_rows_(delta, basis)
    if baseline is not None:
        tensor.copy_(baseline + delta)


def resolve_protected_conflict_(
    risk_delta: Any, safety_delta: Any, basis: Any, epsilon: float = 1e-12
) -> dict[str, float | bool]:
    """Remove only risk components that oppose safety inside the protected span."""
    import torch

    if risk_delta.shape != safety_delta.shape:
        raise ValueError("risk and safety gradients must have identical shapes")
    basis_value = basis.detach().to(device=risk_delta.device, dtype=risk_delta.dtype)
    risk_protected = (risk_delta @ basis_value.T) @ basis_value
    safety_protected = (safety_delta @ basis_value.T) @ basis_value
    dot_before = torch.sum(risk_protected.float() * safety_protected.float())
    safety_norm_sq = torch.sum(safety_protected.float().square())
    conflict = bool(dot_before < 0 and safety_norm_sq > epsilon)
    correction_norm_sq = torch.zeros((), device=risk_delta.device, dtype=torch.float32)
    if conflict:
        coefficient = dot_before / safety_norm_sq
        correction = coefficient.to(risk_delta.dtype) * safety_protected
        risk_delta.sub_(correction)
        correction_norm_sq = torch.sum(correction.float().square())
    corrected_protected = (risk_delta @ basis_value.T) @ basis_value
    dot_after = torch.sum(corrected_protected.float() * safety_protected.float())
    return {
        "conflict": conflict,
        "dot_before": float(dot_before),
        "dot_after": float(dot_after),
        "safety_protected_norm_sq": float(safety_norm_sq),
        "risk_protected_norm_sq_before": float(torch.sum(risk_protected.float().square())),
        "risk_protected_norm_sq_after": float(torch.sum(corrected_protected.float().square())),
        "correction_norm_sq": float(correction_norm_sq),
    }


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()
