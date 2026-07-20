from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_lora_with_checkpoints import lora_factor_and_delta_proxies, regularization_components


class LoraLike(torch.nn.Module):
    def __init__(self, a: torch.Tensor, b: torch.Tensor, scaling: float = 2.0):
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(a.shape[1], a.shape[0], bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(b.shape[1], b.shape[0], bias=False)})
        with torch.no_grad():
            self.lora_A["default"].weight.copy_(a)
            self.lora_B["default"].weight.copy_(b)
        self.scaling = {"default": scaling}


class TinyPeft(torch.nn.Module):
    def __init__(self, modules: list[LoraLike]):
        super().__init__()
        self.blocks = torch.nn.ModuleList(modules)


def assert_close(left: torch.Tensor, right: torch.Tensor, tolerance: float = 1e-5) -> None:
    if not torch.allclose(left, right, atol=tolerance, rtol=tolerance):
        raise AssertionError(f"{float(left)} != {float(right)}")


def direct_delta_norm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(b @ a, ord=2)


def direct_factor_proxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.matrix_norm(b, ord=2) * torch.linalg.matrix_norm(a, ord=2)


def check_numerical_identities() -> None:
    torch.manual_seed(7)
    shapes = [
        (3, 5, 4),
        (8, 16, 12),
        (8, 64, 32),
    ]
    for rank, in_dim, out_dim in shapes:
        a = torch.randn(rank, in_dim)
        b = torch.randn(out_dim, rank)
        scaling = 2.0
        proxies = lora_factor_and_delta_proxies(a, b, scaling)
        factor_unscaled = proxies["factor_proxy_unscaled"]
        factor_effective = proxies["factor_proxy_effective"]
        delta_effective = proxies["delta_spectral_effective"]
        assert_close(factor_unscaled, direct_factor_proxy(a, b))
        if factor_unscaled + 1e-5 < direct_delta_norm(a, b):
            raise AssertionError("factor proxy violated ||BA||_2 upper bound")
        if factor_effective + 1e-5 < delta_effective:
            raise AssertionError("effective factor proxy violated effective delta upper bound")
        if not torch.isfinite(factor_unscaled + factor_effective + delta_effective):
            raise AssertionError("non-finite proxy value")
        if factor_unscaled < 0 or factor_effective < 0 or delta_effective < 0:
            raise AssertionError("negative proxy value")


def check_backward_random() -> None:
    torch.manual_seed(11)
    a = torch.randn(8, 32, requires_grad=True)
    b = torch.randn(24, 8, requires_grad=True)
    proxies = lora_factor_and_delta_proxies(a, b, 2.0)
    loss = proxies["factor_proxy_unscaled"] + 0.1 * proxies["factor_proxy_effective"]
    loss.backward()
    if not torch.isfinite(a.grad).all() or not torch.isfinite(b.grad).all():
        raise AssertionError("non-finite random gradients")
    if a.grad.abs().sum() <= 0 or b.grad.abs().sum() <= 0:
        raise AssertionError("zero random gradients")


def check_backward_zero_b() -> None:
    torch.manual_seed(13)
    a = torch.randn(8, 32, requires_grad=True)
    b = torch.zeros(24, 8, requires_grad=True)
    proxies = lora_factor_and_delta_proxies(a, b, 2.0)
    if abs(float(proxies["factor_proxy_unscaled"])) > 1e-8:
        raise AssertionError("zero-B factor proxy should report zero")
    loss = proxies["factor_proxy_unscaled"]
    loss.backward()
    if not torch.isfinite(a.grad).all() or not torch.isfinite(b.grad).all():
        raise AssertionError("non-finite zero-B gradients")


def check_aggregation_gradients() -> None:
    torch.manual_seed(17)
    modules = []
    for _ in range(3):
        modules.append(LoraLike(torch.randn(8, 32), torch.randn(24, 8), scaling=2.0))
    model = TinyPeft(modules)
    for arm in [
        {"regularization_type": "spectrasafe_factor_proxy", "lambda_mean": 3.0, "lambda_max": 0.0},
        {"regularization_type": "spectrasafe_factor_proxy", "lambda_mean": 0.0, "lambda_max": 0.2},
        {"regularization_type": "spectrasafe_factor_proxy", "lambda_mean": 3.0, "lambda_max": 0.2},
    ]:
        model.zero_grad(set_to_none=True)
        components = regularization_components(model, arm, 0)
        if components is None:
            raise AssertionError("missing regularization components")
        total = components["total"]
        if not torch.isfinite(total):
            raise AssertionError("non-finite aggregate penalty")
        total.backward()
        grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not grads or not all(torch.isfinite(grad).all() for grad in grads):
            raise AssertionError("non-finite aggregate gradients")
        if sum(float(grad.abs().sum()) for grad in grads) <= 0:
            raise AssertionError("zero aggregate gradients")


def check_tied_max_gradients() -> None:
    a = torch.eye(4, 8)
    b = torch.eye(6, 4)
    model = TinyPeft([LoraLike(a.clone(), b.clone()), LoraLike(a.clone(), b.clone())])
    components = regularization_components(
        model,
        {"regularization_type": "spectrasafe_factor_proxy", "lambda_mean": 0.0, "lambda_max": 0.2},
        0,
    )
    if components is None:
        raise AssertionError("missing tied max components")
    components["total"].backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not grads or not all(torch.isfinite(grad).all() for grad in grads):
        raise AssertionError("non-finite tied max gradients")


def main() -> None:
    check_numerical_identities()
    check_backward_random()
    check_backward_zero_b()
    check_aggregation_gradients()
    check_tied_max_gradients()
    print("spectrasafe proxy tests passed")


if __name__ == "__main__":
    main()
