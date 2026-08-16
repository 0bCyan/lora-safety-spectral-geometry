"""Pure route-trie mathematics for TARC experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


ROUTE_MARKERS = (
    "i can't help",
    "i won't help",
    "i won't provide",
    "i won't assist",
    "i won't give",
    "i won't suggest",
    "i can't engage",
    "i won't discuss",
)


def refusal_route_marker(
    text: str, markers: tuple[str, ...] = ROUTE_MARKERS
) -> str:
    normalized = text.strip().lower()
    matches = [marker for marker in markers if normalized.startswith(marker)]
    if not matches:
        raise ValueError("response does not start with a registered refusal route")
    return min(matches, key=len)


def refusal_route_end(text: str, markers: tuple[str, ...] = ROUTE_MARKERS) -> int:
    return len(refusal_route_marker(text, markers))


def trie_nodes(paths: list[list[int]]) -> list[dict[str, Any]]:
    if len(paths) < 2 or any(not path for path in paths):
        raise ValueError("route trie requires at least two non-empty paths")
    children: dict[tuple[int, ...], set[int]] = defaultdict(set)
    occurrences: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for path_index, path in enumerate(paths):
        for depth, token_id in enumerate(path):
            history = tuple(path[:depth])
            children[history].add(int(token_id))
            occurrences[history].append(path_index)
    return [
        {
            "history": list(history),
            "depth": len(history),
            "allowed_token_ids": sorted(children[history]),
            "path_indices": sorted(set(occurrences[history])),
        }
        for history in sorted(children, key=lambda value: (len(value), value))
    ]


def conservative_allowed_margin(
    logits: Any, allowed_token_ids: list[int], temperature: float = 1.0
) -> Any:
    import torch

    if logits.ndim != 1 or logits.numel() < 2:
        raise ValueError("TARC margin requires one vocabulary-logit vector")
    allowed = sorted(set(int(value) for value in allowed_token_ids))
    if (
        not allowed
        or allowed[0] < 0
        or allowed[-1] >= logits.numel()
        or len(allowed) >= logits.numel()
        or temperature <= 0
    ):
        raise ValueError("TARC allowed-token set is invalid")
    indices = torch.tensor(allowed, dtype=torch.long, device=logits.device)
    allowed_logits = logits.float().index_select(0, indices)
    conservative_floor = -temperature * torch.logsumexp(
        -allowed_logits / temperature, dim=0
    )
    mask = torch.ones(logits.numel(), dtype=torch.bool, device=logits.device)
    mask[indices] = False
    strongest_disallowed = logits.float()[mask].max()
    return conservative_floor - strongest_disallowed


def certificate_barrier(margins: Any) -> Any:
    import torch.nn.functional as functional

    if margins.ndim != 1 or margins.numel() == 0:
        raise ValueError("TARC barrier requires a non-empty margin vector")
    return functional.softplus(-margins).mean()


def select_route(gates: dict[str, bool]) -> str:
    invalid = ("integrity", "gradient_valid", "budget_matched", "vram", "state_restored")
    if not all(gates[name] for name in invalid):
        return "STOP_TARC_P0_INVALID"
    if not gates["route_structure"]:
        return "STOP_TARC_P0_ROUTE_STRUCTURE_FAILED"
    if not gates["objective_distinct"]:
        return "STOP_TARC_P0_OBJECTIVE_NOT_DISTINCT"
    if not gates["cross_prompt_directional"]:
        return "STOP_TARC_P0_DIRECTION_NOT_REPLICATED"
    return "PASS_TARC_P0_AUTHORIZE_MATCHED_SMOKE_PROTOCOL"
