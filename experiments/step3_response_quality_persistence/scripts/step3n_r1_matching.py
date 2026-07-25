"""Exact deterministic assignment utilities for the Step3N-R1 data gate.

The public solvers accept integer tuple edge costs.  Objectives compare the
coordinate-wise sum of those tuples lexicographically; no floating-point or
heuristic weighting is used.  The order of ``nodes`` defines the stable final
tie-break signature (the destination index for each source, in source order).
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from typing import Generic, Hashable, Iterable, Mapping, NamedTuple, Sequence, TypeVar


NodeT = TypeVar("NodeT", bound=Hashable)
Cost = tuple[int, ...]
Edge = tuple[NodeT, NodeT]
EdgeCollection = Iterable[Edge[NodeT]] | Mapping[NodeT, NodeT]


class InfeasibleAssignmentError(ValueError):
    """Raised when no complete assignment satisfies the supplied constraints."""


# Keep a descriptive alternative available to callers without two exception types.
AssignmentInfeasibleError = InfeasibleAssignmentError


class CycleCoverResult(NamedTuple, Generic[NodeT]):
    """A retained node set and its closed, non-self permutation."""

    retained: frozenset[NodeT]
    assignment: dict[NodeT, NodeT]


@dataclass(frozen=True)
class _Solution:
    destinations: tuple[int, ...]
    cost: Cost

    @property
    def signature(self) -> tuple[int, ...]:
        return self.destinations


def _prepare_nodes(nodes: Sequence[NodeT]) -> tuple[tuple[NodeT, ...], dict[NodeT, int]]:
    ordered = tuple(nodes)
    try:
        index = {node: position for position, node in enumerate(ordered)}
    except TypeError as exc:
        raise TypeError("all nodes must be hashable") from exc
    if len(index) != len(ordered):
        raise ValueError("nodes must be unique")
    return ordered, index


def _validate_edge_costs(
    node_index: Mapping[NodeT, int],
    edge_costs: Mapping[Edge[NodeT], Cost],
) -> tuple[dict[tuple[int, int], Cost], int]:
    validated: dict[tuple[int, int], Cost] = {}
    dimension: int | None = None

    for edge, cost in edge_costs.items():
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise TypeError(f"edge keys must be (source, destination) tuples: {edge!r}")
        source, destination = edge
        try:
            source_known = source in node_index
            destination_known = destination in node_index
        except TypeError as exc:
            raise TypeError("edge endpoints must be hashable") from exc
        if not source_known or not destination_known:
            raise ValueError(f"edge references a node outside nodes: {edge!r}")
        if not isinstance(cost, tuple) or any(type(value) is not int for value in cost):
            raise TypeError(f"edge cost must be a tuple containing only integers: {edge!r}")
        if dimension is None:
            dimension = len(cost)
        elif len(cost) != dimension:
            raise ValueError("all edge-cost tuples must have the same length")
        validated[(node_index[source], node_index[destination])] = cost

    return validated, 0 if dimension is None else dimension


def _normalise_constraint_edges(
    edges: EdgeCollection[NodeT],
    node_index: Mapping[NodeT, int],
    label: str,
) -> frozenset[tuple[int, int]]:
    raw_edges = edges.items() if isinstance(edges, Mapping) else edges
    normalised: set[tuple[int, int]] = set()
    for edge in raw_edges:
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise TypeError(f"{label} entries must be (source, destination) tuples")
        source, destination = edge
        try:
            source_known = source in node_index
            destination_known = destination in node_index
        except TypeError as exc:
            raise TypeError(f"{label} endpoints must be hashable") from exc
        if not source_known or not destination_known:
            raise ValueError(f"{label} edge references a node outside nodes: {edge!r}")
        normalised.add((node_index[source], node_index[destination]))
    return frozenset(normalised)


def _mixed_radix_weights(ranges: Sequence[int]) -> tuple[int, ...]:
    """Return exact integer weights that preserve lexicographic order.

    Each coordinate is known to vary by at most ``ranges[i]``.  Consequently,
    one unit in an earlier coordinate outweighs the entire possible suffix.
    This is an exact order embedding, not a numerical approximation.
    """

    if not ranges:
        return ()
    weights = [1] * len(ranges)
    for position in range(len(ranges) - 2, -1, -1):
        weights[position] = weights[position + 1] * (ranges[position + 1] + 1)
    return tuple(weights)


def _hungarian(cost_matrix: Sequence[Sequence[int | None]]) -> list[int]:
    """Solve a square sparse assignment matrix using Hungarian augmentation."""

    size = len(cost_matrix)
    if any(len(row) != size for row in cost_matrix):
        raise ValueError("Hungarian input must be square")
    if size == 0:
        return []

    # One-based arrays follow the shortest-augmenting-path Hungarian notation.
    row_potential = [0] * (size + 1)
    column_potential = [0] * (size + 1)
    matched_row = [0] * (size + 1)
    predecessor = [0] * (size + 1)

    for row in range(1, size + 1):
        matched_row[0] = row
        minimum_slack: list[int | None] = [None] * (size + 1)
        used_column = [False] * (size + 1)
        column = 0

        while True:
            used_column[column] = True
            active_row = matched_row[column]
            next_column = 0
            delta: int | None = None

            for candidate in range(1, size + 1):
                if used_column[candidate]:
                    continue
                edge_cost = cost_matrix[active_row - 1][candidate - 1]
                if edge_cost is not None:
                    slack = edge_cost - row_potential[active_row] - column_potential[candidate]
                    if minimum_slack[candidate] is None or slack < minimum_slack[candidate]:
                        minimum_slack[candidate] = slack
                        predecessor[candidate] = column
                candidate_slack = minimum_slack[candidate]
                if candidate_slack is not None and (
                    delta is None
                    or candidate_slack < delta
                    or (candidate_slack == delta and candidate < next_column)
                ):
                    delta = candidate_slack
                    next_column = candidate

            if delta is None:
                raise InfeasibleAssignmentError("no complete assignment exists")

            for candidate in range(size + 1):
                if used_column[candidate]:
                    row_potential[matched_row[candidate]] += delta
                    column_potential[candidate] -= delta
                elif minimum_slack[candidate] is not None:
                    minimum_slack[candidate] -= delta

            column = next_column
            if matched_row[column] == 0:
                break

        while True:
            previous = predecessor[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break

    row_to_column = [-1] * size
    for column in range(1, size + 1):
        row = matched_row[column]
        if row == 0:
            raise InfeasibleAssignmentError("no complete assignment exists")
        row_to_column[row - 1] = column - 1
    if any(column < 0 for column in row_to_column):
        raise InfeasibleAssignmentError("no complete assignment exists")
    return row_to_column


class _AssignmentProblem(Generic[NodeT]):
    def __init__(
        self,
        nodes: Sequence[NodeT],
        edge_costs: Mapping[Edge[NodeT], Cost],
        *,
        allow_self: bool,
    ) -> None:
        self.nodes, self.node_index = _prepare_nodes(nodes)
        all_costs, self.cost_dimension = _validate_edge_costs(self.node_index, edge_costs)
        self.costs = {
            edge: cost for edge, cost in all_costs.items() if allow_self or edge[0] != edge[1]
        }
        self.scalar_costs = self._encode_costs_exactly()

    def _encode_costs_exactly(self) -> dict[tuple[int, int], int]:
        size = len(self.nodes)
        objective_ranges: list[int] = []
        for coordinate in range(self.cost_dimension):
            values = [cost[coordinate] for cost in self.costs.values()]
            span = 0 if not values else size * (max(values) - min(values))
            objective_ranges.append(span)

        # Exactly one destination rank contributes to each source coordinate.
        objective_ranges.extend([max(0, size - 1)] * size)
        weights = _mixed_radix_weights(objective_ranges)
        encoded: dict[tuple[int, int], int] = {}
        for (source, destination), cost in self.costs.items():
            scalar = sum(
                value * weights[coordinate] for coordinate, value in enumerate(cost)
            )
            if size:
                scalar += destination * weights[self.cost_dimension + source]
            encoded[(source, destination)] = scalar
        return encoded

    def constraints(
        self,
        required_edges: EdgeCollection[NodeT],
        forbidden_edges: EdgeCollection[NodeT],
    ) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
        return (
            _normalise_constraint_edges(required_edges, self.node_index, "required_edges"),
            _normalise_constraint_edges(forbidden_edges, self.node_index, "forbidden_edges"),
        )

    def solve(
        self,
        required: frozenset[tuple[int, int]],
        forbidden: frozenset[tuple[int, int]],
    ) -> _Solution:
        required_by_source: dict[int, int] = {}
        required_by_destination: dict[int, int] = {}
        for source, destination in sorted(required):
            if source in required_by_source and required_by_source[source] != destination:
                raise InfeasibleAssignmentError("required edges assign one source more than once")
            if (
                destination in required_by_destination
                and required_by_destination[destination] != source
            ):
                raise InfeasibleAssignmentError("required edges use one destination more than once")
            if (source, destination) in forbidden:
                raise InfeasibleAssignmentError("an edge is both required and forbidden")
            if (source, destination) not in self.costs:
                raise InfeasibleAssignmentError("a required edge is unavailable")
            required_by_source[source] = destination
            required_by_destination[destination] = source

        remaining_sources = [
            source for source in range(len(self.nodes)) if source not in required_by_source
        ]
        remaining_destinations = [
            destination
            for destination in range(len(self.nodes))
            if destination not in required_by_destination
        ]
        if len(remaining_sources) != len(remaining_destinations):
            raise InfeasibleAssignmentError("required edges do not define a partial bijection")

        matrix: list[list[int | None]] = []
        for source in remaining_sources:
            row = [
                None
                if (source, destination) in forbidden
                else self.scalar_costs.get((source, destination))
                for destination in remaining_destinations
            ]
            if row and all(value is None for value in row):
                raise InfeasibleAssignmentError("a source has no available destination")
            matrix.append(row)

        residual_assignment = _hungarian(matrix)
        destinations = [-1] * len(self.nodes)
        for source, destination in required_by_source.items():
            destinations[source] = destination
        for row, column in enumerate(residual_assignment):
            destinations[remaining_sources[row]] = remaining_destinations[column]

        if any(destination < 0 for destination in destinations):
            raise InfeasibleAssignmentError("no complete assignment exists")
        if len(set(destinations)) != len(destinations):
            raise AssertionError("internal assignment is not one-to-one")

        total = [0] * self.cost_dimension
        for source, destination in enumerate(destinations):
            cost = self.costs[(source, destination)]
            for coordinate, value in enumerate(cost):
                total[coordinate] += value
        return _Solution(tuple(destinations), tuple(total))

    def public_assignment(self, solution: _Solution) -> dict[NodeT, NodeT]:
        return {
            source: self.nodes[solution.destinations[position]]
            for position, source in enumerate(self.nodes)
        }


def minimum_cost_assignment(
    nodes: Sequence[NodeT],
    edge_costs: Mapping[Edge[NodeT], Cost],
    required_edges: EdgeCollection[NodeT] = (),
    forbidden_edges: EdgeCollection[NodeT] = (),
    *,
    allow_self: bool = True,
) -> dict[NodeT, NodeT]:
    """Return the exact lexicographically minimum complete assignment.

    ``required_edges`` must form a partial bijection and ``forbidden_edges``
    remove otherwise available edges.  Constraint conflicts and a missing full
    matching raise :class:`InfeasibleAssignmentError`.
    """

    problem = _AssignmentProblem(nodes, edge_costs, allow_self=allow_self)
    required, forbidden = problem.constraints(required_edges, forbidden_edges)
    return problem.public_assignment(problem.solve(required, forbidden))


def k_best_assignments(
    nodes: Sequence[NodeT],
    edge_costs: Mapping[Edge[NodeT], Cost],
    k: int = 4,
    *,
    required_edges: EdgeCollection[NodeT] = (),
    forbidden_edges: EdgeCollection[NodeT] = (),
) -> list[dict[NodeT, NodeT]]:
    """Return exactly ``k`` distinct complete derangements in exact rank order.

    Results are ordered first by the lexicographic sum of edge-cost tuples and
    then by the destination-index signature induced by ``nodes``.  A Lawler
    partition enumerates alternatives without approximation.  If fewer than
    ``k`` solutions exist, :class:`InfeasibleAssignmentError` makes the
    Step3N-R1 hard gate explicit rather than silently reducing ``k``.
    """

    if type(k) is not int or k < 0:
        raise ValueError("k must be a non-negative integer")
    problem = _AssignmentProblem(nodes, edge_costs, allow_self=False)
    base_required, base_forbidden = problem.constraints(required_edges, forbidden_edges)
    if k == 0:
        return []

    root_solution = problem.solve(base_required, base_forbidden)
    required_sources = {source for source, _ in base_required}
    free_sources = [
        source for source in range(len(problem.nodes)) if source not in required_sources
    ]

    serials = count()
    heap: list[tuple[
        Cost,
        tuple[int, ...],
        int,
        int,
        frozenset[tuple[int, int]],
        frozenset[tuple[int, int]],
        _Solution,
    ]] = []
    seen_subproblems: set[
        tuple[int, frozenset[tuple[int, int]], frozenset[tuple[int, int]]]
    ] = set()

    def enqueue(
        start: int,
        required: frozenset[tuple[int, int]],
        forbidden: frozenset[tuple[int, int]],
        solution: _Solution,
    ) -> None:
        state = (start, required, forbidden)
        if state in seen_subproblems:
            return
        seen_subproblems.add(state)
        heappush(
            heap,
            (
                solution.cost,
                solution.signature,
                next(serials),
                start,
                required,
                forbidden,
                solution,
            ),
        )

    enqueue(0, base_required, base_forbidden, root_solution)
    results: list[dict[NodeT, NodeT]] = []
    seen_signatures: set[tuple[int, ...]] = set()

    while heap and len(results) < k:
        _, _, _, start, parent_required, parent_forbidden, solution = heappop(heap)
        if solution.signature not in seen_signatures:
            seen_signatures.add(solution.signature)
            results.append(problem.public_assignment(solution))
            if len(results) == k:
                break

        prefix_required = set(parent_required)
        for position in range(start, len(free_sources)):
            source = free_sources[position]
            selected_edge = (source, solution.destinations[source])
            child_required = frozenset(prefix_required)
            child_forbidden = frozenset(set(parent_forbidden) | {selected_edge})
            try:
                child_solution = problem.solve(child_required, child_forbidden)
            except InfeasibleAssignmentError:
                pass
            else:
                enqueue(position, child_required, child_forbidden, child_solution)
            prefix_required.add(selected_edge)

    if len(results) != k:
        raise InfeasibleAssignmentError(
            f"requested {k} distinct complete derangements, but only "
            f"{len(results)} exist"
        )
    return results


def maximum_cardinality_cycle_cover(
    nodes: Sequence[NodeT],
    eligible_edge_costs: Mapping[Edge[NodeT], Cost],
) -> CycleCoverResult[NodeT]:
    """Return the largest retained-node directed cycle cover.

    A self edge means that its node is dropped.  Self-drop edges are inserted
    implicitly with a zero secondary cost, or use their supplied cost when an
    explicit ``(node, node)`` entry exists.  The exact objective minimizes
    ``(drop_count, *sum_edge_cost)`` and then the stable assignment signature.
    The returned mapping strips all self edges, so its keys and values are both
    exactly ``retained`` and every remaining cycle has length at least two.
    """

    ordered_nodes, node_index = _prepare_nodes(nodes)
    validated_costs, dimension = _validate_edge_costs(node_index, eligible_edge_costs)
    zero_cost = (0,) * dimension
    augmented_costs: dict[Edge[NodeT], Cost] = {}

    for (source, destination), cost in validated_costs.items():
        if source != destination:
            augmented_costs[(ordered_nodes[source], ordered_nodes[destination])] = (0,) + cost
    for position, node in enumerate(ordered_nodes):
        drop_cost = validated_costs.get((position, position), zero_cost)
        augmented_costs[(node, node)] = (1,) + drop_cost

    problem = _AssignmentProblem(ordered_nodes, augmented_costs, allow_self=True)
    solution = problem.solve(frozenset(), frozenset())
    full_assignment = problem.public_assignment(solution)
    retained = frozenset(
        source for source, destination in full_assignment.items() if source != destination
    )
    closed_assignment = {
        source: destination
        for source, destination in full_assignment.items()
        if source in retained
    }

    if frozenset(closed_assignment) != retained:
        raise AssertionError("cycle-cover sources do not match retained nodes")
    if frozenset(closed_assignment.values()) != retained:
        raise AssertionError("cycle-cover destinations do not match retained nodes")
    if any(source == destination for source, destination in closed_assignment.items()):
        raise AssertionError("retained cycle cover contains a self edge")
    return CycleCoverResult(retained, closed_assignment)


__all__ = [
    "AssignmentInfeasibleError",
    "CycleCoverResult",
    "InfeasibleAssignmentError",
    "k_best_assignments",
    "maximum_cardinality_cycle_cover",
    "minimum_cost_assignment",
]
