from __future__ import annotations

import random
import unittest
from itertools import combinations, permutations

from step3n_r1_matching import (
    InfeasibleAssignmentError,
    k_best_assignments,
    maximum_cardinality_cycle_cover,
    minimum_cost_assignment,
)


def _sum_costs(
    assignment: dict[int, int],
    edge_costs: dict[tuple[int, int], tuple[int, ...]],
) -> tuple[int, ...]:
    if not assignment:
        return ()
    dimension = len(next(iter(edge_costs.values())))
    total = [0] * dimension
    for edge in assignment.items():
        for coordinate, value in enumerate(edge_costs[edge]):
            total[coordinate] += value
    return tuple(total)


def _signature(nodes: list[int], assignment: dict[int, int]) -> tuple[int, ...]:
    index = {node: position for position, node in enumerate(nodes)}
    return tuple(index[assignment[node]] for node in nodes)


def _exhaustive_derangements(
    nodes: list[int],
    edge_costs: dict[tuple[int, int], tuple[int, ...]],
    *,
    required: set[tuple[int, int]] | None = None,
    forbidden: set[tuple[int, int]] | None = None,
) -> list[dict[int, int]]:
    required = set() if required is None else required
    forbidden = set() if forbidden is None else forbidden
    candidates: list[dict[int, int]] = []
    for destinations in permutations(nodes):
        assignment = dict(zip(nodes, destinations, strict=True))
        edges = set(assignment.items())
        if any(source == destination for source, destination in edges):
            continue
        if not required <= edges or forbidden & edges:
            continue
        if any(edge not in edge_costs for edge in edges):
            continue
        candidates.append(assignment)
    candidates.sort(key=lambda item: (_sum_costs(item, edge_costs), _signature(nodes, item)))
    return candidates


def _exhaustive_cycle_cover(
    nodes: list[int],
    edge_costs: dict[tuple[int, int], tuple[int, ...]],
) -> tuple[frozenset[int], dict[int, int]]:
    dimension = len(next(iter(edge_costs.values()))) if edge_costs else 0
    best_key: tuple[int, tuple[int, ...], tuple[int, ...]] | None = None
    best_result: tuple[frozenset[int], dict[int, int]] | None = None
    index = {node: position for position, node in enumerate(nodes)}

    for retained_count in range(len(nodes) + 1):
        for retained_tuple in combinations(nodes, retained_count):
            retained = frozenset(retained_tuple)
            for destinations in permutations(retained_tuple):
                assignment = dict(zip(retained_tuple, destinations, strict=True))
                if any(source == destination for source, destination in assignment.items()):
                    continue
                if any(edge not in edge_costs for edge in assignment.items()):
                    continue
                total = [0] * dimension
                for edge in assignment.items():
                    for coordinate, value in enumerate(edge_costs[edge]):
                        total[coordinate] += value
                full_signature = tuple(
                    index[assignment[node]] if node in retained else index[node] for node in nodes
                )
                key = (-retained_count, tuple(total), full_signature)
                if best_key is None or key < best_key:
                    best_key = key
                    best_result = retained, assignment

    assert best_result is not None
    return best_result


class Step3NR1MatchingTests(unittest.TestCase):
    def test_minimum_assignment_matches_exhaustive_with_constraints(self) -> None:
        nodes = [0, 1, 2, 3]
        costs = {
            (source, destination): (
                (source + 2 * destination) % 3,
                abs(source - destination),
                (11 * source + 7 * destination) % 13,
            )
            for source in nodes
            for destination in nodes
            if source != destination
        }
        required = {(0, 2)}
        forbidden = {(1, 0)}
        expected = _exhaustive_derangements(
            nodes, costs, required=required, forbidden=forbidden
        )[0]
        actual = minimum_cost_assignment(
            nodes,
            costs,
            required_edges=required,
            forbidden_edges=forbidden,
            allow_self=False,
        )
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0], 2)
        self.assertNotEqual(actual[1], 0)

    def test_infeasible_and_required_conflicts_are_explicit(self) -> None:
        nodes = [0, 1, 2]
        dense = {
            (source, destination): (0,)
            for source in nodes
            for destination in nodes
            if source != destination
        }
        conflict_cases = [
            ({(0, 1), (0, 2)}, set()),
            ({(0, 2), (1, 2)}, set()),
            ({(0, 1)}, {(0, 1)}),
            ({(0, 0)}, set()),
        ]
        for required, forbidden in conflict_cases:
            with self.subTest(required=required, forbidden=forbidden):
                with self.assertRaises(InfeasibleAssignmentError):
                    k_best_assignments(
                        nodes,
                        dense,
                        required_edges=required,
                        forbidden_edges=forbidden,
                    )

        sparse = {(0, 1): (0,), (1, 0): (0,)}
        with self.assertRaises(InfeasibleAssignmentError):
            k_best_assignments(nodes, sparse)
        with self.assertRaises(InfeasibleAssignmentError):
            minimum_cost_assignment(
                nodes,
                dense,
                forbidden_edges={(0, 1), (0, 2)},
                allow_self=False,
            )

    def test_k_best_matches_small_n_exhaustive_oracle(self) -> None:
        for size in range(2, 7):
            random_source = random.Random(8100 + size)
            nodes = list(range(size))
            costs = {
                (source, destination): (
                    random_source.randint(-2, 2),
                    random_source.randint(0, 8),
                    random_source.randint(0, 4),
                )
                for source in nodes
                for destination in nodes
                if source != destination
            }
            expected = _exhaustive_derangements(nodes, costs)
            count = min(12, len(expected))
            actual = k_best_assignments(nodes, costs, k=count)
            with self.subTest(size=size):
                self.assertEqual(actual, expected[:count])
                self.assertEqual(len({_signature(nodes, item) for item in actual}), count)
                self.assertTrue(
                    all(source != destination for item in actual for source, destination in item.items())
                )

    def test_k_best_is_explicitly_infeasible_when_fewer_than_k_exist(self) -> None:
        nodes = [0, 1]
        costs = {(0, 1): (4,), (1, 0): (7,)}
        with self.assertRaisesRegex(InfeasibleAssignmentError, "only 1 exist"):
            k_best_assignments(nodes, costs, k=4)

    def test_k_best_valid_constraints_match_exhaustive_oracle(self) -> None:
        nodes = [0, 1, 2, 3, 4]
        costs = {
            (source, destination): (
                (source + destination) % 2,
                (source * 13 + destination * 17) % 19,
            )
            for source in nodes
            for destination in nodes
            if source != destination
        }
        required = {(0, 3)}
        forbidden = {(1, 4), (2, 0)}
        expected = _exhaustive_derangements(
            nodes, costs, required=required, forbidden=forbidden
        )
        count = min(20, len(expected))
        actual = k_best_assignments(
            nodes,
            costs,
            k=count,
            required_edges=required,
            forbidden_edges=forbidden,
        )
        self.assertEqual(actual, expected[:count])

    def test_sparse_random_assignments_match_exhaustive_oracle(self) -> None:
        for size in range(3, 7):
            nodes = list(range(size))
            for trial in range(10):
                random_source = random.Random(9200 + 100 * size + trial)
                costs = {
                    (source, destination): (
                        random_source.randint(-3, 3),
                        random_source.randint(0, 20),
                    )
                    for source in nodes
                    for destination in nodes
                    if source != destination and random_source.random() < 0.68
                }
                expected = _exhaustive_derangements(nodes, costs)
                with self.subTest(size=size, trial=trial):
                    if not expected:
                        with self.assertRaises(InfeasibleAssignmentError):
                            k_best_assignments(nodes, costs, k=5)
                    else:
                        count = min(5, len(expected))
                        self.assertEqual(
                            k_best_assignments(nodes, costs, k=count), expected[:count]
                        )

    def test_equal_cost_assignments_use_stable_signature_order(self) -> None:
        nodes = [0, 1, 2, 3]
        costs = {
            (source, destination): (0, 0)
            for source in nodes
            for destination in nodes
            if source != destination
        }
        expected = _exhaustive_derangements(nodes, costs)
        self.assertEqual(k_best_assignments(nodes, costs, k=9), expected)

    def test_integer_lexicographic_order_is_not_float_or_magnitude_weighted(self) -> None:
        huge = 10**80 + 123
        nodes = [0, 1, 2]
        forward = {(0, 1), (1, 2), (2, 0)}
        costs: dict[tuple[int, int], tuple[int, int]] = {}
        for edge in forward:
            costs[edge] = (0, huge)
        for edge in {(0, 2), (2, 1), (1, 0)}:
            costs[edge] = ((1 if edge == (0, 2) else 0), 0)

        ranked = k_best_assignments(nodes, costs, k=2)
        self.assertEqual(set(ranked[0].items()), forward)
        self.assertEqual(_sum_costs(ranked[0], costs), (0, 3 * huge))
        self.assertEqual(_sum_costs(ranked[1], costs), (1, 0))

    def test_cycle_cover_matches_exhaustive_and_is_closed(self) -> None:
        nodes = [0, 1, 2, 3, 4]
        costs = {
            (0, 1): (0, 4),
            (1, 0): (0, 3),
            (2, 3): (0, 2),
            (3, 2): (0, 1),
            (0, 2): (1, 0),
            (2, 0): (1, 0),
            (1, 3): (1, 0),
            (3, 1): (1, 0),
            (4, 0): (-1000, -1000),
        }
        expected_retained, expected_assignment = _exhaustive_cycle_cover(nodes, costs)
        retained, assignment = maximum_cardinality_cycle_cover(nodes, costs)
        self.assertEqual(retained, expected_retained)
        self.assertEqual(assignment, expected_assignment)
        self.assertEqual(retained, frozenset(assignment))
        self.assertEqual(retained, frozenset(assignment.values()))
        self.assertNotIn(4, retained)
        self.assertTrue(all(source != destination for source, destination in assignment.items()))

    def test_cycle_cover_cardinality_precedes_arbitrarily_large_cost(self) -> None:
        nodes = [0, 1, 2, 3]
        huge = 10**70
        costs = {
            (0, 1): (huge,),
            (1, 2): (huge,),
            (2, 3): (huge,),
            (3, 0): (huge,),
            (0, 2): (-huge,),
            (2, 0): (-huge,),
        }
        retained, assignment = maximum_cardinality_cycle_cover(nodes, costs)
        self.assertEqual(retained, frozenset(nodes))
        self.assertEqual(len(assignment), 4)

    def test_explicit_self_drop_cost_breaks_equal_cardinality_tie(self) -> None:
        nodes = [0, 1, 2]
        costs = {
            (0, 1): (0,),
            (1, 0): (0,),
            (0, 2): (0,),
            (2, 0): (0,),
            (0, 0): (0,),
            (1, 1): (0,),
            (2, 2): (9,),
        }
        retained, assignment = maximum_cardinality_cycle_cover(nodes, costs)
        self.assertEqual(retained, frozenset({0, 2}))
        self.assertEqual(assignment, {0: 2, 2: 0})

    def test_sparse_random_cycle_covers_match_exhaustive_oracle(self) -> None:
        for size in range(2, 7):
            nodes = list(range(size))
            for trial in range(8):
                random_source = random.Random(10300 + 100 * size + trial)
                costs = {
                    (source, destination): (
                        random_source.randint(-5, 5),
                        random_source.randint(0, 15),
                    )
                    for source in nodes
                    for destination in nodes
                    if source != destination and random_source.random() < 0.5
                }
                expected = _exhaustive_cycle_cover(nodes, costs)
                actual = maximum_cardinality_cycle_cover(nodes, costs)
                with self.subTest(size=size, trial=trial):
                    self.assertEqual(actual, expected)

    def test_empty_cycle_cover_is_feasible(self) -> None:
        retained, assignment = maximum_cardinality_cycle_cover([0, 1, 2], {})
        self.assertEqual(retained, frozenset())
        self.assertEqual(assignment, {})

    def test_determinism_is_independent_of_edge_mapping_order(self) -> None:
        nodes = list(range(7))
        costs = {
            (source, destination): (
                (source * 19 + destination * 23) % 5,
                (source * 7 + destination * 11) % 17,
            )
            for source in nodes
            for destination in nodes
            if source != destination
        }
        reversed_costs = dict(reversed(list(costs.items())))
        first = k_best_assignments(nodes, costs, k=8)
        second = k_best_assignments(nodes, reversed_costs, k=8)
        third = k_best_assignments(nodes, costs, k=8)
        self.assertEqual(first, second)
        self.assertEqual(first, third)
        self.assertEqual(
            maximum_cardinality_cycle_cover(nodes, costs),
            maximum_cardinality_cycle_cover(nodes, reversed_costs),
        )

    def test_dense_n40_smoke(self) -> None:
        nodes = list(range(40))
        costs = {
            (source, destination): (
                int(source // 5 != destination // 5),
                abs(source - destination),
                (source * 101 + destination * 37) % 997,
            )
            for source in nodes
            for destination in nodes
            if source != destination
        }
        ranked = k_best_assignments(nodes, costs, k=4)
        self.assertEqual(len(ranked), 4)
        self.assertEqual(len({_signature(nodes, item) for item in ranked}), 4)
        retained, assignment = maximum_cardinality_cycle_cover(nodes, costs)
        self.assertEqual(retained, frozenset(nodes))
        self.assertEqual(frozenset(assignment.values()), retained)


if __name__ == "__main__":
    unittest.main()
