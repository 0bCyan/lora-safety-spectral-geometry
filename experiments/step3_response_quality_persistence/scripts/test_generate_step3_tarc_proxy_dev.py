from __future__ import annotations

import unittest

import generate_step3_tarc_proxy_dev as generation


class RegistryTests(unittest.TestCase):
    def test_route_path_map_is_exact(self) -> None:
        registry = {
            "paths": [
                {"token_ids": [1, 2, 3, 4], "marker": "i can't help"},
                {"token_ids": [1, 5, 3, 4], "marker": "i won't help"},
            ]
        }
        paths = generation.route_path_map(registry)
        self.assertEqual(paths[(1, 2, 3, 4)], "i can't help")
        self.assertNotIn((1, 2, 3), paths)

    def test_registry_hash_is_key_order_invariant(self) -> None:
        self.assertEqual(
            generation.canonical_hash({"b": 2, "a": 1}),
            generation.canonical_hash({"a": 1, "b": 2}),
        )


if __name__ == "__main__":
    unittest.main()
