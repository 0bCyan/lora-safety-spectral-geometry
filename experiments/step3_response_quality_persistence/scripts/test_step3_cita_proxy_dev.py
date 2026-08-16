from __future__ import annotations

import unittest

import step3_cita_proxy_dev as proxy


class ProxyPackageTests(unittest.TestCase):
    def test_default_protocol_is_stage_specific(self) -> None:
        self.assertTrue(proxy.DEFAULT_PROTOCOL.endswith("cita_proxy_dev_effect_protocol.json"))


if __name__ == "__main__":
    unittest.main()
