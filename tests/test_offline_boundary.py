"""Unit tests must stay offline even with a configured local .env."""

import unittest

import httpx


class OfflineBoundaryTests(unittest.TestCase):
    def test_default_http_transport_is_rejected_before_network(self):
        with httpx.Client() as client:
            with self.assertRaisesRegex(AssertionError, "real HTTP is prohibited"):
                client.get("https://unit-test.invalid/never-send")

    def test_explicit_in_memory_transport_is_allowed(self):
        with httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"mock": True})
            )
        ) as client:
            self.assertEqual(
                client.get("https://unit-test.invalid/mock").json(), {"mock": True}
            )
