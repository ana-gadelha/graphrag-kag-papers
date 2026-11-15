"""Test namespace: real HTTP is forbidden even when a local .env has API keys.

The SDK protocol tests use httpx.MockTransport, which does not call HTTPTransport.
Opt-in Neo4j integration uses Bolt and remains restricted by TEST_NEO4J_*.
"""

from unittest.mock import patch


def _reject_real_http(*args, **kwargs):
    raise AssertionError(
        "Unit tests must use httpx.MockTransport; real HTTP is prohibited."
    )


_http_guard = patch("httpx.HTTPTransport.handle_request", _reject_real_http)
_http_guard.start()
