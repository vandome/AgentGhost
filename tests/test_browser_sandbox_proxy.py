"""The in-sandbox Chromium must use the sandbox's own Tor, not the host's.

The host's TOR_PROXY port is meaningless inside the container's netns. It only
worked while the host Tor and the in-container Tor both listened on 9050; a host
Tor on any other port (e.g. 9150 when 9050 is taken) broke every browser call
with net::ERR_PROXY_CONNECTION_FAILED.
"""

from ghost_agent.sandbox.tor_egress import SOCKS_PORT, in_sandbox_socks_proxy
from ghost_agent.tools.browser import _build_op_payload


def test_host_port_is_rewritten_to_sandbox_tor():
    assert in_sandbox_socks_proxy("socks5://127.0.0.1:9150") == f"socks5://127.0.0.1:{SOCKS_PORT}"


def test_scheme_and_isolation_credentials_are_kept():
    assert (in_sandbox_socks_proxy("socks5h://circuit-7:x@localhost:9150")
            == f"socks5h://circuit-7:x@127.0.0.1:{SOCKS_PORT}")


def test_direct_stays_direct():
    assert in_sandbox_socks_proxy(None) is None
    assert in_sandbox_socks_proxy("") == ""


def _payload(tor_proxy):
    return _build_op_payload("navigate", "https://example.com", None, None, None,
                             None, None, 30000, tor_proxy)


def test_the_runner_payload_carries_the_sandbox_tor_not_the_host_port():
    # what the runner hands Chromium's --proxy-server (socks5h -> socks5 too)
    assert _payload("socks5h://127.0.0.1:9150")["proxy"] == f"socks5://127.0.0.1:{SOCKS_PORT}"


def test_the_runner_payload_stays_direct_without_a_proxy():
    assert _payload(None)["proxy"] is None
