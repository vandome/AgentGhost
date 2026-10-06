"""The in-sandbox Chromium must use the sandbox's own Tor, not the host's.

The host's TOR_PROXY port is meaningless inside the container's netns. It only
worked while the host Tor and the in-container Tor both listened on 9050; a host
Tor on any other port (e.g. 9150 when 9050 is taken) broke every browser call
with net::ERR_PROXY_CONNECTION_FAILED.
"""

from ghost_agent.sandbox.tor_egress import SOCKS_PORT
from ghost_agent.tools.browser import _sandbox_socks_proxy


def test_host_port_is_rewritten_to_sandbox_tor():
    assert _sandbox_socks_proxy("socks5://127.0.0.1:9150") == f"socks5://127.0.0.1:{SOCKS_PORT}"


def test_scheme_and_isolation_credentials_are_kept():
    assert (_sandbox_socks_proxy("socks5h://circuit-7:x@localhost:9150")
            == f"socks5h://circuit-7:x@127.0.0.1:{SOCKS_PORT}")


def test_direct_stays_direct():
    assert _sandbox_socks_proxy(None) is None
    assert _sandbox_socks_proxy("") == ""
