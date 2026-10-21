"""2026-10-20, request 52c9a5d2: Wikipedia answered the subjects photo lookup
429 from every Tor exit because the client impersonated Chrome; a plain tool
User-Agent is answered 200. The 429 was also reported to the model as
"network / Tor" with "call again once", and the reason reached no log."""
from unittest.mock import AsyncMock

import pytest

import ghost_agent.tools.subject_photos as SP
from ghost_agent.tools.image_gen import tool_generate_image
from tests.test_4mg_image_subjects import _llm


class _Status:
    def __init__(self, code):
        self.status_code = code


def test_lookup_sends_the_generic_tool_user_agent_not_a_browser(monkeypatch):
    import curl_cffi.requests as creq
    seen = []

    class Empty:
        status_code = 200

        def json(self):
            return {"query": {"pages": {}}}

    def get(url, params=None, **kw):
        seen.append(kw)
        return Empty()
    monkeypatch.setattr(creq, "get", get)
    SP.lookup_photo_url("Joseph Stalin", None)
    assert seen and all(k.get("headers") == {"User-Agent": "curl/8.7.1"} for k in seen)
    assert not any("impersonate" in k for k in seen)
    assert SP.LOOKUP_USER_AGENT == "curl/8.7.1"      # generic: never a name of our own


@pytest.mark.parametrize("code,limited", [(429, True), (503, False), (403, False)])
def test_a_429_marks_the_failure_rate_limited(monkeypatch, code, limited):
    import curl_cffi.requests as creq
    monkeypatch.setattr(creq, "get", lambda *a, **k: _Status(code))
    with pytest.raises(SP.SearchUnavailable) as ei:
        SP.lookup_photo_url("Joseph Stalin", None)
    assert ei.value.rate_limited is limited and f"HTTP {code}" in str(ei.value)


def test_a_429_in_one_language_still_marks_rate_limited(monkeypatch):
    """Greek names ask el. then en.: a 429 on el. must survive en.'s 503
    (a failed title query skips that language's search: one call each)."""
    import curl_cffi.requests as creq
    codes = iter([429, 503])
    monkeypatch.setattr(creq, "get", lambda *a, **k: _Status(next(codes)))
    with pytest.raises(SP.SearchUnavailable) as ei:
        SP.lookup_photo_url("Ζωή Κωνσταντοπούλου", None)
    assert ei.value.rate_limited is True


def _down(monkeypatch, limited):
    def down(subject, proxy):
        raise SP.SearchUnavailable("the encyclopedia could not be reached (HTTP 429)"
                                   if limited else "the encyclopedia could not be reached (HTTP 503)",
                                   rate_limited=limited)
    monkeypatch.setattr(SP, "lookup_photo_url", down)
    import ghost_agent.utils.helpers as H
    monkeypatch.setattr(H, "request_new_tor_identity", lambda *a, **k: None)
    monkeypatch.setattr(SP.asyncio, "sleep", AsyncMock())
    logged = []
    monkeypatch.setattr(SP, "pretty_log", lambda *a, **k: logged.append((a, k)))
    return logged


async def test_rate_limited_result_says_429_and_forbids_the_retry(tmp_path, monkeypatch):
    logged = _down(monkeypatch, True)
    llm = _llm()
    out = await tool_generate_image(prompt="x", subjects=["Zoe Konstantinou", "Joseph Stalin"],
                                    llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("ERROR: the encyclopedia is rate-limiting the photo search")
    assert "HTTP 429" in out and "NOT a network problem" in out and "Do NOT call image_generation" in out
    assert "Call image_generation again" not in out and "Tor" not in out
    llm.generate_image.assert_not_called()
    # the reason reaches the log, once per subject, as a warning
    warns = [a for a, k in logged if k.get("level") == "WARNING"]
    assert len(warns) == 2 and all("HTTP 429" in a[1] for a in warns)


async def test_a_plain_outage_keeps_the_retry_once_advice(tmp_path, monkeypatch):
    _down(monkeypatch, False)
    llm = _llm()
    out = await tool_generate_image(prompt="x", subjects=["Joseph Stalin"], llm_client=llm, sandbox_dir=tmp_path)
    assert out.startswith("ERROR: the photo search could not reach the encyclopedia")
    assert "Call image_generation again once" in out and "rate-limiting" not in out
