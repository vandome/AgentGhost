"""§4LN helpers for the browser tool — split out of `browser.py` so it stays
under its post-extraction size gate. Every name is re-exported there."""
import asyncio
import re
from pathlib import Path

_BROWSER_PROFILE_DIR = ".browser_profile"     # the runner's profile dir (browser.py imports it)


#: screenshot paths this process wrote (overwriting those is the point)
_BROWSER_WRITTEN: set = set()
_PROTECTED_IMAGE_NAME_RE = re.compile(r"^(?:gen_[0-9a-f]{8}|vision_[0-9a-f]{12})\.[A-Za-z]{3,4}$")


def _overwrite_refusal(host_path) -> str:
    """Why a screenshot must NOT be written over this existing file, or "".
    A screenshot is a PNG; an existing file that is not one (a user's
    photo.jpg, a document) cannot be an earlier screenshot, and a generated
    image or a pasted attachment is never one — `out_path="photo.jpg"`
    silently replaced the user's photo (§4LN)."""
    try:
        p = Path(host_path)
        if not p.exists() or str(p) in _BROWSER_WRITTEN:
            return ""
        with open(p, "rb") as fh:
            head = fh.read(8)
        if head == b"\x89PNG\r\n\x1a\n" and not _PROTECTED_IMAGE_NAME_RE.match(p.name):
            return ""
    except OSError:
        return ""
    return (f"'{Path(host_path).name}' already exists and is not a browser screenshot — "
            "a screenshot will not overwrite it. Pick a new out_path (e.g. shot_1.png).")

async def _exec_holding_lock(fn, *args, **kwargs):
    """Run the (blocking) runner exec in a thread — and when the CALLER is
    cancelled, keep waiting for that thread before letting the lock go.
    A hard cancel freed `_BROWSER_PROFILE_LOCK` at once while Chromium ran
    on in the container for up to the exec timeout, and the next call
    launched a second Chromium on the same profile — the crash this lock
    exists to prevent (§4LN). Call it INSIDE `async with` the lock."""
    fut = asyncio.ensure_future(asyncio.to_thread(fn, *args, **kwargs))
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        while not fut.done():                    # a second cancel must not free it either
            try:
                await asyncio.shield(fut)
            except asyncio.CancelledError:
                continue
            except BaseException:  # noqa: BLE001 — the cancel is what we re-raise
                break
        raise

# Keep outputs reasonable — a single page's HTML can be 5+ MB and would
# blow the LLM context window, so we cap before returning. These caps
# match `helper_fetch_url_content`'s 5 MB ceiling.

def _runner_failure_hint(err: str) -> str:
    """The advice for THIS failure (§4LN): every runner error used to carry
    one paragraph naming every failure mode — Tor timeouts, click chains and
    a stale `/root/.supercharged` repair — so a missing `h1` was answered
    with Chromium-install instructions (live probe B5)."""
    e = str(err or "")
    parts = []
    if "did not match any element" in e:
        parts.append("The page has no element matching that selector. Read the page first "
                     "(extract_text with no selector) and pick a selector from what is there; "
                     "do not try more guessed selectors.")
    if re.search(r"Page\.goto: Timeout|ERR_TIMED_OUT|navigation timeout|Timeout \d+ms exceeded navigating", e, re.I):
        parts.append("The page did not load over Tor — another attempt usually times out the "
                     "same way; use a different source.")
    if re.search(r"not found within|never became clickable|waiting for locator", e, re.I):
        parts.append("Each atomic op reloads the page in a fresh context, so elements created by "
                     "a previous click (opened windows, menus, dialogs) are GONE — run the whole "
                     "flow in one context with operation='interact' and an actions list.")
    if "headless_shell" in e or "Executable doesn't exist" in e:
        parts.append("The sandbox's browser install is broken — a retry fails the same way. "
                     "Tell the user the browser is unavailable.")
    if "needs a URL" in e:
        parts.append("Call operation=\"navigate\" once first, or pass url=... on this call.")
    return " ".join(parts) or "Read the error above — an identical retry usually fails the same way."


def _last_url_filename() -> str:
    """The sidecar this REQUEST's url-less ops read (§4LN): `.last_url.<req>`,
    or the shared `.last_url` outside a request (background flows)."""
    try:
        from ..utils.logging import request_id_context
        rid = re.sub(r"[^A-Za-z0-9_-]", "", str(request_id_context.get() or ""))[:24]
    except Exception:  # noqa: BLE001
        rid = ""
    return f".last_url.{rid}" if rid and rid != "SYSTEM" else ".last_url"

#: §4LN: three live replies said "the screenshot shows …" with no vision call
#: in the turn; one was contradicted by the user minutes later.
_UNSEEN_SCREENSHOT_NOTE = (
    "\nNOTE: you have NOT seen this image. Before describing what it shows, call "
    "vision_analysis on it (or read the page text with extract_text).\n"
    "A NEW screenshot is a working file: at the end of this turn it is removed from the workspace "
    "unless your reply links it (e.g. ![page](/api/download/<path>)) or the user asked for it.")


def _runner_first_url(operation, url, actions, sandbox_dir):
    """The URL the runner will ACTUALLY dial first — the only URL the
    memo may refuse or blame.

    Mirrors `_runner_script`'s own rule, which R4 measured the host side
    disagreeing with in BOTH directions:

      * the runner tests `actions[0]["action"] == "goto"` — nothing else.
        The host also accepted `"navigate"` (not a valid interact action
        at all: the runner's dispatch is goto-only) and additionally
        required a url. Consequences measured: `actions[0]=navigate` made
        the host skip the check while the runner dialled the top-level
        url (banned host re-dialled, nothing learned); and
        `actions[0]={"action":"goto"}` with no url made the host blame
        the top-level url the runner never touched — a ban on an
        uncontacted host, the inversion R3 called the worst outcome.
      * when the first action is NOT a goto, the runner performs an
        IMPLICIT initial navigation to `url` — or, when there is no url,
        to the `.last_url` sidecar. The memo was blind to that entire
        path (R4): `navigate(A)` then `click`/`extract_text` with no url
        is the flow this tool's own docstring teaches, and on it the memo
        never armed and a banned host was re-dialled at full Tor cost.
    """
    # R5: ops that dial NOTHING. `close` only rmtree's the profile, and
    # `navigate` without a url is a parameter error the runner raises on —
    # neither consults the sidecar. Falling through to it meant `close`
    # was REFUSED whenever the sidecar happened to name a banned host,
    # i.e. the memo's own state blocked the one operation that clears the
    # profile, for the full TTL; and a plain missing-parameter mistake was
    # answered with "pick a DIFFERENT result from your search".
    if operation in ("close", "navigate"):
        return url or ""
    if operation == "interact" and actions:
        first = actions[0] if isinstance(actions[0], dict) else {}
        if first.get("action") == "goto":
            return first.get("url") or ""
    if url:
        return url
    # Sidecar fallback, read host-side from the file the runner writes.
    #
    # ⚠ UN-SCOPE FIRST (R5). The runner's profile is hardcoded to
    # `/workspace/.browser_profile`, and `/workspace` is the bind mount of
    # the sandbox ROOT — but `registry.py` hands this function the
    # PROJECT-scoped dir (`<root>/projects/<id>`) whenever a project is
    # active. Reading `<root>/projects/<id>/.browser_profile/.last_url`
    # finds nothing, so in a project session — the majority of real work —
    # the memo neither refused a banned host nor learned from the failure,
    # and the dead onion was re-dialled over Tor exactly as before the fix.
    # `_to_container_path` un-scopes for the same reason.
    try:
        root = Path(str(sandbox_dir or "."))
        if root.parent.name == "projects":
            root = root.parent.parent
        with open(root / _BROWSER_PROFILE_DIR / _last_url_filename(),
                  "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except Exception:  # noqa: BLE001
        return ""


def click_miss_summary(action_results) -> list:
    """§4NW (req 92dfdd7d): one up-front line when an interact's clicks did
    not reach their target — the per-action lines read "OK" (the click ran),
    so twelve such calls read as success. STATUS is left as it is: a PARTIAL
    there cost strikes and could end a working turn as failed (review)."""
    dead = [r for r in (action_results or []) if r.get("ok") and r.get("no_effect")]
    if not dead:
        return []
    return [f"⚠ {len(dead)} click(s) did not do what was intended "
            f"({', '.join(repr(r.get('selector')) for r in dead[:3])}) — what you meant to "
            "open is NOT open; a screenshot after them does not show it. Reason per click below."]


def no_effect_line(r) -> str:
    """The per-action warning under a click that missed (§4NW), or the note
    under a single click on a double-click-only element, or ""."""
    if r.get("no_effect"):
        return f"\n      ⚠ NO EFFECT: {r['no_effect']}"
    return f"\n      NOTE: {r['hint']}" if r.get("hint") else ""


def _note_shot(host_path) -> None:
    """A screenshot this call writes: overwritable by later screenshots, and
    — when it is NEW — noted so the end of the turn moves it to the host trash
    unless the reply or the user names it (§4NY)."""
    from .turn_scratch import note_created
    note_created(host_path, "screenshot")
    _BROWSER_WRITTEN.add(str(host_path))


def interact_notes(action_results) -> list:
    """The lines above an interact's per-action results: clicks that missed
    (§4NW) and, when it took screenshots, that they are working files (§4NY)."""
    out = click_miss_summary(action_results)
    if any(r.get("ok") and r.get("action") == "screenshot" for r in (action_results or [])):
        out.append("NOTE: new screenshots are working files — at the end of this turn they are removed from the "
                   "workspace unless your reply links them (![x](/api/download/<path>)) or the user asked for them.")
    return out


def ensure_runner_file(path, text: str) -> bool:
    """§4OE-5: write the browser runner only when the file's text differs
    (atomic replace). Returns True when it wrote. Called under the browser
    profile lock — written before it, a batch-mate's rewrite raced the exec
    through the bind mount (`can't open file .browser_runner.py`)."""
    import os as _os
    try:
        if path.exists() and path.read_text(encoding="utf-8") == text:
            return False
    except Exception:  # noqa: BLE001 — unreadable: rewrite it
        pass
    tmp = path.with_name(f"{path.name}.{_os.getpid()}.{id(path)}.tmp")
    tmp.write_text(text, encoding="utf-8")
    _os.replace(tmp, path)
    return True


#: §4OG-5 (f): under a client deadline a 65,536-char preview per page is prefill
#: the clock cannot afford (a 240 s request spent 52 % of its wall prefilling
#: 46k tokens of previews, §4OF) — the preview is capped at a quarter of the max
DEADLINE_PREVIEW_CAP = 16 * 1024


def preview_cap_chars(max_chars: int = 64 * 1024) -> int:
    """The nav preview cap: `max_chars`, or DEADLINE_PREVIEW_CAP when the
    request carries a client deadline."""
    try:
        from ..utils.logging import client_deadline_context
        if float(client_deadline_context.get() or 0.0) > 0:
            return min(int(max_chars), DEADLINE_PREVIEW_CAP)
    except Exception:  # noqa: BLE001
        pass
    return int(max_chars)
