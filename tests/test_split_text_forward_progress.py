"""recursive_split_text / semantic_split_text must always terminate.

Bug (2026-10-05): when the chosen separator was punctuation ('. ', '? ',
'! ', '; ', ', ') and a fragment longer than chunk_size followed at least
one shorter fragment, the overlap carry put the whole prefix back in front of
the oversized fragment and the separator re-attached to the LAST fragment
stripped to just '.' — so the re-pushed chunk was ``current_text + '.'``.
The only hard-split trigger was ``chunk == current_text``, which therefore
never fired: every pass appended one more '.', and the loop spun forever
(100% CPU, final_chunks growing without bound). Ordinary inputs hit it:
'Intro. ' + a 1 KB sentence + '. Outro.', an assistant reply carrying a
base64 blob after a sentence, a long comma list behind a header. The KB
ingest path (tools/memory.py tool_gain_knowledge) calls semantic_split_text
directly on the event loop, so the whole agent froze.

Fix: an oversized chunk is re-pushed only if it is strictly shorter than the
text it came from; otherwise it is hard-split. Every pass therefore works on
less text (guaranteed termination), and inputs that terminated before produce
byte-identical chunks — pinned below against a frozen copy of the pre-fix
algorithm on a seeded corpus.

Every test runs under a SIGALRM deadline: pytest-timeout is not guaranteed to
be installed, and a regression here is an infinite loop, not a failure.
"""

from __future__ import annotations

import contextlib
import random  # non-security-sensitive: seeded synthetic test corpus only
import signal
import string
from collections import Counter
from typing import List
from unittest.mock import patch

import pytest

from ghost_agent.utils import helpers
from ghost_agent.utils.helpers import recursive_split_text, semantic_split_text


class _Deadline(Exception):
    pass


@contextlib.contextmanager
def _deadline(seconds: float):
    """Fail (instead of hanging the suite) if the body runs past `seconds`."""
    if not hasattr(signal, "setitimer"):  # pragma: no cover - non-POSIX
        yield
        return

    def _fire(signum, frame):
        raise _Deadline(f"splitter did not terminate within {seconds}s (infinite loop?)")

    old = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _nonspace(s: str) -> Counter:
    return Counter(c for c in s if not c.isspace())


def _assert_no_loss(text: str, chunks: List[str]) -> None:
    # Overlap duplicates characters, so compare as multisets: every
    # non-whitespace character of the input must survive at least as often.
    lost = _nonspace(text) - _nonspace("".join(chunks))
    assert not lost, f"chunks dropped characters: {dict(lost)}"


_B64 = string.ascii_letters + string.digits + "+/"


def _blob(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(_B64) for _ in range(n))


# --------------------------------------------------------------------------
# The reported shapes (each one hung the pre-fix code)
# --------------------------------------------------------------------------

_RNG = random.Random(1)
_TOKEN = _blob(_RNG, 1000)

HUNG_BEFORE = {
    "verifier_repro": (recursive_split_text, "word. " * 2 + "X" * 700),
    "long_middle_sentence": (recursive_split_text,
                             "Intro. " + " ".join(["word"] * 200) + ". Outro."),
    "code_block_then_token_list": (semantic_split_text,
                                   "Run this:\n```bash\nls -la\n```\nOutput. Token list: " + _TOKEN),
    "header_then_encoded_value": (semantic_split_text,
                                  "## Result\nDone. The encoded value is " + _TOKEN),
    "code_block_then_long_line": (semantic_split_text,
                                  "```python\nprint(1)\n```\nNote. See: " + "Z" * 900 + " end"),
    "header_then_long_comma_list": (semantic_split_text,
                                    "# Items\nList: a, b, " + "Q" * 700 + ", c"),
    "semicolon_run": (recursive_split_text, "a; b; " + "S" * 650),
    "question_run": (recursive_split_text, "Why? " + "Y" * 650 + "? ok"),
}


@pytest.mark.parametrize("name", sorted(HUNG_BEFORE))
def test_reported_shapes_terminate(name):
    fn, text = HUNG_BEFORE[name]
    with _deadline(10):
        chunks = fn(text, 600, 100)
    assert chunks
    _assert_no_loss(text, chunks)
    if fn is recursive_split_text:
        assert all(len(c) <= 600 for c in chunks)
    # Document order: the opening text leads the first chunk.
    assert text[:4] in chunks[0]


def test_reported_shapes_hung_the_prefix_algorithm():
    """Guard the guard: the frozen pre-fix algorithm really loops on them."""
    for name, (fn, text) in HUNG_BEFORE.items():
        with _deadline(30), pytest.raises(_RefRunaway):
            _reference_call(fn.__name__, text, 600, 100)


def test_verifier_repro_exact_chunks():
    with _deadline(10):
        chunks = recursive_split_text("word. " * 2 + "X" * 700, 600, 100)
    assert chunks == ["word. word.",
                      "word. word. " + "X" * 588,
                      ("word. word. " + "X" * 700 + ".")[500:]]


# --------------------------------------------------------------------------
# Frozen pre-fix algorithm (HEAD 2f8cc3f) — the differential reference
# --------------------------------------------------------------------------

class _RefRunaway(Exception):
    """The pre-fix algorithm entered its non-terminating growth."""


def _reference_recursive_split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 70) -> List[str]:
    # Pre-fix logic (comments elided), plus a runaway detector: every chunk the old
    # code produced is a stripped substring of (current_text + separator), so
    # a current_text longer than the input + 2 only happens in the '.'-growth
    # loop. The iteration cap is a backstop that must never fire.
    if not text: return []
    chunk_size = max(1, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))
    if len(text) <= chunk_size: return [text]
    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    final_chunks = []
    stack = [text]
    iterations = 0
    while stack:
        current_text = stack.pop()
        iterations += 1
        if len(current_text) > len(text) + 2:
            raise _RefRunaway()
        assert iterations < 40 * len(text) + 5000, "reference iteration cap hit"
        if len(current_text) <= chunk_size:
            final_chunks.append(current_text)
            continue
        found_sep = ""
        for sep in separators:
            if sep in current_text:
                found_sep = sep
                break
        if not found_sep:
            for i in range(0, len(current_text), chunk_size - chunk_overlap):
                final_chunks.append(current_text[i:i+chunk_size])
            continue
        parts = current_text.split(found_sep)
        buffer = ""
        temp_chunks = []
        for p in parts:
            fragment = p + found_sep
            if len(buffer) + len(fragment) <= chunk_size:
                buffer += fragment
            else:
                if buffer:
                    temp_chunks.append(buffer.strip())
                    _carry = min(chunk_overlap, chunk_size // 4)
                    buffer = buffer[-_carry:] if _carry > 0 else ""
                buffer += fragment
        if buffer:
            temp_chunks.append(buffer.strip())
        for chunk in reversed(temp_chunks):
            if len(chunk) > chunk_size and (found_sep == "" or chunk == current_text):
                pieces = [chunk[i:i+chunk_size]
                          for i in range(0, len(chunk), chunk_size - chunk_overlap)]
                stack.extend(reversed(pieces))
            else:
                stack.append(chunk)
    return final_chunks


def _reference_call(fn_name: str, text: str, cs: int, ov: int) -> List[str]:
    if fn_name == "recursive_split_text":
        return _reference_recursive_split_text(text, cs, ov)
    # semantic_split_text is unchanged; it resolves recursive_split_text
    # through the module globals, so swapping that in gives the old pipeline.
    with patch.object(helpers, "recursive_split_text", _reference_recursive_split_text):
        return helpers.semantic_split_text(text, cs, ov)


# --------------------------------------------------------------------------
# Seeded corpus: normal prose/markdown and adversarial shapes
# --------------------------------------------------------------------------

_WORDS = ("the of and to in is was for on that with as by at from it this be are "
          "build config deploy server error value token model chunk split merge "
          "forecast grid python bash docker vector memory ingest").split()


def _sentence(rng, lo=3, hi=25):
    return " ".join(rng.choice(_WORDS) for _ in range(rng.randint(lo, hi))) + \
        rng.choice([".", ".", ".", "?", "!", ";", ""])


def _prose(rng):
    paras = [" ".join(_sentence(rng) for _ in range(rng.randint(1, 8)))
             for _ in range(rng.randint(1, 6))]
    return rng.choice(["\n\n", "\n", " "]).join(paras)


def _code_block(rng):
    lines = [rng.choice(["ls -la", "git status", "for i in range(3): print(i)",
                         "echo done", "x = 1; y = 2", "SELECT * FROM t;"])
             for _ in range(rng.randint(1, 12))]
    return "```" + rng.choice(["bash", "python", ""]) + "\n" + "\n".join(lines) + "\n```"


def _markdown(rng):
    out = []
    for _ in range(rng.randint(1, 5)):
        if rng.random() < 0.8:
            out.append("#" * rng.randint(1, 4) + " " + _sentence(rng, 1, 6))
        out.append(rng.choice([
            lambda: _prose(rng),
            lambda: _code_block(rng) + "\n" + _prose(rng),
            lambda: "- " + "\n- ".join(_sentence(rng) for _ in range(rng.randint(2, 9))),
        ])())
    return "\n".join(out)


def _adversarial(rng):
    big = rng.randint(150, 2200)
    seps = [". ", "? ", "! ", "; ", ", ", " ", "\n", "\n\n"]
    return rng.choice([
        lambda: "word. " * rng.randint(0, 6) + "X" * big,
        lambda: "Run this:\n```bash\nls -la\n```\nOutput. Token list: " + _blob(rng, big),
        lambda: "## Result\nDone. The encoded value is " + _blob(rng, big),
        lambda: _code_block(rng) + "\n" + _sentence(rng) + " " + _blob(rng, big),
        lambda: "# Items\n" + ", ".join(rng.choice(_WORDS) for _ in range(rng.randint(50, 300))),
        lambda: _sentence(rng) + " " + ", ".join(_blob(rng, rng.randint(1, 30)) for _ in range(60)) + ". tail",
        lambda: "".join(rng.choice([_blob(rng, rng.randint(1, 40)), rng.choice(seps),
                                    _blob(rng, rng.randint(100, 900))])
                        for _ in range(rng.randint(2, 10))),
        lambda: rng.choice([" ", "\n", ""]) + _prose(rng) + " " + _blob(rng, big) + rng.choice([" ", ". ", ""]),
        lambda: ". " * rng.randint(1, 4) + _blob(rng, big) + ". " * rng.randint(0, 4),
        lambda: "".join(rng.choice("ab. ,;\n?!") for _ in range(big)),
        lambda: _prose(rng) + rng.choice([". ", "; ", ", "]) + "y" * big + rng.choice(["", ". ", " tail."]),
    ])()


def _params(rng):
    r = rng.random()
    if r < 0.4:
        return rng.choice([(600, 100), (1200, 150), (500, 70)])
    if r < 0.75:
        cs = rng.randint(20, 1300)
        return cs, rng.randint(0, cs // 3)
    cs = rng.randint(1, 200)
    return cs, rng.randint(0, cs + 60)   # includes overlap >= chunk_size


def _corpus(seed: int, n: int):
    rng = random.Random(seed)
    for _ in range(n):
        if rng.random() < 0.5:
            text = rng.choice([_prose, _markdown, lambda r: _prose(r) + "\n\n" + _markdown(r)])(rng)
        else:
            text = _adversarial(rng)
        yield text, _params(rng)


def test_identical_chunks_whenever_prefix_algorithm_terminated():
    """No behaviour change: for every corpus input the old code finished on,
    the new code returns exactly the same chunks; where the old code looped
    forever, the new one terminates without dropping content."""
    same = runaway = 0
    with _deadline(120):
        for text, (cs, ov) in _corpus(seed=20261005, n=350):
            for fn in (recursive_split_text, semantic_split_text):
                new = fn(text, cs, ov)
                try:
                    old = _reference_call(fn.__name__, text, cs, ov)
                except _RefRunaway:
                    runaway += 1
                    _assert_no_loss(text, new)
                    continue
                assert new == old, (fn.__name__, len(text), cs, ov)
                same += 1
    # The corpus must exercise both regimes for the comparison to mean anything.
    assert same >= 300 and runaway >= 50, (same, runaway)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_property_adversarial_inputs_terminate_bounded(seed):
    rng = random.Random(seed)
    with _deadline(60):
        for _ in range(250):
            text = _adversarial(rng)
            cs, ov = _params(rng)
            chunks = recursive_split_text(text, cs, ov)
            assert chunks
            assert all(len(c) <= max(1, cs) for c in chunks), (len(text), cs, ov)
            _assert_no_loss(text, chunks)
            sem = semantic_split_text(text, cs, ov)
            if text.strip():
                assert sem
                _assert_no_loss(text, sem)


def test_property_random_separator_soup_terminates():
    """Dense random mixes of every separator with long runs and odd
    whitespace, at tiny and production chunk sizes."""
    rng = random.Random(77)
    alphabet = ["a", "b", "Z", ".", "?", "!", ";", ",", " ", "  ", "\n", "\n\n", "\t", ". ", "; ", ", "]
    with _deadline(60):
        for _ in range(400):
            text = "".join(rng.choice(alphabet) * rng.choice([1, 1, 1, 3, 40, 400])
                           for _ in range(rng.randint(1, 60)))
            cs = rng.choice([1, 2, 3, 7, 16, 50, 120, 600, 1200])
            ov = rng.choice([0, 1, cs // 4, cs - 1, cs, cs + 10])
            chunks = recursive_split_text(text, cs, ov)
            assert all(len(c) <= cs for c in chunks)
            _assert_no_loss(text, chunks)
            semantic_split_text(text, cs, ov)
