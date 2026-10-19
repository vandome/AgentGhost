"""§4PO: a tool-free reply that ANSWERS and then ends on a promise of work
("…PS4 only. Let me check if there are any workarounds like cloud gaming or
emulation:") gets the §4IW one continuation. §4IW caught only replies that
were narration through and through; the trailing-promise guard exempts
tool-free turns by design. Live replay 10-16: 2 of 10 still ended so;
corpus 30 d: 4 of 367 owner replies, all real, no false positive."""
import pytest

from ghost_agent.core import agent as A
from tests.test_4kw_announced_work_and_outcome import _drive, _directive_sent

ANSWER_THEN_PROMISE = ("**No, Bloodborne is not available on PC natively.** It's a PlayStation exclusive.\n\n"
                       "Let me check if there are any streaming/cloud workarounds:")


@pytest.mark.parametrize("text,want", [
    (ANSWER_THEN_PROMISE, True),
    ("No — PS4 only. Let me check if there are any workarounds like cloud gaming or emulation:", True),
    ("The calculator is rewritten. Let me verify it's actually working by opening the Calculator app:", True),
    ("No — PS4 only.\n\nLet me know if you want me to set up Remote Play.", False),
    ("No — PS4 only. If you want to try RPCS3, I can help you set it up.", False),
    ("No — PS4 only. Want me to look up the latest shadPS4 build?", False),
    ("No — PS4 only.\n\n**Bottom line:** you need a PS4 or PS5.", False),
    ("Let me search for the opening hours.", False),      # narration through and through: §4IW's own arm
    ("", False),
    # review r1 m1: a promise for LATER is a commitment (§4IY pins these as answers)
    ("Yes. I'll check it tomorrow.", False),
    ("Ναι. Θα το ελέγξω αύριο.", False),
    ("Got it. I'll look into it on Monday.", False),
    # r1 m2: only the last LINE; a list step is not an announcement
    ("Steps:\n1. Boil water.\n2. Let me add the pasta now.", False),
    ("Here it is:\n```\nprint(1)\n```\nLet me run it:", False),
])
def test_the_tail_is_judged_by_the_shared_narration_predicate(text, want):
    assert A._ends_on_announced_work(text, request="can i play bloodborne on a pc ?") is want


def test_a_creative_request_never_fires():
    poem = "Roses are red\nI'll search the stars for you.\nI will find my way home."
    assert not A._ends_on_announced_work(poem, request="write a poem about home")


def _trailing_directive_sent(payload):
    from tests import test_4kv_slack12_fixes as K
    return any("The user already has your answer" in u and "Let me check if there are any" in u
               for u in K._user_texts(payload))


@pytest.mark.parametrize("role", ["owner", "member"])
async def test_an_answer_that_ends_on_a_promise_keeps_the_answer(monkeypatch, tmp_path, role):
    """r1/r2 M1: the continuation used to REPLACE the first message — the
    loop now keeps the answer (without the promise) and adds the step's result."""
    final, model, logged, _, _ = await _drive(monkeypatch, tmp_path, [
        ("say", ANSWER_THEN_PROMISE), ("say", "Cloud: PS Plus Premium streams it to a PC."), ("say", "x")],
        request="can i play bloodborne on a pc ?", tools=("web_search",), role=role)
    assert _trailing_directive_sent(model.payloads[1]) and not _directive_sent(model.payloads[1])
    assert any("answers, then ends by announcing work" in c for t, c, _ in logged if t == "Turn Budget")
    assert "not available on PC natively" in final and "PS Plus Premium" in final
    assert "Let me check if there are any streaming" not in final and final.count("not available on PC") == 1


async def test_the_answer_survives_a_continuation_that_calls_a_tool(monkeypatch, tmp_path):
    """r2 M1: text written beside a tool call is trimmed at finalize — the
    kept answer is not such text."""
    final, model, _, _, _ = await _drive(monkeypatch, tmp_path, [
        ("say", ANSWER_THEN_PROMISE), ("call", "web_search", {"query": "bloodborne pc cloud"}),
        ("say", "Cloud gaming: PS Plus Premium streams it to a PC; the community port shadPS4 runs it too."),
        ("say", "x")],
        request="can i play bloodborne on a pc ?", tools=("web_search",), role="owner")
    assert "not available on PC natively" in final and "shadPS4" in final


async def test_a_reply_that_is_only_narration_keeps_the_original_directive(monkeypatch, tmp_path):
    _, model, _, _, _ = await _drive(monkeypatch, tmp_path, [
        ("say", "Let me search for the opening hours of the museum."), ("say", "It opens at 9."), ("say", "x")],
        request="when does it open", role="owner")
    assert _directive_sent(model.payloads[1]) and not _trailing_directive_sent(model.payloads[1])


async def test_an_ordinary_closing_line_gets_none(monkeypatch, tmp_path):
    _, model, _, _, _ = await _drive(monkeypatch, tmp_path, [
        ("say", "No — PS4 only.\n\nLet me know if you want me to set up Remote Play."), ("say", "x")],
        request="can i play bloodborne on a pc ?", tools=("web_search",), role="owner")
    assert len(model.payloads) == 1


async def test_a_second_promise_ships_one_nudge_per_request(monkeypatch, tmp_path):
    _, model, _, _, _ = await _drive(monkeypatch, tmp_path, [
        ("say", ANSWER_THEN_PROMISE), ("say", ANSWER_THEN_PROMISE), ("say", "x")],
        request="can i play bloodborne on a pc ?", tools=("web_search",), role="owner")
    assert len(model.payloads) == 2


async def test_a_tool_turn_keeps_the_trailing_promise_guard(monkeypatch, tmp_path):
    """After a tool ran, the older guard (act-or-admit) owns this shape."""
    _, model, logged, _, _ = await _drive(monkeypatch, tmp_path, [
        ("call", "web_search", {"query": "x"}), ("say", "I found two pages. Let me check the second one for prices:"),
        ("say", "The second page lists no prices."), ("say", "x")],
        request="what does it cost", tools=("web_search",), role="owner")
    assert any(t == "Pending-Promise Guard" for t, _, _ in logged)
    assert not any("answers, then ends" in c for t, c, _ in logged if t == "Turn Budget")


def test_a_system_note_after_the_promise_is_not_the_tail():
    """The loop's own notes are not the model's words (battery Q7)."""
    from ghost_agent.core.reply_smoothing import UNPARSED_TOOL_CALL_NOTE
    assert A._ends_on_announced_work(ANSWER_THEN_PROMISE + "\n\n" + UNPARSED_TOOL_CALL_NOTE,
                                     request="can i play bloodborne on a pc ?")


def test_the_carry_is_recorded_per_request_and_bounded():
    """r3: the rewind and the no-answer fallback read where the carry starts."""
    ctx = type("C", (), {})()
    for k in range(20):
        A._po_note_carry(ctx, f"r{k}", k, f"answer {k}")
    assert A._po_carry(ctx, "r19") == (19, "answer 19") and A._po_carry(ctx, "r0") is None
    assert len(ctx._po_carries) == 16 and A._po_carry(type("D", (), {})(), "r1") is None


def test_nothing_is_kept_when_the_tail_is_not_at_the_end():
    """r3 m1: then the arm asks for the whole answer instead of claiming the user has it."""
    assert A._answer_without_tail("Answer.\n\nLet me check X:\n\n_note_", "Let me check X:") == ""
    assert "will see only your NEXT message" in A._TRAILING_ANNOUNCED_STANDALONE
    assert "already has your answer" in A._TRAILING_ANNOUNCED_DIRECTIVE

