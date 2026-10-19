"""§4QA-e: a results file that cites a probe transcript quotes the model's words.

Project 448d5afa86f0's write-ups (§4QA-c reruns, 2026-10-19) invented no
answers, but six of them put the model's words in quote marks with words
changed: "aggregate of my attention patterns" became "aggregate of attention
patterns", "I am a process, not a substance" became "identity as process, not
substance". The transcripts are the data; a reworded quote of them is not.

``probe_model`` keeps a byte copy of every transcript it saves, OUTSIDE the
sandbox (``<memory_dir>/probe_transcripts/<project>/<sha256>.md`` + an index
of the path it was saved as). A sandbox copy can be edited by the model; this
one cannot, so a check never compares a quote with a transcript the writer
changed to match it (ownership by bytes, §4QA).

ADVISORY (operator 2026-10-19, after reviews r27–r31 each found 3–9 MAJOR in
a blocking version): the close always stands; a write-up that seems to reword
the answers is reported to the model and in the project ledger
(task_deliverable.note_misquotes).

At task close, :func:`quote_problem` takes a deliverable that CITES a kept
transcript (names its path or file name) and reads every quote in it:
``` ```text ``` blocks, ``>`` blockquote paragraphs (only the quoted spans of
one that holds any), and double-quoted spans of four or more words. A quote
is fine when its words stand in that order in ONE kept transcript or in the
task's stored description (case, punctuation, markdown and line breaks
ignored; an ellipsis or a ``[bracketed]`` insertion splits it into pieces,
matched in order). A quote found in neither is flagged only when it REWORDS
a transcript passage (:func:`_reworded`): in a stretch of the transcript as
long as the quote plus SLACK words, at least NEAR_RATIO of its content words
(stopwords dropped) stand in the quote's order, with ≥ 3 of them matched or
two of them side by side in both. Anything else is a quote of a source this
check does not hold (Hofstadter, a web page) and is left alone.

r27/r28 history: "every quote must be in a transcript" refused 7 honest live
quotes; trusting snapshotted project files as sources let an edited
transcript copy and the first run's reworded write-ups launder rewordings; a
bag of ALL words over a quote-length window refused 18 of 30 famous quotes
on stopwords alone. Calibration of this rule on the live 448d5afa86f0 data:
8 of 8 real rewordings refused; 0 of 30 famous quotes, 0 of 94 research and
design quotes refused (scratchpad calib/, r28). r29 judged the answers only
and made the project's goal a source; r30 judges each LINE (cell, sentence)
with labels, list markers and writers' notes removed, refuses a splice of
verbatim sentences, lets a quote as close to the planner's words as to the
answer pass, and checks the clock inside the scan.
"""
from __future__ import annotations

import hashlib
import math
import json
import logging
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger("GhostAgent")

DIRNAME = "probe_transcripts"
INDEX_NAME = "index.jsonl"
#: a double-quoted span is a quote from this many words (shorter ones are
#: scare quotes or terms: "stateless", "the hum")
MIN_QUOTE_WORDS = 4
#: an ellipsis piece is matched from this many words ("…and" is not a quote)
MIN_PIECE_WORDS = 2
#: a quote found nowhere rewords a transcript passage when this share of its
#: content words stands, in order, in a stretch of the transcript as long as
#: the quote plus SLACK words (r28 calibration: real rewordings 0.67–1.0)
NEAR_RATIO = 0.6
SLACK = 3
#: below this many content words a quote is too short to tell a rewording
#: from a famous phrase the model also used ("I am a strange loop")
MIN_CONTENT_WORDS = 3
#: unfound quotes judged per deliverable (each is a scan of every transcript),
#: and the wall clock for judging them (it runs inside task_update)
_MAX_JUDGED = 200
_MAX_JUDGE_S = 2.0
_MAX_LISTED = 5
_STOPWORDS = frozenset("""a an the and or but if then than so as of in on at to for from by with
without into onto over under about above below between through during before after up down out
off again further once here there when where why how all any both each few more most other some
such no nor not only own same too very can will just don should now i me my myself we our ours
ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself
they them their theirs themselves what which who whom this that these those am is are was were be
been being have has had having do does did doing would could ought s t d ll m re ve y let us also
may might must shall yet one""".split())

_WORD_RE = re.compile(r"[^\W_]+")
_FENCE_OPEN_RE = re.compile(r"^(`{3,}|~{3,})([^`]*)$")
_DQ_RE = re.compile(r"[\"“«„]([^\"“”«»„\n]+)[\"”»“]")
#: what splits a quote into pieces: an ellipsis, a [bracketed] insertion, and
#: (r30) a writer's note in parentheses — "(sic)", "(truncated)", "(emphasis mine)"
_SPLIT_RE = re.compile(r"\.\.\.|…|\[[^\]\n]*\]|\((?:[^()\n]{0,30}\b)?(?:sic|truncated|cut(?: off)?|"
                       r"emphasis|abridged|edited|original|added|max_tokens|reached|unfinished|excerpt|"
                       r"verbatim|shortened|ends here|incomplete|continues)\b[^()\n]{0,30}\)", re.I)
#: r30: the writer's label at the head of a quoted line ("**Response 1:**",
#: "*Prompt 1 (control):*", "Layer 2:") is not the model's word
_LABEL_RE = re.compile(r"^\s*(?:(\*\*|\*|_)(?:(?!\1)[^\n]){1,60}?(?::\s*\1|\1\s*:)|"
                       r"[A-Z][\w().-]*(?: [\w().-]+){0,2}:)\s*")
#: r30: a list marker ("1.", "-") — a write-up renumbers the model's list
_LIST_RE = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+")
#: r30: a quoted line longer than this many words is judged by sentences (and
#: then by chunks of it) — one 110-row table took 36 s as a single quote
_LONG_UNIT = 60


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text or "")).lower()
    return re.sub(r"[‘’ʼ`´]", "'", t)


def _words(text: str) -> List[str]:
    return _WORD_RE.findall(_norm(text))


def _joined(text: str) -> str:
    return " " + " ".join(_words(text)) + " "


def project_dir(memory_dir, project_id: str) -> Optional[Path]:
    pid = str(project_id or "")
    if not memory_dir or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", pid):
        return None
    return Path(memory_dir) / DIRNAME / pid


def record(memory_dir, project_id: str, rel: str, text: str) -> Optional[str]:
    """Keep the transcript's bytes host-side; returns its sha256, or None when
    it could not be kept (logged — the probe itself already ran)."""
    d = project_dir(memory_dir, project_id)
    if d is None:
        logger.warning("probe_transcripts: %s not kept (no memory dir / project id %r)", rel, project_id)
        return None
    try:
        data = text.encode("utf-8", errors="replace")
        sha = hashlib.sha256(data).hexdigest()
        d.mkdir(parents=True, exist_ok=True)
        dest = d / f"{sha}.md"
        if not dest.exists():
            tmp = d / f".{sha}.{os.getpid()}.tmp"
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        with open(d / INDEX_NAME, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"rel": str(rel), "sha": sha,
                                 "when": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
        return sha
    except OSError as e:
        logger.warning("probe_transcripts: %s not kept (%s)", rel, e)
        return None


def recorded(memory_dir, project_id: str) -> List[Tuple[str, str]]:
    """(saved-as path, text) of every transcript kept for the project whose
    bytes still hash to their name."""
    d = project_dir(memory_dir, project_id)
    if d is None:
        return []
    try:
        lines = (d / INDEX_NAME).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out, seen = [], set()
    for ln in lines:
        try:
            row = json.loads(ln)
            rel, sha = str(row["rel"]), str(row["sha"])
        except (ValueError, KeyError, TypeError):
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", sha) or (rel, sha) in seen:
            continue
        seen.add((rel, sha))
        try:
            data = (d / f"{sha}.md").read_bytes()
        except OSError:
            continue
        if hashlib.sha256(data).hexdigest() == sha:
            out.append((rel, data.decode("utf-8", errors="replace")))
    return out


def cites(text: str, rel: str) -> bool:
    """The text names the transcript: its saved path or, failing that, its
    file name as a whole name."""
    name = PurePosixPath(rel).name
    return rel in text or bool(name and re.search(r"(?<![\w.-])" + re.escape(name) + r"(?![\w-])", text))


def _split_fences(text: str) -> Tuple[List[str], str]:
    """(```text block bodies, the text without any fenced block) — a line
    scanner: an unclosed fence runs to the end (r27: a regex was quadratic)."""
    blocks, rest, lines, i = [], [], text.split("\n"), 0
    while i < len(lines):
        m = _FENCE_OPEN_RE.match(lines[i])
        if not m:
            rest.append(lines[i])
            i += 1
            continue
        fence, j = m.group(1), i + 1
        while j < len(lines) and not (lines[j].startswith(fence) and not lines[j].strip(fence[0] + " \t")):
            j += 1
        if m.group(2).strip().lower() == "text":
            blocks.append("\n".join(lines[i + 1:j]))
        rest.append("")
        i = j + 1
    return blocks, "\n".join(rest)


def _spans(text: str) -> List[str]:
    return [m.group(1) for m in _DQ_RE.finditer(text) if len(_words(m.group(1))) >= MIN_QUOTE_WORDS]


def _chunks(unit: str) -> List[str]:
    """A unit of at most _LONG_UNIT words: its sentences, then word chunks."""
    if len(_words(unit)) <= _LONG_UNIT:
        return [unit]
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", unit):
        ws = _WORD_RE.findall(sent)
        if len(ws) <= _LONG_UNIT:
            out.append(sent)
        else:
            out.extend(" ".join(ws[k:k + _LONG_UNIT]) for k in range(0, len(ws), _LONG_UNIT))
    return out


def _units(block: str, spans_ok: bool) -> List[str]:
    """r30: the quotes of a ```text block or a blockquote are its LINES (a
    table's cells), each without the writer's label (r32: an attribution
    line is judged like any other — dropping lines that start with a dash let
    a reworded "— I am a continuous stream…" through unread). With ``spans_ok`` a line whose double-quoted spans carry the
    quote, around fewer than MIN_CONTENT_WORDS words of the writer's own, is
    those spans (r27: "**Insight:** \"…\""); any other line is judged whole
    (r30: a reworded line passed on one verbatim span inside it)."""
    out = []
    for line in block.split("\n"):
        if _FENCE_OPEN_RE.match(line.strip()):
            continue
        line = _LABEL_RE.sub("", _LIST_RE.sub("", line, count=1), count=1)
        # a table's cells — also a plain-text one laid side by side (r30)
        cells = line.strip().strip("|").split("|")
        for c in cells:
            if not _words(c):
                continue
            sp = _spans(c) if spans_ok else []
            if sp and len([w for w in _words(_DQ_RE.sub(" ", c)) if w not in _STOPWORDS]) < MIN_CONTENT_WORDS:
                out.extend(sp)
            else:
                out.append(c)
    return out


def quotes(text: str) -> List[str]:
    """The quotes in a markdown text: the lines of ```text blocks and of
    blockquotes (:func:`_units`), and double-quoted spans of MIN_QUOTE_WORDS+
    words outside code."""
    blocks, rest = _split_fences(text)
    found = [u for b in blocks for u in _units(b, False)]
    plain, para = [], []
    for line in rest.split("\n"):
        s = line.lstrip()
        if s.startswith(">"):
            para.append(s.lstrip("> \t"))      # r29: "> ```markdown" is dropped by _units
            continue
        plain.append(line)
    found.extend(_units("\n".join(para), True))
    found.extend(_spans("\n".join(plain)))
    return found


def _pieces(quote: str) -> List[str]:
    """The quote's pieces between ellipses / [brackets]; none unless one
    has MIN_PIECE_WORDS+ words. One-word pieces are kept and matched in
    order too (r29: "my identity is … suffering" passed with a word the
    transcript never had)."""
    ps = [w for w in (_words(p) for p in _SPLIT_RE.split(quote)) if w]
    if not any(len(w) >= MIN_PIECE_WORDS for w in ps):
        return []
    return [" " + " ".join(w) + " " for w in ps]


def _found(quote: str, joined_sources: List[str]) -> bool:
    """Every piece stands in ONE source, in the quote's order (r27/r28)."""
    return _in_order(_pieces(quote), joined_sources)


def _in_order(pieces: List[str], joined_sources: List[str]) -> bool:
    if not pieces:
        return False
    for s in joined_sources:
        at = 0
        for p in pieces:
            at = s.find(p, at)
            if at < 0:
                break
            at += len(p) - 1
        else:
            return True
    return False


def _lcs(a: List[str], b: List[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def _content_pairs(ws: List[str]) -> set:
    return {(a, b) for a, b in zip(ws, ws[1:]) if a not in _STOPWORDS and b not in _STOPWORDS}


def _responses(text: str) -> str:
    """The model's answers in a probe transcript (its ```text fences) — r29:
    judging the prompts and the header too refused the project's own
    hypothesis, quoted word for word, as a rewording of a prompt."""
    blocks, _rest = _split_fences(str(text or ""))
    return "\n\n".join(blocks) if blocks else str(text or "")


def _prompts(text: str) -> str:
    """Everything of a transcript but the answers: the planner's prompts."""
    blocks, rest = _split_fences(str(text or ""))
    return rest if blocks else ""


class _OutOfTime(Exception):
    pass


def _lower_1to1(text: str) -> str:
    """lower() that never changes a length (r29: "İ" lowers to two
    characters and shifted every span after it)."""
    return "".join(c if len(c.lower()) != 1 else c.lower() for c in text)


class _Transcript:
    """A kept transcript's ANSWERS, tokenised once per check: words + spans."""

    def __init__(self, text: str, answers: bool = True):
        self.text = unicodedata.normalize("NFKC", _responses(text) if answers else str(text or ""))
        low = re.sub(r"[‘’ʼ`´]", "'", _lower_1to1(self.text))
        found = [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(low)]
        self.words = [w for w, _, _ in found]
        self.spans = [(a, b) for _, a, b in found]
        self.vocab = set(self.words)

    def passage(self, i: int, n: int) -> str:
        """The transcript's own text for words i..i+n-1 (r28: the shown passage
        was a list of lower-case tokens)."""
        j = min(i + n, len(self.words)) - 1
        return self.text[self.spans[i][0]:self.spans[j][1]] if 0 <= i <= j else ""


def _reworded(quote: str, transcripts: List[_Transcript], deadline: float = math.inf) -> Optional[str]:
    """The transcript passage ``quote`` rewords, or None."""
    return _closest(quote, transcripts, deadline)[1]


def _closest(quote: str, transcripts: List[_Transcript],
             deadline: float = math.inf) -> Tuple[float, Optional[str]]:
    """(share of the quote's content words matched, passage) of the stretch
    ``quote`` rewords most closely, or (0.0, None). A stretch of
    len(quote)+SLACK words qualifies when ≥ NEAR_RATIO of the quote's content
    words stand in it in order, with ≥ 3 matched or a content-word pair side
    by side in both (r28: a bag of all words over a quote-length window
    refused "I think, therefore I am" on stopwords)."""
    qw = _words(_SPLIT_RE.sub(" ", quote))
    qc = [w for w in qw if w not in _STOPWORDS]
    n = len(qc)
    if n < MIN_CONTENT_WORDS:
        return 0.0, None
    need, qset, qpairs = Counter(qc), set(qc), _content_pairs(qw)
    qwords = set(qw)
    thr = max(2, math.ceil(NEAR_RATIO * n - 1e-9))
    best: Tuple[float, Optional[str]] = (0.0, None)
    for tr in transcripts:
        sw = tr.words
        if len(qset & tr.vocab) < 2:
            continue
        span = min(len(qw) + SLACK, len(sw))
        win = Counter(sw[:span])
        ov = sum((win & need).values())
        i = 0
        while True:
            if ov >= thr:
                seg = sw[i:i + span]
                hit = _lcs(qc, [w for w in seg if w not in _STOPWORDS])
                if hit / n > best[0] and hit / n >= NEAR_RATIO and (hit >= 3 or qpairs & _content_pairs(seg)):
                    # r30: shown from its first matched word, not the window's start
                    k = next((k for k, w in enumerate(seg) if w in qwords), 0)
                    best = (hit / n, tr.passage(i + k, span - k))
                    if hit == n:
                        return best
            if i + span >= len(sw):
                break
            if not i & 255 and time.monotonic() > deadline:
                raise _OutOfTime
            a, b = sw[i + span], sw[i]
            win[a] += 1
            if win[a] <= need[a]:
                ov += 1
            if win[b] <= need[b]:
                ov -= 1
            win[b] -= 1
            i += 1
    return best


def _judged_units(wholes: List[str], srcs: List[str], n_trs: int, out: list,
                  deadline: float = math.inf) -> Iterable[str]:
    """The units of the unfound quotes to judge: a long quote by its
    sentences (r30) — but one whose sentences are each verbatim, and not one
    after another, is a splice (r30: the sentence split let the live
    "The cascade does not produce…" splice pass) and is flagged here. The
    clock bounds these verbatim scans too (r33: 1 MB pasted verbatim against
    40 transcripts took 11 s, all in str.find)."""
    for whole in wholes:
        if time.monotonic() > deadline:
            raise _OutOfTime
        if _found(whole, srcs):
            continue
        parts = _chunks(whole)
        if len(parts) > 1 and all(_found_by(p, srcs[:n_trs], deadline) for p in parts):
            out.append((whole, "", "its sentences are the transcript's, but not one after another there "
                                   "— mark each cut with …"))
            continue
        for p in parts:
            if not _found_by(p, srcs, deadline):
                yield p


def _found_by(quote: str, srcs: List[str], deadline: float) -> bool:
    """:func:`_found` under the clock (r34: one 850 KB line of verbatim
    chunks took 8 s with the clock read only between whole quotes)."""
    if time.monotonic() > deadline:
        raise _OutOfTime
    return _found(quote, srcs)


def misquoted(text: str, transcripts: List[str], others: Iterable[str] = ()) -> List[Tuple[str, str, str]]:
    """(quote, the transcript passage it rewords, or "" and a note) for each
    quote of ``text`` found in no source but misquoting a transcript."""
    others = [o for o in others if o]
    srcs = [_joined(s) for s in list(transcripts) + others]
    trs = [_Transcript(t) for t in transcripts]
    # r30: the planner's words — prompts, the task, the project's goal. An
    # answer that echoes the hypothesis made a quote of the hypothesis read
    # as a rewording of the ANSWER; one close to these is the planner's
    planner = [_Transcript(t, answers=False) for t in [_prompts(t) for t in transcripts] + others if t]
    out, t0 = [], time.monotonic()
    deadline = t0 + _MAX_JUDGE_S
    try:
        return _misquoted(text, trs, srcs, planner, out, t0, deadline)
    except _OutOfTime:
        logger.warning("probe_transcripts: stopped checking quotes after %.1f s", time.monotonic() - t0)
        return out


def _misquoted(text, trs, srcs, planner, out, t0, deadline):
    judged = 0
    for q in _judged_units(quotes(text), srcs, len(trs), out, deadline):
        if len([w for w in _words(q) if w not in _STOPWORDS]) < MIN_CONTENT_WORDS:
            continue                           # r31: a number or a heading cell is never judged
        judged += 1
        if judged > _MAX_JUDGED:                # the clock: _judged_units / _closest
            logger.warning("probe_transcripts: stopped judging quotes after %d (%.1f s)",
                           judged - 1, time.monotonic() - t0)
            break
        pieces = _pieces(q)
        if len(pieces) > 1 and any(all(p in s for p in pieces) for s in srcs[:len(trs)]):
            # every piece verbatim in one transcript, but not in this order
            out.append((q, "", "its pieces are the transcript's words in another order — keep its order"))
            continue
        if _in_order([p for p in pieces if p.count(" ") > MIN_PIECE_WORDS], srcs[:len(trs)]):
            # r29: the long pieces are verbatim, a one-word piece is not there
            out.append((q, "", "a single word set apart by … is not in the transcript there"))
            continue
        try:
            near, passage = _closest(q, trs, deadline)
            # r30: as close to the planner's words as to the answer → the
            # planner's (an answer echoing the hypothesis); closer to the answer → judged
            if passage and _closest(q, planner, deadline)[0] < near:
                out.append((q, passage, ""))
        except _OutOfTime:
            logger.warning("probe_transcripts: stopped judging quotes after %.1f s (quote %d)",
                           time.monotonic() - t0, judged)
            break
    return out


def quote_problem(memory_dir, project_id: str, text: str, rel: str,
                  description: str = "", project_text: str = "") -> Optional[str]:
    """What looks misquoted in deliverable ``rel`` — it cites a kept
    transcript and rewords a quote of it — else None (advisory). ``description`` (the task's STORED
    description) and ``project_text`` (the project's title and goal — r29:
    the hypothesis is quoted) may be quoted too."""
    kept = recorded(memory_dir, project_id)
    named = sorted({r for r, _ in kept if cites(text, r)})
    if not named:
        return None
    bad = misquoted(text, [t for _, t in kept], [description, project_text])
    if not bad:
        return None

    def _short(s: str, n: int = 150) -> str:
        s = " ".join(s.split())
        return s[:n] + ("…" if len(s) > n else "")
    shown = "; ".join(f"«{_short(q)}» — " + (f"the transcript says: «{_short(w, 220)}»" if w else note)
                      for q, w, note in bad[:_MAX_LISTED])
    more = f" (+{len(bad) - _MAX_LISTED} more)" if len(bad) > _MAX_LISTED else ""
    cited = ", ".join(named[:3]) + (f" (+{len(named) - 3} more)" if len(named) > 3 else "")
    return (f"the deliverable {rel} cites the probe transcript {cited} but {len(bad)} quote(s) "
            f"in it reword the model's answer: {shown}{more}. In {rel}, copy each quote word for word from "
            "the transcript (formatting and punctuation may differ, the words may not; mark a cut with …), "
            "or remove its quote marks / '>' and say it as your own summary")


__all__ = ["record", "recorded", "quotes", "misquoted", "quote_problem", "cites", "DIRNAME"]
