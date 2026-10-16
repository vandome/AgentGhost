"""§4PK: the PRE-REGISTERED trigger v4, byte-for-byte except this docstring and
the path line (sha256 of the original: 94f9185e…3e9db). Read-only: the
differential test holds core/failure_census.unbacked_no_tool to it (§4PL;
the §4PK hold itself was rolled back)."""
import re, sys
from ghost_agent.memory.episodes import is_moving_target_question, _QUESTION_SHAPE_RE
FENCE = re.compile(r"```.*?```", re.S); INLINE = re.compile(r"`[^`\n]+`")
SYS_PREFIX = re.compile(r"^\*\*Background activity while you were away:\*\*.*?(?:\n\n|\Z)", re.S)
FIN = re.compile(r"\b(?:ETF|ETFs|fund|stock|stocks|shares|ticker|ISIN|UCITS|index|portfolio|broker|dividend)\b", re.I)
SPEC = [
  ("figure", re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w]|\.\d)|(?<![\w.:])\d+\.\d+(?![\w:]|\.\d)|(?<![\w.])\d+(?:\.\d+)?\s?%")),
  ("money", re.compile(r"[$€£]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:USD|EUR|GBP|dollars|euros)\b")),
  ("version", re.compile(r"(?<![\w.])v?\d+\.\d+\.\d+(?![\w]|\.\d)|\bversion\s+\d+(?:\.\d+)*")),
  ("ident", re.compile(r"(?<![\w./-])[a-z]+_[a-z0-9_]+\(\)|(?<![\w./-])[a-z][a-z0-9]*_[a-z0-9_]+(?![\w./-])|(?<![\w.])[A-Za-z_]\w*\(\)")),
]
TICKER = re.compile(r"(?<![\w-])[A-Z]{3,5}(?![\w-])")
CLAIM = re.compile(r"\b(?:I(?:'ve| have)? just (?:checked|looked|ran|run|pulled|searched|verified|tested|generated|fetched)|"
                   r"(?:no|nothing has) change[ds]? since (?:the )?last (?:check|time)|"
                   r"I (?:checked|verified|tested|looked it up|searched)\b|the (?:data|results?) I (?:just )?(?:pulled|fetched|got)|"
                   r"as of (?:today|right now|this morning)|"
                   r"I (?:just )?(?:ran|executed|pulled|fetched|looked up|checked) (?:it|the|this|that|a|your)\b)", re.I)
LIVE = re.compile(r"\b(?:weather|temperature|forecast|right now|at the moment|today's news|headlines|score (?:of|in) the)\b", re.I)
LITERAL = re.compile(r"\b(?:reply|respond|answer|say|repeat|echo)\b[^.\n]{0,30}\b(?:exactly|verbatim|only with|nothing else|word for word)\b", re.I)
CREATIVE = re.compile(r"\b(?:write|compose|draft|make up|invent|imagine)\b.{0,40}\b(?:poem|story|song|joke|haiku|limerick|fiction|lyrics|toast|speech)\b", re.I)
SMALLTALK = re.compile(r"^\W*(?:hi|hello|hey|good (?:morning|evening|afternoon)|yo)\b|\bhow (?:are|r) (?:you|u)\b|how(?:'s| is| are) (?:things|it going|you doing|life)|\bhow do you feel\b|\bfeeling today\b|^\W*(?:thanks|thank you|very good|great|nice|cool|ok|okay)\b[^?]{0,40}$", re.I)
def model_text(reply):
    t = SYS_PREFIX.sub("", reply or ""); t = FENCE.sub(" ", t); return INLINE.sub(" ", t)
def norm(s): return re.sub(r"\s+", " ", (s or "").lower())
def trigger(reply, user_texts, context_text, tools_ran, tool_names=()):
    if tools_ran: return None
    req = user_texts[-1] if user_texts else ""
    if CREATIVE.search(req) or SMALLTALK.search(req) or LITERAL.search(req): return None
    t = model_text(reply)
    known = norm(" ".join(user_texts) + " " + context_text + " " + " ".join(tool_names))
    unsup = set()
    for kind, rx in SPEC:
        for m in rx.finditer(t):
            v = m.group(0).strip().rstrip(".,;:")
            if not v: continue
            if norm(v) not in known and norm(v.rstrip("()")) not in known: unsup.add(v)
    if FIN.search(t) or FIN.search(req):
        for m in TICKER.finditer(t):
            if norm(m.group(0)) not in known: unsup.add(m.group(0))
    claims = [m.group(0) for m in CLAIM.finditer(t)]
    moving = is_moving_target_question(req) or bool(LIVE.search(req) and _QUESTION_SHAPE_RE.search(req))
    if claims or unsup or moving:
        return {"unsupported": sorted(unsup), "claims": claims, "moving": moving}
    return None
