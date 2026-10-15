"""§4PD mutation battery: search.py / darkweb_search.py / helpers.py / agent.py mutated IN PLACE,
one mutant at a time, the search-family test set run after each; restored at the end.
Usage: python scripts/mutate_4pd.py [all|rerun]   (do not edit the mutated files while it runs)."""
import os, shutil, subprocess, sys, signal, filecmp, glob, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); PY = sys.executable
S = "src/ghost_agent/tools/search.py"; D = "src/ghost_agent/tools/darkweb_search.py"
H = "src/ghost_agent/utils/helpers.py"; AG = "src/ghost_agent/core/agent.py"
def _tests():
    pat = re.compile(r"tools\.search|tools\.darkweb|darkweb|deep_research|fact_check|_clean_for_cpp|_filter_junk|_sanitize_query|helper_fetch_url_content|link_grounding|_reformulate_query|prune_off_topic|research_outcome|FetchError")
    return sorted(os.path.relpath(t, ROOT) for t in glob.glob(os.path.join(ROOT, "tests", "test_*.py")) if pat.search(open(t, errors="ignore").read()))
M = [
    ("CONTROL no-op (must SURVIVE)", S, "def omitted_hosts_note(hosts) -> str:", "# ctl\ndef omitted_hosts_note(hosts) -> str:"),
    ("CONTROL known-bad (must DIE): junk matched as a substring again", S,
     "        if host == j or host.endswith(\".\" + j):\n            return j", "        if j in url:\n            return j"),
    ("M1 omitted hosts never named", S, "            if omitted is not None and j not in omitted:\n                omitted.append(j)", "            pass"),
    ("M2 boolean strip case-insensitive and boundary-only again", S,
     "_BOOLEAN_OPERATOR_RE = re.compile(r'(?<=\\S\\s)(?:OR|AND|or|and)(?=\\s+[^\\s-])')", "_BOOLEAN_OPERATOR_RE = re.compile(r'(?<!\\w)(?:or|and)(?!\\w)', re.IGNORECASE)"),
    ("M3 adjacent-operator strip dropped (dangling or/and left)", S, "    q = _ADJACENT_BOOLEAN_RE.sub(' ', q)              # or/and beside a removed operator\n", ""),
    ("M43 adjacent rule widened to a leading-operator strip", S, "_ADJACENT_BOOLEAN_RE = re.compile(r'(?:(?<!\\w)(?:or|and)\\s+)*\\x00(?:\\s+(?:or|and)(?!\\w))*', re.IGNORECASE)", "_ADJACENT_BOOLEAN_RE = re.compile(r'^\\s*(?:or|and)\\s+|\\x00', re.IGNORECASE)"),
    ("M4 operator note never built", S, "    if _BOOLEAN_OPERATOR_RE.search(q):\n        found.append(\"OR/AND\")", "    pass"),
    ("M5 brace rewrite regardless of the grammar", S, "        if not _grammar_enabled():\n            return text", "        pass"),
    ("M6 prune note dropped", S, "    if n <= 0 or low:\n        return kept, low, \"\"", "    if True:\n        return kept, low, \"\""),
    ("M7 prune log without titles", S, "                   + (\" — hidden: \" + \" | \".join(_dropped[:3]) if _dropped else \"\"),", "                   ,"),
    ("M8 greek stem rule dropped", S, "        if len(t) >= 5 and _GREEK_RE.search(t):", "        if False:"),
    ("M9 ddgs executor never sized", S, "        if _cur < _RACE_POOL._max_workers:\n            _DDGS_cls.threads = _RACE_POOL._max_workers", "        pass"),
    ("M10 wave stats: omitted hosts not recorded", S, "                    if stats is not None:\n                        stats[\"omitted_hosts\"] = list(_omitted_by_engine.get(engine, []))\n", ""),
    ("M11 wave stats: failure tallies not recorded", S, "        if stats is not None:\n            _tally = stats.setdefault(\"cats\", {})", "        if False:\n            _tally = stats.setdefault(\"cats\", {})"),
    ("M12 transport failure blamed on the query", S, "    if _transport and _transport >= _empty:", "    if False:"),
    ("M13 cache hit unmarked", S, "        return _cached + \"\\n\\n[cached: the same query ran within the last 5 minutes — these are its results again, not a new search]\"", "        return _cached"),
    ("M14 low batch cached again", S, "            if not _low:                                  # §4PD: a low-relevance batch is not an answer to repeat for 5 min\n                _cache_put(_cache_key, clean_output)", "            _cache_put(_cache_key, clean_output)"),
    ("M15 wiki future exception unretrieved", S, "        _wiki_task.add_done_callback(lambda t: (t.exception() if not t.cancelled() else None))\n", ""),
    ("M16 neutraliser quotes nothing", S, "    t = _REPORT_GRAMMAR_RE.sub(\"> \", t)\n    # a block whose FIRST line", "    # a block whose FIRST line"),
    ("M41 a leading Error line of content not quoted", S, "    return _LEADING_ERROR_RE.sub(\"> \", t, count=1)", "    return t"),
    ("M42 raw page text spliced without the neutraliser", S, "        raw_preview = neutralise_report_grammar(raw_preview)        # §4PD: page text is content (every return site)\n", ""),
    ("M17 neutraliser keeps the extractor marker", S, "    t = _EXTRACT_MARK_RE.sub(\"[extracted-facts marker in page text]\", t)\n", ""),
    ("M18 neutraliser skipped on the extractor output", S, "            preview = \"[EDGE EXTRACTED FACTS]:\\n\" + neutralise_report_grammar(\n                (summary_data[\"choices\"][0][\"message\"].get(\"content\") or \"\").strip())", "            preview = \"[EDGE EXTRACTED FACTS]:\\n\" + (summary_data[\"choices\"][0][\"message\"].get(\"content\") or \"\").strip()"),
    ("M19 fetch failure judged by the first word again", S, "                if isinstance(res, str) and not isinstance(res, _FetchError):\n                    return res", "                if isinstance(res, str) and not res.lstrip().startswith(\"Error\"):\n                    return res"),
    ("M20 consumer judges by the first word again", S, "        if isinstance(text, _FetchError):                  # §4PD: the fetcher's type, not the page's first word", "        if isinstance(text, str) and text.startswith(\"Error:\"):"),
    ("M21 429 definitive again", S, "    if \"received status 4\" in e and \"received status 429\" not in e:", "    if \"received status 4\" in e:"),
    ("M22 duplicate hrefs fill slots again", S, "        if _k and _k in _seen:\n            continue", "        if False:\n            continue"),
    ("M23 deep_research gives up without reformulating", S, "        for _ri, _rq in enumerate(_refs):\n            pretty_log(\"Search Retry\", f\"deep_research: reformulated ‹{truncate_query(_rq)}›\", icon=Icons.TOOL_SEARCH)", "        for _ri, _rq in enumerate([]):\n            pretty_log(\"Search Retry\", f\"deep_research: reformulated ‹{truncate_query(_rq)}›\", icon=Icons.TOOL_SEARCH)"),
    ("M24 reformulated query not named in the report", S, "    _text = (f\"--- DEEP RESEARCH RESULT ---\\n{_search_note}{full_report}\\n\\n\"", "    _text = (f\"--- DEEP RESEARCH RESULT ---\\n{full_report}\\n\\n\""),
    ("M25 fact_check partial arm keyed on the status alone", S, "                   or _research_reason in (\"research_sources_partial\", \"research_no_relevant_facts\"))", "                   or False)"),
    ("M26 fact_check verify call unbounded", S, "            timeout=_FACTCHECK_VERIFY_S, total_budget=_FACTCHECK_VERIFY_S), _FACTCHECK_VERIFY_S + 5.0)", "            ), 10 ** 6)"),
    ("M27 fact_check off the batch wall", AG, "                               \"fact_check\"})                                   # §4PD", "                               })"),
    ("M28 names get the how-to prefix again", S, "    elif _namelike:", "    elif False:"),
    ("M29 month-day not held", S, "    broader = _MONTH_DAY_RE.sub(_hold, broader)        # §4PD: \"October 14\" is a date, not a number to drop\n", ""),
    ("M30 image bodies read as text", H, "                             or (isinstance(content_type, str)\n                                 and content_type.strip().startswith((\"image/\", \"video/\", \"audio/\", \"font/\")))  # §4PD", ""),
    ("M31 onion parser back on the loop", D, "                    parsed = await asyncio.to_thread(_parse_onion_results, body, exclude_hosts)  # §4PD: off the loop", "                    parsed = _parse_onion_results(body, exclude_hosts)"),
    ("M32 onion parser unbounded", D, "            if _anchors > _PARSE_MAX_ANCHORS:\n                break", "            pass"),
    ("M33 block text recomputed per anchor", D, "                if _bid not in _block_text:\n                    _block_text[_bid] = \" \".join(block.get_text(\" \", strip=True).split())[:4000]", "                _block_text[_bid] = \" \".join(block.get_text(\" \", strip=True).split())[:4000]"),
    ("M34 onion redirect to another host read as the onion", D, "    if _final and _redirected_away(url, _final):", "    if False:"),
    ("M35 result rows can forge rows again", D, "    t = \" \".join(str(text or \"\").split()).replace(\"[\", \"(\").replace(\"]\", \")\")\n    return re.sub(r\"#{2,}\", \"#\", t)", "    return str(text or \"\")"),
    ("M36 no-hits trips the breaker again", D, "        if _res or not _diag[\"nohits\"]:\n            _breaker_record(engine[\"name\"], bool(_res))", "        _breaker_record(engine[\"name\"], bool(_res))"),
    ("M37 form token fetch without a clock", D, "        status, body = await asyncio.wait_for(                 # §4PD: curl's timeout is a low-speed limit, not a clock\n            _fetch_raw_html(token_url, proxy, _FORM_TOKEN_TIMEOUT), timeout=_FORM_TOKEN_TIMEOUT + 2.0)", "        status, body = await _fetch_raw_html(token_url, proxy, _FORM_TOKEN_TIMEOUT)"),
    ("M38 v2 onions harvested again", D, "    r\"https?://[a-z2-7]{56}\\.onion(?:/[^\\s\\\"'<>]*)?\",", "    r\"https?://(?:[a-z2-7]{56}|[a-z2-7]{16})\\.onion(?:/[^\\s\\\"'<>]*)?\","),
    ("M39 darkweb consumer judges by the first word again", D, "        if isinstance(text, _FetchError):                      # §4PD: the fetcher's TYPE, not the page's first word", "        if isinstance(text, str) and text.startswith(\"Error:\"):"),
    ("M40 max_results not coerced", D, "        max_results = max(1, min(50, int(max_results)))          #       raised inside gather() and read as 'zero results'", "        pass"),
]
def run_tests(tests):
    env = dict(os.environ, PYTHONPATH="src", GHOST_API_KEY=open(os.path.expanduser("~/Data/AI/.ghost_api_key")).read().strip()); env.pop("FORCE_COLOR", None)
    r = subprocess.run([PY, "-m", "pytest", *tests, "-q", "--no-header", "-p", "no:cacheprovider", "-x", "-n", "8", "--dist", "loadfile"], capture_output=True, text=True, env=env, cwd=ROOT)
    return r.returncode, (r.stdout.strip().splitlines() or ["?"])[-1]
def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    only = set(sys.argv[2:])
    batch = M if which == "all" else [m for m in M if m[0].startswith("CONTROL") or any(m[0].startswith(o + " ") for o in only)]
    tests = _tests(); files = sorted({m[1] for m in batch}); backups = {f: f + ".4pd.bak" for f in files}
    for f, b in backups.items(): shutil.copy2(os.path.join(ROOT, f), os.path.join(ROOT, b))
    def restore(*_):
        for f, b in backups.items(): shutil.copy2(os.path.join(ROOT, b), os.path.join(ROOT, f))
        subprocess.run(["find", os.path.join(ROOT, "src"), "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"], capture_output=True)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP): signal.signal(sig, lambda *a: (restore(), sys.exit(130)))
    killed = survived = 0
    try:
        for label, f, old, new in batch:
            restore(); path = os.path.join(ROOT, f); src = open(path).read()
            if src.count(old) != 1: print(f"SKIP   {label}: anchor count {src.count(old)}", flush=True); continue
            open(path, "w").write(src.replace(old, new))
            if subprocess.run([PY, "-m", "py_compile", path], capture_output=True).returncode != 0: print(f"NOCOMP {label}", flush=True); continue
            rc, tail = run_tests(tests); status = "SURVIVED" if rc == 0 else "KILLED"; killed += rc != 0; survived += rc == 0
            print(f"{status:8} {label} — {tail}", flush=True)
    finally:
        restore(); ok = all(filecmp.cmp(os.path.join(ROOT, f), os.path.join(ROOT, b), shallow=False) for f, b in backups.items())
        for b in backups.values(): os.remove(os.path.join(ROOT, b))
        print(f"RESTORED identical={ok} killed={killed} survived={survived}", flush=True)
if __name__ == "__main__": main()
