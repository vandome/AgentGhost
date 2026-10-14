"""§4OU step 2 (2026-10-14): the quote checker's false "missing" verdicts.
Labelled: every mark since quote checking became the default (52, of which
10 right, 0 of 10 on real traffic; system/eval/4ou/labels.json). Three
classes fixed here; the figure rule stays strict (loosening it cut false
marks 20 → 6 but accepted 26 more of 101 claims with a figure changed —
an operator decision)."""
from __future__ import annotations

import pytest

from ghost_agent.core import quote_bound as QB

HEALTH = QB.evidence_sources(["--- SYSTEM HEALTH DIAGNOSTICS ---\nTor: Connected (Anonymous)\n"
                              "Agent Internals: LLM=Active, Memory=Active, Idle Scheduler=Running"])
JSON = QB.evidence_sources(['{"id": "9009ff89d8bb", "status": "RELEASED", "released_at": "2026-08-11 12:49:30"}'])
FORUM = QB.evidence_sources(["its different,i believe tnr is probably quite close to Chris Kenner's Torn Asunder"])
MODELS = QB.evidence_sources(["Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive... Qwen3.6-35B-A3B uncensored"])


# ── label-value forms: "LLM: Active" is "LLM=Active" ──────────────────

@pytest.mark.parametrize("q", ["LLM: Active", "LLM=Active", "Idle Scheduler: Running", "Memory: Active"])
def test_a_label_value_quote_is_found_in_either_spelling(q):
    assert QB.quote_found(q, HEALTH)


@pytest.mark.parametrize("q", ["LLM: Inactive", "GPU: Active", "Bus: Active", "LLM Active"])
def test_a_label_value_quote_must_still_match(q):
    assert not QB.quote_found(q, HEALTH)


# ── short quotes ───────────────────────────────────────────────────────

@pytest.mark.parametrize("q,found", [
    ("Connected (Anonymous)", True),                       # two words, long enough to name a thing
    ("Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive", True),   # one exact identifier
    ("it is", False), ("is the", False), ("Connected", False),  # "September" is not a source
    ("Anonymous)", False),
    ("Qwen3.6-35B-A3B-Uncensored-HauhauCS-Passive", False),
    ("uncensored", False),                                 # one plain word, however long
])
def test_short_quotes(q, found):
    srcs = HEALTH + MODELS
    assert QB.quote_found(q, srcs) is found


def test_an_identifier_must_stand_at_word_edges():
    assert not QB.quote_found("Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggr", MODELS)


# ── a JSON pair is one quote, not two ──────────────────────────────────

@pytest.mark.parametrize("q", ['"status": "RELEASED"', '"released_at": "2026-08-11 12:49:30"'])
def test_a_json_pair_is_one_quote(q):
    assert QB.quote_found(q, JSON)


@pytest.mark.parametrize("q", ['"status": "FAILED"', '"status": "RELEASED", "id": "0000"'])
def test_a_json_pair_must_still_match(q):
    assert not QB.quote_found(q, JSON)


def test_several_quotes_in_one_marker_are_still_each_checked():
    srcs = QB.evidence_sources(["Temp: 19.0°C\nWind: 5.0 km/h\nHumidity: 83%"])
    assert QB.quote_found('"Temp: 19.0°C", "Humidity: 83%"', srcs)
    assert not QB.quote_found('"Temp: 19.0°C", "Humidity: 84%"', srcs)


# ── a source's missing space after a comma ─────────────────────────────

def test_a_missing_space_after_a_comma_in_the_source():
    assert QB.quote_found("its different, I believe tnr is probably quite close", FORUM)


def test_numbers_keep_their_commas():
    srcs = QB.evidence_sources(["the fleet has 1,067 aircraft and 1,5 ratio"])
    assert QB.quote_found("the fleet has 1,067 aircraft", srcs)
    assert not QB.quote_found("the fleet has 1,068 aircraft", srcs)


# ── Greek month names are months ───────────────────────────────────────

@pytest.mark.parametrize("text,want", [
    ("Την 1η Νοεμβρίου 2025 φτάνει", {"1", "11", "2025"}),
    ("στις 6 Οκτωβρίου 2026", {"6", "10", "2026"}),
    ("από 15 Μαΐου", {"15", "5"}),
    ("ο Οκτώβριος ήταν ζεστός", set()),                    # no day: not a date
])
def test_greek_month_names_count_as_their_number(text, want):
    assert QB._figures(text) == want


def test_a_greek_date_backs_an_english_one():
    q = "Την 1η Νοεμβρίου 2025 φτάνει στην Αθήνα"
    src = QB.evidence_sources([q + " με ένα πλατύ χαμόγελο"])
    assert QB.marker_ok(q, "arrives in Athens on 1 November 2025", src) == (True, True)
    assert QB.marker_ok(q, "arrives in Athens on 1 December 2025", src) == (True, False)


# ── the figure rule is unchanged ───────────────────────────────────────

def test_the_figure_rule_stays_strict():
    src = QB.evidence_sources(["nginx-1.30.5 stable and nginx-1.31.6 mainline versions have been released, "
                               "with fixes for (CVE-2026-90439)."])
    q = "nginx-1.30.5 stable and nginx-1.31.6 mainline versions have been released"
    assert QB.marker_ok(q, "the latest stable is 1.30.5", src) == (True, True)
    assert QB.marker_ok(q, "fixes CVE-2026-90439", src) == (True, False)     # beside the quote is not IN it
    assert QB.marker_ok("nothing like this", "1.30.5", src) == (False, False)


def test_both_checkers_use_the_same_verdict():
    """The inline rewrite and the streamed closing line read one function."""
    from tests import pin_ast as P
    assert P.calls(P.tree(QB.apply_quotes), "marker_ok")
    assert P.calls(P.tree(QB.MarkerFilter.failed_claims), "marker_ok")
    assert not P.calls(P.tree(QB.apply_quotes), "quote_found")


# ── pins from the mutation battery ─────────────────────────────────────

def test_a_colon_without_its_space_is_the_same_label():
    srcs = QB.evidence_sources(["Tor:Connected (Anonymous) · listing LSE:EIMI"])
    assert QB.quote_found("Tor: Connected (Anonymous)", srcs)
    assert QB.quote_found("listing LSE: EIMI", srcs)


@pytest.mark.parametrize("raw", ["3,14", "1,5 ratio", "x;y", "10:30"])
def test_figures_keep_their_punctuation(raw):
    assert QB._norm(raw) == raw.lower().replace("x;y", "x; y")


@pytest.mark.parametrize("q", ["A3B", "35B-A3B", "internationalization"])
def test_a_short_or_plain_single_token_is_not_a_source(q):
    srcs = QB.evidence_sources([f"text {q} text"])
    assert not QB.quote_found(q, srcs)


def test_every_quote_of_a_multi_marker_must_stand():
    srcs = QB.evidence_sources(["Temp: 19.0°C\nWind: 5.0 km/h\nHumidity: 83%"])
    assert not QB.quote_found('"Temp: 20.0°C", "Humidity: 83%"', srcs)


# ── r1 review: no new false passes ─────────────────────────────────────

@pytest.mark.parametrize("q", ["is headquartered", "executive officer", "state-of-the-art", "q: a",
                               "status: ok"])
def test_generic_short_quotes_back_nothing(q):
    srcs = QB.evidence_sources(["Acme Corp is headquartered in Munich; the chief executive officer said "
                                "a state-of-the-art plant opens. FAQ q: a. status: ok"])
    assert not QB.quote_found(q, srcs)


@pytest.mark.parametrize("q,src", [
    ("qwen3-coder-30b", "pull qwen3-coder-30b-a3b-instruct today"),
    ("coder-30b-a3b-instruct", "pull qwen3-coder-30b-a3b-instruct today"),
    ("requests-2.31", "install types-requests-2.31.0.6 now"),
])
def test_an_identifier_is_not_part_of_a_longer_one(q, src):
    assert not QB.quote_found(q, QB.evidence_sources([src]))


def test_a_whole_identifier_still_stands():
    srcs = QB.evidence_sources(["pull qwen3-coder-30b-a3b-instruct today, or qwen3-coder-30b."])
    assert QB.quote_found("qwen3-coder-30b-a3b-instruct", srcs)
    assert QB.quote_found("qwen3-coder-30b", srcs)            # the second, whole occurrence


@pytest.mark.parametrize("text", ["5 μάρτυρες", "2 μαιευτήρες", "3 Ιουλίες", "1 Αυγουστίνος"])
def test_greek_words_that_start_like_months_are_not_months(text):
    assert QB._figures(text) == {text.split()[0]}


def test_a_witness_count_does_not_back_a_date():
    q = "5 μάρτυρες είδαν το τροχαίο στην Αθήνα"
    assert QB.marker_ok(q, "Το τροχαίο έγινε στις 5 Μαρτίου", QB.evidence_sources([q])) == (True, False)


def test_a_namespaced_name_is_still_the_whole_name():
    srcs = QB.evidence_sources(["3. fredrezones55/qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive... more"])
    assert QB.quote_found("Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive", srcs)
    assert not QB.quote_found("a3b-uncensored-hauhaucs-aggressive", srcs)


def test_mutation_pins_for_short_quotes_and_greek_forms():
    srcs = QB.evidence_sources(["Acme Corp Is Headquartered in Munich. record id: abcd here"])
    assert not QB.quote_found("Is Headquartered", srcs)          # a capital is not enough: "is" is a function word
    assert not QB.quote_found("id: abcd", srcs)                  # a label pair this short is no source
    assert QB._figures("3 Αυγουστούπολη") == {"3"}               # a month form that runs on is a word


# ── r2 review ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("q,src", [
    ("max_tokens: 4096", "set max_tokens >= 4096"),
    ("Status: Active", "if status >= active then"),
    ("mode: strict", "default => strict"),
    ("default: strict", "default => strict"),
    ("x: y", "if x == y"),
])
def test_comparison_operators_are_not_labels(q, src):
    assert not QB.quote_found(q, QB.evidence_sources([src]))


@pytest.mark.parametrize("q,src", [
    ("enabled: true", "server:\n enabled: false\nclient:\n enabled: true"),
    ("Status: Active", "nginx Status: Down … postgres Status: Active"),
    ('"status": "RELEASED"', '[{"id": "a", "status": "FAILED"}, {"id": "b", "status": "RELEASED"}]'),
])
def test_a_label_pair_with_no_figure_must_name_one_row(q, src):
    assert not QB.quote_found(q, QB.evidence_sources([src]))


def test_a_label_that_appears_once_still_stands():
    assert QB.quote_found("LLM: Active", HEALTH)
    assert QB.quote_found('"status": "RELEASED"', QB.evidence_sources(['{"id": "b", "status": "RELEASED"}']))


@pytest.mark.parametrize("q", ["Executive Officer", "Customer Service", "General Availability",
                               "Open-Source Software", "Πρωθυπουργός Μητσοτάκης"])
def test_a_capitalised_pair_is_not_a_name(q):
    srcs = QB.evidence_sources([f"the {q} of the company and non-open-source software; ο Πρωθυπουργός Μητσοτάκης"])
    assert not QB.quote_found(q, srcs)


def test_compact_json_is_the_same_pair():
    srcs = QB.evidence_sources(['{"id":"9","status":"RELEASED"}'])
    assert QB.quote_found('"status": "RELEASED"', srcs)
    assert QB.quote_found('"status":"RELEASED"', srcs)


@pytest.mark.parametrize("src", ["llama-3.1-8b@2", "llama-3.1-8b~1", "x=llama-3.1-8b-instruct"])
def test_more_identifier_edges(src):
    assert not QB.quote_found("llama-3.1-8b", QB.evidence_sources([src]))


# ── r3 review ──────────────────────────────────────────────────────────

def test_a_figure_row_with_the_same_label_is_another_row():
    srcs = QB.evidence_sources(["Memory: 45.1% used (16GB / 36GB)\nAgent Internals: LLM=Active, Memory=Active"])
    assert QB.quote_found("Memory: Active", srcs)


@pytest.mark.parametrize("q,src", [
    ('"status": "released"', '{"status":"released","sub_status":"x"}'),
    ("status: released now", "sub_status: draft\nstatus: released now"),
    ("status: released now", "build-status: draft\nstatus: released now"),
])
def test_a_longer_label_is_not_the_same_label(q, src):
    assert QB.quote_found(q, QB.evidence_sources([src]))


@pytest.mark.parametrize("q,src", [('"temp": -5 degrees', '{"temp":-5 degrees}')])
def test_compact_json_before_any_value(q, src):
    assert QB.quote_found(q, QB.evidence_sources([src]))


@pytest.mark.parametrize("src", ["**LLM**=Active and more", "`LLM`=Active and more"])
def test_markdown_around_a_label(src):
    assert QB.quote_found("LLM: Active", QB.evidence_sources([src]))


def test_snake_case_survives_normalisation_on_both_sides():
    srcs = QB.evidence_sources(['{"workspace_dir": "/data/ai", "released_at": "2026-08-11"}'])
    assert QB.quote_found('"released_at": "2026-08-11"', srcs)
    assert QB._norm("_emphasis_ and snake_case") == "emphasis and snake_case"
