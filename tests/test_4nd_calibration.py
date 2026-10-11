"""§4ND (2026-10-10, operator decisions): experiment / calibration machinery
that changed nothing is switched off; confidence is a RANKING (deep
verification on the lowest 15% of recent owner turns, no probability claims,
no competence term); the competence table leaves prompts; a missing or broken
experiments.json enables nothing; coding practice is off.
Each test CLEARS the conftest switches — this file pins PRODUCTION."""
from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest

SWITCHES = ("GHOST_COMPETENCE_PROMPT", "GHOST_CONF_COMPETENCE")   # §4NM: GEPA autonomy + coding practice removed


@pytest.fixture
def production(monkeypatch):
    for k in SWITCHES:
        monkeypatch.delenv(k, raising=False)


def test_every_switch_is_off_unless_set(production, monkeypatch):
    from ghost_agent.utils.helpers import env_flag
    assert not any(env_flag(k) for k in SWITCHES)
    monkeypatch.setenv("GHOST_COMPETENCE_PROMPT", "1")
    assert env_flag("GHOST_COMPETENCE_PROMPT")


def _agent_tree():
    from ghost_agent.core import agent
    return ast.parse(inspect.getsource(agent))


def _flags_read(tree):
    return {n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
            and (getattr(n.func, "id", "") .startswith("_ef_") or getattr(n.func, "attr", "") == "env_flag")
            and n.args and isinstance(n.args[0], ast.Constant)}


def test_the_competence_prompt_switch_is_gone():
    """§4NM: the competence block and its switch were deleted from agent.py."""
    got = _flags_read(_agent_tree())
    assert "GHOST_COMPETENCE_PROMPT" not in got


# ── confidence: a ranking, the lowest 15% ─────────────────────────────

def _cc():
    from ghost_agent.core.confidence import CompositeConfidence
    c = CompositeConfidence()
    c.w_entropy, c.w_competence, c.w_effort = 0.1, 0.6, 0.3
    return c


def test_competence_is_out_of_the_score(production, monkeypatch):
    c = _cc()
    lo = c.score(normalised_entropy=0.2, competence_p_success=0.0, n_observations=1000)
    hi = c.score(normalised_entropy=0.2, competence_p_success=1.0, n_observations=1000)
    assert lo.raw_pre_penalty_composite == hi.raw_pre_penalty_composite
    # §4NM: the switch that re-added it was deleted — the old env var is inert
    monkeypatch.setenv("GHOST_CONF_COMPETENCE", "1")
    assert (c.score(normalised_entropy=0.2, competence_p_success=1.0, n_observations=1000).raw_pre_penalty_composite
            == c.score(normalised_entropy=0.2, competence_p_success=0.0, n_observations=1000).raw_pre_penalty_composite)


def _rows(n, vals):
    return [{"entropy_component": v, "entropy_observed": True, "effort_component": 0.5,
             "effort_observed": False, "competence_component": 0.9, "source": "turn", "origin": "user"}
            for v in (vals * (n // len(vals) + 1))[:n]]


def test_the_cut_is_the_lowest_fifteen_percent_of_recent_turns(production):
    c = _cc()
    c.set_rank_threshold(_rows(100, [i / 100 for i in range(100)]))
    below = [c.score(normalised_entropy=1 - v, competence_p_success=0.9, n_observations=99).below_threshold
             for v in [i / 100 for i in range(100)]]
    assert 13 <= sum(below) <= 17


def test_a_refit_cannot_move_the_cut_and_a_penalty_still_pulls_below(production):
    c = _cc()
    c.set_rank_threshold(_rows(100, [i / 100 for i in range(100)]))
    c.platt_a, c.platt_b, c.threshold = 3.0, -2.0, 0.99          # a refit's map and τ
    r = c.score(normalised_entropy=0.1, competence_p_success=0.9, n_observations=99)
    assert r.below_threshold is False                             # the rank decides, not τ
    r = c.score(normalised_entropy=0.1, competence_p_success=0.9, n_observations=99, outcome_penalty=1.0)
    assert r.below_threshold is True


def test_too_few_turns_leave_the_fitted_threshold(production):
    c = _cc()
    c.set_rank_threshold(_rows(10, [0.5]))
    assert c.rank_threshold_raw is None


def test_the_cut_is_taken_from_owner_turns_only(tmp_path):
    import json
    from ghost_agent.core.calibration import recent_turn_rows
    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"source": "turn", "origin": "user", "x": 1}, {"source": "bench", "origin": "user"},
        {"source": "turn", "origin": "probe"}, "garbage"]))
    assert [r.get("x") for r in recent_turn_rows(p)] == [1]


def test_boot_and_refit_both_take_the_cut():
    import ghost_agent.main as M
    from ghost_agent.core import agent
    for mod in (M, agent):
        tree = ast.parse(inspect.getsource(mod))
        assert any(isinstance(n, ast.Attribute) and n.attr == "set_rank_threshold" for n in ast.walk(tree)), mod


# ── no probability claims ─────────────────────────────────────────────

def _cal(**kw):
    from tests.test_calibration_audit import _cal as _full     # the producer-shaped block
    return _full(auc=0.666, outcome_pos=1364, outcome_neg=84, outcome_placeholder=794, **kw)


def test_the_report_header_is_a_ranking_not_a_probability(monkeypatch, tmp_path):
    from ghost_agent.core import learning_health as LH
    monkeypatch.setattr(LH, "collect_learning_health", lambda md: {"calibration": _cal()})
    out = LH.render_learning_health(tmp_path)
    head = next(l for l in out.splitlines() if l.startswith("CALIBRATION:"))
    assert "RANKING, not a probability" in head and "lowest 15%" in head and "AUC 0.67" in head
    assert "794 are the unverified placeholder" in out


# ── r1 review ─────────────────────────────────────────────────────────

def test_zero_weights_never_fall_back_to_competence(production):
    """r1 MAJOR: a fit with w_entropy=0 and no effort scored the turn on
    competence alone — ~31% of turns could never reach the cut."""
    c = _cc()
    c.w_entropy, c.w_effort = 0.0, 0.0
    a = c.score(normalised_entropy=0.9, competence_p_success=0.99, n_observations=500)
    b = c.score(normalised_entropy=0.9, competence_p_success=0.01, n_observations=500)
    assert a.raw_pre_penalty_composite == b.raw_pre_penalty_composite == pytest.approx(0.1)


def test_the_fit_scores_the_formula_that_runs(production):
    from ghost_agent.core import calibration as C
    s = SimpleNamespace(entropy_component=0.3, competence_component=0.95, effort_component=0.5,
                        entropy_observed=True, effort_observed=False, uncertainty_pressure=0.0)
    assert C._composite_for(s, 0.1, 0.0, 0.3) == pytest.approx(0.3)     # competence not blended in


def test_the_feature_change_started_a_new_epoch():
    from ghost_agent.core import calibration as C
    assert C.CURRENT_EPOCH == "2026-10-10.ranking"
