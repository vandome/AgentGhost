"""Learning-health telemetry — read-only aggregation of the on-disk stores
the agent's learning/cognitive loops write to.

The 2026-07 loop-closing work shipped a large stack (outcome-gated lesson
utility, competence context, episode recovery, calibration) each carrying an
explicit "watch/keep-or-kill in ~2 weeks" criterion — but nothing surfaced
the data to make that call. This turns those pending watches into one
screen. Pure reads; never mutates; never raises (every store access is
defensive so a missing/corrupt file degrades to "n/a", not a crash).

Exposed as ``introspect action='learning'`` and ``scripts/learning_health.py``.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

# The "a report must not die" handlers log here. ⚠ This name was READ before
# it was ever DEFINED (2026-09-13 review): the lowest-confidence block's
# except-handler raised NameError instead of swallowing, so the one failure
# it existed to absorb killed the report.
logger = logging.getLogger("GhostAgent")

# FALLBACK mirrors of gates OWNED by other modules. The live values are
# imported lazily at report time (see the ``_live_*`` helpers) so the
# instrument cannot drift from the mechanism it reports on — hand-copied
# mirrors here already skewed two verdicts (competence inject gate,
# entropy fit gate; both found 2026-07-27). These literals are used only
# when the owning module cannot be imported, and match its values as of
# 2026-07-27.
_STALE_MIN_RETRIEVALS = 5    # skills._STALE_MIN_RETRIEVALS
_STALE_HIT_RATE = 0.35       # skills._STALE_HIT_RATE
_OUTCOME_MIN_OBS = 4         # skills._OUTCOME_MIN_OBS ("decisive" floor)
_MIN_ENTROPY_SAMPLES = 30    # calibration._MIN_ENTROPY_SAMPLES


def _live_entropy_floor() -> int:
    """calibration.py's minimum observed-entropy sample count, read from
    the owning module so the LEARNABLE verdict can't drift from the fit."""
    try:
        from .calibration import _MIN_ENTROPY_SAMPLES as floor
        return int(floor)
    except Exception:  # noqa: BLE001
        return _MIN_ENTROPY_SAMPLES


def _live_separation(rows: List[dict], attr: str):
    """The fit's separation gate, evaluated on raw JSONL rows.

    Returns ``(sigmas, required)``. Enough observations is necessary but not
    sufficient for a weight to leave zero — the column must also separate the
    outcome classes — so the LEARNABLE verdict has to apply BOTH tests or it
    resumes lying in the same direction it did before: printing "LEARNABLE"
    over a corpus the fit pins to 0. Calls calibration's own function rather
    than reimplementing it, for the same reason.
    """
    try:
        from .calibration import _MIN_SEPARATION_SIGMAS, _separation_sigmas
        return _separation_sigmas(rows, attr), float(_MIN_SEPARATION_SIGMAS)
    except Exception:  # noqa: BLE001 — telemetry must never break on import
        return 0.0, 0.0


def _live_epoch_filter(samples: List[dict], bench_verdict: str = ""):
    """Scope raw JSONL rows to the epoch `calibration.fit` actually reads.

    Read from the owning module so this cannot drift from the fit — the whole
    point of the epoch is that cross-epoch rows are incomparable, and a
    telemetry report that pools them would go on describing a population no
    fit ever sees (which is how competence read as DEAD: its separation was
    measured across a label-scheme change). Returns
    ``(capped, n_other_epochs, n_superseded, precap)`` — ``capped`` is the
    fit population (epoch + supersede + origin cap + DIRECTION gate),
    ``precap`` the same minus the cap (the bench-flood detector's
    population). Degrades to "everything, nothing excluded" if the import
    fails.

    ⚠ `bench_verdict` (§4EC) is the fit's RECORDED direction verdict,
    passed in rather than recomputed. Without it this twin applied THREE
    of the fit's four filters, and the report headline said 1275 samples
    where the fit consumed 1022 — "a population the agent never fits",
    the exact defect this function exists to prevent, reintroduced by
    adding a filter to the fit and not to its telemetry twin.
    """
    try:
        from .calibration import CURRENT_EPOCH, epoch_for_ts
    except Exception:  # noqa: BLE001 — telemetry must never break on import
        return samples, 0, 0, samples
    scoped = [s for s in samples
              if str(s.get("epoch") or epoch_for_ts(s.get("ts"))) == CURRENT_EPOCH]
    n_other = len(samples) - len(scoped)
    # Same supersession resolution as the fit (§4L R3 MINOR-1): a
    # corrected request must contribute ONE label to separations and
    # class counts, not both. Counted APART from the epoch exclusion
    # (§4L R4: the combined number was reported under a key named
    # "other_epochs" — off by exactly the superseded count).
    n_sup = 0
    try:
        from .calibration import resolve_superseded_rows
        resolved = resolve_superseded_rows(scoped)
        n_sup = len(scoped) - len(resolved)
        scoped = resolved
    except Exception:  # noqa: BLE001 — telemetry must never break on import
        pass
    # §4NM: the fit's equal-mass bench cap was deleted (nothing writes
    # origin="bench" any more and the current epoch holds none), so the
    # epoch-scoped, supersede-resolved rows ARE the pre-cap population the
    # origin-mix line reports.
    precap = scoped
    # §4EC: and the DIRECTION gate — the fit's stored verdict, APPLIED, not
    # re-derived. Last in the chain, mirroring `_fit_population`'s order.
    try:
        from .calibration import apply_bench_direction_rows
        scoped = apply_bench_direction_rows(scoped, bench_verdict)
    except Exception:  # noqa: BLE001 — telemetry must never break on import
        pass
    return scoped, n_other, n_sup, precap


def _live_stale_gates():
    """skills.py's stale-lesson demotion signal (min retrievals, hit-rate
    floor), read from the owning module with a last-known fallback."""
    try:
        from ..memory.skills import _STALE_MIN_RETRIEVALS as r, _STALE_HIT_RATE as h
        return int(r), float(h)
    except Exception:  # noqa: BLE001
        return _STALE_MIN_RETRIEVALS, _STALE_HIT_RATE


def _live_outcome_min_obs() -> int:
    """skills.py's decisive-outcome floor (the tick count at which the
    outcome arm may influence utility)."""
    try:
        from ..memory.skills import _OUTCOME_MIN_OBS as n
        return int(n)
    except Exception:  # noqa: BLE001
        return _OUTCOME_MIN_OBS


def _load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _load_jsonl(path: Path, limit: Optional[int] = None) -> List[dict]:
    # With a limit, a bounded deque keeps memory O(limit) instead of
    # materialising the whole file first (the activity ledger never
    # rotates). The limit also holds on a mid-read failure — the old
    # early return skipped the tail slice.
    buf = deque(maxlen=limit) if limit else []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    buf.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass  # partial read — return what we have (still bounded)
    return list(buf)


# Lessons CREATED before this date accumulated their retrieval counters
# during the double-booking era, so their hit-rate denominator is inflated and
# cannot be un-mixed retroactively — the counters are cumulative with no
# per-period breakdown. Lessons created AFTER it accumulated entirely in the
# clean era, which gives an honest denominator without discarding anything.
#
# ⚠ AUDIT 2026-08-10. The report printed the contaminated mean with a
# "do not trend across that date" caveat — a number the reader was told not to
# use, next to no number they could. Measured: contaminated 0.620 vs CLEAN
# 0.557 over 13 lessons / 600 retrievals. The caveat was hiding a 0.06
# OVERSTATEMENT that was computable all along.
_LESSON_CLEAN_EPOCH = "2026-08-01"


def _lesson_created_ts(lesson: dict):
    """Creation time as epoch seconds, or None when unparseable."""
    v = lesson.get("timestamp")
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) or None
    import datetime as _dt
    try:
        return _dt.datetime.fromisoformat(
            str(v).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _lesson_hit_rate(lesson: dict) -> float:
    # Laplace-smoothed hit rate, matching skills.compute_lesson_utility's
    # (helpful+1)/(retrievals+2) convention.
    h = int(lesson.get("helpful_retrievals") or 0)
    r = int(lesson.get("retrievals") or 0)
    return (h + 1) / (r + 2)


def usable_number(v, *, allow_negative: bool = False):
    """A value off `calibration_params.json` we may print or compare, else None.

    ⚠ ONE GUARD, BECAUSE SITE-BY-SITE GUARDING CANNOT CONVERGE. Six review
    rounds each hardened the numeric reads the previous round had looked at
    and missed a neighbour: the bool clause reached `_num` and not
    `n_negative`; `math.isfinite` reached `_baseline_ok`, `_have_ci` and
    `_show` and not `_num` or `_hdr_num`; the `>= 0` sentinel check reached
    three of the four Brier sites. Every round was a real defect and every
    round was the same defect, because the reviewer's attention defined the
    boundary. This is the boundary instead, and
    `test_every_numeric_read_goes_through_the_guard` walks the AST to prove
    nothing bypasses it.

    Rejected: non-numbers, `bool` (`isinstance(True, int)` is True), and
    non-finite values. Also rejected by default: negatives — because `-1.0`
    is this module's OWN "not recorded" sentinel, written by `fit()` for
    `brier_raw`, `brier_base_rate` and `brier_cv`. It was recognised as such
    at three of the four Brier sites, so a params file the agent certifies
    as IN FORCE rendered "beats the base-rate predictor" off a number that
    means "we never computed this".
    """
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return None
    f = float(v)
    if not math.isfinite(f):
        return None
    if not allow_negative and f < 0.0:
        return None
    return f


def collect_learning_health(memory_dir, args: Any = None) -> Dict[str, Any]:
    """Aggregate the learning stores under ``memory_dir`` (the
    system/memory directory). Returns a nested dict of sections; each is
    self-contained so a failed read leaves the others intact."""
    md = Path(memory_dir)
    calib_dir = md.parent / "calibration"
    report: Dict[str, Any] = {}

    # -- Lessons (playbook) ------------------------------------------------
    pb = _load_json(md / "skills_playbook.json", []) or []
    if isinstance(pb, list):
        outcome_min = _live_outcome_min_obs()
        stale_min_r, stale_hr = _live_stale_gates()
        decisive = [l for l in pb
                    if (int(l.get("succeeded_retrievals") or 0)
                        + int(l.get("failed_retrievals") or 0)) >= outcome_min]
        # Single O(n) pass; the old list-membership version compared dicts
        # by equality (O(n²), and duplicate lessons misclassified).
        n_pass_only = n_fail_only = n_mixed = 0
        for l in decisive:
            s_ticks = int(l.get("succeeded_retrievals") or 0)
            f_ticks = int(l.get("failed_retrievals") or 0)
            if s_ticks > 0 and f_ticks == 0:
                n_pass_only += 1
            elif f_ticks > 0 and s_ticks == 0:
                n_fail_only += 1
            else:
                n_mixed += 1
        stale = [l for l in pb
                 if int(l.get("retrievals") or 0) >= stale_min_r
                 and _lesson_hit_rate(l) < stale_hr]
        hrs = [_lesson_hit_rate(l) for l in pb if int(l.get("retrievals") or 0) > 0]
        # CLEAN denominator: lessons created after the double-booking era, so
        # every retrieval they counted was counted once. See _LESSON_CLEAN_EPOCH.
        import datetime as _dt
        _cut = _dt.datetime.fromisoformat(_LESSON_CLEAN_EPOCH).timestamp()
        _clean = [l for l in pb
                  if int(l.get("retrievals") or 0) > 0
                  and (_lesson_created_ts(l) or 0) >= _cut]
        _clean_hrs = [_lesson_hit_rate(l) for l in _clean]
        report["lessons"] = {
            "total": len(pb),
            "graduated": sum(1 for l in pb if l.get("graduated")),
            "quarantined": sum(1 for l in pb if l.get("quarantined")),
            "verified": sum(1 for l in pb if l.get("verified")),
            # §4FD provenance: who wrote the lessons the prompt retrieves.
            # "" = legacy (pre-stamp). A "probe" count above zero is a leak.
            "by_origin": {
                k: sum(1 for l in pb if str(l.get("origin") or "") == k)
                for k in ("user", "auto", "probe", "bench", "")},
            "with_outcome_ticks": sum(
                1 for l in pb
                if (int(l.get("succeeded_retrievals") or 0)
                    + int(l.get("failed_retrievals") or 0)) > 0),
            "decisive": len(decisive),
            "present_on_pass_only": n_pass_only,
            "present_on_failure_only": n_fail_only,
            "present_on_both": n_mixed,
            "stale_prune_candidates": len(stale),
            "mean_hit_rate": round(sum(hrs) / len(hrs), 3) if hrs else None,
            # The honest headline. `mean_hit_rate` above spans the
            # double-booking era and OVERSTATES (measured 0.620 vs 0.557).
            "mean_hit_rate_clean": (round(sum(_clean_hrs) / len(_clean_hrs), 3)
                                    if _clean_hrs else None),
            "clean_lessons": len(_clean_hrs),
            "clean_epoch": _LESSON_CLEAN_EPOCH,
            "clean_retrievals": sum(int(l.get("retrievals") or 0)
                                    for l in _clean),
            # Raw outcome-tick totals (2026-07-27): the honest FAILURE-arm
            # liveness test. At the live ~96% turn pass rate a retrieved
            # lesson almost surely accrues ≥1 success, so the fail-ONLY
            # bucket is near-impossible by construction and its emptiness
            # says nothing about the arm — total failed ticks does.
            "succeeded_ticks_total": sum(
                int(l.get("succeeded_retrievals") or 0) for l in pb),
            "failed_ticks_total": sum(
                int(l.get("failed_retrievals") or 0) for l in pb),
        }

    # -- Competence --------------------------------------------------------
    cp = _load_json(md / "competence_profile.json", {}) or {}
    if isinstance(cp, dict):
        # Per-domain rollup cells are keyed "domain|*".
        domains = {}
        for k, v in cp.items():
            if not isinstance(v, dict):
                continue
            parts = str(k).split("|")
            if len(parts) == 2 and parts[1] == "*" and parts[0] != "*":
                a = float(v.get("alpha", 1.0)); b = float(v.get("beta", 1.0))
                n = int(v.get("n", 0))
                domains[parts[0]] = {"p_success": round(a / (a + b), 3), "n": n}
        # §4NM: the competence block is no longer injected into any prompt
        # (it was off in production since §4ND — it counted error-free calls,
        # not success — and the code behind its switch was deleted). The
        # profile is still kept and reported; mirror the mechanism: nothing
        # injects.
        total_n = sum(s["n"] for s in domains.values())
        report["competence"] = {
            "cells_total": sum(1 for v in cp.values() if isinstance(v, dict)),
            "domains": domains,
            "total_observations": total_n,
            "injects_into_prompt": False,
        }

    # -- Adaptive threshold ------------------------------------------------
    at = _load_json(md / "adaptive_threshold.json", {}) or {}
    if isinstance(at, dict) and "threshold" in at:
        win = at.get("window") or []
        report["adaptive_threshold"] = {
            "learned_threshold": round(float(at.get("threshold", 0.0)), 4),
            "window_samples": len(win),
        }

    # -- Episodes ----------------------------------------------------------
    epi = _episode_stats(md / "episodic_memory.db")
    if epi is not None:
        report["episodes"] = epi

    # -- Calibration -------------------------------------------------------
    params = _load_json(calib_dir / "calibration_params.json", {}) or {}
    # ⚠ TWO READERS, TWO POLICIES, ONE FILE. `calibration.load_params`
    # refuses a params file whose `schema` does not match SCHEMA_VERSION and
    # the agent then runs on hardcoded defaults — while this reader accepted
    # it and rendered a full CALIBRATION section, base-rate licence included,
    # for a fit that is IN FORCE NOWHERE. Flag it rather than blanking the
    # section: the numbers are still the last fit's and worth seeing, but the
    # report must not imply they are running.
    # ⚠ ASK THE LOADER, DO NOT MIRROR ONE OF ITS FOUR REFUSALS. The first
    # version compared `schema` only, while `load_params` also returns None
    # on bad JSON, a non-dict, and any KeyError/TypeError/ValueError from the
    # required numeric fields. Measured: `"threshold": "abc"` and a deleted
    # `w_competence` each make the agent fall back to hardcoded defaults
    # while this report still printed the fit AND its base-rate licence —
    # the exact scenario the flag was added to catch. One authority.
    # ⚠ NON-DICT, AND A LOADER THAT CAN RAISE. `params` is whatever the
    # file held: `[1,2,3]` or `"hello"` made `params.get("schema")` six
    # lines down raise AttributeError — while THIS flag's own comment
    # enumerates "a non-dict" as one of the refusals it reports. And
    # `load_params` does `int(d.get(...))`, which raises OverflowError (not
    # covered by its KeyError/TypeError/ValueError catch) on a JSON
    # `Infinity` or `1e400`. Calling it here therefore introduced a NEW way
    # for one malformed field to blank the entire report — a file that
    # rendered fine before this flag existed.
    # ⚠ AND REMEMBER WHY IT IS EMPTY. `_load_json` returns {} for BAD JSON
    # and this line empties a non-dict — after which `not params`
    # short-circuits the check below to "in force". The notice's own sentence
    # says it reports "bad JSON, a non-dict, and unusable required numerics";
    # it could not fire for the first two, which are the two cases this
    # normalisation creates.
    _params_malformed = bool(params) and not isinstance(params, dict)
    _raw_present = (calib_dir / "calibration_params.json").exists()
    if not isinstance(params, dict):
        params = {}
    from .calibration import CalibrationTracker as _CalTracker
    try:
        _params_in_force = (
            (not _raw_present)                      # no file at all: fine
            and not _params_malformed
        ) or (_CalTracker(calib_dir).load_params() is not None)
    except Exception:       # noqa: BLE001 — a report must never break on data
        _params_in_force = False
    all_samples = _load_jsonl(calib_dir / "calibration.jsonl")
    # Every count below is EPOCH-SCOPED, matching the fit. Pooling epochs is
    # what made this report contradict itself: 1709 rows on disk, of which
    # only 541 were fittable, while the feature verdicts were computed over
    # all of them and so measured a label-scheme change rather than a signal.
    samples, n_other_epochs, n_superseded, precap_samples = \
        _live_epoch_filter(all_samples,
                           str(params.get("bench_verdict") or ""))
    # §4BF 1c (R2+R3 reviews): `label_origins` inside _label_health
    # describes the CAPPED population (correct — it must match the fit),
    # which saturates at 1:1 by construction. The PRE-CAP mix is the
    # bench-flood detector — counted on the CURRENT-EPOCH, supersede-
    # resolved rows (R3: counting all epochs let retired real rows mask
    # a flooded current epoch, and a bench-heavy retired epoch could
    # false-alarm).
    raw_origins: Dict[str, int] = {}
    for s in precap_samples:
        raw_origins[str(s.get("origin") or "user")] = raw_origins.get(
            str(s.get("origin") or "user"), 0) + 1
    if params or all_samples:
        # Count entropy variety over OBSERVED samples only. Counting all of
        # them conflated "the model was 50/50" with "no logprobs came back",
        # which is what made this metric read as a mysterious degeneracy
        # instead of the plain coverage problem it is.
        obs = [s for s in samples if s.get("entropy_observed")]
        ent = [s.get("entropy_component") for s in obs
               if s.get("entropy_component") is not None]
        ent_distinct = len(set(round(float(e), 3) for e in ent)) if ent else 0
        def _row_outcome(s) -> float:
            # Defensive: corrupt/hand-edited rows count as the negative
            # class rather than taking down the report.
            try:
                return float(s.get("outcome", 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        # §4M (Lens D F4): this list was built UNCOERCED and compared with
        # `o < 0.5` below — one non-numeric outcome row raised TypeError
        # and blanked the entire report ("Learning health unavailable"),
        # violating the module's never-raises contract while the coercer
        # sat unused right above.
        outs = [_row_outcome(s) for s in samples if "outcome" in s]

        # Mirror calibration.py's fit gate EXACTLY: >= floor observed
        # samples AND both outcome classes represented among them. The old
        # formula here (distinct >= 3 and observed >= 30) could print
        # "LEARNABLE — w_entropy is fit on these" while the fit pinned
        # w_entropy to 0 on a one-class observed corpus (and vice versa
        # could deny a fit calibration would happily run).
        obs_pos = sum(1 for s in obs if _row_outcome(s) >= 0.5)
        obs_neg = len(obs) - obs_pos
        ent_floor = _live_entropy_floor()
        ent_sigmas, sep_floor = _live_separation(obs, "entropy_component")
        eff_rows = [s for s in samples if s.get("effort_observed")]
        eff_sigmas, _ = _live_separation(eff_rows, "effort_component")
        report["calibration"] = {
            "params_in_force": _params_in_force,
            "schema_on_disk": params.get("schema"),
            "samples_on_disk": len(all_samples),
            "samples_this_epoch": len(samples),
            "samples_other_epochs": n_other_epochs,
            "samples_superseded": n_superseded,
            "epoch": params.get("epoch"),
            "n_fitted": params.get("n_samples"),
            "brier": params.get("brier"),
            # Honest, out-of-sample counterparts (audit 2026-08-10). The
            # renderer scores the WIN/LOSS verdict on `brier_cv` and shows
            # `brier` explicitly labelled as in-sample.
            "brier_cv": params.get("brier_cv"),
            # The interval the WIN/LOSS verdict is scored on (queue #8).
            "brier_cv_delta_lo": params.get("brier_cv_delta_lo"),
            "brier_cv_delta_hi": params.get("brier_cv_delta_hi"),
            # ⚠ The COLLECTOR is the only path from the producer to the
            # renderer. Adding the field to both ends and not to this list
            # left the renderer's `cal.get("beats_base_rate")` permanently
            # None, so it fell back to re-deriving from lo/hi — the "ask
            # once" fix was inert. Migrate the whole reader set, not the two
            # ends of it.
            # ⚠ THE ABSENT-VS-NULL RULE WENT TO THE NEIGHBOUR, NOT TO THIS
            # KEY. `map_status` got a helper, a guarded forward, a comment
            # and a test; `beats_base_rate` — the field this whole change is
            # about — kept a bare `.get()`, so absent and null collapsed and
            # the renderer read null as "legacy file, please derive". That
            # made an explicit null the MOST PERMISSIVE value the field
            # accepts: a stored "unknown" withholds the licence, an outright
            # typo withholds it, and `null` gets "beats the base-rate
            # predictor" plus a DERIVED tag asserting the file predates a
            # field it explicitly contains.
            "beats_base_rate": params.get("beats_base_rate"),
            "beats_base_rate_present": "beats_base_rate" in params,
            # ⚠ §4EB, AND THIS COMMENT BLOCK IS WHY. The rank verdict was
            # added to `fit()` and to the agent's CALIB line and NOT here —
            # the exact "both ends but not the collector" mistake the note
            # above records, committed against the note. Absent-vs-null gets
            # its own flag for the same reason `beats_base_rate` needed one.
            "ranks_outcomes": params.get("ranks_outcomes"),
            "ranks_outcomes_present": "ranks_outcomes" in params,
            "auc": params.get("auc"),
            "auc_ci_lo": params.get("auc_ci_lo"),
            "auc_ci_hi": params.get("auc_ci_hi"),
            # ⚠ §4EO, AND THE SAME COMMENT BLOCK IS WHY, AGAIN. Both verdicts
            # above are measured on the rows carrying a VERDICT, not on
            # `n_fitted` — 402 of 1074 live. Forwarded here so the renderer
            # states the population it is reporting on rather than letting
            # `n_fitted` be read as the evidence behind the verdict, and so
            # `indistinguishable` can be told apart from "cannot measure"
            # (`delta_halfwidth` is the smallest resolvable difference).
            "n_verdict_rows": params.get("n_verdict_rows"),
            "n_unverified_prior": params.get("n_unverified_prior"),
            "delta_halfwidth": params.get("delta_halfwidth"),
            # The rank measurement per feature — the statistic with power at
            # this base rate, beside `feature_contrib`'s Brier deltas.
            "component_auc": params.get("component_auc"),
            # ⚠ THE RENDERER NEEDS THE CAUSE, NOT JUST THE REFUSAL. Its
            # `unknown` branch used to state "the Platt map was rejected" as
            # fact while having no field to check it against -- and `unknown`
            # has a second source: a fit with no usable interval, which keeps
            # `map_status="applied"`. Reproduced on a 9-row fit
            # (`min_samples_for_fit` is a per-tracker argument): applied map,
            # `unknown` verdict, no interval, and all three clauses of that
            # sentence false. Carry the field and say what happened.
            # ⚠ ABSENT -> "applied" (the loader's documented default for a
            # file older than the field); an explicit null stays None and
            # fails closed. Passing `.get()` through collapsed the two, so a
            # `"map_status": null` read as APPLIED here while `load_params`
            # (which stringifies it to "None") failed closed.
            "map_status": (params["map_status"] if "map_status" in params
                           else "applied"),
            "n_negative": params.get("n_negative"),
            "n_samples": params.get("n_samples"),
            "feature_contrib": params.get("feature_contrib"),
            "w_entropy": params.get("w_entropy"),
            "w_competence": params.get("w_competence"),
            "threshold": params.get("threshold"),
            "entropy_observed_samples": len(obs),
            "entropy_observed_pct": (round(100.0 * len(obs) / len(samples), 1)
                                     if samples else 0.0),
            "entropy_distinct_values": ent_distinct,
            "entropy_observed_pos": obs_pos,
            "entropy_observed_neg": obs_neg,
            "entropy_min_samples_gate": ent_floor,
            # BOTH halves of the fit gate: enough observations AND measurable
            # class separation. Reporting only the first is how this line
            # came to read "LEARNABLE" while the fit pinned w_entropy to 0.
            "entropy_separation_sigmas": round(ent_sigmas, 2),
            "effort_separation_sigmas": round(eff_sigmas, 2),
            "separation_min_sigmas": sep_floor,
            "entropy_learnable": (len(obs) >= ent_floor
                                  and obs_pos > 0 and obs_neg > 0
                                  and ent_sigmas >= sep_floor),
            # Split at 0.5, the way every gate in calibration.py splits it.
            # These used to count only EXACT 0.0/1.0, which silently stopped
            # describing the corpus when labels went graded: on the current
            # epoch that covers 162 of 541 rows, so "outcomes 155+/7-" left
            # 70% of the population invisible. The exact-anchor counts are
            # kept alongside — a verifier PASS/FAIL is a checked fact and
            # worth seeing apart from the graded priors.
            "outcome_neg": sum(1 for o in outs if o < 0.5),
            "outcome_pos": sum(1 for o in outs if o >= 0.5),
            "outcome_verified_neg": sum(1 for o in outs if o == 0.0),
            "outcome_verified_pos": sum(1 for o in outs if o == 1.0),
            "outcome_placeholder": sum(1 for o in outs if abs(o - 0.83) < 1e-9),   # §4ND
            "platt_a": params.get("platt_a"),
            "brier_raw": params.get("brier_raw"),
            "brier_base_rate": params.get("brier_base_rate"),
            "w_effort": params.get("w_effort"),
            "effort_observed_samples": sum(
                1 for s in samples if s.get("effort_observed")),
            # Current-epoch, supersede-resolved, PRE-cap origin mix — the
            # bench-flood detector (the capped `label_origins` below
            # saturates at 1:1 by construction).
            "label_origins_raw": raw_origins,
            "bench_verdict": str(params.get("bench_verdict") or ""),
            **_label_health(samples),
            **_feature_health(samples),
        }

    # -- Background firing (dream / etc.) ---------------------------------
    report["activity"] = _activity_counts(md.parent / "autonomous_activity.jsonl")
    # Registry-driven, so a mechanism that never wrote is a ROW, not a silence.
    report["liveness"] = activity_liveness(
        md.parent / "autonomous_activity.jsonl")

    # -- Cognitive-subsystem wiring ---------------------------------------
    # Observability for the "wire-or-retire" question (improvement #5): which
    # producers run and which consumers are live vs gated OFF. Retiring a
    # producer is a product decision for the operator — this makes it an
    # informed one instead of guessing.
    report["cognitive_wiring"] = _cognitive_wiring(args, memory_dir)

    # -- GEPA prompt-optimization activation (§4F Phase 0) ----------------
    report["optim"] = _optim_activation(md.parent / "optim")

    # -- verifier escalation rate (§4F false-positive watch metric) -------
    report["verifier_escalation"] = _escalation_health(
        md.parent / "verifier" / "escalations.jsonl")

    # -- tool-call framing leak: RECURRENCE watch ------------------------
    report["framing_leak"] = _framing_leak_health(md.parent / "trajectories")

    # ── LOOP YIELD, machine-readable (§4CS item B, review round 2) ──
    # The axis existed only in the human RENDER, so nothing could alarm on
    # it programmatically and `--json` carried no yield key at all: the
    # §4CS failure mode ("the loop produced nothing and no instrument said
    # so") was fixed only for a human who reads the report. Same read-only
    # probes, same GHOST_HOME resolution as the rendered block.
    try:
        from .liveness import yield_all as _yield_all
        report["loop_yield"] = _yield_all(md.parent.parent)
    except Exception as e:  # noqa: BLE001 — telemetry must never break a report
        report["loop_yield"] = {"error": f"{type(e).__name__}: {e}"}


    return report


def _optim_activation(optim_dir: Path) -> Dict[str, Any]:
    """Tuned-prompt artifacts on disk + in-process application counters.

    The pairing is the point: an artifact WITH zero in-process applies means
    the read-site is not firing (the write-only defect this loop already had
    once). Counters live in optim.loader and reset per process — "tuned file
    present, applied 0 since boot" is the actionable line."""
    out: Dict[str, Any] = {"artifacts": {}, "activation": {}}
    try:
        if optim_dir.is_dir():
            for p in sorted(optim_dir.glob("*.json")):
                if p.name.endswith(".candidate"):
                    continue  # staging files are not live artifacts
                data = _load_json(p, {}) or {}
                opt = data.get("optimized_instruction")
                out["artifacts"][p.stem] = {
                    "chars": len(opt) if isinstance(opt, str) else 0,
                    "valid": bool(isinstance(opt, str) and opt.strip()),
                }
    except Exception:
        pass
    try:
        from ..optim.loader import activation_stats
        out["activation"] = activation_stats()
    except Exception:
        pass
    return out


def _escalation_health(ledger_path: Path) -> Dict[str, Any]:
    """Verifier escalation outcomes from the durable ledger.

    This is the §4F false-positive-rate watch metric. It spent weeks being
    "recorded" into `VerifyResult.to_dict()`, which has no production caller,
    so the only way to read it was hand-run log archaeology — and a naive grep
    over-counts by 9 points because the OVERTURNED line is a WARNING mirrored
    to a second logger while "verdict stands" is INFO. The ledger writes both
    outcomes at the point escalation RESOLVES, which is what makes a rate
    computable at all.

    Reported per (route, kind) — claim-refute and code-refute are different
    populations and averaging them hides both (measured 2026-08-04 from the
    LOG, pre-ledger: claim refutes overturned 84%, all 7 code refutes upheld
    on replay — note a ``code/*`` arm appears HERE only once a code refute
    escalates after the ledger shipped 2026-08-04; its absence from the file
    is not evidence it never fires). ``unavailable`` stays in the
    denominator: the call was spent.
    """
    out: Dict[str, Any] = {"present": False, "arms": {}, "total": 0}
    rows = _load_jsonl(ledger_path, limit=5000)
    if not rows:
        # Distinguish "no ledger yet" from "ledger with nothing in it" — the
        # file is created on the FIRST escalation after a restart, so absence
        # right after a deploy is expected, not a broken instrument.
        out["reason"] = ("no ledger file" if not Path(ledger_path).exists()
                         else "ledger empty")
        return out
    out["present"] = True
    out["total"] = len(rows)
    # §4M (Lens D F6b): the counters were undated — the ledger begins
    # 2026-08-04 and pre-ledger history (log-only) is invisible, so an
    # undated "100% overturned" invited trending against a population the
    # file does not contain.
    out["first_ts"] = min((str(r.get("ts") or "") for r in rows), default="")
    out["last_ts"] = max((str(r.get("ts") or "") for r in rows), default="")
    for rec in rows:
        arm = f"{rec.get('route') or '?'}/{rec.get('kind') or '?'}"
        slot = out["arms"].setdefault(
            arm, {"overturned": 0, "upheld": 0, "withheld": 0,
                  "unavailable": 0, "downgraded": 0,
                  "replaced_uncertain": 0, "truncation_guard": 0,
                  "mechanically_upheld": 0, "mechanically_dismissed": 0,
                  "claim_binding": 0,          # §4IN/§4IP: the binder's validated verdict shipped over the incumbent's
                  "n": 0})
        outcome = str(rec.get("outcome") or "").strip()
        if outcome in slot:
            slot[outcome] += 1
        else:
            # ⚠ Vocabulary drift must be VISIBLE (§4L Lens-C MINOR-3):
            # the outcome vocabulary grew 5 new strings in one month,
            # and an unknown string used to bump only `n` — buckets
            # quietly stopped summing to n.
            slot["other"] = slot.get("other", 0) + 1
        slot["n"] += 1
    for arm, slot in out["arms"].items():
        decided = slot["overturned"] + slot["upheld"]
        slot["overturn_rate"] = (round(slot["overturned"] / decided, 3)
                                 if decided else None)
    return out


# The native-dialect fix (`QWEN_TOOL_PROMPT_NATIVE` split) deployed
# 2026-07-31 ~18:54 local. Every one of the 17 known framing leaks predates
# it; 161 trajectories recorded after it carry zero. So the watch is not
# "is the count above zero" — the corpus is append-only, so it never will be —
# it is "is the LATEST occurrence newer than the fix".
# §4M (Lens D F5): this constant is string-compared against trajectory
# `last_seen` values that are UTC ("...Z"). The fix landed ~18:54 LOCAL
# (UTC+3) = ~15:54Z — the old value 19:00 (local, unconverted) left a
# 3h06m blind window where a real recurrence rendered "all historical".
_FRAMING_FIX_TS = "2026-07-31T15:54"


def _framing_leak_health(traj_root: Path) -> Dict[str, Any]:
    """Recurrence watch for tool-call framing leaking into argument values.

    The journal recorded "repair fires ≈0 now, each one is news" — and then
    nothing WATCHED, so 17 historical cases were re-discovered by hand four
    days later and briefly mistaken for a live defect. This is the listener
    that note assumed existed.

    Verdict is keyed on the newest occurrence, not the count.
    """
    out: Dict[str, Any] = {"available": False}
    try:
        from ..utils.leaked_framing import scan_trajectories
        from ..distill.collector import TrajectoryCollector
        if not Path(traj_root).is_dir():
            out["reason"] = "no trajectory corpus"
            return out
        stats = scan_trajectories(
            # A corruption DETECTOR wants the whole corpus, probe turns included —
            # a dialect regression that appears first in a probe must still register
            # (§4FS review). include_probes=True is the documented exception.
            TrajectoryCollector(root=Path(traj_root),
                                session_id="reader").iter_trajectories(include_probes=True))
    except Exception as e:  # noqa: BLE001 — telemetry never breaks a turn
        out["reason"] = f"scan failed ({e})"
        return out
    out.update({
        "available": True,
        "scanned": stats["scanned"],
        "calls": stats["calls"],
        "corrupt_calls": stats["corrupt_calls"],
        "by_tool": stats["by_tool"],
        "last_seen": stats["last_seen"],
        # The load-bearing field. A post-fix timestamp means the dual-dialect
        # corruption is BACK, which historically wasted whole turns.
        "regression": bool(stats["last_seen"]
                           and stats["last_seen"] > _FRAMING_FIX_TS),
    })
    return out


def _consumer_state(value: Optional[bool], unknown: str = "unknown") -> str:
    """ON / OFF / unknown for a wiring row. A row must never print a
    state it did not check (§4BM R2 MIN-I: the sibling rows printed OFF
    for a None that meant 'core.agent was not importable')."""
    if value is None:
        return unknown
    return "ON" if value else "OFF"


def _cognitive_wiring(args: Any = None, memory_dir=None) -> Dict[str, Any]:
    """Report each cognitive subsystem's producer/consumer wiring. Reads
    the gate flags from core.agent when importable; degrades to 'unknown'
    otherwise (e.g. a headless run where importing agent is undesirable).

    ``args`` (§4BM R1 MIN-2): the runtime args namespace, for consumers
    gated by CLI flags rather than module constants. None ⇒ a row reports
    "unknown", never a state it did not check."""
    flags: Dict[str, Any] = {}
    try:
        from . import agent as _ag
        flags = {
            "selfhood_prefix_consumer": getattr(_ag, "_SELFHOOD_PREFIX_ENABLED", None),
            "metacog_arbiter_consumer": getattr(_ag, "_METACOG_ARBITER_ENABLED", None),
        }
    except Exception:
        flags = {}
    return {
        # producer runs every turn; consumer (wake-up prefix) gated by flag,
        # but ALSO read by `introspect` (summary/narrative/recent) — so not
        # write-only. Store is compaction-bounded (2MB/512KB).
        "selfhood": {
            "producer": "per-turn (capture_turn/record_outcome)",
            "prefix_consumer_enabled": flags.get("selfhood_prefix_consumer"),
            "also_read_by": "introspect",
            "store_bounded": True,
        },
        # confidence/entropy → calibration record → learning-health telemetry
        # + idle Brier refit. NOT write-only. Keep.
        "calibration": {
            "producer": "per-turn confidence reading",
            "consumers": ["idle Brier refit", "introspect learning telemetry"],
            "write_only": False,
        },
        "metacog_arbiter": {
            "consumer_enabled": flags.get("metacog_arbiter_consumer"),
        },
        # RETIRED 2026-07-27 (operator-approved): module + export removed
        # after the INERT flag held — no production or offline caller
        # anywhere in the tree. Entry kept so the telemetry documents the
        # decision instead of the subsystem silently vanishing.
        "self_consistency": {
            "status": "RETIRED 2026-07-27 — module removed (was INERT, "
                      "no caller)",
        },
    }


def _episode_stats(db_path: Path) -> Optional[Dict[str, Any]]:
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            def _c(sql):
                return conn.execute(sql).fetchone()[0]
            total = _c("SELECT COUNT(*) FROM episodes")
            with_ctx = _c("SELECT COUNT(*) FROM episodes WHERE context != ''")
            with_cluster = _c("SELECT COUNT(*) FROM episodes WHERE cluster_id != ''")
            success = _c("SELECT COUNT(*) FROM episodes WHERE outcome_success = 1")
            consolidated = _c("SELECT COUNT(*) FROM episodes WHERE consolidated = 1")
            return {
                "total": total,
                "with_context": with_ctx,
                "with_cluster": with_cluster,
                "context_coverage_pct": round(100 * with_ctx / total, 1) if total else 0.0,
                "success": success,
                "consolidated": consolidated,
                "pending_consolidation": total - consolidated,
            }
        finally:
            conn.close()
    except Exception:
        return None


def _label_health(samples: List[dict]) -> Dict[str, Any]:
    """Label variance and provenance mix.

    Variance is the quantity that actually gates learning: the binary label
    was 96.1% one class, and a near-constant target cannot teach a fit
    anything no matter how many samples accumulate. Provenance is reported
    beside it because the graded end-of-turn label is a PROXY — if the
    ground-truth sources (user corrections) ever vanish from the mix, the
    agent is calibrating purely against its own notion of a tidy turn.
    """
    out: Dict[str, Any] = {}
    if not samples:
        return out
    vals = []
    for s in samples:
        try:
            v = s.get("outcome")
            if v is not None:
                vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if vals:
        mean = sum(vals) / len(vals)
        out["label_variance"] = round(
            sum((v - mean) ** 2 for v in vals) / len(vals), 5)
        out["label_distinct_values"] = len(set(round(v, 3) for v in vals))
        out["label_mean"] = round(mean, 4)
    counts: Dict[str, int] = {}
    for s in samples:
        counts[str(s.get("source") or "turn")] = counts.get(
            str(s.get("source") or "turn"), 0) + 1
    out["label_sources"] = counts
    # §4BF 1c per-origin denominator (the equal-mass cap went in §4NM; the
    # mix stays visible) —
    # a corpus that is quietly half bench reads very differently from a
    # real-turn one. Legacy rows have no origin field = real turns.
    origins: Dict[str, int] = {}
    for s in samples:
        origins[str(s.get("origin") or "user")] = origins.get(
            str(s.get("origin") or "user"), 0) + 1
    out["label_origins"] = origins
    return out


# Reporting floor for a feature-liveness verdict: below this many eligible
# samples, separation is statistical noise and the honest verdict is
# "insufficient", not live/dead. A reporting constant, not a mechanism
# mirror — no fit gates on it.
_FEATURE_MIN_SAMPLES = 10

# Features whose rows carry an explicit "was this actually measured?" flag.
# Only flagged rows are eligible for that feature's verdict — see
# `_feature_health`. Competence and uncertainty_pressure are recorded on
# every sample and so have no flag.
_OBSERVED_FLAG = {
    "entropy_component": "entropy_observed",
    "effort_component": "effort_observed",
}


def _feature_health(samples: List[dict]) -> Dict[str, Any]:
    """Per-feature liveness for the confidence composite.

    A feature is only useful if it (a) VARIES and (b) SEPARATES successes
    from failures. Measured 2026-07-27, all three inputs failed at least one
    test — entropy had 2 distinct values, uncertainty_pressure had 1 (always
    0.0), and competence, the only one that varied, separated the classes by
    −0.0008. That is why the composite's leak-free AUC was 0.473, i.e. no
    discrimination, and why no amount of recalibration can help. Surfacing
    per-feature separation makes a dead input obvious instead of leaving it
    to be inferred from a bad Brier months later.

    Entropy is judged over OBSERVED samples only (2026-07-27, later):
    unobserved samples carry the neutral 0.5 stand-in, and blending ~1300
    stand-ins with a handful of real observations forced separation to ~0
    by construction — the report branded the feature DEAD hours after the
    n_probs fix started producing real values. Same "no signal" vs
    "neutral measurement" conflation as the entropy_distinct_values fix.
    Verdicts: "live" | "dead" | "insufficient" (fewer than
    ``_FEATURE_MIN_SAMPLES`` eligible rows, or only one outcome class
    among them — nothing honest can be said yet).
    """
    out: Dict[str, Any] = {}
    if not samples:
        return out

    def _outcome(s) -> float:
        # Corrupt/hand-edited rows must not take down the report.
        try:
            return float(s.get("outcome", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    feats = {}
    # §4NN: `uncertainty_pressure` left the list with the uncertainty
    # tracker — new rows carry a constant 0.0.
    for name in ("entropy_component", "competence_component",
                 "effort_component"):
        # Judge a feature ONLY over rows that actually MEASURED it. An
        # unobserved row carries the neutral 0.5 stand-in, and averaging
        # stand-ins with real readings drives separation toward 0 by
        # construction. This was fixed for entropy and never generalised, so
        # `effort_component` was still judged over 497 rows when only 394
        # measured it — reporting a separation the fit (which reads observed
        # rows only) does not compute, from a population it does not use.
        _flag = _OBSERVED_FLAG.get(name)
        rows = [s for s in samples if s.get(_flag)] if _flag else samples
        vals = []
        for s in rows:
            try:
                v = s.get(name)
                if v is not None:
                    vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if not vals and not any(name in s for s in samples):
            continue  # feature never recorded at all — nothing to report

        def _mean(subset):
            got = []
            for s in subset:
                try:
                    v = s.get(name)
                    if v is not None:
                        got.append(float(v))
                except (TypeError, ValueError):
                    continue
            return (sum(got) / len(got)) if got else None

        ok = [s for s in rows if _outcome(s) >= 0.5]
        bad = [s for s in rows if _outcome(s) < 0.5]
        m_ok, m_bad = _mean(ok), _mean(bad)
        sep = (round(m_ok - m_bad, 4)
               if (m_ok is not None and m_bad is not None) else None)
        sigmas, sig_floor = _live_separation(rows, name)
        distinct = len(set(round(v, 3) for v in vals))
        # "Dead" = constant (nothing to learn from), or varying but with no
        # measurable ability to tell the two outcome classes apart. Note the
        # rule is NOT "few distinct values": a two-valued feature that splits
        # the classes cleanly is perfectly useful, while the live
        # competence signal has 270 distinct values and separates by
        # −0.0008. Separation is what matters; distinctness alone only
        # catches the fully-constant case.
        #
        # The live/dead cut is the FIT's gate (`_MIN_SEPARATION_SIGMAS`), not
        # a local constant. It used to be `abs(sep) < 0.02`, a raw delta with
        # no notion of noise — which put this report in direct contradiction
        # with the mechanism it describes: live entropy separates by 0.0421
        # (> 0.02 → "live" here) at 0.63σ (< 2.5σ → PINNED by the fit), so
        # two adjacent lines of the same report disagreed and the
        # "features: N/4 live" headline counted a feature the fit refuses to
        # weight. A raw delta is scale-dependent anyway: 0.02 means different
        # things for a feature spanning 0.01 than one spanning the unit
        # interval.
        if len(vals) < _FEATURE_MIN_SAMPLES or sep is None:
            verdict = "insufficient"
        elif distinct < 2 or sigmas < sig_floor:
            verdict = "dead"
        else:
            verdict = "live"
        feats[name] = {
            "n": len(vals),
            "distinct": distinct,
            "separation": sep,
            # Reported beside the raw delta because the delta alone cannot
            # be judged without knowing the spread it sits in.
            "separation_sigmas": (round(sigmas, 2)
                                  if math.isfinite(sigmas) else "inf"),
            "verdict": verdict,
            "dead": verdict == "dead",
        }
    out["feature_health"] = feats
    out["live_features"] = [k for k, v in feats.items()
                            if v["verdict"] == "live"]
    return out


def _activity_counts(ledger_path: Path, *, window_hours: float = 168.0) -> Dict[str, int]:
    """Count ledger records by phase over a recent window.

    ``ActivityRecord.to_dict`` serializes the kind as ``phase`` — keying on
    kind/type/category matched NOTHING, so this always returned {} and the
    "BACKGROUND ACTIVITY" section silently never rendered against a ledger
    with 1500+ real records (2026-07-27: indistinguishable from "nothing
    ran", in the very report meant to answer whether the loops fire).
    """
    counts: Dict[str, int] = {}
    cutoff = time.time() - max(0.0, float(window_hours)) * 3600.0
    for rec in _load_jsonl(ledger_path, limit=_ACTIVITY_READ_LIMIT):
        try:
            ts = float(rec.get("ts") or 0)
        except (TypeError, ValueError):
            ts = 0.0
        if not ts:
            # Unparseable/zero timestamp: SKIP, don't count. `if ts and
            # ts < cutoff` kept these rows in EVERY window — a single
            # garbage-ts row would keep a dead PERIODIC phase looking
            # alive in the liveness table forever, and that table's whole
            # job is telling benign zeros from fatal ones.
            continue
        if ts < cutoff:
            continue
        kind = str(rec.get("phase") or rec.get("kind") or rec.get("type")
                   or rec.get("category") or "").strip()
        try:                                  # §4NM: dream → maintenance
            from .autonomous_activity import canonical_phase as _cp
            kind = _cp(kind)
        except Exception:  # noqa: BLE001
            pass
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
    return counts


# The ledger is append-only and already past 2600 records. A tail-read cap of
# 2000 does not bite at today's ~565 records/week, but it is a SILENT cap: the
# day it bites, phases quietly lose counts and the report keeps its confident
# tone. Raised, and `activity_liveness` reports when the cap was actually hit.
_ACTIVITY_READ_LIMIT = 20000


def activity_liveness(ledger_path: Path,
                      *, alarm_window_hours: float = 24.0,
                      context_window_hours: float = 168.0) -> Dict[str, Any]:
    """Did every registered background mechanism actually fire?

    ⚠ WHY THIS IS NOT `_activity_counts`. That function enumerates phases
    OBSERVATIONALLY — it can only report slugs that are already in the ledger.
    A mechanism that stopped writing, or never wrote, produces no key and is
    therefore invisible, which is indistinguishable from a mechanism that does
    not exist. Measured 2026-08-10: 15 phases instrumented in source, 7 in the
    ledger, and the report rendered seven green rows with no hint of the
    other eight. This function iterates the REGISTRY instead, so absence is a
    row rather than a silence.

    Two windows on purpose. The alarm fires on 24h, but a phase that last ran
    three days ago reads very differently from one dark for a week, and
    collapsing that to a single boolean throws away the only evidence that
    distinguishes "slow" from "dead".
    """
    from .autonomous_activity import (
        EXPECT_PERIODIC, PHASE_EXPECTATION, phase_expectation,
    )

    # ⚠ MISSING LEDGER != IDLE AGENT (caught in fresh-eye review, 2026-08-10).
    #
    # Without this the two render IDENTICALLY: every registered phase at zero,
    # `agent_silent` true, no alarms — the same screen for "the agent was
    # quiet" and "this function is pointed at a path that does not exist".
    #
    # It is the SAME defect this file's sibling fix removed from the response
    # cache hours earlier (an empty cache dir reading exactly like a stale
    # one), reproduced in the code written to detect defects of that shape.
    # Absence of the instrument is a louder fact than absence of activity, and
    # it must never be inferable only by noticing that everything is zero.
    ledger_missing = not Path(ledger_path).exists()

    recent = _activity_counts(ledger_path, window_hours=alarm_window_hours)
    context = _activity_counts(ledger_path, window_hours=context_window_hours)
    n_read = len(_load_jsonl(ledger_path, limit=_ACTIVITY_READ_LIMIT))

    # ⚠ AGENT-DOWN GUARD. If NOTHING fired, the agent was off or idle — that
    # is ONE fact, not five independent dead loops. Emitting five alarms for
    # a stopped agent is how an operator learns to scroll past this section,
    # which would rebuild by hand exactly the blindness it exists to remove.
    silent_everywhere = sum(recent.values()) == 0

    # The attempt heartbeat, read from beside the ledger. Failure here is
    # not an alarm condition: `read_attempts` returns {} and every phase
    # falls back to the outcome-only rule.
    try:
        from .autonomous_activity import (
            attempt_suppresses_alarm as _suppresses,
            read_attempts as _read_attempts)
        attempts = _read_attempts(ledger_path) if ledger_path else {}
    except Exception:  # noqa: BLE001
        attempts = {}
        def _suppresses(_e, _n, _w):  # noqa: E306
            return False
    now_ts = time.time()
    alarm_window_s = float(alarm_window_hours) * 3600.0

    rows = []
    # ⚠ `set(attempts)` is in the union. Without it a phase that has ONLY
    # ever stamped a heartbeat — a new loop wired to the heartbeat before it
    # ever produces an outcome — appears nowhere at all, which is the exact
    # invisibility this registry exists to remove, re-entered through the new
    # door.
    for phase in sorted(set(PHASE_EXPECTATION) | set(context) | set(recent)
                        | set(attempts)):
        exp = phase_expectation(phase)
        n24, n168 = recent.get(phase, 0), context.get(phase, 0)
        rows.append({
            "phase": phase,
            "n_alarm_window": n24,
            "n_context_window": n168,
            "expectation": exp,
            # Only PERIODIC can alarm: the others are at zero by design, and
            # a monitor that cries on a benign zero gets ignored on a real one.
            # ⚠ A RECENT ATTEMPT SUPPRESSES THE ALARM, and is reported.
            #
            # The ledger records OUTCOMES, so "produced nothing in 24h" and
            # "did not run in 24h" were the same row. Measured 2026-08-30:
            # this printed `✗ DEAD router_train 0/24h 0/7d` while the loop
            # had run 30 minutes earlier and correctly declined an 89%
            # unchanged corpus. An alarm that fires on a healthy loop is how
            # the column gets learned-ignored — and then misses a real one.
            #
            # The heartbeat is a SEPARATE signal (`idle_attempts.json`), so
            # this is not the ledger vouching for itself. Absent heartbeat →
            # unchanged behaviour: an unreadable or never-written heartbeat
            # must degrade to the old alarm, never to a false GREEN.
            # ⚠ ASK THE PREDICATE, do not re-derive the comparison here.
            # The first version inlined `(now - ts) <= window`, which is
            # one-sided: any FUTURE timestamp — an NTP step back, a
            # hand-edited file, a JSON `Infinity` (json.loads accepts the
            # literal) — satisfied it forever and turned a real DEAD alarm
            # permanently green. This file already hardens the LEDGER
            # timestamps against exactly that ~100 lines above; the new
            # signal, the only one that can SILENCE an alarm, did not
            # inherit it. One predicate, one place.
            "attempted_at": (attempts.get(phase) or {}).get("ts"),
            "attempted_result": (attempts.get(phase) or {}).get("result"),
            "attempted_age_s": (
                None if not attempts.get(phase)
                else now_ts - float(attempts[phase]["ts"])),
            "attempted_recently": _suppresses(
                attempts.get(phase), now_ts, alarm_window_s),
            "alarm": bool(exp == EXPECT_PERIODIC and n24 == 0
                          and not silent_everywhere
                          and not _suppresses(attempts.get(phase), now_ts,
                                              alarm_window_s)),
            "registered": phase in PHASE_EXPECTATION,
        })
    # Zeros FIRST. The old renderer sorted by count descending and cut to the
    # top 8, so the dead loop — the one thing worth looking at — was
    # structurally the first row dropped.
    rows.sort(key=lambda r: (not r["alarm"], r["n_alarm_window"], r["phase"]))
    return {
        "rows": rows,
        "alarms": [r["phase"] for r in rows if r["alarm"]],
        "agent_silent": silent_everywhere,
        # Distinct from `agent_silent`: no instrument, vs an instrument
        # reporting nothing. The renderer must never merge these.
        "ledger_missing": ledger_missing,
        "ledger_path": str(ledger_path),
        "alarm_window_hours": alarm_window_hours,
        "context_window_hours": context_window_hours,
        # Loud rather than silent if the tail-read ever truncates the window.
        "read_truncated": n_read >= _ACTIVITY_READ_LIMIT,
    }


def render_learning_health(memory_dir, args: Any = None) -> str:
    """Human-readable one-screen learning-health report.

    ``args`` threads the runtime namespace to the wiring section so
    flag-gated consumers report their real state (§4BM R1 MIN-2).
    Passed POSITIONALLY only when present (§4BM R2 CRIT-A): nine existing
    tests monkeypatch ``collect_learning_health`` with one-arg lambdas,
    and an unconditional 2-arg call broke every one of them — including
    the MISSING-INSTRUMENT and CONSUMER-DEAD guards, i.e. the other
    broken-instrument tests."""
    r = (collect_learning_health(memory_dir) if args is None
         else collect_learning_health(memory_dir, args))
    lines: List[str] = ["### LEARNING HEALTH"]

    les = r.get("lessons")
    if les:
        # §4M (Lens D F6d): at the cap, "N total" is a saturated window
        # (the store trims on every write), not a population count — say so.
        _cap_note = ""
        try:
            from ..memory.skills import PLAYBOOK_MAX as _PBMAX
            if int(les["total"]) >= int(_PBMAX):
                _cap_note = f" (AT CAP {_PBMAX} — trims on every write)"
        except Exception:
            pass
        lines.append(
            f"\nLESSONS: {les['total']} total{_cap_note} "
            f"({les['graduated']} graduated, {les['verified']} verified, "
            f"{les['quarantined']} quarantined)")
        _bo = les.get("by_origin") or {}
        if _bo:
            _probe = int(_bo.get("probe") or 0)
            lines.append(
                f"  origin: user {int(_bo.get('user') or 0)}, "
                f"auto {int(_bo.get('auto') or 0)}, bench {int(_bo.get('bench') or 0)}, "
                f"legacy {int(_bo.get('') or 0)}"
                + (f", ⚠ probe {_probe} (diagnostics must never teach)" if _probe else ""))
        lines.append(
            f"  outcome ticks on {les['with_outcome_ticks']} lessons; "
            f"decisive (≥4 obs): {les['decisive']} "
            f"[pass-only {les['present_on_pass_only']}, "
            f"fail-only {les['present_on_failure_only']}, "
            f"both {les['present_on_both']}]")
        # ⚠ The CLEAN figure leads. The all-lessons mean spans the
        # double-booking era and overstates; printing it with a "do not trend"
        # caveat gave the reader a number they were told not to use and no
        # number they could — while the clean one was computable all along.
        _chr, _cn = les.get("mean_hit_rate_clean"), les.get("clean_lessons")
        if _chr is not None and _cn:
            lines.append(
                f"  mean hit-rate: {_chr} (CLEAN — {_cn} lessons created after "
                f"{les.get('clean_epoch')}, {les.get('clean_retrievals')} "
                f"retrievals, denominator counted once)")
            # Direction COMPUTED, not asserted. This line used to print a
            # hardcoded "OVERSTATES": true when written (0.620 vs 0.557),
            # false on live data within a month (0.673 vs 0.681 — the
            # "overstating" number was LOWER). A report that renders both
            # numbers and then asserts their ordering from memory is an
            # instrument stating a stale fact as a measurement. The claim
            # that matters — not comparable, double-counted denominator —
            # does not depend on the direction.
            try:
                _all_v, _clean_v = float(les["mean_hit_rate"]), float(_chr)
                _dir = ("currently overstates" if _all_v > _clean_v
                        else "currently understates" if _all_v < _clean_v
                        else "coincidentally matches")
            except (TypeError, ValueError):
                _dir = "cannot be ordered against"
            lines.append(
                f"    all-lessons {les['mean_hit_rate']} spans the "
                f"double-booking era and {_dir} the clean figure — not "
                "comparable either way (denominator double-counted)")
        else:
            lines.append(
                f"  mean hit-rate: {les['mean_hit_rate']} ⚠ CONTAMINATED — no "
                f"post-{les.get('clean_epoch')} lesson has retrievals yet, so "
                f"no clean denominator exists")
        lines.append(
            f"  stale/prune candidates: {les['stale_prune_candidates']}")
        _fail_ticks = int(les.get("failed_ticks_total") or 0)
        _succ_ticks = int(les.get("succeeded_ticks_total") or 0)
        lines.append(
            f"  outcome ticks total: {_succ_ticks} succeeded / "
            f"{_fail_ticks} failed")
        # Inertness test rewritten 2026-07-27: fail-ONLY lessons are
        # near-impossible at a ~96% pass rate (any retrieved lesson
        # accrues a success), so their absence was a metric artifact,
        # not evidence. The arm is inert only if the success side is
        # clearly flowing while not a single failure tick ever landed.
        if _fail_ticks == 0 and _succ_ticks >= 20:
            lines.append(
                "  ⚠ FAILURE arm has recorded ZERO failed-retrieval ticks "
                "while the success arm flows — the demotion loop looks "
                "inert on this model.")
        elif _fail_ticks > 0 and les["present_on_failure_only"] == 0:
            lines.append(
                "  (failure ticks flow; no fail-only lesson yet — expected "
                "at a high turn pass rate)")

    comp = r.get("competence")
    if comp:
        # §4NM: the profile is recorded, never injected into a prompt
        _gate_note = (f"{comp['total_observations']} total obs — "
                      f"recorded only, not in any prompt")
        lines.append(
            f"\nCOMPETENCE: {comp['cells_total']} cells — {_gate_note}")
        for d, s in sorted(comp["domains"].items(),
                           key=lambda kv: kv[1]["p_success"]):
            lines.append(f"    {d}: {s['p_success']:.0%} (n={s['n']})")

    epi = r.get("episodes")
    if epi:
        lines.append(
            f"\nEPISODES: {epi['total']} "
            f"({epi['pending_consolidation']} pending consolidation, "
            f"{epi['success']} success)")
        # §4M (Lens D F6c): the old single line said "context/cluster"
        # with a context-only numerator — the two counts differ live.
        lines.append(
            f"  context coverage: {epi['with_context']}/{epi['total']} "
            f"({epi['context_coverage_pct']}%), cluster coverage: "
            f"{epi.get('with_cluster', '?')}/{epi['total']} — populated only "
            f"on episodes recorded AFTER the 2026-07-26 field-population fix")

    cal = r.get("calibration")
    if cal:
        # EVERY ratio below is epoch-scoped, because every number beside it
        # is. Rendering an epoch-scoped numerator over the whole-file
        # denominator prints a fraction that contradicts its own percentage
        # (418/1709 shown as 77.3%) and implies the Brier was measured over
        # rows no fit ever read.
        _n_epoch = cal.get("samples_this_epoch", cal["samples_on_disk"])
        _n_other = cal.get("samples_other_epochs") or 0
        # ⚠ The HEADLINE carries the cross-validated Brier, not the in-sample
        # one (audit 2026-08-10). This is the line an operator actually reads;
        # leading it with a number the fit scored on its own rows is where the
        # optimism entered every downstream quote of it.
        # ⚠ THE THIRD SITE READING THIS NUMBER, AND THE ONLY ONE IN THE
        # HEADER. Same missing bool clause as the verdict block below had:
        # `"brier_cv": true` rendered `Brier True (CV)` in the section
        # title. Guarding one site and not its siblings is how the same
        # defect keeps reappearing one line over.
        def _show(v):
            """Render a number, or say it is unreadable.

            ⚠ THE THIRD VALUE IN A GUARDED F-STRING. `_hdr_num` covers
            `brier_cv` and `brier` in the header line; `threshold` sits in
            the SAME f-string unguarded, and `threshold: true` loads fine
            (`float(True)` is 1.0), so the section printed `threshold True`
            on a file certified in force. Same for the three weights on the
            next line. Guarding the values that have bitten so far is how
            the neighbour keeps biting.
            """
            n = usable_number(v)
            return "UNREADABLE" if n is None else v

        def _hdr_num(v):
            return usable_number(v)

        _cv_hdr = _hdr_num(cal.get("brier_cv"))
        _b_hdr = _hdr_num(cal.get("brier"))
        if _cv_hdr is not None and _cv_hdr >= 0:
            _hdr_brier = f"Brier {_cv_hdr} (CV)"
        elif _b_hdr is not None:
            _hdr_brier = f"Brier {_b_hdr} (in-sample)"
        else:
            _hdr_brier = "Brier UNREADABLE"
        # §4ND (operator: "ranking only"): the header the overview shows is
        # no probability claim — the Brier it carried ("0.05") reached
        # greetings as a fact. It names what the score is for.
        _auc_hdr = _hdr_num(cal.get("auc"))
        lines.append(
            f"\nCALIBRATION: confidence is a RANKING, not a probability — {_n_epoch} samples, "
            + (f"ranking AUC {_auc_hdr:.2f}, " if isinstance(_auc_hdr, (int, float)) and _auc_hdr >= 0 else "")
            + "deep verification on the lowest 15% of recent turns"
            + (f" · {_n_other} older-epoch rows excluded from the fit"
               if _n_other else "")
            + f" · ({_hdr_brier} — not a calibrated probability)")
        # ⚠ ABSENT MAY BE OMITTED; PRESENT-BUT-UNUSABLE MUST BE NAMED.
        # Dropping the clause for an unusable value is the silent-omission
        # anti-pattern this same round fixed for the negative-class line.
        _w_effort = (_show(cal["w_effort"]) if "w_effort" in cal else None)
        lines.append(
            f"  weights: entropy {_show(cal.get('w_entropy'))}, "
            f"competence {_show(cal.get('w_competence'))}"
            + (f", effort {_show(_w_effort)}"
               if _w_effort is not None else "")
            + f"; outcomes {cal['outcome_pos']}+/{cal['outcome_neg']}-"
            # §4ND: the unverified placeholder is not a success
            + (f" (of which {cal['outcome_placeholder']} are the unverified "
               f"placeholder, not outcomes)" if cal.get("outcome_placeholder") else "")
            + (f" ({cal['outcome_verified_pos']}+/"
               f"{cal['outcome_verified_neg']}- verifier-checked)"
               if cal.get("outcome_verified_pos") is not None else ""))
        if cal.get("label_variance") is not None:
            _srcs = cal.get("label_sources") or {}
            lines.append(
                f"  labels: variance {cal['label_variance']} over "
                f"{cal['label_distinct_values']} distinct values "
                f"(mean {cal['label_mean']}) · sources "
                + ", ".join(f"{k}={v}" for k, v in sorted(_srcs.items())))
            if cal["label_distinct_values"] <= 2:
                lines.append(
                    "    → BINARY/near-constant: a label this flat caps what "
                    "any fit can learn, regardless of sample count")
        # §4BF 1c (R3 review): the bench-flood detector must reach the
        # OPERATOR surface, not just --json — render the pre-cap origin
        # mix whenever bench rows exist in the current epoch. .get()s
        # throughout (R4 review, REPRODUCED: a bench-only epoch empties
        # the capped population, `_label_health([])` returns {}, and the
        # first cut of this block — which had also swallowed the
        # binary-flat warning above into its guard — took the whole
        # report down with a KeyError at exactly maximal flood).
        _raw_org = cal.get("label_origins_raw") or {}
        if _raw_org.get("bench"):
            # §4MI: say what the fit DID with the bench rows — the
            # direction gate (`bench_verdict`) can exclude them wholesale,
            # and then "blends at ≤1:1" describes a population it removed
            _bv = str(cal.get("bench_verdict") or "")
            lines.append(
                f"  origin mix (current epoch, pre-cap): "
                + ", ".join(f"{k}={v}" for k, v in sorted(_raw_org.items()))
                + (" — the direction gate EXCLUDED bench (bench_verdict=no): the fit is user rows only"
                   if _bv == "no" else
                   " — admitted to the fit (the equal-mass cap was removed in §4NM)"))
            if not _raw_org.get("user"):
                lines.append(
                    "    → BENCH-ONLY epoch: zero real rows — the fit "
                    "consumes nothing until real traffic arrives")
        if cal.get("effort_observed_samples") is not None:
            lines.append(
                f"  turn-effort measured on {cal['effort_observed_samples']}/"
                f"{_n_epoch} samples"
                + (f", separation {cal['effort_separation_sigmas']}σ"
                   if cal.get("effort_separation_sigmas") is not None else ""))
        _sep, _sep_min = (cal.get("entropy_separation_sigmas"),
                          cal.get("separation_min_sigmas"))
        _obs_short = cal["entropy_observed_samples"] < cal["entropy_min_samples_gate"]
        _class_short = (cal["entropy_observed_pos"] == 0
                        or cal["entropy_observed_neg"] == 0)
        if cal["entropy_learnable"]:
            _ent_note = "LEARNABLE — w_entropy is fit on these"
        elif _obs_short or _class_short:
            _ent_note = (
                f"w_entropy pinned to 0 until >={cal['entropy_min_samples_gate']} "
                "observed samples of both outcome classes accumulate. Most "
                "stored samples predate the 2026-07-27 n_probs fix "
                "(tool-attached streamed generations now request "
                "llama.cpp-native n_probs; the OAI logprobs flag is a hard "
                "400 on tools+stream) — coverage should climb from here. If "
                "it stays at 0 the probe is broken again, not the corpus")
        else:
            # The OTHER half of the gate. Saying "wait for 30 samples" while
            # 418 sit on disk is the report contradicting itself — the corpus
            # is sufficient and the FEATURE is the problem.
            _ent_note = (
                f"w_entropy pinned to 0 — sample coverage is sufficient, but "
                f"entropy separates the outcome classes by only {_sep}σ "
                f"(need >={_sep_min}σ). This is a FEATURE verdict, not a "
                f"coverage one: more samples will not move it, a better "
                f"signal would")
        lines.append(
            f"  entropy observed on {cal['entropy_observed_samples']}/"
            f"{_n_epoch} samples ({cal['entropy_observed_pct']}%), "
            f"{cal['entropy_distinct_values']} distinct values "
            f"({cal['entropy_observed_pos']}+/{cal['entropy_observed_neg']}- observed)"
            + (f", separation {_sep}σ" if _sep is not None else ""))
        lines.append(f"  → {_ent_note}")
        # ⚠ FOUR NUMBERS IN ONE EXPRESSION, TWO OF THEM GUARDED. `_num` was
        # applied to `brier` and `brier_cv` and not to these, so
        # `"brier_base_rate": true` printed `the base-rate predictor (True)`
        # AND drove the comparison against 1.0, manufacturing "beats" for a
        # 0.9-Brier model. Guarding a name list is how the same defect keeps
        # reappearing on the neighbour.
        def _n(v):
            return usable_number(v)

        _br, _bb = _n(cal.get("brier_raw")), _n(cal.get("brier_base_rate"))
        # ⚠ A WARNING NESTED INSIDE AN UNRELATED GUARD IS NOT A WARNING.
        # This notice sat below the `_br`/`_bb` check, so a params file
        # `load_params` REFUSES that also lacks `brier_raw` or
        # `brier_base_rate` — or carries their -1.0 "not computed" sentinel —
        # rendered a complete, confident section with no warning at all. The
        # same nesting silently deleted the under-powered notice, the
        # CONSUMER DEAD line and the whole ablation table: eight lines
        # vanishing on one bad number, with the suite green because the pin
        # only asserted the report still contained "LESSONS:".
        if not cal.get("params_in_force", True):
            lines.append(
                f"  ⚠ NOT IN FORCE — `load_params` refuses this file "
                f"(schema on disk {cal.get('schema_on_disk')!r}; it also "
                f"refuses bad JSON, a non-dict, and unusable required "
                f"numerics), so the agent is running on hardcoded "
                f"defaults and NOTHING below describes the live scorer")
        # ⚠ THESE TWO PREDICATES MUST BE COMPLEMENTS, AND WERE NOT. `NaN < 0`
        # and `NaN >= 0` are BOTH False, so a NaN baseline fell through the
        # announcement AND the block: no `NOT IN FORCE` (the file loads
        # fine), no `NO USABLE BASELINE`, and the verdict line, the in-sample
        # line, the negative-class line, the CONSUMER DEAD notice and the
        # whole ablation table silently gone. That is the same eight-line
        # amputation the announcement was added to abolish, surviving for the
        # one value class `_base_rate_verdict` next door explicitly hardened
        # with `math.isfinite`. One predicate now, used twice.
        # ⚠ THE isfinite AND >= 0 CLAUSES USED TO LIVE HERE AND ARE NOW DEAD:
        # `_br`/`_bb` come from `_n` -> `usable_number`, which already
        # rejects bools, non-finite values and the `-1.0` "not recorded"
        # sentinel. A mutation run flagged them as an equivalent mutant,
        # which is the tell: a condition no input can falsify reads as
        # protection and provides none. Deleted rather than left standing —
        # the guarantee now has exactly one home, and it is `usable_number`.
        _baseline_ok = _br is not None and _bb is not None
        if not _baseline_ok:
            lines.append(
                f"  ⚠ NO USABLE BASELINE — brier_raw={cal.get('brier_raw')!r},"
                f" brier_base_rate={cal.get('brier_base_rate')!r}. The verdict"
                f" and the feature table below are omitted, not passed")
        if _baseline_ok:
            # Equality is not a loss: when the map converges to predicting the
            # base rate (the honest outcome while no feature carries weight)
            # the two are identical, and calling that "LOSES TO" reads as a
            # regression that isn't there.
            # ⚠ THE VERDICT IS SCORED ON THE CROSS-VALIDATED BRIER, not the
            # in-sample one (audit 2026-08-10). `brier` is a 2-parameter Platt
            # map scored on the rows it was fitted to; printing it beside
            # `brier_base_rate` is a comparison only one side of which paid
            # for its parameters. Both are shown, labelled, and the WIN/LOSS
            # call is made on the honest number.
            # ⚠ EVERY NUMBER OFF THIS FILE IS UNTRUSTED. `brier_cv` had an
            # isinstance guard with no bool clause (`"brier_cv": true`
            # rendered `Brier True (5-fold CV)`), and `cal["brier"]` had no
            # guard at all — a missing key, a null, or a string reached the
            # `_score < _bb` comparison and raised TypeError, which
            # `introspect` turns into "Learning health unavailable",
            # destroying lessons, episodes, PRM and router reporting over one
            # malformed calibration field. That is the same blast radius the
            # verdict coercion was added to prevent, one variable over.
            def _num(v):
                return usable_number(v)

            _cv = _num(cal.get("brier_cv"))
            _honest = _cv if _cv is not None and _cv >= 0 else None
            _score = _honest if _honest is not None else _num(cal.get("brier"))
            # ⚠ GUARD THE DISPLAY, NOT ONLY THE COMPARISON. `_score` was
            # made safe while three lines below still formatted
            # `cal['brier']` raw, so a bool printed as `Brier True (IN-SAMPLE
            # ...)`. Computing a safe value and then not using it is the
            # same defect as never computing it.
            _brier_shown = (cal.get("brier") if _num(cal.get("brier")) is not None
                            else "UNREADABLE")
            # ⚠ ANNOUNCING THAT A VERDICT CANNOT BE COMPUTED, THEN COMPUTING
            # ONE. Setting `_score = nan` fell through to the comparison
            # chain, where `nan < x` and `nan > x` are both False, so it
            # landed on `else: "matches"` — a MEASURED word, the fabricated
            # neutral `fit()` explicitly refuses to record, printed directly
            # under a notice saying the comparison is impossible. Identical
            # shape to the `_baseline_ok` defect fixed 60 lines above, which
            # is why this is a flag rather than a sentinel value: a sentinel
            # that flows into arithmetic gets compared.
            _score_unusable = _score is None
            if _score_unusable:
                lines.append(
                    "  ⚠ NO USABLE BRIER on file (brier="
                    f"{cal.get('brier')!r}, brier_cv={cal.get('brier_cv')!r})"
                    " — the calibration verdict cannot be computed")
                # ⚠ NO SENTINEL VALUE HERE, DELIBERATELY. This line used to
                # be `_score = float("nan")`, which flowed into the
                # comparison chain below: `nan < x` and `nan > x` are both
                # False, so it fell to `else: "matches"` — a measured word
                # printed under a notice saying the comparison is
                # impossible. `_score_unusable` gates that chain now, so any
                # assignment here is dead code; a mutation run confirmed it
                # (no value is distinguishable), and dead code that looks
                # like a fallback is what invites the next reader to compare
                # it. `_score` stays None.
            # ⚠ THE VERDICT NEEDS THE INTERVAL, NOT THE POINT (queue #8,
            # 2026-08-21). A 1e-6 tolerance makes any wobble a victory: on
            # the live store the model beat the base rate by 0.00108 and this
            # rendered "beats", while the 95% paired-bootstrap CI of that
            # delta is [-0.00246, +0.00017] — it straddles zero, so the
            # honest word is "is indistinguishable from". Same defect as the
            # experiment report's "no difference detected yet": a verdict
            # stated without the power to support it. When no CI is recorded
            # (params written before this existed) fall back to the point
            # comparison, but SAY so rather than implying a tested claim.
            # ⚠ ASK THE STORED VERDICT, do not re-derive it. This block used
            # to recompute the comparison from the raw `lo`/`hi` fields at
            # render time, so the producer's measurement and the consumer's
            # rendering of it were two implementations of one decision and
            # could drift. `beats_base_rate` is decided once by
            # `calibration._base_rate_verdict` and stored on the params;
            # older files predate the field, so the lo/hi path remains as a
            # LABELLED fallback rather than a silent second opinion.
            from .calibration import (BEATS_INDISTINGUISHABLE, BEATS_NO,
                                      BEATS_UNKNOWN, BEATS_YES,
                                      _base_rate_verdict, _known_verdict,
                                      downgrade_for_map, map_applied,
                                      map_applied_in_params)
            # ⚠ CI BOUNDS ARE LEGITIMATELY NEGATIVE — that is what a WIN
            # looks like — so this is the one numeric read besides the
            # ablation deltas that allows them. Routed through the single
            # guard rather than an inline copy: `_have_ci` had drifted from
            # `_base_rate_verdict` on finiteness once already.
            _dlo = usable_number(cal.get("brier_cv_delta_lo"),
                                 allow_negative=True)
            _dhi = usable_number(cal.get("brier_cv_delta_hi"),
                                 allow_negative=True)
            # ⚠ COERCE HERE TOO. This reader takes the RAW params dict
            # (`_load_json`), not `load_params`, so the vocabulary check
            # wired into the loader never ran on the live path. An
            # unhashable value (`[]`, `{}`) then raised out of the
            # membership test below and blanked the ENTIRE learning-health
            # report — lessons, episodes, PRM, router — over one malformed
            # advisory field.
            # ⚠ ABSENCE IS OBSERVABLE HERE TOO, AND MEANS SOMETHING ELSE.
            # `cal` is the COLLECTOR's dict, not the params file. A dict with
            # no `map_status` key is output from a collector older than the
            # field — it reads as applied, exactly as an old params file
            # does. Only an explicit None (a null the collector passed
            # through) fails closed. Reading `.get()` collapsed the two and
            # made every pre-existing caller with a hand-rolled block render
            # as a rejected map.
            _map_in_force = map_applied_in_params(cal)
            _ms = cal.get("map_status")
            _stored_raw = cal.get("beats_base_rate")
            # Absence is the ONLY licence to derive. A present-but-null value
            # is a malformed verdict and fails closed like any other.
            _verdict_present = cal.get("beats_base_rate_present",
                                       _stored_raw is not None)
            # ⚠ THE STORED VERDICT GOES THROUGH THE AUTHORITY TOO. The first
            # version applied `downgrade_for_map` only where the renderer
            # DERIVED a verdict, protecting files where the field is absent
            # and trusting files where it is WRONG. A params file carrying
            # `beats_base_rate="yes"` with `map_status="rejected_inverted"`
            # printed "beats the base-rate predictor" one line above its own
            # admission that the map was discarded — and that combination is
            # not hypothetical: it is exactly what `fit()` wrote before this
            # change, so any file from that build has it.
            _stored = (None if not _verdict_present
                       else (_known_verdict(_stored_raw) if _map_in_force
                             else BEATS_UNKNOWN))
            # ⚠ FOUR WORDS, NOT THREE. `_words` maps only the three
            # PRINTABLE verdicts; quoting it as the vocabulary in an error
            # message omitted `unknown`, which is a verdict.
            _VERDICT_WORDS = (BEATS_YES, BEATS_NO, BEATS_INDISTINGUISHABLE,
                              BEATS_UNKNOWN)
            _words = {BEATS_YES: "beats",
                      BEATS_NO: "LOSES TO",
                      BEATS_INDISTINGUISHABLE: "is INDISTINGUISHABLE from"}
            _ci_note = ""
            # ⚠ BOOLS ARE INTS. `"brier_cv_delta_lo": true` in a params file
            # satisfied this and printed `[95% CI +1.00000..+1.00000]`. The
            # verdict was safe (`_base_rate_verdict` rejects bools) but the
            # interval beside it was fabricated.
            # ⚠ TWO GUARDS, ONE QUESTION, AND THE LOOSER ONE DRIVES THE
            # DISPLAY. `_base_rate_verdict` requires `math.isfinite` on
            # exactly these two values; this display guard got the bool
            # clause and not the finiteness one, so infinite bounds printed
            # `[95% CI of the delta -inf..+inf]` beside a real stored
            # verdict — a fabricated interval, word-for-word the defect the
            # comment above claims the bool clause fixed.
            _have_ci = _dlo is not None and _dhi is not None
            # ⚠ ABSENT IS NOT THE SAME FACT AS WITHHELD, and conflating them
            # let the consumer overturn the producer.
            #
            # This fallback exists for params files written BEFORE the field
            # (key absent → `cal.get` returns None). The first version tested
            # `_stored not in _words`, which is also true of a stored
            # `"unknown"` — and a stored `"unknown"` has exactly one source:
            # `fit()` downgrading the verdict because the map was rejected
            # and the shipped predictor is the RAW composite, so the interval
            # describes a model we refused. Legacy files have the key absent,
            # and `fit()` refuses below 40 samples so `_cv_delta_ci` always
            # returns an interval — meaning `not in _words` fired on the ONE
            # population where re-deriving is provably wrong. Measured: a
            # `discarded_worse` fit storing `unknown` rendered "beats the
            # base-rate predictor", the exact licence the producer withheld.
            _derived = False
            if _stored is None and _have_ci:
                # Pre-field params: derive with the SAME function the
                # producer uses, never a private copy of its logic — AND
                # apply the same rejection rule it applies afterwards.
                #
                # ⚠ THIS USED TO STOP AT `_base_rate_verdict`, WHICH SEES
                # ONLY AN INTERVAL. The rule "a rejected map ships the raw
                # composite, so this interval describes a predictor we
                # refused" lived only in `fit()`. Every params file written
                # before the field existed takes this path — the live one
                # included — so on the anti-correlated corpus this module
                # documents at length, the operator's keep/kill instrument
                # printed "beats the base-rate predictor" for a scorer 3.5x
                # WORSE than a constant. The producer refused the licence and
                # the consumer granted it, one branch above the branch that
                # had just been fixed to stop exactly that.
                _stored = (_base_rate_verdict((None, _dlo, _dhi))
                           if _map_in_force else BEATS_UNKNOWN)
                _derived = True
            if _stored == BEATS_UNKNOWN:
                # The producer measured and REFUSED to answer. Say that;
                # never substitute an answer of our own -- and state the
                # cause we can actually SEE, not the one we assume.
                if not _map_in_force:
                    # ⚠ ONLY CLAIM THE INTERVAL WHEN THERE IS ONE. The first
                    # wording said "while the interval measures the map we
                    # discarded" unconditionally — a dangling reference on a
                    # fit whose `brier_cv_delta_lo` is None (reachable: a
                    # 9-row fit, since `min_samples_for_fit` is a per-tracker
                    # argument). That is the same defect this very sentence
                    # replaced, committed inside its replacement.
                    _why = (" while the interval measures the map we "
                            "discarded" if _have_ci else "")
                    # A blank or non-string status must not render as an
                    # empty gap in the sentence.
                    _shown = _ms if isinstance(_ms, str) and _ms else repr(_ms)
                    _verdict = (f"has NO base-rate comparison — the Platt map "
                                f"was {_shown}, so the shipped predictor is "
                                f"the raw composite{_why}. Not compared "
                                f"against")
                    if _have_ci:
                        _ci_note = (
                            f" [the discarded map's CI {_dlo:+.5f}"
                            f"..{_dhi:+.5f} — does NOT describe what ships]")
                elif (_stored_raw is not None
                      and _stored_raw != BEATS_UNKNOWN):
                    # ⚠ COERCED, NOT RECORDED. The params carried a value
                    # outside the vocabulary — a typo, a hand edit, a verdict
                    # from a future version. Saying only "no comparison" would
                    # hide that the file is malformed, so name it. (The
                    # separate "junk" branch that used to do this became dead
                    # code the moment coercion moved ahead of it; a guard that
                    # cannot run is worse than no guard.)
                    _verdict = (f"has NO usable base-rate verdict — the "
                                f"params record {_stored_raw!r}, which is not "
                                f"one of {sorted(_VERDICT_WORDS)}. Not "
                                f"compared "
                                f"against")
                elif _have_ci:
                    # ⚠ THE THIRD ARM SAID "no usable interval was recorded"
                    # AND THEN PRINTED THE INTERVAL. Arms 1 and 2 were both
                    # fixed to claim only what they can observe; this one
                    # kept a cause its own next clause refutes on the same
                    # line. What is true here is narrower: there IS an
                    # interval, and the fit declined to draw a verdict from
                    # it.
                    _verdict = ("has NO base-rate comparison — the fit "
                                "recorded no verdict from its interval. Not "
                                "compared against")
                    # The interval IS the evidence here; keep showing it.
                    _ci_note = (f" [interval on file {_dlo:+.5f}"
                                f"..{_dhi:+.5f}, but the fit recorded no "
                                f"verdict from it]")
                else:
                    _verdict = ("has NO base-rate comparison — no usable "
                                "cross-validated interval was recorded for "
                                "this fit. Not compared against")
                    if _have_ci:
                        _ci_note = (f" [interval on file {_dlo:+.5f}"
                                    f"..{_dhi:+.5f}, but the fit recorded no "
                                    f"verdict from it]")
            elif _stored in _words:
                _verdict = _words[_stored]
                if _have_ci:
                    # ⚠ AND SAY WHEN IT WAS DERIVED. The comment here called
                    # this a "LABELLED fallback rather than a silent second
                    # opinion", and the docs repeated the claim — but the
                    # output was BYTE-IDENTICAL to a stored verdict, so an
                    # operator could not tell a measurement the fit recorded
                    # from one this renderer invented at display time. That
                    # is the live store's path today: its params file
                    # predates the field entirely.
                    _ci_note = (
                        f" [95% CI of the delta {_dlo:+.5f}..{_dhi:+.5f}"
                        f"{' — straddles zero' if _stored == BEATS_INDISTINGUISHABLE else ''}"
                        f"{' — DERIVED at render time; this params file predates the stored verdict' if _derived else ''}]")
                else:
                    # Keep the honesty qualifier. A stored verdict with no
                    # interval behind it is still a point comparison, and
                    # dropping the caveat was this change's own thesis
                    # running backwards.
                    _verdict += " (point estimate only — no CI recorded)"
            elif not _map_in_force:
                # ⚠ THE ARMS THE RULE STOPPED ONE `elif` SHORT OF.
                #
                # `_map_in_force` gated the stored path, the derive path, the
                # refusal wording and both in-sample lines — and not these
                # three. So a params file with `map_status` present but no CI
                # and no verdict (any build between 2026-07-29 and
                # 2026-08-21) printed "beats the base-rate predictor" from a
                # point comparison, for EVERY status, directly above the line
                # explaining that the number behind it describes the map that
                # was thrown away. Byte-identical output for `applied`,
                # `rejected_inverted`, `rejected_step`, `discarded_worse` and
                # `""`.
                #
                # Four review rounds each found the rule reaching every
                # branch the reviewer was looking at and stopping just short
                # of one more. Gating the arms individually is what kept
                # producing that; this is the last arm, and the `elif` chain
                # is now exhaustively covered.
                _verdict = ("has NO base-rate comparison — the Platt map was "
                            f"{cal.get('map_status')!r}, so the shipped "
                            "predictor is the raw composite and this is a "
                            "point comparison against the DISCARDED map. Not "
                            "compared against")
            elif _score_unusable:
                _verdict = ("has NO base-rate comparison — the Brier on file "
                            "is unusable, so nothing was compared against")
            elif _score < _bb - 1e-6:
                _verdict = "beats (point estimate only — no CI recorded)"
            elif _score > _bb + 1e-6:
                _verdict = "LOSES TO (point estimate only — no CI recorded)"
            else:
                # ⚠ THE THIRD ARM HAD NO CAVEAT. `beats` and `LOSES TO` both
                # carry "(point estimate only — no CI recorded)"; `matches`
                # asserted a bare equivalence, which is a stronger claim than
                # either of them and the one this block's own comment says
                # must never be implied without a test behind it.
                _verdict = "matches (point estimate only — no CI recorded)"
            if _honest is not None:
                lines.append(
                    f"  Brier {_honest} (5-fold CV, out-of-sample) {_verdict} "
                    f"the base-rate predictor ({_bb}){_ci_note}")
                # §4EB: the OTHER question, on its own line, so a reader
                # cannot take the Brier sentence for the whole answer. Same
                # absent-vs-null discipline, and the interval is printed
                # beside the point estimate because the verdict is decided
                # from it.
                _rk_present = cal.get("ranks_outcomes_present",
                                      cal.get("ranks_outcomes") is not None)
                _rk = (_known_verdict(cal.get("ranks_outcomes"))
                       if _rk_present else BEATS_UNKNOWN)
                _auc, _alo, _ahi = (cal.get("auc"), cal.get("auc_ci_lo"),
                                    cal.get("auc_ci_hi"))
                _rk_words = {BEATS_YES: "RANKS outcomes",
                             BEATS_NO: "ranks them BACKWARDS",
                             BEATS_INDISTINGUISHABLE:
                                 "does NOT rank outcomes (spans chance)"}
                if _rk in _rk_words and isinstance(_auc, (int, float)) \
                        and not isinstance(_auc, bool) and _auc >= 0:
                    _auc_ci = ""
                    if all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           for v in (_alo, _ahi)):
                        _auc_ci = f" [95% CI {_alo:.3f}..{_ahi:.3f}]"
                    lines.append(
                        f"  AUC {_auc:.3f}{_auc_ci} — the score "
                        f"{_rk_words[_rk]}. A RANKING use is licensed by "
                        f"THIS line; a probability claim needs the one above.")
                else:
                    lines.append(
                        "  AUC not recorded — discrimination is UNKNOWN for "
                        "this fit, not absent.")
                # ⚠ ON A REJECTION PATH THESE TWO ARE THE SAME NUMBER, and
                # the CV figure above them describes a map that was thrown
                # away. `fit()` sets `brier = brier_raw` and
                # `calibrated = composites` on all three rejection paths, so
                # "in-sample X · raw composite X" printed one value twice and
                # read as two independent measurements agreeing. Meanwhile
                # the shipped predictor -- the raw composite -- has no
                # out-of-sample number anywhere on this line. Say which of
                # these describes what actually runs.
                # ⚠ A SECOND COPY OF THE RULE, IN THE SAME BLOCK, ALREADY
                # DISAGREEING. This kept the pre-fix vocabulary
                # (`not in ("", "applied")`), so for `map_status=""` the
                # verdict line above said the map was rejected while this
                # line said it was applied and dropped the "SAME number"
                # qualification — on the exact value the shared helper was
                # written for. Extracting an authority does nothing if the
                # old expression survives beside it.
                _rejected = not _map_in_force
                if _rejected:
                    lines.append(
                        f"    in-sample {_brier_shown} · raw composite {_br} "
                        f"— the map was {cal.get('map_status')!r}, so these "
                        f"are the SAME "
                        f"number and the CV figure above describes the "
                        f"DISCARDED map, not what ships")
                else:
                    lines.append(
                        f"    in-sample {_brier_shown} · raw composite {_br} "
                        f"— in-sample is NOT performance, it is the fit "
                        f"scored on its own rows")
            else:
                # ⚠ THE OTHER HALF OF THE SAME LINE. `_ci_note` was silently
                # dropped on this branch, and it never received the
                # "these are the SAME number" qualification the CV branch got
                # — so a rejected map with no CV recorded printed
                # `in-sample X … raw composite X`, one value twice, exactly
                # the shape that qualification exists to abolish. Fixing a
                # line means fixing every branch that prints it.
                _same = ("" if _map_in_force
                         else f" — the map was {cal.get('map_status')!r}, so "
                              f"these are the SAME number")
                lines.append(
                    f"  Brier {_brier_shown} (IN-SAMPLE — no CV recorded, "
                    f"treat as optimistic) {_verdict} the base-rate predictor "
                    f"({_bb}); raw composite {_br}{_same}{_ci_note}")
            # ⚠ AND WHICH ROWS THEY REST ON (§4EO). The verdicts above are
            # measured on rows carrying a VERDICT, not on `n_fitted` — 402 of
            # 1074 live. Printing only the fitted count invites reading
            # `n=1074` as the evidence behind an `indistinguishable`, when
            # two thirds of those rows are the unverified prior whose value
            # IS the base-rate comparand. `delta_halfwidth` then separates the
            # two states the word covers: a measured tie, and a comparison
            # that never had the resolution to find one.
            _nvr = usable_number(cal.get("n_verdict_rows"))
            _nup = usable_number(cal.get("n_unverified_prior"))
            _hw = usable_number(cal.get("delta_halfwidth"))
            if _nvr is not None and _nup is not None:
                lines.append(
                    f"  verdict population: {int(_nvr)} row(s) carrying a "
                    f"verdict; {int(_nup)} placeholder row(s) (unverified "
                    f"prior) fitted but NOT judged on")
            if _hw is not None:
                _dlo2 = usable_number(cal.get("brier_cv_delta_lo"),
                                      allow_negative=True)
                _dhi2 = usable_number(cal.get("brier_cv_delta_hi"),
                                      allow_negative=True)
                _obs = (abs((_dlo2 + _dhi2) / 2.0)
                        if _dlo2 is not None and _dhi2 is not None else None)
                lines.append(
                    f"  resolution: this corpus cannot resolve a Brier "
                    f"difference smaller than {_hw:+.5f}"
                    # §4ND: the "cannot measure" reading only when the
                    # effect really is smaller than the resolution (it printed
                    # whatever the verdict, 0.0049 resolution vs 0.0058 effect)
                    + ((f" — the observed effect is {_obs:.5f}, "
                        f"{_hw / _obs:.1f}x smaller, so `indistinguishable` "
                        f"here means CANNOT MEASURE, not measured-equal")
                       if _obs and _obs < _hw else
                       (f" — the observed effect ({_obs:.5f}) is above it" if _obs else "")))
            # THE NUMBER EVERY OTHER VERDICT RESTS ON. Every feature verdict,
            # weight and Brier comparison here is estimated from the negative
            # class, and the live store carries 18 of them in 694 rows.
            _nneg, _n = cal.get("n_negative"), cal.get("n_samples")
            # ⚠ `isinstance(True, int)` IS TRUE, and this is the line the
            # comment above calls "THE NUMBER EVERY OTHER VERDICT RESTS ON".
            # `_n`, `_num`, `_hdr_num` and the feature filter all reject
            # bools; this one did not, so `n_negative: true` rendered
            # `negative class: True/1258 (0.1%) ⚠ UNDER-POWERED` on a file
            # certified in force, with no warning anywhere.
            _counts_ok = (isinstance(_nneg, int) and not isinstance(_nneg, bool)
                          and isinstance(_n, int) and not isinstance(_n, bool)
                          and _n)
            if not _counts_ok:
                # ⚠ THE GUARD ADDED LAST ROUND DELETED THE LINE IN SILENCE —
                # the line this block calls "THE NUMBER EVERY OTHER VERDICT
                # RESTS ON". The sibling fix in the same commit counts and
                # names what it drops; this one did not. Announcing an
                # omission is the rule, not a courtesy.
                lines.append(
                    f"  ⚠ NEGATIVE-CLASS COUNT UNREADABLE (n_negative="
                    f"{_nneg!r}, n_samples={_n!r}) — every verdict above is "
                    f"estimated from a population this report cannot size")
            if _counts_ok:
                _pct = _nneg / _n
                _warn = ("  ⚠ UNDER-POWERED" if _nneg < 50 else "")
                lines.append(
                    f"  negative class: {_nneg}/{_n} ({_pct:.1%}){_warn}"
                    + (" — every verdict on this screen is estimated from "
                       f"those {_nneg} rows" if _warn else ""))
            # ⚠ THE ONE THING THIS SCORE IS LICENSED TO DO (§4ER). Everything
            # above says what it may NOT be read as; this is the use it MAY
            # be put to, and until 2026-09-04 it had none at all — confidence
            # computed 865 times, `below_threshold` 118 times, arbitrations 0.
            # Gated on the licence itself, so if the score stops ranking the
            # list goes silent instead of quietly becoming decoration.
            try:
                from .calibration import CalibrationTracker as _CT
                _cal_dir = Path(str(memory_dir)).parent / "calibration"
                _lows, _why = _CT(_cal_dir).lowest_confidence_turns(
                    limit=5, days=7)
                if _lows:
                    lines.append(
                        "  LOWEST-CONFIDENCE turns this week (the ranking use "
                        "`ranks_outcomes` licenses — NOT probabilities):")
                    for _s in _lows:
                        _tag = ("" if _s.outcome >= 0.5
                                else "  ← and it did go wrong")
                        lines.append(
                            f"    {_s.ts[:16]}  score {_s.composite:.3f}  "
                            f"{_s.domain or '—'}  req {(_s.req_id or '?')[:12]}"
                            f"{_tag}")
                elif _why:
                    # Absent is not withheld: say which.
                    lines.append(f"  lowest-confidence ranking withheld: {_why}")
            except Exception as _lcx:  # noqa: BLE001 — a report must not die
                logger.debug("lowest-confidence list skipped: %s", _lcx)

            # ⚠⚠ WHO ACTS ON THIS (audit 2026-08-10). The threshold's ONLY
            # consumer is metacog arbitration, and that call site is hard-
            # gated by `_METACOG_ARBITER_ENABLED` in core/agent.py — a module
            # constant, False, independent of every flag. Measured over 209
            # metacog summaries in the live log: confidence computed 865
            # times, `below_threshold` fired 118 times, arbitrations **0**.
            #
            # Stated HERE, beside the number, because a reader who sees
            # "threshold 0.837" reasonably assumes something happens at 0.837.
            # Nothing does. That is not an argument for deleting the stack —
            # the samples are what a future consumer would learn from — but
            # quoting its quality without this line overstates what it buys.
            try:
                from . import agent as _agent_mod
                _arb_on = bool(getattr(_agent_mod,
                                       "_METACOG_ARBITER_ENABLED", False))
            except Exception:  # noqa: BLE001
                _arb_on = None
            # §4ND: since §4EC (09-04) the low-confidence verification depth
            # ALSO acts on this score — the "CONSUMER DEAD" line was false
            # for five weeks and the model repeated it to the owner
            try:
                from .verifier import _conf_depth_enabled as _cde
                _depth_on = bool(_cde())
            except Exception:  # noqa: BLE001
                _depth_on = None
            if _arb_on is False and _depth_on:
                lines.append(
                    "  consumer: low-confidence verification depth (§4EC) — "
                    "the lowest 15% of recent turns get a 3-sample verdict; "
                    "metacog arbitration stays off.")
            elif _arb_on is False and _depth_on is False:
                lines.append(
                    "  ⚠ CONSUMER DEAD: metacog arbitration is hard-disabled "
                    "(_METACOG_ARBITER_ENABLED) and low-confidence depth is "
                    "off (GHOST_VERIFY_DEPTH_CONF=0) — NOTHING acts on this "
                    "score. Treat these numbers as a measurement.")
            # ⚠ THE GUARD WENT ON A NAME LIST, NOT ON THE BOUNDARY. `brier`,
            # `brier_cv` and `brier_base_rate` were each guarded by name
            # while this dict raised three separate ways — a list has no
            # `.items()`, and a string or None value dies in `-kv[1]` and in
            # `_v > 1e-6`. Each one blanked the entire report through
            # `introspect`. Anything read off this file is untrusted; filter
            # to what is actually usable instead of naming the fields that
            # have bitten so far.
            _fc_raw = cal.get("feature_contrib")
            _fc = {}
            if isinstance(_fc_raw, dict):
                # ⚠ SAME GUARD. This filter rejected bools and admitted
                # `Infinity`, which sorts to the TOP of the table as the
                # largest possible held-out delta — on the table the report
                # itself tells the operator to believe over the σ-gate.
                # Negatives ARE meaningful here (dropping the feature helps),
                # so this is the one numeric read that allows them.
                _fc = {k: v for k, v in _fc_raw.items()
                       if usable_number(v, allow_negative=True) is not None}
                _dropped = len(_fc_raw) - len(_fc)
                if _dropped:
                    lines.append(
                        f"  ⚠ {_dropped} feature-ablation entr"
                        f"{'y' if _dropped == 1 else 'ies'} unusable and "
                        f"omitted (non-numeric values on file)")
            elif _fc_raw is not None:
                lines.append(
                    f"  ⚠ feature_contrib is {type(_fc_raw).__name__}, not a "
                    f"mapping — the ablation table is omitted")
            if _fc:
                lines.append("  feature ABLATION (held-out Brier delta; "
                             ">0 = dropping it hurts, i.e. it helps):")
                for _k, _v in sorted(_fc.items(), key=lambda kv: -kv[1]):
                    # ⚠ TWO WORDS FOR THREE OUTCOMES. A NEGATIVE delta means
                    # dropping the feature IMPROVES held-out Brier — the
                    # feature is actively hurting — and it rendered as "no
                    # measurable contribution". The live params file carries
                    # `competence_component: -4.3e-05` and reads exactly that
                    # today, beside an `effort_component` 20x smaller in
                    # magnitude labelled "helps".
                    if _v > 1e-6:
                        _m = "helps"
                    elif _v < -1e-6:
                        _m = "HURTS — dropping it IMPROVES held-out Brier"
                    else:
                        _m = "no measurable contribution"
                    lines.append(f"    {_k:<24} {_v:+.6f}  {_m}")
                lines.append(
                    "    ⚠ where ABLATION and the σ-gate disagree, believe the "
                    "ablation: the gate is a difference-of-means test and at "
                    "this negative count it cannot resolve what it judges")
            # ⚠ THE SAME TABLE ON THE STATISTIC THAT HAS POWER (§4EO). Every
            # ablation delta above is under 0.001 at this base rate — a table
            # nobody can act on, which is how a feature sat at the largest
            # weight for a month with no reading on whether it works. Same
            # untrusted-input discipline: filter to what is usable and SAY
            # what was dropped.
            _ca_raw = cal.get("component_auc")
            _ca = {}
            if isinstance(_ca_raw, dict):
                _ca = {k: v for k, v in _ca_raw.items()
                       if usable_number(v) is not None}
                _ca_dropped = len(_ca_raw) - len(_ca)
                if _ca_dropped:
                    lines.append(
                        f"  ⚠ {_ca_dropped} component-rank entr"
                        f"{'y' if _ca_dropped == 1 else 'ies'} unusable and "
                        f"omitted (non-numeric values on file)")
            elif _ca_raw is not None:
                lines.append(
                    f"  ⚠ component_auc is {type(_ca_raw).__name__}, not a "
                    f"mapping — the component-rank table is omitted")
            if _ca:
                lines.append("  component RANK (AUC on the verdict rows; "
                             "0.5 = chance):")
                for _k, _v in sorted(_ca.items(), key=lambda kv: -kv[1]):
                    if _v > 0.55:
                        _m = "orders outcomes"
                    elif _v < 0.45:
                        _m = "orders outcomes BACKWARDS"
                    else:
                        _m = "chance — no ordering"
                    lines.append(f"    {_k:<24} {_v:.4f}  {_m}")
        fh = cal.get("feature_health") or {}
        if fh:
            live = cal.get("live_features") or []
            lines.append(
                f"  features: {len(live)}/{len(fh)} live"
                + (f" ({', '.join(live)})" if live else " — NONE discriminate"))
            for name, st in fh.items():
                _flag = str(st.get("verdict") or
                            ("DEAD" if st.get("dead") else "live"))
                _flag = "DEAD" if _flag == "dead" else _flag
                lines.append(
                    f"    {name}: n={st.get('n')} distinct={st['distinct']} "
                    f"separation={st['separation']}"
                    # The σ is what decides the verdict; printing the raw
                    # delta alone leaves DEAD looking arbitrary next to a
                    # non-zero separation.
                    + (f" ({st['separation_sigmas']}σ)"
                       if st.get("separation_sigmas") is not None else "")
                    + f" [{_flag}]")
            if any(v.get("verdict") == "insufficient" for v in fh.values()):
                lines.append(
                    "    (insufficient = <10 eligible samples or one outcome "
                    "class — no honest verdict yet; entropy is judged over "
                    "OBSERVED samples only)")

    lv = r.get("liveness")
    if lv and lv.get("rows"):
        aw = int(lv["alarm_window_hours"])
        cw = int(lv["context_window_hours"] / 24)
        lines.append(f"\nMECHANISM LIVENESS (registry-driven — every "
                     f"registered loop appears, firing or not):")
        if lv.get("ledger_missing"):
            # ⚠ Checked BEFORE agent_silent: a missing ledger is also silent,
            # so reporting idleness first would describe the symptom and hide
            # the cause. This whole table is derived from that file — if it is
            # not there, every zero below means nothing at all.
            lines.append(
                f"  ⚠⚠ LEDGER NOT FOUND at {lv.get('ledger_path')} — the "
                f"counts below are not evidence of anything. This is a "
                f"MISSING INSTRUMENT, not an idle agent.")
        elif lv.get("agent_silent"):
            lines.append(
                f"  ⚠ NO activity of ANY kind in {aw}h — the agent was off or "
                f"fully idle. That is ONE fact, so per-loop alarms are "
                f"withheld; they would all be this.")
        if lv.get("read_truncated"):
            lines.append("  ⚠ ledger tail-read hit its cap — counts below "
                         "UNDERSTATE the window.")
        for row in lv["rows"]:
            mark = "  ✗ DEAD " if row["alarm"] else "    "
            note = "" if row["registered"] else "  ← UNREGISTERED phase"
            lines.append(
                f"{mark}{row['phase']:<24} {row['n_alarm_window']:>4} /{aw}h "
                f"{row['n_context_window']:>5} /{cw}d   "
                f"{row['expectation']}{note}")
            # ⚠ SAY IT, don't just withhold the alarm. A zero with the alarm
            # quietly suppressed is indistinguishable from a zero nobody
            # looked at — the operator has no way to tell "ran and declined"
            # from "we stopped checking". The heartbeat is positive evidence,
            # so it gets a line.
            from .autonomous_activity import (
                EXPECT_PERIODIC as _EXPECT_PERIODIC_NAME)
            # Only a row that COULD have alarmed needs the explanation, and
            # the age comes from the row — a second `time.time()` here can
            # print an age that contradicts the `attempted_recently` it is
            # explaining.
            if (row["expectation"] == _EXPECT_PERIODIC_NAME
                    and row["n_alarm_window"] == 0
                    and row.get("attempted_recently")):
                _age_h = float(row["attempted_age_s"] or 0.0) / 3600.0
                lines.append(
                    f"        └ RAN {_age_h:.1f}h ago and produced nothing — "
                    f"this loop records an OUTCOME, and declining (e.g. an "
                    f"unchanged corpus) is a healthy outcome-free run. Not "
                    f"dead; the heartbeat is a separate signal from the "
                    f"ledger.")
                if int(row.get("n_context_window") or 0) == 0:
                    # §4MI: a decline every 3 h for 17 days silenced the
                    # alarm while the router checkpoint aged to 409 h —
                    # the heartbeat vouched for a loop whose gate never
                    # opened. Not an alarm (the gate may be right), but
                    # said.
                    lines.append(
                        f"        └ ⚠ and no outcome in 7 d: a loop that only ever "
                        f"declines is one whose input never changes — check its gate")
        if lv.get("alarms"):
            lines.append(
                f"  ✗ {len(lv['alarms'])} PERIODIC loop(s) silent for {aw}h: "
                + ", ".join(lv["alarms"]))
        else:
            lines.append("  all periodic loops fired within the alarm window")
        # ⚠ Only PERIODIC alarms. on_output/on_demand/gated phases sit at zero
        # by design, and a monitor that cries on a benign zero is one the
        # operator learns to scroll past — rebuilding by hand the blindness
        # this section exists to remove.

    # ── SUBSYSTEM LIVENESS (probe-per-mechanism) ─────────────────────────
    # The section above covers phases that write to autonomous_activity.jsonl.
    # This one covers everything that does NOT — the gap that let metacog
    # arbitration sit dead and invisible (0 firings against 118 opportunities)
    # until it was found by grepping the log by hand.
    try:
        from .liveness import render as _liveness_render
        # ⚠ GHOST_HOME, not its `system/` child. `memory_dir` is
        # $GHOST_HOME/system/memory, so the home is TWO levels up — the probes
        # resolve "system/<store>/..." relative to it. Passing `.parent`
        # made all 8 probes report NO_SOURCE. That the view said NO_SOURCE
        # rather than quietly reporting zeros is the third state earning its
        # keep on its first real outing.
        lines.append("\n" + _liveness_render(Path(memory_dir).parent.parent))
    except Exception as e:  # noqa: BLE001 — telemetry must never break a report
        lines.append(f"\nSUBSYSTEM LIVENESS: unavailable ({type(e).__name__})")

    # ── LOOP YIELD (§4CS) ────────────────────────────────────────────────
    # The SECOND axis, and the one the macro loop had to be found by hand
    # to expose. Liveness above asks "did this mechanism RUN?"; this asks
    # "did it produce anything anyone CONSUMED?". Dream fired on schedule
    # for six weeks and minted 25 macros that were invoked ZERO times, and
    # every liveness probe would have read FIRED throughout.
    #
    # Extended here rather than given its own introspect action on purpose:
    # a parallel surface is one more thing an operator has to remember to
    # look at, and this report already carries the probe-per-mechanism
    # section it belongs beside.
    try:
        from .liveness import render_yield as _yield_render
        # Same two-levels-up resolution as the liveness block above:
        # `memory_dir` is $GHOST_HOME/system/memory.
        lines.append("\n" + _yield_render(Path(memory_dir).parent.parent))
    except Exception as e:  # noqa: BLE001 — telemetry must never break a report
        lines.append(f"\nLOOP YIELD: unavailable ({type(e).__name__})")

    act = r.get("activity")
    if act:
        # NO top-N CUT. It used to sort by count descending and slice [:8], so
        # a phase at zero — the only kind worth acting on — sorted last and was
        # structurally the first row dropped.
        top = sorted(act.items(), key=lambda kv: -kv[1])
        lines.append("\nBACKGROUND ACTIVITY (recent ledger): "
                     + ", ".join(f"{k}={v}" for k, v in top))
        # These counts are NOT comparable across phases, and reading them as
        # a workload budget is a live trap: reflection appeared to run ~10x
        # less than dream, which looks like a starved loop but is a recording
        # artifact. Reflection only writes a ledger event when it produced
        # OUTCOMES, and it deliberately skips ticks whose trajectory corpus
        # is unchanged since an all-duplicate pass. §4CB R1/R2: dream too now
        # ledgers ONLY cycles that did consolidation work — skip AND error
        # cycles mint nothing (so the zero-rows liveness alarm can fire on a
        # permanently failing dream). A low count here means "little new
        # material", not "under-scheduled" — check the cooldowns instead.
        lines.append(
            "  (per-phase recording policies differ — reflection logs only "
            "outcome-producing runs and skips unchanged-corpus ticks; "
            "maintenance logs only passes that wrote something (digests, "
            "manifests, store care), so a zero can mean quiet OR erroring — "
            "do NOT read these as a workload budget)")

    cw = r.get("cognitive_wiring")
    if cw:
        lines.append("\nCOGNITIVE WIRING (producer → consumer status):")
        sf = cw.get("selfhood", {})
        lines.append(
            f"  selfhood: writes every turn; wake-up prefix "
            + _consumer_state(sf.get("prefix_consumer_enabled"),
                              "unknown (gate not read)")
            + ", also read by introspect (bounded store)")
        # R5 MIN-6: this row was collected and rendered NOWHERE, so the
        # wiring report silently omitted one of its five flag rows — the
        # metacog arbiter, whose 0-lifetime deadness the docs already
        # describe. A collected-but-unrendered row is the same
        # instrument-blindness class in its quietest form.
        _ma = cw.get("metacog_arbiter", {})
        lines.append(
            "  metacog arbiter: dual-solver divergence consumer "
            + _consumer_state(_ma.get("consumer_enabled"),
                              "unknown (gate not read)"))
        lines.append("  calibration: per-turn → idle refit + this telemetry (live)")
        sc = cw.get("self_consistency", {})
        if sc.get("status"):
            lines.append(f"  self_consistency: {sc['status']}")

    op = r.get("optim")
    if op and (op.get("artifacts") or op.get("activation")):
        lines.append("\nPROMPT OPTIMIZATION (GEPA artifacts → read-site activation):")
        arts = op.get("artifacts") or {}
        acts = op.get("activation") or {}
        for name in sorted(set(arts) | set(acts)):
            a = arts.get(name)
            c = acts.get(name) or {}
            applied = c.get("applied", 0)
            fallback = c.get("fallback", 0)
            if a and a.get("valid"):
                state = f"tuned ({a['chars']} chars)"
                # ⚠ The applies counter is PER-PROCESS (§4L Lens-C
                # MINOR-2): a headless `scripts/learning_health.py` run
                # has trivially 0 for every artifact and used to render
                # the alarm regardless of live-agent truth. Only the
                # in-process view (introspect action='learning') can
                # distinguish "read-site never ran" from "wrong
                # process"; the script now says which one it is.
                import sys as _sys
                _headless = not _sys.argv[0].endswith(("main.py", "-m"))
                if applied == 0 and _headless:
                    flag = ("  (0 applies IN THIS PROCESS — per-process "
                            "counter; run introspect action='learning' "
                            "in the agent for live truth)")
                elif applied == 0:
                    flag = "  ⚠ tuned but 0 applies since boot"
                else:
                    flag = ""
            elif a:
                state = "artifact INVALID"
                flag = ""
            else:
                state = "no artifact (baseline)"
                flag = ""
            lines.append(
                f"  {name}: {state}; applied {applied} / fallback {fallback}{flag}")

    esc = r.get("verifier_escalation") or {}
    if esc.get("present"):
        _span = ""
        if esc.get("first_ts"):
            _span = (f" — ledger span {str(esc['first_ts'])[:10]} → "
                     f"{str(esc['last_ts'])[:10]}; pre-ledger history is "
                     f"log-only and NOT counted here")
        lines.append(f"\nVERIFIER ESCALATION (§4F false-positive watch){_span}:")
        for arm in sorted(esc.get("arms") or {}):
            s = esc["arms"][arm]
            rate = s.get("overturn_rate")
            rate_s = "n/a" if rate is None else f"{rate:.0%} overturned"
            extra = ""
            if s.get("withheld"):
                extra += f", {s['withheld']} withheld"
            if s.get("unavailable"):
                extra += f", {s['unavailable']} unavailable (call spent)"
            if s.get("downgraded"):
                extra += (f", {s['downgraded']} downgraded "
                          f"(gloss-tier, no call)")
            # Every category renders or the arithmetic doesn't add up to
            # `n` and the mechanical layer reads as "never fired"
            # (round-2 review). ⚠ overturn_rate is conditioned on
            # SURVIVING the mechanical layer since 2026-08-07, and
            # strong-UNCERTAIN moved from `overturned` to
            # `replaced_uncertain` — do not trend the rate across that
            # date without re-baselining.
            if s.get("mechanically_dismissed"):
                extra += (f", {s['mechanically_dismissed']} mech-dismissed "
                          f"(no call)")
            if s.get("mechanically_upheld"):
                extra += (f", {s['mechanically_upheld']} mech-upheld "
                          f"(protected, no call)")
            if s.get("truncation_guard"):
                extra += f", {s['truncation_guard']} truncation-guarded"
            if s.get("replaced_uncertain"):
                extra += (f", {s['replaced_uncertain']} replaced-uncertain "
                          f"(strong judge punted)")
            if s.get("claim_binding"):
                extra += (f", {s['claim_binding']} claim-binding "
                          f"(validated quotes decided, no appeal)")
            if s.get("other"):
                # §4L R2 NEW-6: the counter existed but the renderer
                # omitted it — vocabulary drift invisible again, one
                # layer up.
                extra += (f", {s['other']} OTHER (unknown outcome "
                          f"strings — vocabulary drift, investigate)")
            lines.append(f"  {arm}: {s['n']} escalations — {rate_s} "
                         f"({s['overturned']}/{s['overturned'] + s['upheld']} "
                         f"decided){extra}")
    elif esc:
        # An absent ledger is only news once the agent has been up a while;
        # say WHICH silence it is rather than printing nothing.
        lines.append(f"\nVERIFIER ESCALATION: {esc.get('reason', 'unknown')} "
                     "— the file appears on the first escalation after a boot")

    fl = r.get("framing_leak") or {}
    if fl.get("available"):
        n, tot = fl["corrupt_calls"], fl["calls"]
        if fl.get("regression"):
            lines.append("\n⚠ TOOL-CALL FRAMING LEAK — REGRESSION:")
            lines.append(
                f"  {n}/{tot} calls carry leaked framing; newest "
                f"{fl['last_seen'][:16]} is AFTER the 2026-07-31 fix "
                f"— the dual-dialect corruption is back {fl.get('by_tool')}")
        elif n:
            lines.append(
                f"\nTOOL-CALL FRAMING LEAK: {n}/{tot} calls, all historical "
                f"(newest {fl['last_seen'][:16]}, pre-fix) — no action")
        else:
            lines.append(
                f"\nTOOL-CALL FRAMING LEAK: none in {tot} calls "
                f"across {fl['scanned']} trajectories")
    elif fl:
        lines.append(f"\nTOOL-CALL FRAMING LEAK: {fl.get('reason', 'unknown')}")

    return "\n".join(lines)
