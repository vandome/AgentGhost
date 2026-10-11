"""§4CS — a macro whose steps carry `$slot` templates is activatable.

Measured 2026-08-23 on the live store: 26 composed skills, 25 auto-mined and
all at status="proposed", **0 invocations across every auto-mined macro, all
time** — a param-less macro advertises no inputs, so it parks at "proposed"
forever. §4CS made the auto-mint produce a param SCHEMA.

§4NM removed the dream-side miner and skills_auto; the post-§4NN tidy-up
removed the now-unreachable mint (`mint_param_schema`, the tool-schema index,
the macro identity / admission / harvest helpers) with their tests. What is
pinned here is the registry side that stored macros still use: an approved
macro advertises its slots as REQUIRED, runtime args reach the tools, and the
monotone param-less upgrade in `compile_from_pattern`.
"""

import json
import tempfile
from pathlib import Path

import pytest

from ghost_agent.tools.composed_skills import (
    ComposedSkill, ComposedSkillRegistry, SkillStep,
    macro_step_inputs,
)


@pytest.fixture
def registry():
    d = tempfile.mkdtemp()
    try:
        yield ComposedSkillRegistry(storage_dir=Path(d))
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


# ── the artifact is actually activatable, end to end ─────────────────────
class TestActivatable:
    def _mint(self, registry):
        # The schema the (removed) auto-mint produced for this sequence: the
        # declared mode fixed, every other argument a `$slot`.
        tpl = [{"operation": "replace", "path": "$path", "content": "$content",
                "pattern": "$pattern", "replace_with": "$replace_with"},
               {"action": "restart", "name": "$name"}]
        slots = ["path", "content", "pattern", "replace_with", "name"]
        registry.compile_from_pattern(
            "auto_file_system_replace_manage_services_restart",
            [{"tool": "file_system", "description": "", "params": tpl[0]},
             {"tool": "manage_services", "description": "", "params": tpl[1]}],
            "edit then restart", status="proposed")
        return slots

    def test_proposed_macro_is_not_advertised(self, registry):
        self._mint(registry)
        assert registry.to_tool_definitions() == []

    def test_once_approved_it_advertises_its_slots_as_REQUIRED(self, registry):
        slots = self._mint(registry)
        sk = registry.skills["auto_file_system_replace_manage_services_restart"]
        sk.status = "active"
        defs = registry.to_tool_definitions()
        assert len(defs) == 1
        schema = defs[0]["function"]["parameters"]
        assert sorted(schema["required"]) == sorted(slots)
        assert sorted(schema["properties"]) == sorted(slots)
        assert schema["required"], "a param-less macro is the defect itself"

    async def test_runtime_args_reach_the_tools(self, registry):
        # ⚠ `async def`, driven by pytest-asyncio (pytest.ini sets
        # asyncio_mode=auto), NOT `asyncio.run` inside a sync test.
        # `asyncio.run` creates and CLOSES a loop and leaves the policy's
        # current loop closed, which broke 16 later tests in
        # tests/test_auth_rejection_logging.py when the two ran in the same
        # session — the file passed alone and failed in a suite chunk.
        # Same pattern as the sibling tests in test_composed_skills.py.
        self._mint(registry)
        sk = registry.skills["auto_file_system_replace_manage_services_restart"]
        sk.status = "active"
        seen = []

        async def _exec(tool, args):
            seen.append((tool, dict(args)))
            return "OK"

        res = await registry.execute(
            sk.name, _exec,
            {"path": "/live/target.py", "content": "NEW", "pattern": "a",
             "replace_with": "b", "name": "live-svc"})
        assert res["success"], res
        assert seen[0][0] == "file_system"
        assert seen[0][1]["operation"] == "replace"      # the fixed mode
        assert seen[0][1]["path"] == "/live/target.py"   # the runtime slot
        assert seen[1][1] == {"action": "restart", "name": "live-svc"}
        assert "$" not in json.dumps(seen)                # every slot was filled


# ── the parked backlog can acquire a schema ──────────────────────────────
class TestParamlessUpgrade:
    def _paramless(self, registry, status="proposed"):
        sk = ComposedSkill(
            name="auto_file_system_execute", trigger_description="old",
            status=status,
            steps=[SkillStep("file_system", "file_system (step 1)", {}),
                   SkillStep("execute", "execute (step 2)", {})])
        registry.register(sk)
        return sk

    def _incoming(self):
        return [{"tool": "file_system", "description": "read it",
                 "params": {"operation": "read", "path": "$path"}},
                {"tool": "execute", "description": "run it",
                 "params": {"command": "$command"}}]

    def test_a_parked_paramless_macro_gets_its_slots(self, registry):
        sk = self._paramless(registry)
        registry.compile_from_pattern("auto_file_system_execute",
                                      self._incoming(), "d", status="proposed")
        got = {n for st in sk.steps for n in macro_step_inputs(st.param_template)}
        assert got == {"path", "command"}
        assert sk.status == "proposed", "the upgrade must not activate it"

    def test_an_ACTIVE_macro_is_never_rewritten(self, registry):
        sk = self._paramless(registry, status="active")
        before = [dict(st.param_template) for st in sk.steps]
        registry.compile_from_pattern("auto_file_system_execute",
                                      self._incoming(), "d", status="proposed")
        assert [dict(st.param_template) for st in sk.steps] == before

    def test_a_macro_that_already_has_slots_is_not_rewritten(self, registry):
        sk = self._paramless(registry)
        registry.compile_from_pattern("auto_file_system_execute",
                                      self._incoming(), "d", status="proposed")
        first = [dict(st.param_template) for st in sk.steps]
        other = [{"tool": "file_system", "description": "",
                  "params": {"operation": "write", "path": "$elsewhere"}},
                 {"tool": "execute", "description": "",
                  "params": {"command": "$other"}}]
        registry.compile_from_pattern("auto_file_system_execute", other,
                                      "d", status="proposed")
        assert [dict(st.param_template) for st in sk.steps] == first, \
            "the upgrade is monotone: one step, then never again"

    def test_a_tool_mismatch_blocks_the_upgrade(self, registry):
        sk = self._paramless(registry)
        wrong = [{"tool": "browser", "description": "",
                  "params": {"operation": "navigate", "url": "$url"}},
                 {"tool": "execute", "description": "",
                  "params": {"command": "$command"}}]
        registry.compile_from_pattern("auto_file_system_execute", wrong,
                                      "d", status="proposed")
        assert all(st.param_template == {} for st in sk.steps)

    def test_the_upgrade_persists(self, registry):
        self._paramless(registry)
        registry.compile_from_pattern("auto_file_system_execute",
                                      self._incoming(), "d", status="proposed")
        reloaded = ComposedSkillRegistry(storage_dir=registry.storage_dir)
        sk = reloaded.skills["auto_file_system_execute"]
        got = {n for st in sk.steps for n in macro_step_inputs(st.param_template)}
        assert got == {"path", "command"}


# ── the guards review round 1 found unpinned ─────────────────────────────
class TestUpgradeGuardsArePinned:
    def _parked(self, registry, n_steps=3):
        tools = ["file_system", "execute", "manage_services"][:n_steps]
        sk = ComposedSkill(
            name="auto_parked", trigger_description="old", status="proposed",
            steps=[SkillStep(t, f"{t} (step {i+1})", {})
                   for i, t in enumerate(tools)])
        registry.register(sk)
        return sk

    def test_a_SHORTER_incoming_cannot_half_upgrade_a_macro(self, registry):
        """`zip` truncates silently: steps 1-2 would take slots and step 3
        would stay `{}` forever, because the "already has slots" guard
        then blocks every future repair. Irreversibly half-broken, saved
        to disk in that state."""
        sk = self._parked(registry, n_steps=3)
        registry.compile_from_pattern("auto_parked", [
            {"tool": "file_system", "description": "",
             "params": {"operation": "read", "path": "$path"}},
            {"tool": "execute", "description": "", "params": {"command": "$cmd"}},
        ], "d", status="proposed")
        assert all(st.param_template == {} for st in sk.steps), \
            "a length mismatch must block the whole upgrade, not part of it"

    def test_a_LITERAL_ONLY_incoming_does_not_rewrite_forever(self, registry):
        """Without the "incoming carries at least one slot" guard the
        upgrade is not monotone: a literal-only mint rewrites the macro on
        every cycle, each time saving and logging "it now carries runtime
        slots ()"."""
        sk = self._parked(registry, n_steps=2)
        first = [{"tool": "file_system", "description": "",
                  "params": {"operation": "read"}},
                 {"tool": "execute", "description": "", "params": {}}]
        registry.compile_from_pattern("auto_parked", first, "d",
                                      status="proposed")
        assert all(st.param_template == {} for st in sk.steps), \
            "a literal-only incoming carries no slots — nothing to upgrade to"


# ── ROUND 2: the surviving mutants that named real behaviour ────────────
class TestRoundTwoMutationSurvivors:
    def test_the_paramless_UPGRADE_is_announced(self, registry, capsys):
        """The only operator-visible signal that a parked macro was healed
        — the sibling of round 1's CRITICAL, added by the same change and
        never pinned."""
        registry.register(ComposedSkill(
            name="auto_parked2", trigger_description="old", status="proposed",
            steps=[SkillStep("file_system", "", {}), SkillStep("execute", "", {})]))
        registry.compile_from_pattern("auto_parked2", [
            {"tool": "file_system", "description": "",
             "params": {"operation": "read", "path": "$path"}},
            {"tool": "execute", "description": "", "params": {"command": "$c"}},
        ], "d", status="proposed")
        out = capsys.readouterr().out.lower()
        assert "macro schema" in out
        assert "auto_parked2" in out
