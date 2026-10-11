"""Tests for optim.signatures."""

import pytest

from ghost_agent.optim.signatures import (
    OptimizableSignature,
    PLANNING_SIGNATURE,
    TOOL_SELECTION_SIGNATURE,
    SIGNATURES,
)




def test_planning_signature_inputs_outputs():
    assert "user_request" in PLANNING_SIGNATURE.inputs
    assert "plan" in PLANNING_SIGNATURE.outputs
    assert PLANNING_SIGNATURE.scope == "planning"


def test_tool_selection_signature_scope():
    assert TOOL_SELECTION_SIGNATURE.scope == "tool_selection"




def test_out_of_scope_signature_rejected():
    with pytest.raises(ValueError):
        OptimizableSignature(
            name="forbidden",
            scope="safety",  # not in allow-list
            inputs={"x": "x"},
            outputs={"y": "y"},
            instruction="do dangerous things",
        )


def test_signature_requires_name():
    with pytest.raises(ValueError):
        OptimizableSignature(
            name="",
            scope="planning",
            inputs={"x": "x"},
            outputs={"y": "y"},
            instruction="hi",
        )


def test_signature_requires_inputs():
    with pytest.raises(ValueError):
        OptimizableSignature(
            name="n",
            scope="planning",
            inputs={},
            outputs={"y": "y"},
            instruction="hi",
        )


def test_signature_requires_outputs():
    with pytest.raises(ValueError):
        OptimizableSignature(
            name="n",
            scope="planning",
            inputs={"x": "x"},
            outputs={},
            instruction="hi",
        )


def test_compile_baseline_contains_instruction_and_fields():
    compiled = PLANNING_SIGNATURE.compile_baseline()
    assert "Decompose" in compiled
    assert "user_request" in compiled
    assert "plan" in compiled


def test_dream_prompt_is_not_in_registry():
    """Safety: the dream prompt must NEVER appear in the optimizable
    registry. CLAUDE.md documents its idempotency invariants — GEPA
    would risk violating them."""
    for sig in SIGNATURES.values():
        assert "dream" not in sig.scope.lower()
        assert "dream" not in sig.name.lower()
