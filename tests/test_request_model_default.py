"""A chat request without `model` runs on the configured model.

The web UI leaves `model` out on purpose (app.js: the agent "uses its
configured model"), and the API route computes `requested_model or
configured_model`, but handle_chat read `body.get("model",
"qwen-3.6-35b-a3")`, so that literal went to the upstream as the model name.
llama.cpp ignores the field; Ollama answers 404, so a deployment whose main
model is served by Ollama could not chat from the web UI at all.

World where each pin fails: handle_chat stops taking its model from
_request_model, the helper falls back to the literal while a model is
configured, or a model the client did name is overridden.
"""
import ast
import inspect
import types
from unittest.mock import MagicMock

from ghost_agent.core import agent as ag
from ghost_agent.core.agent import _request_model

ARGS = types.SimpleNamespace(model="ghost-main:32k")


def test_a_request_without_model_uses_the_configured_model():
    assert _request_model({"messages": []}, ARGS) == "ghost-main:32k"
    assert _request_model({"model": ""}, ARGS) == "ghost-main:32k"


def test_a_named_model_is_kept():
    assert _request_model({"model": "qwen3.8:27b"}, ARGS) == "qwen3.8:27b"


def test_no_configured_model_keeps_the_old_default():
    assert _request_model({}, types.SimpleNamespace()) == "qwen-3.6-35b-a3"
    assert _request_model({}, MagicMock()) == "qwen-3.6-35b-a3"     # test doubles' args


def test_handle_chat_takes_its_model_from_the_helper():
    for n in ast.walk(ast.parse(inspect.getsource(ag))):
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "handle_chat":
            calls = [c for c in ast.walk(n) if isinstance(c, ast.Call)
                     and getattr(c.func, "id", "") == "_request_model"]
            assert calls, "handle_chat no longer resolves the model with _request_model"
            assert 'body.get("model", "qwen-3.6-35b-a3")' not in ast.unparse(n)
            return
    raise AssertionError("handle_chat not found")
