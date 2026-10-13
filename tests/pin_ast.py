"""AST helpers for WIRING pins (R1/R4): a pin that walks the parsed tree
fails when the code is deleted even if its text survives in a comment —
which a substring check on `inspect.getsource` does not (the pin-quality
ratchet rejects those). Not a test module."""
import ast
import inspect
import textwrap


def tree(obj) -> ast.AST:
    return ast.parse(textwrap.dedent(inspect.getsource(obj)))


def U(node) -> str:
    """One canonical spelling of a node (ast.unparse, whitespace-collapsed)."""
    return " ".join(ast.unparse(node).split()) if node is not None else ""


def _callee(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    return getattr(f, "id", "")


def calls(t: ast.AST, name: str):
    """Every Call whose callee's last name is `name` (`x.y.name(...)` or `name(...)`)."""
    return [n for n in ast.walk(t) if isinstance(n, ast.Call) and _callee(n) == name]


def call_kw(call: ast.Call, kw: str):
    for k in call.keywords:
        if k.arg == kw:
            return k.value
    return None


def assigns(t: ast.AST, target: str):
    """Every Assign/AugAssign/AnnAssign whose (one) target unparses to `target`."""
    out = []
    for n in ast.walk(t):
        if isinstance(n, ast.Assign):
            if any(U(x) == target for x in n.targets):
                out.append(n)
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            if U(n.target) == target:
                out.append(n)
    return out


def assigned_values(t: ast.AST, target: str):
    return [U(n.value) for n in assigns(t, target) if getattr(n, "value", None) is not None]


def ifs(t: ast.AST, test: str):
    """Every If whose test unparses to `test` (an `elif` is an If in orelse)."""
    return [n for n in ast.walk(t) if isinstance(n, ast.If) and U(n.test) == test]


def whiles(t: ast.AST, test: str):
    return [n for n in ast.walk(t) if isinstance(n, ast.While) and U(n.test) == test]


def strings(t: ast.AST):
    return {n.value for n in ast.walk(t) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def getattr_consts(t: ast.AST):
    """The attribute-name constants of every `getattr(obj, 'name', …)` call."""
    out = set()
    for c in calls(t, "getattr"):
        if len(c.args) >= 2 and isinstance(c.args[1], ast.Constant):
            out.add(c.args[1].value)
    return out


def first_line(node) -> int:
    return getattr(node, "lineno", 0)
