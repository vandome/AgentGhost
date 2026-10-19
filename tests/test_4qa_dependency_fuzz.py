"""§4QA-d review r25: random task trees against the dependency rule.

Four review rounds (r21–r24) each found a plan shape the hand-written
cases missed — a deadlock, a task run before its prerequisite, or a task
the picker hands out that task_update then refuses. This drives the real
picker (ProjectPlan.next_ready_leaf → update_status DONE) over seeded
random trees (≤8 nodes, random parents, ALL/ANY/BEST, depends_on incl.
cycles) and checks, per pick:
  * the gate agrees — a picked task's `_unmet_dependencies` is empty;
  * nothing runs before a dependency that does not REALLY wait on it
    (oracle `_really_waits`, written from the rules, not from the code);
and that no tree deadlocks where the pre-§4QA-d rule (waive every
dependency of a task in a depends_on cycle) drained.
"""
import random

from ghost_agent.core.planning import DependencyType as D
from ghost_agent.core.planning import ProjectPlan, TaskNode
from ghost_agent.core.planning import TaskStatus as S
from ghost_agent.tools.projects import _unmet_dependencies


class _Store:
    def list_tasks(self, pid):
        return []

    def update_task(self, *a, **k):
        pass


def _build(spec):
    p = ProjectPlan(_Store(), "p")
    for nid, par, dt, deps in spec:
        p.tree.nodes[nid] = TaskNode(id=nid, description=nid, parent_id=par,
                                     dependency_type=dt, depends_on=list(deps))
    for nid, par, _dt, _deps in spec:
        if par:
            p.tree.nodes[par].children.append(nid)
        elif p.tree.root_id is None:
            p.tree.root_id = nid
    return p


def _gen(rng):
    n = rng.randint(2, 8)
    ids = [chr(65 + i) for i in range(n)]
    spec = []
    for i, nid in enumerate(ids):
        par = rng.choice(ids[:i]) if i and rng.random() < 0.8 else None
        spec.append([nid, par, rng.choice([D.ALL, D.ALL, D.ANY, D.BEST]), []])
    for s in spec:
        k = rng.choice([0, 0, 1, 1, 2])
        s[3] = rng.sample([x for x in ids if x != s[0]], min(k, n - 1))
    return [tuple(s) for s in spec]


def _ancestors(tree, nid):
    a, c = set(), tree.nodes[nid]
    while c.parent_id and c.parent_id in tree.nodes and c.parent_id not in a:
        a.add(c.parent_id)
        c = tree.nodes[c.parent_id]
    return a


def _really_waits(tree, start, target):
    """A leaf waits on its known, non-ancestor depends_on; an ALL/BEST
    parent on any open child; an ANY parent only when every open child
    does (and none is DONE); a DONE node on nothing."""
    def w(n, path):
        if n == target:
            return True
        if n in path or n not in tree.nodes:
            return False
        node = tree.nodes[n]
        if node.status == S.DONE:
            return False
        path = path | {n}
        if node.children:
            kids = [c for c in node.children if c in tree.nodes]
            if node.dependency_type == D.ANY:
                return (bool(kids) and not any(tree.nodes[c].status == S.DONE for c in kids)
                        and all(w(c, path) for c in kids))
            return any(w(c, path) for c in kids)
        anc = _ancestors(tree, n)
        return any(w(d, path) for d in node.depends_on if d in tree.nodes and d not in anc)
    return w(start, frozenset())


def _old_rule_drains(spec):
    p = _build(spec)
    tree = p.tree

    def old_open(nid):
        n = tree.nodes[nid]

        def reach(start):
            st, seen = [start], set()
            while st:
                c = st.pop()
                if c == nid:
                    return True
                if c in seen or c not in tree.nodes:
                    continue
                seen.add(c)
                st.extend(tree.nodes[c].depends_on)
            return False
        if any(reach(d) for d in n.depends_on):
            return []
        return [d for d in n.depends_on if d in tree.nodes and tree.nodes[d].status != S.DONE]

    for _ in range(50):
        nxt = None
        for node in tree.nodes.values():
            if node.children or node.status not in (S.PENDING, S.READY):
                continue
            if not old_open(node.id):
                nxt = node
                break
        if nxt is None:
            break
        p.update_status(nxt.id, S.DONE, result="ok")
    return not [i for i, x in tree.nodes.items() if not x.children and x.status in (S.PENDING, S.READY)]


def test_random_trees_agree_run_in_order_and_do_not_newly_deadlock():
    rng = random.Random(4025)
    disagree, out_of_order, new_deadlocks = [], [], []
    for _ in range(3000):
        spec = _gen(rng)
        p = _build(spec)
        for _ in range(50):
            nxt = p.next_ready_leaf()
            if nxt is None:
                break
            if _unmet_dependencies(p, nxt.id):
                disagree.append(spec)
            anc = _ancestors(p.tree, nxt.id)
            bad = [d for d in nxt.depends_on
                   if d in p.tree.nodes and p.tree.nodes[d].status != S.DONE
                   and d not in anc and not _really_waits(p.tree, d, nxt.id)]
            if bad:
                out_of_order.append((spec, nxt.id, bad))
            p.update_status(nxt.id, S.DONE, result="ok")
        stuck = [i for i, x in p.tree.nodes.items()
                 if not x.children and x.status in (S.PENDING, S.READY)]
        if stuck and _old_rule_drains(spec):
            new_deadlocks.append(spec)
    assert not disagree, disagree[:2]
    assert not out_of_order, out_of_order[:2]
    # the reviewer's 30,000-tree run found ONE genuine OR-cycle through
    # mixed ANY children; this seed's 3,000 trees have none
    assert len(new_deadlocks) <= 1, new_deadlocks[:2]


def test_depends_back_matches_the_rule_on_every_pair_with_random_statuses():
    # r26: the memo cached a False that was False only because the search hit
    # its own path, and another branch reused it (328 pairs in 30,000 trees).
    # Statuses are set directly so an ANY parent can hold a DONE child.
    rng = random.Random(2626)
    wrong = []
    for _ in range(3000):
        spec = _gen(rng)
        p = _build(spec)
        for node in p.tree.nodes.values():
            node.status = rng.choice([S.PENDING, S.PENDING, S.PENDING, S.DONE])
        for a in p.tree.nodes:
            for b in p.tree.nodes:
                if p.tree.depends_back(a, b) != _really_waits(p.tree, a, b):
                    wrong.append((spec, a, b))
    assert not wrong, wrong[:2]


def test_a_path_cut_answer_is_not_reused_and_the_plan_drains():
    # r26 reviewer's repro: the cached False held F on A, G ran first, and
    # E, F, H, J were stuck — the rule (and the pre-§4QA-d rule) drain it.
    spec = [("A", None, D.ALL, []), ("B", None, D.ALL, []), ("C", "A", D.ALL, []),
            ("D", "C", D.ANY, []), ("E", None, D.ALL, ["A"]), ("F", "B", D.ALL, ["A"]),
            ("G", None, D.ALL, ["H"]), ("H", "D", D.ALL, ["G", "F"]),
            ("J", "D", D.ALL, ["E", "G"])]
    p = _build(spec)
    assert p.tree.depends_back("A", "F") == _really_waits(p.tree, "A", "F") is True
    assert p.tree.open_dependencies("F") == []
    for _ in range(20):
        n = p.next_ready_leaf()
        if n is None:
            break
        p.update_status(n.id, S.DONE, result="ok")
    assert not [i for i, x in p.tree.nodes.items() if not x.children and x.status == S.PENDING]
