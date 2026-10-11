"""§4HZ / §4IB — an execute call is keyed by what it RUNS, not by `cd`.

`command_head` annotates the same-error strike path (`core/strikes.py`):
the head is the program after `cd … &&`, `timeout [opts] N`, `sudo`,
`env`, `nohup`, …  World where each pin fails: the head is the first token
again (`cd`, `timeout`, `sudo`, `nohup` become heads), or a wrapper
swallows the real command and returns "".
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest

from ghost_agent.core import strikes as st
from ghost_agent.core.strikes import command_head


@pytest.mark.parametrize("command,head", [
    # the live shapes
    ("cd /workspace && python3 ifs_grid_oxford.py", "python3"),
    ("cd /workspace && timeout 400 python3 -u ifs_grid_oxford.py 2>&1; echo \"exit=$?\"", "python3"),
    ("timeout -k 5s 60s python3 -u .browser_runner.py '{...}'", "python3"),
    ("cd /tmp && python3 download_grib.py 2>&1 | tail -20", "python3"),
    # wrappers
    ("sudo -n -u debian-tor curl http://x", "curl"),
    ("sudo apt-get install -y cdo", "apt-get"),
    ("env FOO=1 python3 x.py", "python3"),
    ("nohup python3 srv.py &", "python3"),
    ("nice -n 10 ./build.sh", "./build.sh"),
    ("set -o pipefail; cd /tmp && pip install eccodes", "pip"),
    ("export X=1; ./run.sh", "./run.sh"),
    ("timeout --signal=KILL 30 make test", "make"),
    # the old behaviours that must survive
    ("FOO=1 python3 x.py", "python3"),
    ("python3 -m pytest", "python3"),
    ("echo hi; ls", "echo"),
    ("ls -la", "ls"),
    # nothing runs
    ("cd /tmp", ""),
    ("A=1 B=2", ""),
    ("", ""),
    ("cd /a && cd /b", ""),
])
def test_command_head_table(command, head):
    assert command_head(command) == head


def test_no_wrapper_is_ever_the_head():
    """The prefixes that were heads by accident cannot be produced any
    more: no command's head is one of the wrappers."""
    for c in ("cd /x && ls", "timeout 5 ls", "sudo ls", "nohup ls", "env ls", "nice ls"):
        assert command_head(c) == "ls", c
    for w in sorted(st._HEAD_WRAPPERS | {"cd", "timeout", "set", "export"}):
        assert command_head(f"{w} ls") != w


def test_head_walk_is_bounded():
    """A pathological command (thousands of `cd x &&`) terminates and
    returns "" rather than walking forever or recursing."""
    assert command_head(" && ".join(["cd /x"] * 500)) == ""
