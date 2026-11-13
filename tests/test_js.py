"""Syntax check of our own JS (no build step): `node --check`, skipped when node is missing."""

import shutil
import subprocess

import pytest

from clipador.web.deps import STATIC_DIR

NODE = shutil.which("node")
OWN_JS = sorted(p for p in STATIC_DIR.glob("*.js"))


@pytest.mark.skipif(NODE is None, reason="node não instalado")
@pytest.mark.parametrize("path", OWN_JS, ids=[p.name for p in OWN_JS])
def test_node_check(path):
    r = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_own_js_present():
    assert {"review.js", "caption_editor.js"} <= {p.name for p in OWN_JS}
