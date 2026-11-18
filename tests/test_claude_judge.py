"""Claude as the AI judge (Claude Code CLI with the user's subscription) - with a FAKE `claude`
executable, so tests never use the subscription: isolation flags, structured output, limits,
login, and the full enrich stage choosing with Claude + the stricter v2 prompt."""

import json
import stat
import sys
from pathlib import Path

import pytest
from sqlmodel import Session, select

from clipador.llm import client as llm_client
from clipador.models import CandidateEnrichment
from tests.fake_llm import good_answer

REAL_FACTORY = llm_client.make_llm_client  # conftest replaces it with an offline fake per test

FAKE = r"""#!{python}
import json, os, sys
args = sys.argv[1:]
log = os.environ["FAKE_CLAUDE_LOG"]
mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
if args[:2] == ["auth", "status"]:
    print(json.dumps({{"loggedIn": mode != "logout"}}))
    sys.exit(0)
prompt = sys.stdin.read()
with open(log, "a") as f:
    f.write(json.dumps({{"args": args, "prompt": prompt, "cwd": os.getcwd()}}) + "\n")
if mode == "limit":
    print(json.dumps({{"is_error": True, "subtype": "error", "result": "Claude AI usage limit reached"}}))
    sys.exit(1)
answer = json.loads(os.environ["FAKE_CLAUDE_ANSWER"])
print(json.dumps({{"type": "result", "subtype": "success", "is_error": False,
                   "structured_output": answer, "result": json.dumps(answer),
                   "usage": {{"input_tokens": 2, "cache_creation_input_tokens": 1200,
                              "cache_read_input_tokens": 0, "output_tokens": 180}}}}))
"""


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    exe = tmp_path / "claude"
    exe.write_text(FAKE.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setattr(llm_client, "make_llm_client", REAL_FACTORY)
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("FAKE_CLAUDE_ANSWER", json.dumps(good_answer(nota=9)))

    def calls():
        return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []

    return exe, calls


def _settings(settings, exe):
    settings.llm_backend, settings.claude_binary = "claude", str(exe)
    return settings


def test_isolated_call_and_structured_output(settings, fake_claude):
    exe, calls = fake_claude
    s = _settings(settings, exe)
    c = llm_client.make_llm_client(s)
    assert isinstance(c, llm_client.ClaudeCodeClient)
    assert "claude:sonnet" in c.list_models()
    assert llm_client.default_model(s) == "claude:sonnet"
    assert llm_client.prompt_name(s) == "enrich_v2"
    r = c.chat("claude:opus", [{"role": "system", "content": "SISTEMA"},
                               {"role": "user", "content": "trecho <<<TRANSCRICAO_INICIO>>>"}],
               schema={"type": "object"}, temperature=0.2, num_ctx=0, keep_alive=0, timeout=30)  # fmt: skip
    assert json.loads(r.content)["scores"]["nota_geral"] == 9 and r.tokens == 1382
    call = calls()[-1]
    a = call["args"]
    # isolated: no tools (untrusted transcript), no MCP, no settings/skills, no saved session
    assert a[a.index("--tools") + 1] == "" and "--no-session-persistence" in a
    assert "--strict-mcp-config" in a and a[a.index("--setting-sources") + 1] == ""
    assert "--disable-slash-commands" in a and a[a.index("--model") + 1] == "opus"
    assert a[a.index("--system-prompt") + 1] == "SISTEMA" and "--json-schema" in a
    assert call["prompt"].startswith("trecho") and Path(call["cwd"]).name == "claude"


def test_limits_and_login_are_clear(settings, fake_claude, monkeypatch):
    exe, _calls = fake_claude
    c = llm_client.make_llm_client(_settings(settings, exe))
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "limit")
    with pytest.raises(llm_client.LLMUnavailable, match="Limite de uso"):
        c.chat("claude:sonnet", [{"role": "user", "content": "x"}], schema=None, temperature=0,
               num_ctx=0, keep_alive=0, timeout=30)  # fmt: skip
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "logout")
    with pytest.raises(llm_client.LLMUnavailable, match="não está logado"):
        c.list_models()
    settings.claude_binary = "/nao/existe/claude"
    st = llm_client.ollama_status(settings)
    assert not st.ok and "não encontrado" in st.message


def test_ollama_stays_the_default(settings):
    assert settings.llm_backend == "ollama"
    assert isinstance(REAL_FACTORY(settings), llm_client.OllamaClient)
    assert llm_client.prompt_name(settings) == "enrich_v1"


from tests.conftest import requires_ffmpeg  # noqa: E402
from tests.test_analysis import _analyze, _drain, _source, asettings, fake_tx  # noqa: E402, F401


@requires_ffmpeg
def test_enrich_stage_with_claude(asettings, engine, fake_tx, fake_claude):
    from clipador.services import analysis
    from clipador.worker.runner import Worker

    exe, calls = fake_claude
    asettings.target_clips = 1  # K = 3
    sid = _source(engine, asettings)
    _analyze(engine, sid)
    w = Worker(asettings, engine)
    _drain(w)  # Ollama "offline" here: enrich skipped
    _settings(asettings, exe)
    # transcribed with another Whisper setup (e.g. on the PC): the existing transcript is used
    asettings.whisper_model = "small"
    with Session(engine) as s:
        analysis.enqueue_enrich(s, sid)
        s.commit()
    _drain(w)
    with Session(engine) as s:
        rows = s.exec(select(CandidateEnrichment)).all()
    assert rows and all(r.model == "claude:sonnet" and r.status == "ok" for r in rows)
    assert all(r.prompt_version.startswith("enrich_v2") for r in rows)
    assert len(calls()) == len(rows)
    assert (
        "O que vale um corte" in calls()[0]["args"][calls()[0]["args"].index("--system-prompt") + 1]
    )
