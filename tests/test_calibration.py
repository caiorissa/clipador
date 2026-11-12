import csv
from unittest import mock

import pytest
from sqlmodel import Session, select

from clipador.cli import main
from clipador.detection.evaluate import RankedCandidate, evaluate, parse_golden
from clipador.models import Candidate, CandidateFeedback
from clipador.services import doctor
from clipador.services.calibration import dump, run_eval
from clipador.services.doctor import CheckStatus
from clipador.worker.runner import Worker
from tests.conftest import requires_ffmpeg
from tests.test_analysis import (  # noqa: F401  (fixtures)
    BURSTS,
    _analyze,
    _drain,
    _source,
    asettings,
    fake_tx,
)


def test_parse_golden():
    text = "# meus momentos\n00:01:00\n1:30  # risada\n\n90\n1:00:00\n"
    assert parse_golden(text) == [60, 90, 90, 3600]
    with pytest.raises(ValueError, match="linha 2"):
        parse_golden("00:01:00\nabc\n")


def test_recall_and_precision_known_case():
    ranked = [
        RankedCandidate(1, 100, 130, 90, "bom"),  # contains 110
        RankedCandidate(2, 300, 330, 80, "ruim"),
        RankedCandidate(3, 500, 530, 70),  # contains 520
        RankedCandidate(4, 700, 730, 60),
        RankedCandidate(5, 900, 930, 50),
        RankedCandidate(6, 1000, 1030, 40),  # contains 1031 only with tolerance
    ]
    golden = [110, 520, 1031, 2000]
    res = evaluate(golden, ranked, ns=(1, 5, 10), tolerance=2.0, curve_at=lambda t: 12.0)
    assert res.recall == {1: (1, 4), 5: (2, 4), 10: (3, 4)}
    assert res.precision[1] == (1, 1)
    assert res.precision[5] == (2, 3)  # #1 bom+golden, #2 ruim, #3 golden; #4,#5 unjudged
    assert [m.t for m in res.missed] == [2000]
    assert res.missed[0].curve_score == 12.0 and res.missed[0].nearest_rank == 6


@requires_ffmpeg
def test_dump_and_eval_on_synthetic_analysis(asettings, engine, fake_tx, tmp_path):  # noqa: F811
    sid = _source(engine, asettings)
    _analyze(engine, sid)
    _drain(Worker(asettings, engine))
    out = tmp_path / "candidates.csv"
    text = dump(engine, sid, 5, out)
    assert "análise #1" in text and out.is_file()
    with open(out, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert rows and rows[0]["rank"] == "1" and rows[0]["chat"] == ""

    with Session(engine) as s:
        top = s.exec(select(Candidate).order_by(Candidate.rank)).first()
        s.add(CandidateFeedback(candidate_id=top.id, verdict="bom"))
        s.commit()
    golden = tmp_path / "golden.txt"
    golden.write_text(
        "\n".join(f"00:{int(t // 60):02d}:{int(t % 60) + 1:02d}" for t in BURSTS) + "\n00:03:30\n",
        encoding="utf-8",
    )
    res, report = run_eval(engine, asettings, sid, golden)
    assert res.recall[5] == (2, 3)  # both bursts found; 03:30 (after the end) is not
    assert res.precision[5][0] >= 1
    assert "recall@5" in report and "00:03:30" in report and "interesse nesse instante" in report


@requires_ffmpeg
def test_cli_dump_and_eval(asettings, engine, fake_tx, tmp_path, monkeypatch, capsys):  # noqa: F811
    sid = _source(engine, asettings)
    _analyze(engine, sid)
    _drain(Worker(asettings, engine))
    monkeypatch.setattr("clipador.cli.get_settings", lambda: asettings)
    assert main(["dump", str(sid), "--top", "3", "--out", str(tmp_path / "c.csv")]) == 0
    assert "CSV salvo" in capsys.readouterr().out
    g = tmp_path / "g.txt"
    g.write_text("00:00:41\n", encoding="utf-8")
    assert main(["eval", str(sid), "--golden", str(g)]) == 0
    assert "recall@5" in capsys.readouterr().out
    assert main(["eval", "999", "--golden", str(g)]) == 1
    assert "não existe" in capsys.readouterr().err


def test_doctor_new_checks(settings, tmp_path):
    assert doctor.check_faster_whisper().status == CheckStatus.OK
    settings.twitch_downloader_path = str(tmp_path / "nope.exe")
    r = doctor.check_twitch_downloader(settings)
    assert r.status == CheckStatus.WARN and "lay295" in r.hint
    assert doctor.check_models_dir(settings).status in (CheckStatus.OK, CheckStatus.WARN)
    from clipador.services.cuda import CudaStatus

    with mock.patch(
        "clipador.services.cuda.cuda_status", return_value=CudaStatus(1, ["cudnn64_9.dll"])
    ):
        r = doctor.check_cuda_whisper(settings)
    assert r.status == CheckStatus.FAIL and "cudnn64_9.dll" in r.detail
    with mock.patch("clipador.services.cuda.cuda_status", return_value=CudaStatus(0, [])):
        assert doctor.check_cuda_whisper(settings).status == CheckStatus.WARN
