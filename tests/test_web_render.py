import shutil

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from clipador.models import Candidate, CandidateStatus, Render, RenderStatus, StageArtifact
from clipador.web.app import create_app
from clipador.worker.runner import Worker
from tests.conftest import requires_ffmpeg
from tests.render_helpers import drain, make_video, seed

pytestmark = requires_ffmpeg


@pytest.fixture(scope="module")
def src_video(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg não instalado")
    return make_video(tmp_path_factory.mktemp("webrender") / "src.mp4")


@pytest.fixture
def client(settings, src_video):
    settings.render_encoder = "x264"
    app = create_app(settings)
    video = settings.vods_dir / "1" / "video.mp4"
    video.parent.mkdir(parents=True)
    shutil.copy(src_video, video)
    sid, (c1, c2), stid = seed(settings, app.state.engine, video, rect=(0.7, 0.55, 0.25, 0.4))
    with TestClient(app) as c:
        c.engine, c.worker = app.state.engine, Worker(settings, app.state.engine)
        c.sid, c.c1, c.c2, c.stid, c.settings = sid, c1, c2, stid, settings
        yield c


def latest(client, cid, kind):
    with Session(client.engine) as s:
        return s.exec(select(Render).where(Render.candidate_id == cid, Render.kind == kind)
                      .order_by(Render.id.desc())).first()  # fmt: skip


def test_page_shows_render_controls(client):
    page = client.get(f"/fontes/{client.sid}").text
    assert "Renderizar aprovados (2)" in page and "Vídeo longo 16:9" in page
    assert "Prévia rápida" in page and "Renderizar final" in page
    assert "Jogo cheio" in page and "Foco horizontal" in page and "Destaque amarelo" in page
    assert "olha a polícia!" in page  # caption text prefilled from the transcript


def test_preview_then_final_with_options_and_range(client):
    form = {"layout": "jogo_cheio", "style": "pop", "zoom": "1.2", "foco_x": "0.2",
            "opening": "on", "credit": "on", "peak_zoom": "on", "overlay_title": "FUGIU",
            "subtitle_text": "olha a VIATURA! corre - corre", "kind": "preview"}  # fmt: skip
    r = client.post(f"/candidatos/{client.c1}/corte", data=form)
    assert r.status_code == 200 and "Prévia na fila" in r.text
    with Session(client.engine) as s:
        c = s.get(Candidate, client.c1)
        assert (c.render_layout, c.render_style, c.overlay_title) == ("jogo_cheio", "pop", "FUGIU")
        assert c.render_options["foco_x"] == 0.2 and c.render_options["zoom"] == 1.2
        assert c.caption_overrides == {"2": "VIATURA!", "4": "", "5": "corre"}
    drain(client.worker)
    pv = latest(client, client.c1, "preview")
    assert pv.status == RenderStatus.DONE and pv.params["size"] == [540, 960]

    # same options again -> reused, nothing queued
    r = client.post(f"/candidatos/{client.c1}/corte", data=form)
    assert "já existe: reaproveitada" in r.text

    form["kind"] = "final"
    client.post(f"/candidatos/{client.c1}/corte", data=form)
    drain(client.worker)
    fn = latest(client, client.c1, "final")
    assert fn.status == RenderStatus.DONE and fn.loudness_lufs == pytest.approx(-14, abs=1.5)
    page = client.get(f"/fontes/{client.sid}").text
    assert fn.path in page and pv.path in page and "Baixar" in page

    full = client.get(f"/arquivos/{fn.path}")
    assert full.status_code == 200 and full.headers.get("accept-ranges") == "bytes"
    part = client.get(f"/arquivos/{fn.path}", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and part.content == full.content[:100]
    assert "Clip de @fulano" in client.get(f"/arquivos/{fn.package_path}").text
    with Session(client.engine) as s:
        assert s.get(Candidate, client.c1).status == CandidateStatus.RENDERED


def test_caption_word_count_mismatch_is_explained(client):
    r = client.post(f"/candidatos/{client.c1}/corte",
                    data={"subtitle_text": "só três palavras", "kind": "preview"})  # fmt: skip
    assert "corrija palavra por palavra" in r.text
    with Session(client.engine) as s:
        assert s.exec(select(Render)).first() is None


def test_render_approved_and_long_video(client):
    r = client.post(f"/fontes/{client.sid}/renderizar-aprovados", follow_redirects=False)
    assert r.status_code == 303
    client.post(f"/fontes/{client.sid}/video-longo")
    drain(client.worker)
    with Session(client.engine) as s:
        finals = s.exec(select(Render).where(Render.kind == "final")).all()
        assert len(finals) == 2 and all(f.status == RenderStatus.DONE for f in finals)
        assert s.exec(select(StageArtifact).where(StageArtifact.stage == "compile_long")).one()
    page = client.get(f"/fontes/{client.sid}").text
    assert "Renderizar aprovados (2)" in page and "longo.mp4 (2 momentos)" in page


def test_streamer_frame_for_facecam_tool(client):
    r = client.get(f"/streamers/{client.stid}/frame.jpg")
    assert (
        r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and len(r.content) > 1000
    )
    page = client.get(f"/streamers/{client.stid}/editar").text
    assert "fc-tool" in page and 'value="0.7"' in page


def test_long_video_requires_approved(client):
    with Session(client.engine) as s:
        for c in s.exec(select(Candidate)).all():
            c.status = CandidateStatus.SUGGESTED
            s.add(c)
        s.commit()
    assert "Aprove pelo menos um" in client.post(f"/fontes/{client.sid}/video-longo").text
    assert (
        "pelo menos um candidato" in client.post(f"/fontes/{client.sid}/renderizar-aprovados").text
    )
