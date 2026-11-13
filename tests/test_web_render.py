import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from clipador.models import Candidate, CandidateStatus, RenderedClip, StageArtifact
from clipador.web.app import create_app
from clipador.worker.runner import Worker
from tests.conftest import requires_ffmpeg
from tests.test_render import drain, rsettings, setup  # noqa: F401  (fixtures)

pytestmark = requires_ffmpeg


@pytest.fixture
def client(rsettings, setup):
    app = create_app(rsettings)
    with TestClient(app) as c:
        c.engine = app.state.engine
        c.worker = Worker(rsettings, c.engine)
        c.sid, c.c1, c.c2 = setup
        yield c


def test_page_shows_render_controls(client):
    page = client.get(f"/fontes/{client.sid}").text
    assert "Renderizar aprovados (2)" in page and "Vídeo longo 16:9" in page
    assert "Corte vertical 9:16" in page and "Renderizar corte" in page
    assert "OLHA" not in page and "olha a polícia!" in page  # subtitle text prefilled from words
    assert "Meu deus" in page  # overlay title of candidate 2


def test_render_single_with_options_then_download(client):
    r = client.post(f"/candidatos/{client.c1}/corte",
                    data={"layout": "blur", "style": "neon", "overlay_title": "FUGIU",
                          "subtitle_text": "olha a VIATURA"})  # fmt: skip
    assert r.status_code == 200 and "Renderizando" in r.text
    with Session(client.engine) as s:
        c = s.get(Candidate, client.c1)
        assert (c.render_layout, c.render_style, c.overlay_title) == ("blur", "neon", "FUGIU")
        assert c.subtitle_text == "olha a VIATURA"
    drain(client.worker)
    with Session(client.engine) as s:
        clip = s.exec(select(RenderedClip)).one()
    assert clip.layout == "blur" and clip.style == "neon"
    page = client.get(f"/fontes/{client.sid}").text
    assert "Baixar .mp4" in page and clip.file_path in page
    mp4 = client.get(f"/arquivos/{clip.file_path}")
    assert mp4.status_code == 200 and len(mp4.content) > 1000
    assert "Clip de @fulano" in client.get(f"/arquivos/{clip.package_path}").text


def test_unchanged_subtitles_are_not_stored_as_override(client):
    client.post(f"/candidatos/{client.c1}/corte",
                data={"layout": "", "style": "classico", "subtitle_text": "olha a polícia!  corre (corre) /Ncorre"})  # fmt: skip
    with Session(client.engine) as s:
        assert s.get(Candidate, client.c1).subtitle_text is None  # same words -> keep auto


def test_render_approved_and_long_video(client):
    r = client.post(f"/fontes/{client.sid}/renderizar-aprovados", follow_redirects=False)
    assert r.status_code == 303
    r = client.post(f"/fontes/{client.sid}/video-longo", follow_redirects=False)
    assert r.status_code == 303
    drain(client.worker)
    with Session(client.engine) as s:
        assert len(s.exec(select(RenderedClip)).all()) == 2
        assert s.exec(select(StageArtifact).where(StageArtifact.stage == "compile_long")).one()
    page = client.get(f"/fontes/{client.sid}").text
    assert "longo.mp4 (2 momentos)" in page and "Capítulos" in page


def test_long_video_requires_approved(client):
    with Session(client.engine) as s:
        for c in s.exec(select(Candidate)).all():
            c.status = CandidateStatus.SUGGESTED
            s.add(c)
        s.commit()
    r = client.post(f"/fontes/{client.sid}/video-longo")
    assert "Aprove pelo menos um" in r.text
    r = client.post(f"/fontes/{client.sid}/renderizar-aprovados")
    assert "pelo menos um candidato" in r.text
