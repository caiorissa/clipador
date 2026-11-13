from datetime import date
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from clipador.models import Job, JobStatus, Source, Streamer
from clipador.services.doctor import CheckResult, CheckStatus
from clipador.web.app import create_app

VOD = "https://www.twitch.tv/videos/2283746281"


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        c.engine = app.state.engine
        yield c


def _create_streamer(client, confirmed=True, **extra) -> int:
    data = {
        "display_name": "Fulano",
        "permission_date": date.today().isoformat(),
        "permission_note": "liberou no Discord em 10/11",
        "channel_twitch": "https://twitch.tv/Fulano",
        "channel_youtube": "@fulanogamer",
        "channel_kick": "fulano",
        "default_layout": "blur",
        **extra,
    }
    if confirmed:
        data["permission_confirmed"] = "on"
    r = client.post("/streamers", data=data, follow_redirects=False)
    assert r.status_code == 303, r.text
    with Session(client.engine) as s:
        return s.exec(select(Streamer).order_by(Streamer.id.desc())).first().id


def test_pages_render(client):
    for path in ("/fontes", "/streamers", "/streamers/novo", "/jobs", "/sistema"):
        r = client.get(path)
        assert r.status_code == 200, path
        assert "Clipador" in r.text
    assert client.get("/", follow_redirects=False).headers["location"] == "/fontes"


def test_create_streamer_with_three_channels(client):
    sid = _create_streamer(client)
    page = client.get("/streamers").text
    assert "autorizado" in page and "fulanogamer" in page
    with Session(client.engine) as s:
        st = s.get(Streamer, sid)
        assert st.permission_confirmed and st.permission_date == date.today()
        assert sorted(c.platform for c in st.channels) == ["kick", "twitch", "youtube"]
        assert st.credit_text == "Clip de @fulano na Twitch"
    assert "Autorização confirmada" in client.get(f"/streamers/{sid}/editar").text


def test_confirmed_permission_requires_date(client):
    r = client.post(
        "/streamers",
        data={"display_name": "X", "permission_confirmed": "on", "permission_date": ""},
    )
    assert r.status_code == 422
    assert "data em que o streamer autorizou" in r.text


def test_edit_add_remove_channel_and_deactivate(client):
    sid = _create_streamer(client)
    r = client.post(
        f"/streamers/{sid}/canais", data={"platform": "kick", "channel_handle": "fulano_2"}
    )
    assert "fulano_2" in r.text
    with Session(client.engine) as s:
        ch = next(c for c in s.get(Streamer, sid).channels if c.channel_handle == "fulano_2")
    client.post(f"/streamers/{sid}/canais/{ch.id}/remover")
    r = client.post(
        f"/streamers/{sid}",
        data={"display_name": "Fulano Novo", "default_layout": "facecam_embaixo", "fc_x": "0.75",
              "fc_y": "0,6", "fc_w": "0.2", "fc_h": "0.3", "default_caption_style": "pop",
              "show_credit": "on"},
    )  # fmt: skip
    assert r.status_code == 200
    client.post(f"/streamers/{sid}/ativo")
    with Session(client.engine) as s:
        st = s.get(Streamer, sid)
        assert st.display_name == "Fulano Novo" and st.facecam_rect == (0.75, 0.6, 0.2, 0.3)
        assert st.default_layout == "facecam_embaixo" and st.default_caption_style == "pop"
        assert st.show_credit and not st.show_opening_text
        assert not st.permission_confirmed and not st.active
        assert len(st.channels) == 3
    r = client.post(f"/streamers/{sid}", data={"display_name": "X", "fc_x": "0.9", "fc_y": "0.1",
                                               "fc_w": "0.5", "fc_h": "0.2"})  # fmt: skip
    assert r.status_code == 422 and "caber no quadro" in r.text
    assert client.get(f"/streamers/{sid}/frame.jpg").status_code == 404  # no source yet
    assert "Ainda não há nenhuma fonte baixada" in client.get(f"/streamers/{sid}/editar").text


def test_add_source_blocked_without_permission(client):
    sid = _create_streamer(client, confirmed=False)
    r = client.post("/fontes", data={"url": VOD, "streamer_id": str(sid)})
    assert r.status_code == 422
    assert "ainda não tem autorização" in r.text
    with Session(client.engine) as s:
        assert s.exec(select(Source)).first() is None
        assert s.exec(select(Job)).first() is None


def test_add_source_enqueues_ingest(client):
    sid = _create_streamer(client)
    r = client.post(
        "/fontes",
        data={"url": VOD, "streamer_id": str(sid), "start_offset": "1:00", "end_offset": "1h"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    with Session(client.engine) as s:
        src = s.exec(select(Source)).one()
        job = s.exec(select(Job)).one()
    assert (src.start_offset, src.end_offset) == (60, 3600)
    assert job.type == "ingest" and job.source_id == src.id and job.status == JobStatus.QUEUED
    assert "2283746281" in client.get("/fontes").text


def test_add_source_invalid_url_and_bad_time(client):
    sid = _create_streamer(client)
    r = client.post(
        "/fontes", data={"url": "https://www.twitch.tv/fulano", "streamer_id": str(sid)}
    )
    assert r.status_code == 422 and "canal ou live" in r.text
    r = client.post("/fontes", data={"url": VOD, "streamer_id": str(sid), "start_offset": "abc"})
    assert r.status_code == 422 and "Tempo inválido" in r.text


def test_detect_badge(client):
    r = client.post("/fontes/detectar", data={"url": "https://youtu.be/dQw4w9WgXcQ"})
    assert "YouTube" in r.text and "VOD" in r.text
    r = client.post("/fontes/detectar", data={"url": "https://kick.com/fulano/clips/clip_01ABC"})
    assert "Kick" in r.text and "Clip" in r.text
    r = client.post("/fontes/detectar", data={"url": "https://vimeo.com/1"})
    assert "não reconhecida" in r.text


def test_jobs_table_cancel_retry_and_log(client):
    sid = _create_streamer(client)
    client.post("/fontes", data={"url": VOD, "streamer_id": str(sid)})
    with Session(client.engine) as s:
        job_id = s.exec(select(Job)).one().id
    table = client.get("/jobs/tabela").text
    assert "Download" in table and "Cancelar" in table
    r = client.post(f"/jobs/{job_id}/cancelar")
    assert "Cancelado" in r.text and "Tentar de novo" in r.text
    r = client.post(f"/jobs/{job_id}/cancelar")
    assert "Só dá para cancelar" in r.text
    r = client.post(f"/jobs/{job_id}/tentar-de-novo")
    assert "Na fila" in r.text
    log = client.get(f"/jobs/{job_id}/log").text
    assert "Cancelado pelo usuário" in log and "Recolocado na fila" in log
    assert client.get(f"/jobs/{job_id}").status_code == 200
    assert client.get("/jobs/999").status_code == 404


def test_system_checks_partial(client):
    fake = [
        CheckResult("ffmpeg", CheckStatus.OK, "ffmpeg version 9"),
        CheckResult("Ollama", CheckStatus.WARN, "não acessível", "instale"),
    ]
    with mock.patch("clipador.services.doctor.run_all", return_value=fake):
        r = client.get("/sistema/checagens")
    assert "ffmpeg version 9" in r.text and "AVISO" in r.text


def test_data_file_serving_blocks_traversal(client, settings):
    (settings.data_dir / "vods").mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "vods" / "t.jpg").write_bytes(b"jpg")
    assert client.get("/arquivos/vods/t.jpg").content == b"jpg"
    assert client.get("/arquivos/../pyproject.toml").status_code == 404
    assert client.get("/arquivos/..%2F..%2Fpyproject.toml").status_code == 404
