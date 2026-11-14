"""Step 5E part A: automatic faithful previews of the top N, 'Gerar prévias de todos' and the
faithful preview in the source page's card. No ffmpeg: renders are only planned/queued."""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from clipador.models import Candidate, CandidateStatus, Job, Render, RenderStatus
from clipador.pipeline.base import StageContext
from clipador.pipeline.stages.enrich import queue_auto_previews
from clipador.services import previews
from clipador.web.app import create_app
from tests.review_helpers import seed_review


class _Rep:
    def __init__(self):
        self.lines = []

    def report_progress(self, job_id, percent):
        pass

    def append_log(self, job_id, line):
        self.lines.append(line)

    def is_canceled(self, job_id):
        return False

    def is_stopping(self):
        return False


def _ctx(settings, engine, source_id):
    return StageContext(job_id=0, job_type="enrich", source_id=source_id, payload={}, attempt=1,
                        settings=settings, engine=engine, reporter=_Rep())  # fmt: skip


def test_top_candidates_by_final_score_skip_rejected(settings, engine):
    seeded = seed_review(settings, engine)
    sid = seeded.sources[0]
    a1, a2, a3 = seeded.cands[sid]
    with Session(engine) as s:
        c = s.get(Candidate, a1)
        c.status = CandidateStatus.REJECTED
        s.add(c)
        s.commit()
        top = previews.top_candidates(s, seeded.runs[sid], 2)
    assert [c.id for c in top] == [a2, a3]  # final scores 85 (rejected), 80, 40


def test_auto_previews_after_analysis_low_priority_and_once(settings, engine):
    settings.auto_preview_top_n = 2
    seeded = seed_review(settings, engine)
    sid = seeded.sources[0]
    ctx = _ctx(settings, engine, sid)
    queue_auto_previews(ctx, seeded.runs[sid])
    with Session(engine) as s:
        jobs = s.exec(select(Job).where(Job.type == "render")).all()
        rows = s.exec(select(Render)).all()
    assert len(jobs) == 1 and jobs[0].priority == previews.AUTO_PREVIEW_PRIORITY < 0
    assert sorted(r.candidate_id for r in rows) == sorted(seeded.cands[sid][:2])
    assert all(r.kind == "preview" and r.status == RenderStatus.QUEUED for r in rows)
    assert any("Prévias fiéis dos 2 melhores" in ln for ln in ctx.reporter.lines)
    queue_auto_previews(ctx, seeded.runs[sid])  # nothing twice
    with Session(engine) as s:
        assert len(s.exec(select(Job).where(Job.type == "render")).all()) == 1


def test_auto_previews_off_and_without_video(settings, engine):
    settings.auto_preview_top_n = 0
    seeded = seed_review(settings, engine)
    sid = seeded.sources[0]
    queue_auto_previews(_ctx(settings, engine, sid), seeded.runs[sid])
    settings.auto_preview_top_n = 12
    (settings.vods_dir / "1" / "video.mp4").unlink()  # best effort: no video, nothing queued
    queue_auto_previews(_ctx(settings, engine, sid), seeded.runs[sid])
    with Session(engine) as s:
        assert s.exec(select(Job).where(Job.type == "render")).all() == []


def test_hand_render_goes_ahead_of_auto_previews(settings, engine):
    """A preview the user asks for bumps the queued auto one to priority 0."""
    from clipador.services import analysis

    settings.auto_preview_top_n = 3
    seeded = seed_review(settings, engine)
    sid = seeded.sources[0]
    queue_auto_previews(_ctx(settings, engine, sid), seeded.runs[sid])
    with Session(engine) as s:
        analysis.enqueue_render(s, settings, sid, [seeded.cands[sid][0]], "preview", priority=0)
        s.commit()
        jobs = s.exec(select(Job).where(Job.type == "render")).all()
    assert [j.priority for j in jobs] == [0]


@pytest.fixture
def client(settings):
    app = create_app(settings)
    seeded = seed_review(settings, app.state.engine)
    with TestClient(app) as c:
        c.engine, c.settings, c.seeded = app.state.engine, settings, seeded
        yield c


def test_generate_previews_for_all_button(client):
    sid = client.seeded.sources[0]
    page = client.get(f"/fontes/{sid}")
    assert "Gerar prévias de todos" in page.text
    r = client.post(f"/fontes/{sid}/previas", follow_redirects=False)
    assert r.status_code == 303 and "pr%C3%A9via" in r.headers["location"]
    with Session(client.engine) as s:
        rows = s.exec(select(Render)).all()
        job = s.exec(select(Job).where(Job.type == "render")).one()
    assert sorted(r.candidate_id for r in rows) == sorted(client.seeded.cands[sid])
    assert job.priority == 0
    r = client.post(f"/fontes/{sid}/previas", follow_redirects=True)
    assert "já têm prévia" in r.text


def test_card_shows_faithful_preview_when_done(client):
    sid = client.seeded.sources[0]
    a1 = client.seeded.cands[sid][0]
    page = client.get(f"/fontes/{sid}").text
    assert "faithful-player" not in page  # only the original excerpt
    out = client.settings.clips_dir / str(sid) / "p.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"x")
    with Session(client.engine) as s:
        c = s.get(Candidate, a1)
        s.add(Render(candidate_id=a1, source_id=sid, kind="preview", params_hash="h",
                     params={"start": c.start, "end": c.end}, status=RenderStatus.DONE,
                     path=client.settings.rel_path(out), duration=c.end - c.start,
                     encoder="libx264", size_bytes=1))  # fmt: skip
        s.commit()
    page = client.get(f"/fontes/{sid}").text
    assert 'class="faithful-player"' in page and f"/arquivos/clips/{sid}/p.mp4" in page
    assert "trecho original" in page and 'class="clip-player"' in page
