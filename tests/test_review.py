"""services/review.py: transitions, decisions + undo stack, queue, range checks, render triggers."""

import pytest
from sqlmodel import select

from clipador.errors import ValidationError
from clipador.models import (
    AnalysisRun,
    Candidate,
    CandidateFeedback,
    Decision,
    Job,
    JobStatus,
    Render,
    RenderStatus,
)
from clipador.services import rendering, review
from clipador.services.review import ReviewFilters, snap_edge, validate_range
from tests.review_helpers import seed_review


@pytest.fixture
def seeded(settings, engine):
    return seed_review(settings, engine)


def _ids(seeded):
    s1, s2 = seeded.sources
    return seeded.cands[s1], seeded.cands[s2]


def _fb(session, cid):
    return session.exec(
        select(CandidateFeedback).where(CandidateFeedback.candidate_id == cid)
    ).first()


# --- transitions ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("current", "target", "ok"),
    [
        ("suggested", "approved", True), ("suggested", "rejected", True),
        ("suggested", "skipped", True), ("approved", "rejected", True),
        ("rejected", "approved", True), ("skipped", "approved", True),
        ("skipped", "rejected", True), ("skipped", "suggested", True),
        ("approved", "skipped", False), ("rejected", "skipped", False),
        ("suggested", "rendered", False), ("approved", "rendered", False),
        ("rendered", "approved", False), ("rendered", "rejected", True),
    ],
)  # fmt: skip
def test_transitions(current, target, ok):
    assert review.can_transition(current, target) is ok


def test_invalid_transition_and_reason_raise(session, seeded):
    (a, *_), _ = _ids(seeded)
    review.apply_decision(session, a, "approve")
    with pytest.raises(ValidationError, match="Não dá para mudar"):
        review.apply_decision(session, a, "skip")
    with pytest.raises(ValidationError, match="Motivo"):
        review.apply_decision(session, a, "reject", reason_tag="feio")
    with pytest.raises(ValidationError):
        review.apply_decision(session, a, "voar")


def test_decisions_write_feedback_and_prev_state(session, seeded):
    (a, b, _c), _ = _ids(seeded)
    r = review.apply_decision(session, a, "approve", note="ótimo")
    assert r.candidate.status == "approved" and r.candidate.reviewed_at is not None
    assert (
        r.decision.prev_state["status"] == "suggested" and r.decision.prev_state["feedback"] is None
    )
    assert _fb(session, a).verdict == "bom" and _fb(session, a).note == "ótimo"

    r = review.apply_decision(session, a, "reject", reason_tag="sem_contexto")
    assert r.decision.prev_state["status"] == "approved"
    assert r.decision.prev_state["feedback"]["verdict"] == "bom"
    fb = _fb(session, a)
    assert (fb.verdict, fb.reason_tag) == ("ruim", "sem_contexto")
    assert r.decision.reason_tag == "sem_contexto"

    review.apply_decision(session, b, "skip")
    assert _fb(session, b) is None
    assert session.get(Candidate, b).status == "skipped"


def test_undo_last_repeatedly(session, seeded):
    (a, b, c), _ = _ids(seeded)
    review.apply_decision(session, a, "approve")
    review.apply_decision(session, b, "reject", reason_tag="audio_ruim")
    review.apply_decision(session, c, "skip")
    session.commit()

    assert review.undo_last(session).candidate_id == c
    assert session.get(Candidate, c).status == "suggested"
    assert review.undo_last(session).candidate_id == b
    assert session.get(Candidate, b).status == "suggested" and _fb(session, b) is None
    # new decision after undos, then undo it and keep going back
    review.apply_decision(session, c, "approve")
    assert review.undo_last(session).candidate_id == c
    assert session.get(Candidate, c).status == "suggested"
    undone = review.undo_last(session)
    assert undone.candidate_id == a and undone.action == "approve"
    got = session.get(Candidate, a)
    assert got.status == "suggested" and got.reviewed_at is None and _fb(session, a) is None
    assert review.undo_last(session) is None
    actions = [d.action for d in session.exec(select(Decision).order_by(Decision.id))]
    assert actions.count("undo") == 4


def test_undo_restores_feedback_and_range(session, settings, seeded):
    (a, *_), _ = _ids(seeded)
    session.add(CandidateFeedback(candidate_id=a, verdict="ruim", note="antes"))
    session.flush()
    review.edit_range(session, settings, a, 9.0, 21.0)
    review.apply_decision(session, a, "approve")
    review.undo_last(session)
    fb = _fb(session, a)
    assert (fb.verdict, fb.note) == ("ruim", "antes")
    review.undo_last(session)
    assert session.get(Candidate, a).user_start is None


# --- queue ---------------------------------------------------------------------------------------


def test_order_nota_ignores_auto_clipped_boost(session, seeded):
    (a1, a2, a3), (b1, b2) = _ids(seeded)
    last = session.get(Candidate, b2)  # lowest score
    last.auto_clipped = True
    session.add(last)
    session.flush()
    assert review.scope(session, ReviewFilters(order="nota"))[0].id != b2
    assert review.scope(session, ReviewFilters())[0].id == b2  # default keeps live-first


def test_queue_order_and_filters(session, seeded):
    (a1, a2, a3), (b1, b2) = _ids(seeded)
    s2 = seeded.sources[1]
    order = [c.id for c in review.scope(session, ReviewFilters())]
    assert order == [b1, a1, a2, b2, a3]  # final_rank, then higher score
    assert [c.id for c in review.scope(session, ReviewFilters(order="nota"))] == [
        b1,
        a1,
        a2,
        a3,
        b2,
    ]
    assert [c.id for c in review.scope(session, ReviewFilters(order="horario"))] == [
        a1,
        a2,
        a3,
        b1,
        b2,
    ]
    assert [c.id for c in review.scope(session, ReviewFilters(source_id=s2))] == [b1, b2]
    assert [c.id for c in review.scope(session, ReviewFilters(platform="twitch"))] == [a1, a2, a3]
    st_b = seeded.streamers[1]
    assert [c.id for c in review.scope(session, ReviewFilters(streamer_id=st_b))] == [b1, b2]
    assert [c.id for c in review.scope(session, ReviewFilters(min_score=80))] == [b1, a1, a2]

    f = ReviewFilters()
    assert review.next_pending(session, f).id == b1
    review.apply_decision(session, b1, "approve")
    review.apply_decision(session, a1, "skip")
    assert review.next_pending(session, f, after_id=a1).id == a2
    # suggested ones come first; skipped come back at the end
    for cid in (a2, b2, a3):
        review.apply_decision(session, cid, "reject", reason_tag="sem_graca")
    assert review.next_pending(session, f).id == a1
    assert review.next_pending(session, f, after_id=a1).id == a1  # the only one left
    review.apply_decision(session, a1, "approve")
    assert review.next_pending(session, f) is None
    approved = ReviewFilters(status="approved")
    assert review.next_pending(session, approved).id == b1
    assert review.next_pending(session, approved, after_id=b1).id == a1

    p = review.progress(session, f, a2)
    assert (p["total"], p["position"], p["approved"], p["rejected"], p["reviewed"]) == (
        5,
        3,
        2,
        3,
        5,
    )
    assert (p["prev_id"], p["next_id"]) == (a1, b2)


def test_queue_uses_only_latest_run(session, seeded):
    s1 = seeded.sources[0]
    run = AnalysisRun(source_id=s1, params_hash="y", params={})
    session.add(run)
    session.flush()
    new = Candidate(source_id=s1, analysis_run_id=run.id, rank=1, start=1, end=9, peak_t=5,
                    score=10)  # fmt: skip
    session.add(new)
    session.flush()
    ids = [c.id for c in review.scope(session, ReviewFilters(source_id=s1))]
    assert ids == [new.id]


# --- range ---------------------------------------------------------------------------------------


def test_validate_range_limits_and_overlap():
    ok = validate_range(10, 40, duration=600, min_s=3, max_s=180, others=[(2, 35, 60), (3, 80, 90)])
    assert (ok.start, ok.end) == (10, 40)
    assert len(ok.warnings) == 1 and "#2" in ok.warnings[0]
    for s, e, msg in [(50, 40, "O fim precisa"), (-1, 10, "O fim precisa"),
                      (590, 610, "O fim precisa"), (0, 200, "máx. 3 min"),
                      (10, 11, "mín. 3 s")]:  # fmt: skip
        with pytest.raises(ValidationError, match=msg):
            validate_range(s, e, duration=600, min_s=3, max_s=180)
    assert validate_range(0, 1000, duration=None, min_s=1, max_s=2000).end == 1000


def test_snap_edge_words_and_pauses():
    words = [(10.0, 10.4), (10.5, 10.8), (10.9, 11.6), (13.0, 13.5)]
    assert snap_edge(10.1, words, "start") == 10.0
    assert snap_edge(11.5, words, "end") == 11.6
    assert snap_edge(12.3, words, "end") == 12.3  # pause midpoint (11.6..13.0)
    assert snap_edge(12.2, words, "start") == 12.3
    assert snap_edge(16.0, words, "start") == 16.0  # nothing near: unchanged
    assert snap_edge(10.6, [], "start") == 10.6


def test_edit_range_coalesces_and_invalidates_renders(session, settings, seeded):
    (a, *_), _ = _ids(seeded)
    c = session.get(Candidate, a)
    old = rendering.plan_render(session, session.get_bind(), settings, c, "preview").phash
    out = settings.clips_dir / "x.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"x")
    session.add(Render(candidate_id=a, source_id=c.source_id, kind="preview", params_hash=old,
                       status=RenderStatus.DONE, path=settings.rel_path(out)))  # fmt: skip
    session.flush()
    _todo, reused = rendering.request_renders(session, session.get_bind(), settings, [a], "preview")
    assert reused  # cached before the edit

    review.edit_range(session, settings, a, 7.5, 22.0)
    review.edit_range(session, settings, a, 7.0, 22.0)
    edits = session.exec(select(Decision).where(Decision.action == "edit_range")).all()
    assert len(edits) == 1 and edits[0].prev_state["user_start"] is None
    assert session.get(Candidate, a).user_start == 7.0
    new = rendering.plan_render(session, session.get_bind(), settings, c, "preview").phash
    assert new != old
    todo, reused = rendering.request_renders(session, session.get_bind(), settings, [a], "preview")
    assert todo and not reused  # the old preview is not reused for the new range

    c2, warnings = review.edit_range(session, settings, a, 10.1, 21.1,
                                     snap_words=[(10.0, 10.4), (20.6, 21.2)])  # fmt: skip
    assert (c2.user_start, c2.user_end) == (10.0, 21.2) and warnings == []
    review.undo_last(session)
    assert session.get(Candidate, a).user_start is None


def test_edit_range_warns_on_overlap_with_approved(session, settings, seeded):
    (a1, a2, _), _ = _ids(seeded)
    review.apply_decision(session, a1, "approve")
    _c, warnings = review.edit_range(session, settings, a2, 20, 45)
    assert warnings and "#1" in warnings[0]


# --- render triggers ---------------------------------------------------------------------------


def test_render_on_approve_queues_once(session, settings, seeded):
    (a, b, _), _ = _ids(seeded)
    r = review.apply_decision(session, a, "approve", settings=settings)
    assert r.render_job is not None and "render final na fila" in r.notice
    session.commit()
    review.apply_decision(session, a, "reject", reason_tag="outro", settings=settings)
    r2 = review.apply_decision(session, a, "approve", settings=settings)
    assert r2.render_job is None and "já estava na fila" in r2.notice
    finals = session.exec(select(Render).where(Render.kind == "final")).all()
    assert len(finals) == 1
    assert len(session.exec(select(Job).where(Job.type == "render")).all()) == 1

    settings.render_on_approve = False
    assert review.apply_decision(session, b, "approve", settings=settings).render_job is None


def test_undo_approve_cancels_its_queued_render(session, settings, seeded):
    (a, *_), _ = _ids(seeded)
    job = review.apply_decision(session, a, "approve", settings=settings).render_job
    session.commit()
    review.undo_last(session, settings)
    session.commit()
    assert session.get(Job, job.id).status == JobStatus.CANCELED
    assert session.exec(select(Render)).one().status == RenderStatus.CANCELED


def test_render_on_approve_without_video_is_a_notice(session, settings, engine):
    seeded = seed_review(settings, engine, video=False, transcript=False)
    cid = seeded.cands[seeded.sources[0]][0]
    r = review.apply_decision(session, cid, "approve", settings=settings)
    assert r.candidate.status == "approved" and "não foi para a fila" in r.notice
    assert session.exec(select(Render)).first() is None


def test_prefetch_does_not_duplicate_and_has_low_priority(session, settings, seeded):
    f = ReviewFilters()
    first = review.next_pending(session, f)
    nxt = review.upcoming(session, f, first.id, 3)
    assert len(nxt) == 3 and first.id not in [c.id for c in nxt]
    jobs = review.prefetch_previews(session, settings, nxt)
    session.commit()
    assert len(jobs) == 3
    assert all(session.get(Job, j).priority == review.PREFETCH_PRIORITY for j in jobs)
    assert review.prefetch_previews(session, settings, nxt) == []
    assert len(session.exec(select(Render).where(Render.kind == "preview")).all()) == 3

    # asking for the same preview by hand bumps its job ahead of the prefetch queue
    from clipador.services import analysis

    job, _todo, _reused = analysis.enqueue_render(session, settings, nxt[0].source_id,
                                                  [nxt[0].id], "preview")  # fmt: skip
    assert job is None
    pv = session.exec(select(Render).where(Render.candidate_id == nxt[0].id)).one()
    assert session.get(Job, pv.job_id).priority == 0


def test_filters_roundtrip_query():
    f = ReviewFilters.from_params({"fonte": "2", "nota_min": "70,5", "status": "skipped",
                                   "ordem": "nota", "plataforma": "kick", "streamer": "x"})  # fmt: skip
    assert (f.source_id, f.min_score, f.status, f.order, f.platform, f.streamer_id) == (
        2,
        70.5,
        "skipped",
        "nota",
        "kick",
        None,
    )
    assert ReviewFilters.from_params(dict(f.params())) == f
    assert ReviewFilters.from_params({"status": "zzz"}).status == "pendentes"
