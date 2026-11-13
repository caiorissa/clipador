"""Data migrations: pixel facecam (1920x1080 frame) -> normalized box; layout rename."""

import json

import sqlalchemy as sa
from alembic import command

from clipador.db import alembic_config, make_engine


def test_0005_converts_facecam_pixels_to_relative(settings):
    settings.ensure_dirs()
    cfg = alembic_config(settings)
    command.upgrade(cfg, "0004")
    eng = make_engine(settings)
    with eng.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO streamer (id, display_name, permission_confirmed, permission_note, "
            "credit_text, default_layout, facecam_x, facecam_y, facecam_w, facecam_h, active, "
            "created_at, updated_at) VALUES (1, 'F', 1, '', '', 'facecam', 1440, 648, 384, 324, 1, "
            "'2026-11-13 00:00:00', '2026-11-13 00:00:00')"
        ))  # fmt: skip
    eng.dispose()
    command.upgrade(cfg, "head")
    eng = make_engine(settings)
    with eng.connect() as conn:
        row = conn.execute(sa.text(
            "SELECT facecam_box, default_layout, show_credit, show_opening_text FROM streamer"
        )).one()  # fmt: skip
    eng.dispose()
    assert json.loads(row[0]) == [0.75, 0.6, 0.2, 0.3]
    assert row[1] == "facecam_embaixo" and row[2] == 1 and row[3] == 1


def _columns(eng, table):
    return {c["name"] for c in sa.inspect(eng).get_columns(table)}


def test_0006_review_upgrade_and_downgrade(settings):
    settings.ensure_dirs()
    cfg = alembic_config(settings)
    command.upgrade(cfg, "0005")
    eng = make_engine(settings)
    with eng.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO job (id, type, payload, status, progress, attempts, max_attempts, log, "
            "needs_gpu, run_after, created_at, updated_at) VALUES (1, 'ingest', '{}', 'queued', 0, "
            "0, 3, '', 0, '2026-11-13 00:00:00', '2026-11-13 00:00:00', '2026-11-13 00:00:00')"
        ))  # fmt: skip
    eng.dispose()

    command.upgrade(cfg, "0006")
    eng = make_engine(settings)
    assert "decision" in sa.inspect(eng).get_table_names()
    assert {"reviewed_at", "checks"} <= _columns(eng, "candidate")
    assert "reason_tag" in _columns(eng, "candidate_feedback")
    with eng.connect() as conn:
        assert conn.execute(sa.text("SELECT priority FROM job")).scalar_one() == 0
    eng.dispose()

    command.downgrade(cfg, "0005")
    eng = make_engine(settings)
    assert "decision" not in sa.inspect(eng).get_table_names()
    assert "reviewed_at" not in _columns(eng, "candidate")
    assert "priority" not in _columns(eng, "job")
    eng.dispose()
    command.upgrade(cfg, "head")


def test_0007_caption_editor_upgrade_and_downgrade(settings):
    settings.ensure_dirs()
    cfg = alembic_config(settings)
    command.upgrade(cfg, "0006")
    eng = make_engine(settings)
    with eng.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO streamer (id, display_name, permission_confirmed, permission_note, "
            "credit_text, default_layout, show_credit, show_opening_text, active, created_at, "
            "updated_at) VALUES (1, 'F', 1, '', '', 'blur', 1, 1, 1, '2026-11-13 00:00:00', "
            "'2026-11-13 00:00:00')"
        ))  # fmt: skip
    eng.dispose()
    command.upgrade(cfg, "0007")
    eng = make_engine(settings)
    assert "caption_override_originals" in _columns(eng, "candidate")
    assert "glossary" in _columns(eng, "streamer")
    with eng.connect() as conn:
        assert conn.execute(sa.text("SELECT glossary FROM streamer")).scalar_one() is None
    eng.dispose()
    command.downgrade(cfg, "0006")
    eng = make_engine(settings)
    assert "caption_override_originals" not in _columns(eng, "candidate")
    assert "glossary" not in _columns(eng, "streamer")
    with eng.connect() as conn:
        assert conn.execute(sa.text("SELECT display_name FROM streamer")).scalar_one() == "F"
    eng.dispose()
    command.upgrade(cfg, "head")
