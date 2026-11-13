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
