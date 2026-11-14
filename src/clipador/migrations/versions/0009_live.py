"""live capture: live_session, live_segment, source.is_live, streamer.auto_watch, candidate flags

Revision ID: 0009
Revises: 0008
Create Date: 2026-11-14 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_str = sqlmodel.sql.sqltypes.AutoString
_dt = sqlmodel.sql.sqltypes.UTCDateTime


def upgrade() -> None:
    op.create_table(
        "live_session",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("status", _str(), nullable=False),
        sa.Column("started_at", _dt(), nullable=False),
        sa.Column("stopped_at", _dt(), nullable=True),
        sa.Column("heartbeat_at", _dt(), nullable=True),
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("warning", _str(), nullable=True),
        sa.Column("segment_seconds", sa.Integer(), nullable=False),
        sa.Column("live_from_start", sa.Boolean(), nullable=False),
        sa.Column("auto_clip", sa.Boolean(), nullable=False),
        sa.Column("min_score", sa.Float(), nullable=False),
        sa.Column("stop_requested", sa.Boolean(), nullable=False),
        sa.Column("restarts", sa.Integer(), nullable=False),
        sa.Column("reconnects", sa.Integer(), nullable=False),
        sa.Column("sim_file", _str(), nullable=True),
        sa.Column("sim_speed", sa.Float(), nullable=True),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("created_at", _dt(), nullable=False),
        sa.Column("updated_at", _dt(), nullable=False),
        sa.ForeignKeyConstraint(["source_id"], ["source.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["analysis_run.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("live_session", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_live_session_source_id"), ["source_id"])
        batch_op.create_index(batch_op.f("ix_live_session_status"), ["status"])

    op.create_table(
        "live_segment",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("path", _str(), nullable=True),
        sa.Column("start_offset_s", sa.Float(), nullable=False),
        sa.Column("duration_s", sa.Float(), nullable=False),
        sa.Column("status", _str(), nullable=False),
        sa.Column("error", _str(), nullable=True),
        sa.Column("analyzed_at", _dt(), nullable=True),
        sa.Column("created_at", _dt(), nullable=False),
        sa.Column("updated_at", _dt(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["live_session.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "index", name="uq_segment_session_index"),
    )
    with op.batch_alter_table("live_segment", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_live_segment_session_id"), ["session_id"])

    with op.batch_alter_table("source", schema=None) as batch_op:
        batch_op.add_column(sa.Column("is_live", sa.Boolean(), nullable=False, server_default="0"))
    with op.batch_alter_table("streamer", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("auto_watch", sa.Boolean(), nullable=False, server_default="0")
        )
    with op.batch_alter_table("candidate", schema=None) as batch_op:
        batch_op.add_column(sa.Column("live_session_id", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("auto_clipped", sa.Boolean(), nullable=False, server_default="0")
        )
        batch_op.create_index(batch_op.f("ix_candidate_live_session_id"), ["live_session_id"])
        batch_op.create_foreign_key(
            "fk_candidate_live_session_id", "live_session", ["live_session_id"], ["id"]
        )


def downgrade() -> None:
    with op.batch_alter_table("candidate", schema=None) as batch_op:
        batch_op.drop_constraint("fk_candidate_live_session_id", type_="foreignkey")
        batch_op.drop_index(batch_op.f("ix_candidate_live_session_id"))
        batch_op.drop_column("auto_clipped")
        batch_op.drop_column("live_session_id")
    with op.batch_alter_table("streamer", schema=None) as batch_op:
        batch_op.drop_column("auto_watch")
    # A source still marked live can't exist without its session: it goes back to failed.
    op.execute("UPDATE source SET status = 'failed', kind = 'vod' WHERE kind = 'live'")
    with op.batch_alter_table("source", schema=None) as batch_op:
        batch_op.drop_column("is_live")
    with op.batch_alter_table("live_segment", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_live_segment_session_id"))
    op.drop_table("live_segment")
    with op.batch_alter_table("live_session", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_live_session_status"))
        batch_op.drop_index(batch_op.f("ix_live_session_source_id"))
    op.drop_table("live_session")
