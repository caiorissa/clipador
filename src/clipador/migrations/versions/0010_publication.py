"""publishing: publication table (YouTube uploads and their numbers)

Revision ID: 0010
Revises: 0009
Create Date: 2026-11-17 20:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_str = sqlmodel.sql.sqltypes.AutoString
_dt = sqlmodel.sql.sqltypes.UTCDateTime


def upgrade() -> None:
    op.create_table(
        "publication",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("candidate_id", sa.Integer(), nullable=False),
        sa.Column("render_id", sa.Integer(), nullable=False),
        sa.Column("platform", _str(), nullable=False),
        sa.Column("status", _str(), nullable=False),
        sa.Column("privacy", _str(), nullable=False),
        sa.Column("publish_at", _dt(), nullable=True),
        sa.Column("title", _str(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("external_id", _str(), nullable=True),
        sa.Column("url", _str(), nullable=True),
        sa.Column("job_id", sa.Integer(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("views", sa.Integer(), nullable=True),
        sa.Column("likes", sa.Integer(), nullable=True),
        sa.Column("comments", sa.Integer(), nullable=True),
        sa.Column("metrics_at", _dt(), nullable=True),
        sa.Column("created_at", _dt(), nullable=False),
        sa.Column("updated_at", _dt(), nullable=False),
        sa.ForeignKeyConstraint(["candidate_id"], ["candidate.id"]),
        sa.ForeignKeyConstraint(["render_id"], ["render.id"]),
        sa.ForeignKeyConstraint(["job_id"], ["job.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("publication", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_publication_candidate_id"), ["candidate_id"])
        batch_op.create_index(batch_op.f("ix_publication_status"), ["status"])


def downgrade() -> None:
    with op.batch_alter_table("publication", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_publication_status"))
        batch_op.drop_index(batch_op.f("ix_publication_candidate_id"))
    op.drop_table("publication")
