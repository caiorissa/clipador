"""review queue: decision history, review columns, job priority

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-13 20:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "decision",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("candidate_id", sa.Integer(), nullable=False),
        sa.Column("action", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("reason_tag", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("note", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("prev_state", sa.JSON(), nullable=False),
        sa.Column("created_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=False),
        sa.ForeignKeyConstraint(["candidate_id"], ["candidate.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("decision", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_decision_candidate_id"), ["candidate_id"])
        batch_op.create_index(batch_op.f("ix_decision_created_at"), ["created_at"])

    with op.batch_alter_table("candidate", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("reviewed_at", sqlmodel.sql.sqltypes.UTCDateTime(), nullable=True)
        )
        batch_op.add_column(sa.Column("checks", sa.JSON(), nullable=True))

    with op.batch_alter_table("candidate_feedback", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("reason_tag", sqlmodel.sql.sqltypes.AutoString(), nullable=True)
        )

    with op.batch_alter_table("job", schema=None) as batch_op:
        batch_op.add_column(sa.Column("priority", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    with op.batch_alter_table("job", schema=None) as batch_op:
        batch_op.drop_column("priority")
    with op.batch_alter_table("candidate_feedback", schema=None) as batch_op:
        batch_op.drop_column("reason_tag")
    # 'skipped' only exists from this revision on: back to 'suggested'.
    op.execute("UPDATE candidate SET status = 'suggested' WHERE status = 'skipped'")
    with op.batch_alter_table("candidate", schema=None) as batch_op:
        batch_op.drop_column("checks")
        batch_op.drop_column("reviewed_at")
    with op.batch_alter_table("decision", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_decision_created_at"))
        batch_op.drop_index(batch_op.f("ix_decision_candidate_id"))
    op.drop_table("decision")
