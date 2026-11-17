"""social accounts (several YouTube channels / TikTok) and the auto-post schedule

Revision ID: 0011
Revises: 0010
Create Date: 2026-11-17 21:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_str = sqlmodel.sql.sqltypes.AutoString
_dt = sqlmodel.sql.sqltypes.UTCDateTime


def upgrade() -> None:
    op.create_table(
        "social_account",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("platform", _str(), nullable=False),
        sa.Column("name", _str(), nullable=False),
        sa.Column("external_id", _str(), nullable=False),
        sa.Column("credentials", sa.JSON(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("paused", sa.Boolean(), nullable=False),
        sa.Column("autopost", sa.Boolean(), nullable=False),
        sa.Column("autopost_since", _dt(), nullable=True),
        sa.Column("streamer_ids", sa.JSON(), nullable=False),
        sa.Column("slots", sa.JSON(), nullable=False),
        sa.Column("post_now", sa.Boolean(), nullable=False),
        sa.Column("max_per_day", sa.Integer(), nullable=False),
        sa.Column("min_gap_minutes", sa.Integer(), nullable=False),
        sa.Column("privacy", _str(), nullable=False),
        sa.Column("hold_on_alerts", sa.Boolean(), nullable=False),
        sa.Column("created_at", _dt(), nullable=False),
        sa.Column("updated_at", _dt(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("platform", "external_id", name="uq_account_external"),
    )
    with op.batch_alter_table("publication", schema=None) as batch_op:
        batch_op.add_column(sa.Column("account_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("scheduled_for", _dt(), nullable=True))
        batch_op.add_column(sa.Column("auto", sa.Boolean(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("hold_reason", _str(), nullable=True))
        batch_op.create_index(batch_op.f("ix_publication_account_id"), ["account_id"])
        batch_op.create_index(batch_op.f("ix_publication_scheduled_for"), ["scheduled_for"])
        batch_op.create_foreign_key(
            "fk_publication_account_id", "social_account", ["account_id"], ["id"]
        )


def downgrade() -> None:
    with op.batch_alter_table("publication", schema=None) as batch_op:
        batch_op.drop_constraint("fk_publication_account_id", type_="foreignkey")
        batch_op.drop_index(batch_op.f("ix_publication_scheduled_for"))
        batch_op.drop_index(batch_op.f("ix_publication_account_id"))
        batch_op.drop_column("hold_reason")
        batch_op.drop_column("auto")
        batch_op.drop_column("scheduled_for")
        batch_op.drop_column("account_id")
    op.drop_table("social_account")
