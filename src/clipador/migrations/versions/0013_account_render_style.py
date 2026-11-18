"""per-account edit style: social_account.render_style, render.account_id/variant

Revision ID: 0013
Revises: 0012
Create Date: 2026-11-18 14:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("social_account", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("render_style", sa.JSON(), nullable=False, server_default="{}")
        )
    with op.batch_alter_table("render", schema=None) as batch_op:
        batch_op.add_column(sa.Column("account_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("variant", sa.JSON(), nullable=True))
        batch_op.create_index(batch_op.f("ix_render_account_id"), ["account_id"])


def downgrade() -> None:
    with op.batch_alter_table("render", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_render_account_id"))
        batch_op.drop_column("variant")
        batch_op.drop_column("account_id")
    with op.batch_alter_table("social_account", schema=None) as batch_op:
        batch_op.drop_column("render_style")
