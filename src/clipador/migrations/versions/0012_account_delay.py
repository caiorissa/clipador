"""social_account.delay_minutes (post on this account N minutes after the approval)

Revision ID: 0012
Revises: 0011
Create Date: 2026-11-18 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("social_account", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("delay_minutes", sa.Integer(), nullable=False, server_default="0")
        )


def downgrade() -> None:
    with op.batch_alter_table("social_account", schema=None) as batch_op:
        batch_op.drop_column("delay_minutes")
