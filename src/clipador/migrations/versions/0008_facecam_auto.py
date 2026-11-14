"""facecam auto-detection: facecam_auto flag and facecam_confidence on streamer

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-13 23:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Existing boxes were all drawn by hand: facecam_auto defaults to false (never overwritten).
    with op.batch_alter_table("streamer", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("facecam_auto", sa.Boolean(), nullable=False, server_default="0")
        )
        batch_op.add_column(sa.Column("facecam_confidence", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("streamer", schema=None) as batch_op:
        batch_op.drop_column("facecam_confidence")
        batch_op.drop_column("facecam_auto")
