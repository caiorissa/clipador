"""caption editor: originals of corrected words, per-streamer glossary

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-13 22:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("candidate", schema=None) as batch_op:
        batch_op.add_column(sa.Column("caption_override_originals", sa.JSON(), nullable=True))
    with op.batch_alter_table("streamer", schema=None) as batch_op:
        batch_op.add_column(sa.Column("glossary", sa.JSON(), nullable=True))


def downgrade() -> None:
    # 'edit_checks' decisions only exist from this revision on (their undo needs `checks`,
    # which 0006 already has), so they can stay.
    with op.batch_alter_table("streamer", schema=None) as batch_op:
        batch_op.drop_column("glossary")
    with op.batch_alter_table("candidate", schema=None) as batch_op:
        batch_op.drop_column("caption_override_originals")
