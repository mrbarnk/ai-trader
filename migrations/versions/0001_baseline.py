"""Baseline schema

Revision ID: 0001_baseline
Revises: 
Create Date: 2026-01-07 00:00:00.000000
"""

from alembic import op

from server.models import Base

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    pass
