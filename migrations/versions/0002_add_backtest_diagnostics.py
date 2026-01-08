"""Add backtest diagnostics

Revision ID: 0002_add_backtest_diagnostics
Revises: 0001_baseline
Create Date: 2026-01-08 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "0002_add_backtest_diagnostics"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("backtests", sa.Column("diagnostics_json", sa.Text()))


def downgrade() -> None:
    op.drop_column("backtests", "diagnostics_json")
