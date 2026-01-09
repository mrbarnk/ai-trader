"""Add backtest csv_path

Revision ID: 0003_add_backtest_csv_path
Revises: 0002_add_backtest_diagnostics
Create Date: 2026-01-09 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "0003_add_backtest_csv_path"
down_revision = "0002_add_backtest_diagnostics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("backtests", sa.Column("csv_path", sa.Text()))


def downgrade() -> None:
    op.drop_column("backtests", "csv_path")
