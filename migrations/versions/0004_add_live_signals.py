"""add live signals table

Revision ID: 0004_add_live_signals
Revises: 0003_add_backtest_csv_path
Create Date: 2024-01-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "0004_add_live_signals"
down_revision = "0003_add_backtest_csv_path"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    table_names = set(inspector.get_table_names())
    if "live_signals" not in table_names:
        op.create_table(
            "live_signals",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("account_id", sa.Integer(), sa.ForeignKey("mt5_accounts.id"), nullable=False),
            sa.Column("trade_id", sa.Integer(), sa.ForeignKey("trades.id"), nullable=True),
            sa.Column("model", sa.String(length=20)),
            sa.Column("decision", sa.String(length=20)),
            sa.Column("symbol", sa.String(length=20)),
            sa.Column("direction", sa.String(length=4)),
            sa.Column("entry_price", sa.Numeric(10, 5)),
            sa.Column("stop_loss", sa.Numeric(10, 5)),
            sa.Column("take_profit_1", sa.Numeric(10, 5)),
            sa.Column("take_profit_2", sa.Numeric(10, 5)),
            sa.Column("take_profit_3", sa.Numeric(10, 5)),
            sa.Column("rules_passed", sa.Text()),
            sa.Column("rules_failed", sa.Text()),
            sa.Column("status", sa.String(length=20), server_default="new"),
            sa.Column("error_message", sa.Text()),
            sa.Column("signal_time", sa.DateTime(timezone=True)),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                onupdate=sa.func.now(),
            ),
        )
    indexes = {index["name"] for index in inspector.get_indexes("live_signals")}
    if "ix_live_signals_account_id" not in indexes:
        op.create_index("ix_live_signals_account_id", "live_signals", ["account_id"])


def downgrade() -> None:
    op.drop_index("ix_live_signals_account_id", table_name="live_signals")
    op.drop_table("live_signals")
