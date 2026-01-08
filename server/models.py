from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from .settings import DATABASE_URL


def _normalize_database_url() -> tuple[str, dict[str, Any]]:
    url = DATABASE_URL
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    connect_args: dict[str, Any] = {}
    if url.startswith("sqlite:///"):
        connect_args = {"check_same_thread": False}
    return url, connect_args


_db_url, _db_connect_args = _normalize_database_url()
engine = create_engine(_db_url, connect_args=_db_connect_args, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    avatar_url = Column(Text)
    timezone = Column(String(50), server_default="UTC", nullable=False)
    email_verified = Column(Boolean, server_default="0", nullable=False)
    two_factor_enabled = Column(Boolean, server_default="0", nullable=False)
    two_factor_secret = Column(String(255))
    reset_token_hash = Column(String(255))
    reset_token_expires_at = Column(DateTime(timezone=True))
    reset_token_sent_at = Column(DateTime(timezone=True))
    email_verify_token_hash = Column(String(255))
    email_verify_expires_at = Column(DateTime(timezone=True))
    email_verify_sent_at = Column(DateTime(timezone=True))
    subscription_tier = Column(String(20), server_default="free", nullable=False)
    subscription_expires_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    tokens = relationship("ApiToken", cascade="all, delete-orphan", back_populates="user")
    configs = relationship("UserConfig", cascade="all, delete-orphan", back_populates="user")
    backtests = relationship("Backtest", cascade="all, delete-orphan", back_populates="user")
    accounts = relationship("Mt5Account", cascade="all, delete-orphan", back_populates="user")


class ApiToken(Base):
    __tablename__ = "api_tokens"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    user = relationship("User", back_populates="tokens")


class UserConfig(Base):
    __tablename__ = "user_configs"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    config_json = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    user = relationship("User", back_populates="configs")


class Mt5Account(Base):
    __tablename__ = "mt5_accounts"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    nickname = Column(String(100))
    account_number = Column(String(50), nullable=False)
    platform = Column(String(10), nullable=False)
    account_type = Column(String(10), nullable=False)
    broker = Column(String(100), nullable=False)
    server = Column(String(100), nullable=False)
    password_encrypted = Column(Text, nullable=False)
    metaapi_account_id = Column(String(100))
    metaapi_region = Column(String(50))
    trade_tag = Column(String(50))
    magic_number = Column(Integer)
    balance = Column(Numeric(15, 2), default=0)
    equity = Column(Numeric(15, 2), default=0)
    margin = Column(Numeric(15, 2), default=0)
    free_margin = Column(Numeric(15, 2), default=0)
    leverage = Column(Integer, default=100)
    currency = Column(String(10), default="USD")
    status = Column(String(20), default="disconnected")
    last_sync_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    user = relationship("User", back_populates="accounts")
    settings = relationship(
        "AccountSettings", cascade="all, delete-orphan", back_populates="account", uselist=False
    )
    trades = relationship("Trade", cascade="all, delete-orphan", back_populates="account")


class AccountSettings(Base):
    __tablename__ = "account_settings"

    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey("mt5_accounts.id"), unique=True, nullable=False)
    ai_enabled = Column(Boolean, default=False)
    selected_model = Column(String(20))
    risk_per_trade = Column(Numeric(4, 2), default=0.5)
    use_real_balance = Column(Boolean, default=True)
    balance_override = Column(Numeric(15, 2), default=10000)
    min_lot = Column(Numeric(5, 2), default=0.01)
    max_lot = Column(Numeric(5, 2), default=5.0)
    enable_drawdown_limit = Column(Boolean, default=True)
    max_drawdown_percent = Column(Numeric(4, 2), default=3)
    enable_consecutive_loss = Column(Boolean, default=True)
    max_consecutive_losses = Column(Integer, default=4)
    enable_max_daily_losses = Column(Boolean, default=True)
    max_daily_losses = Column(Integer, default=10)
    enable_profit_target = Column(Boolean, default=False)
    profit_target_percent = Column(Numeric(4, 2), default=8)
    sl_lookback = Column(Integer, default=3)
    sl_buffer = Column(Integer, default=3)
    enable_break_even = Column(Boolean, default=True)
    be_buffer = Column(Integer, default=2)
    tp1_percent = Column(Integer, default=50)
    tp2_percent = Column(Integer, default=90)
    enable_tp3 = Column(Boolean, default=False)
    tp3_percent = Column(Integer, default=100)
    london_enabled = Column(Boolean, default=True)
    ny_enabled = Column(Boolean, default=False)
    notify_trade_open = Column(Boolean, default=True)
    notify_tp_hit = Column(Boolean, default=True)
    notify_sl_hit = Column(Boolean, default=True)
    notify_limit_hit = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    account = relationship("Mt5Account", back_populates="settings")


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey("mt5_accounts.id"), nullable=False, index=True)
    mt_ticket_id = Column(Integer)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(4), nullable=False)
    entry_price = Column(Numeric(10, 5), nullable=False)
    exit_price = Column(Numeric(10, 5))
    stop_loss = Column(Numeric(10, 5), nullable=False)
    take_profit_1 = Column(Numeric(10, 5))
    take_profit_2 = Column(Numeric(10, 5))
    take_profit_3 = Column(Numeric(10, 5))
    position_size = Column(Numeric(10, 2), nullable=False)
    risk_amount = Column(Numeric(15, 2))
    pips = Column(Numeric(10, 1))
    pnl = Column(Numeric(15, 2))
    r_multiple = Column(Numeric(5, 2))
    outcome = Column(String(20))
    balance_before = Column(Numeric(15, 2))
    balance_after = Column(Numeric(15, 2))
    session = Column(String(10))
    entry_time = Column(DateTime(timezone=True), nullable=False)
    exit_time = Column(DateTime(timezone=True))
    duration_seconds = Column(Integer)
    model_used = Column(String(20))
    model_tag = Column(String(50))
    model_magic = Column(Integer)
    rules_passed = Column(Text)
    entry_reasoning = Column(Text)
    is_live = Column(Boolean, default=True)
    backtest_id = Column(String(64))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    account = relationship("Mt5Account", back_populates="trades")


class TradingModel(Base):
    __tablename__ = "trading_models"

    id = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False)
    badge = Column(String(50))
    description = Column(Text)
    entry_style = Column(String(255))
    filters = Column(String(255))
    tp_strategy = Column(String(255))
    risk_description = Column(String(255))
    trading_description = Column(String(255))
    best_for = Column(String(255))
    default_risk_per_trade = Column(Numeric(4, 2))
    default_min_lot = Column(Numeric(5, 2))
    default_max_lot = Column(Numeric(5, 2))
    default_sl_buffer = Column(Integer)
    default_tp1_percent = Column(Integer)
    default_tp2_percent = Column(Integer)
    default_break_even = Column(Boolean)
    default_max_drawdown = Column(Numeric(4, 2))
    default_consecutive_losses = Column(Integer)
    default_max_daily_losses = Column(Integer)
    active_accounts = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Backtest(Base):
    __tablename__ = "backtests"

    id = Column(String(64), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255))
    model = Column(String(20))
    date_start = Column(Date)
    date_end = Column(Date)
    starting_balance = Column(Numeric(15, 2))
    symbol = Column(String(20), default="GBPUSD")
    settings_json = Column(Text)
    status = Column(String(20), default="pending")
    progress = Column(Integer, default=0)
    error_message = Column(Text)
    ending_balance = Column(Numeric(15, 2))
    net_pnl = Column(Numeric(15, 2))
    net_pnl_percent = Column(Numeric(10, 2))
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    break_even_trades = Column(Integer)
    win_rate = Column(Numeric(5, 2))
    total_buys = Column(Integer)
    buy_win_rate = Column(Numeric(5, 2))
    total_sells = Column(Integer)
    sell_win_rate = Column(Numeric(5, 2))
    total_r = Column(Numeric(10, 2))
    avg_r = Column(Numeric(5, 2))
    best_r = Column(Numeric(5, 2))
    worst_r = Column(Numeric(5, 2))
    session_performance = Column(Text)
    equity_data = Column(Text)
    rows_json = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True))
    user = relationship("User", back_populates="backtests")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


init_db()


@contextmanager
def db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
