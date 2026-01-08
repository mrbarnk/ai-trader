# AlgoTrade AI - Backend API Specification

> **Version:** 1.0.0  
> **Last Updated:** January 2025  
> **Frontend Stack:** React 18, TypeScript, Vite, TailwindCSS, React Router

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Database Schema](#database-schema)
4. [API Endpoints](#api-endpoints)
5. [Data Models](#data-models)
6. [WebSocket Events](#websocket-events)
7. [External Integrations](#external-integrations)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)

---

## Overview

AlgoTrade AI is an automated forex trading platform that generates model signals and executes trades on connected MetaTrader 4/5 accounts. The backend must support:

- **User authentication** (email/password, OAuth)
- **MT4/MT5 account connections** via MetaApi.cloud (primary integration)
- **Real-time trade execution** and monitoring
- **Backtesting engine** for historical strategy testing
- **Notification system** (in-app, email, push)
- **Subscription/billing** management

### Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   React App     │────▶│   REST API       │────▶│   PostgreSQL    │
│   (Frontend)    │     │   (Backend)      │     │   (Database)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                        
         │                       ▼                        
         │              ┌──────────────────┐              
         │              │   WebSocket      │              
         └─────────────▶│   (Real-time)    │              
                        └──────────────────┘              
                                 │                        
                                 ▼                        
                        ┌──────────────────┐              
                        │   MetaApi.cloud  │              
                        │   (MT4/MT5)      │              
                        └──────────────────┘              
```

---

## Authentication

### Requirements

- JWT-based authentication (access token)
- Refresh token rotation (refresh token stored server-side)
- Email verification flow
- Password reset flow
- Optional 2FA support

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/signup` | Create new user account |
| POST | `/auth/login` | Authenticate user |
| POST | `/auth/logout` | Invalidate session |
| POST | `/auth/refresh` | Refresh access token |
| POST | `/auth/forgot-password` | Request password reset |
| POST | `/auth/reset-password` | Complete password reset |
| POST | `/auth/verify-email` | Verify email address |
| GET | `/auth/me` | Get current user profile |
| PATCH | `/auth/me` | Update user profile |

### Password Reset Flow (Implemented)

1. `POST /auth/forgot-password`
   - Body: `{ "email": "user@example.com" }`
   - Always returns `{ "ok": true }` (prevents email enumeration).
   - Sends a reset code (logged server-side by default).
2. `POST /auth/reset-password`
   - Body: `{ "token": "<reset_code>", "password": "<new_password>" }`
   - Resets password and revokes existing tokens.

### Email Verification Flow (Implemented)

- `POST /auth/verify-email`
  - If `token` is provided: verifies email using the code.
  - If no token is provided: uses bearer token to send a new verification code.

### User Profile Fields

```typescript
interface User {
  id: string;                    // UUID
  email: string;
  full_name: string;
  avatar_url?: string;
  timezone: string;              // e.g., "America/New_York"
  created_at: string;            // ISO 8601
  updated_at: string;
  email_verified: boolean;
  two_factor_enabled: boolean;
  subscription_tier: "free" | "demo" | "live";
  subscription_expires_at?: string;
}
```

### Token Responses

```json
{
  "access_token": "<jwt>",
  "refresh_token": "<refresh_token>",
  "user": { "...": "..." }
}
```

---

## Database Schema

### Entity Relationship Diagram

```
┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│     users       │──────▶│   mt5_accounts   │──────▶│     trades      │
└─────────────────┘       └──────────────────┘       └─────────────────┘
         │                         │                          │
         │                         │                          │
         ▼                         ▼                          ▼
┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│ subscriptions   │       │ account_settings │       │  notifications  │
└─────────────────┘       └──────────────────┘       └─────────────────┘
         │                                                    
         ▼                                                    
┌─────────────────┐       ┌──────────────────┐                
│   backtests     │──────▶│  backtest_trades │                
└─────────────────┘       └──────────────────┘                
```

### Tables

#### `users`
```sql
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  full_name VARCHAR(255),
  avatar_url TEXT,
  timezone VARCHAR(50) DEFAULT 'UTC',
  email_verified BOOLEAN DEFAULT FALSE,
  two_factor_enabled BOOLEAN DEFAULT FALSE,
  two_factor_secret VARCHAR(255),
  reset_token_hash VARCHAR(255),
  reset_token_expires_at TIMESTAMPTZ,
  reset_token_sent_at TIMESTAMPTZ,
  email_verify_token_hash VARCHAR(255),
  email_verify_expires_at TIMESTAMPTZ,
  email_verify_sent_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### `mt5_accounts`
```sql
CREATE TABLE mt5_accounts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  nickname VARCHAR(100),
  account_number VARCHAR(50) NOT NULL,
  platform VARCHAR(10) NOT NULL CHECK (platform IN ('MT4', 'MT5')),
  account_type VARCHAR(10) NOT NULL CHECK (account_type IN ('demo', 'live')),
  broker VARCHAR(100) NOT NULL,
  server VARCHAR(100) NOT NULL,
  password_encrypted TEXT NOT NULL,  -- Encrypted investor password
  metaapi_account_id VARCHAR(100),   -- MetaApi account identifier
  metaapi_region VARCHAR(50),        -- e.g., "new-york"
  trade_tag VARCHAR(50),             -- order comment/tag for model tracking
  magic_number INTEGER,              -- MT5 magic number for model tracking
  balance DECIMAL(15, 2) DEFAULT 0,
  equity DECIMAL(15, 2) DEFAULT 0,
  margin DECIMAL(15, 2) DEFAULT 0,
  free_margin DECIMAL(15, 2) DEFAULT 0,
  leverage INTEGER DEFAULT 100,
  currency VARCHAR(10) DEFAULT 'USD',
  status VARCHAR(20) DEFAULT 'disconnected' CHECK (status IN ('connected', 'disconnected', 'syncing', 'error')),
  last_sync_at TIMESTAMPTZ,
  error_message TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(user_id, account_number, broker)
);
```

#### `account_settings`
```sql
CREATE TABLE account_settings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  account_id UUID NOT NULL REFERENCES mt5_accounts(id) ON DELETE CASCADE,
  
  -- Trading Model
  ai_enabled BOOLEAN DEFAULT FALSE,
  selected_model VARCHAR(20) CHECK (selected_model IN ('aggressive', 'passive')),
  
  -- Risk Management
  risk_per_trade DECIMAL(4, 2) DEFAULT 0.5,     -- Percentage
  use_real_balance BOOLEAN DEFAULT TRUE,
  balance_override DECIMAL(15, 2) DEFAULT 10000,
  min_lot DECIMAL(5, 2) DEFAULT 0.01,
  max_lot DECIMAL(5, 2) DEFAULT 5.0,
  
  -- Daily Limits
  enable_drawdown_limit BOOLEAN DEFAULT TRUE,
  max_drawdown_percent DECIMAL(4, 2) DEFAULT 3,
  enable_consecutive_loss BOOLEAN DEFAULT TRUE,
  max_consecutive_losses INTEGER DEFAULT 4,
  enable_max_daily_losses BOOLEAN DEFAULT TRUE,
  max_daily_losses INTEGER DEFAULT 10,
  enable_profit_target BOOLEAN DEFAULT FALSE,
  profit_target_percent DECIMAL(4, 2) DEFAULT 8,
  
  -- Stop Loss & Take Profit
  sl_lookback INTEGER DEFAULT 3,               -- Candles
  sl_buffer INTEGER DEFAULT 3,                 -- Pips
  enable_break_even BOOLEAN DEFAULT TRUE,
  be_buffer INTEGER DEFAULT 2,                 -- Pips
  tp1_percent INTEGER DEFAULT 50,              -- % position close
  tp2_percent INTEGER DEFAULT 90,              -- % position close
  enable_tp3 BOOLEAN DEFAULT FALSE,
  tp3_percent INTEGER DEFAULT 100,
  
  -- Session
  london_enabled BOOLEAN DEFAULT TRUE,
  ny_enabled BOOLEAN DEFAULT FALSE,
  
  -- Notifications
  notify_trade_open BOOLEAN DEFAULT TRUE,
  notify_tp_hit BOOLEAN DEFAULT TRUE,
  notify_sl_hit BOOLEAN DEFAULT TRUE,
  notify_limit_hit BOOLEAN DEFAULT TRUE,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(account_id)
);
```

#### `trades`
```sql
CREATE TABLE trades (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  account_id UUID NOT NULL REFERENCES mt5_accounts(id) ON DELETE CASCADE,
  
  -- Trade Identification
  mt_ticket_id BIGINT,                         -- MetaTrader ticket number
  symbol VARCHAR(20) NOT NULL,                 -- e.g., "GBPUSD"
  
  -- Trade Details
  direction VARCHAR(4) NOT NULL CHECK (direction IN ('BUY', 'SELL')),
  entry_price DECIMAL(10, 5) NOT NULL,
  exit_price DECIMAL(10, 5),
  stop_loss DECIMAL(10, 5) NOT NULL,
  take_profit_1 DECIMAL(10, 5),
  take_profit_2 DECIMAL(10, 5),
  take_profit_3 DECIMAL(10, 5),
  
  -- Position Sizing
  position_size DECIMAL(10, 2) NOT NULL,       -- Lot size
  risk_amount DECIMAL(15, 2),                  -- $ risked
  
  -- Results
  pips DECIMAL(10, 1),
  pnl DECIMAL(15, 2),
  r_multiple DECIMAL(5, 2),                    -- Risk-adjusted return
  outcome VARCHAR(20) CHECK (outcome IN ('TP1', 'TP2', 'TP3', 'SL', 'BE', 'TP1_THEN_BE', 'MANUAL', 'OPEN')),
  
  -- Balance
  balance_before DECIMAL(15, 2),
  balance_after DECIMAL(15, 2),
  
  -- Timing
  session VARCHAR(10) CHECK (session IN ('LONDON', 'NY', 'ASIAN', 'OTHER')),
  entry_time TIMESTAMPTZ NOT NULL,
  exit_time TIMESTAMPTZ,
  duration_seconds INTEGER,
  
  -- AI Analysis
  model_used VARCHAR(20),                    -- 'aggressive' or 'passive'
  model_tag VARCHAR(50),                     -- copied from account trade_tag
  model_magic INTEGER,                       -- copied from account magic_number
  rules_passed TEXT[],                         -- Array of rule names that triggered entry
  entry_reasoning TEXT,
  
  -- Metadata
  is_live BOOLEAN DEFAULT TRUE,                -- FALSE for backtests
  backtest_id UUID REFERENCES backtests(id),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_trades_account_id ON trades(account_id);
CREATE INDEX idx_trades_entry_time ON trades(entry_time);
CREATE INDEX idx_trades_backtest_id ON trades(backtest_id);
```

#### `trading_models`
```sql
CREATE TABLE trading_models (
  id VARCHAR(20) PRIMARY KEY,                  -- 'aggressive' or 'passive'
  name VARCHAR(100) NOT NULL,
  badge VARCHAR(50),
  description TEXT,
  
  -- Characteristics
  entry_style VARCHAR(255),
  filters VARCHAR(255),
  tp_strategy VARCHAR(255),
  risk_description VARCHAR(255),
  trading_description VARCHAR(255),
  best_for VARCHAR(255),
  
  -- Default Settings (can be overridden per account)
  default_risk_per_trade DECIMAL(4, 2),
  default_min_lot DECIMAL(5, 2),
  default_max_lot DECIMAL(5, 2),
  default_sl_buffer INTEGER,
  default_tp1_percent INTEGER,
  default_tp2_percent INTEGER,
  default_break_even BOOLEAN,
  default_max_drawdown DECIMAL(4, 2),
  default_consecutive_losses INTEGER,
  default_max_daily_losses INTEGER,
  
  -- Stats (updated periodically)
  active_accounts INTEGER DEFAULT 0,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### `model_performance`
```sql
CREATE TABLE model_performance (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id VARCHAR(20) NOT NULL REFERENCES trading_models(id),
  period_type VARCHAR(20) NOT NULL CHECK (period_type IN ('monthly', 'quarterly', 'yearly', '6month')),
  period_start DATE NOT NULL,
  period_end DATE NOT NULL,
  
  -- Performance Metrics
  total_return DECIMAL(10, 2),
  win_rate DECIMAL(5, 2),
  total_trades INTEGER,
  trades_per_day DECIMAL(5, 2),
  max_drawdown DECIMAL(10, 2),
  sharpe_ratio DECIMAL(5, 2),
  sortino_ratio DECIMAL(5, 2),
  calmar_ratio DECIMAL(5, 2),
  profit_factor DECIMAL(5, 2),
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  UNIQUE(model_id, period_type, period_start)
);
```

#### `backtests`
```sql
CREATE TABLE backtests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name VARCHAR(255) NOT NULL,
  model VARCHAR(20) NOT NULL REFERENCES trading_models(id),
  
  -- Configuration
  date_start DATE NOT NULL,
  date_end DATE NOT NULL,
  starting_balance DECIMAL(15, 2) NOT NULL,
  symbol VARCHAR(20) DEFAULT 'GBPUSD',
  settings JSONB,                              -- Snapshot of account_settings used
  
  -- Status
  status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
  progress INTEGER DEFAULT 0,                  -- 0-100
  error_message TEXT,
  
  -- Results Summary
  ending_balance DECIMAL(15, 2),
  net_pnl DECIMAL(15, 2),
  net_pnl_percent DECIMAL(10, 2),
  peak_balance DECIMAL(15, 2),
  max_drawdown DECIMAL(15, 2),
  max_drawdown_percent DECIMAL(10, 2),
  recovery_time_days INTEGER,
  
  -- Trade Stats
  total_trades INTEGER,
  winning_trades INTEGER,
  losing_trades INTEGER,
  break_even_trades INTEGER,
  win_rate DECIMAL(5, 2),
  total_buys INTEGER,
  buy_win_rate DECIMAL(5, 2),
  total_sells INTEGER,
  sell_win_rate DECIMAL(5, 2),
  
  -- Profit Metrics
  gross_profit DECIMAL(15, 2),
  gross_loss DECIMAL(15, 2),
  profit_factor DECIMAL(5, 2),
  avg_win DECIMAL(15, 2),
  avg_loss DECIMAL(15, 2),
  win_loss_ratio DECIMAL(5, 2),
  largest_win DECIMAL(15, 2),
  largest_loss DECIMAL(15, 2),
  
  -- R Metrics
  total_r DECIMAL(10, 2),
  avg_r DECIMAL(5, 2),
  best_r DECIMAL(5, 2),
  worst_r DECIMAL(5, 2),
  expectancy DECIMAL(5, 2),
  
  -- Time Analysis
  avg_duration_minutes INTEGER,
  shortest_trade_minutes INTEGER,
  longest_trade_minutes INTEGER,
  trades_per_day DECIMAL(5, 2),
  most_active_hour INTEGER,                    -- 0-23
  
  -- Daily Limit Hits
  drawdown_limit_hits INTEGER DEFAULT 0,
  consecutive_loss_hits INTEGER DEFAULT 0,
  max_loss_hits INTEGER DEFAULT 0,
  profit_target_hits INTEGER DEFAULT 0,
  avg_daily_pnl DECIMAL(15, 2),
  
  -- Risk Metrics
  sharpe_ratio DECIMAL(5, 2),
  sortino_ratio DECIMAL(5, 2),
  calmar_ratio DECIMAL(5, 2),
  max_consecutive_wins INTEGER,
  max_consecutive_losses INTEGER,
  avg_consecutive_losses DECIMAL(5, 2),
  
  -- Session Performance (stored as JSONB)
  session_performance JSONB,
  
  -- Equity Curve Data (stored as JSONB for chart)
  equity_data JSONB,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);

CREATE INDEX idx_backtests_user_id ON backtests(user_id);
CREATE INDEX idx_backtests_status ON backtests(status);
```

#### `notifications`
```sql
CREATE TABLE notifications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  
  type VARCHAR(50) NOT NULL,                   -- 'trade_executed', 'model_update', 'account_synced', etc.
  title VARCHAR(255) NOT NULL,
  message TEXT NOT NULL,
  metadata JSONB,                              -- Additional context (trade_id, account_id, etc.)
  
  read BOOLEAN DEFAULT FALSE,
  read_at TIMESTAMPTZ,
  
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_notifications_user_id ON notifications(user_id);
CREATE INDEX idx_notifications_read ON notifications(read);
```

#### `subscriptions`
```sql
CREATE TABLE subscriptions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  
  tier VARCHAR(20) NOT NULL CHECK (tier IN ('free', 'demo', 'live')),
  status VARCHAR(20) NOT NULL CHECK (status IN ('active', 'canceled', 'past_due', 'expired')),
  
  -- Stripe Integration
  stripe_customer_id VARCHAR(255),
  stripe_subscription_id VARCHAR(255),
  stripe_price_id VARCHAR(255),
  
  -- Billing
  current_period_start TIMESTAMPTZ,
  current_period_end TIMESTAMPTZ,
  cancel_at_period_end BOOLEAN DEFAULT FALSE,
  
  -- Limits
  max_live_accounts INTEGER DEFAULT 0,
  max_demo_accounts INTEGER DEFAULT 2,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## API Endpoints

### Accounts API

#### List Connected Accounts
```
GET /api/accounts
```

**Response:**
```json
{
  "accounts": [
    {
      "id": "acc_123",
      "nickname": "FTMO Challenge",
      "account_number": "12345678",
      "platform": "MT5",
      "type": "demo",
      "broker": "FTMO",
      "server": "FTMO-Demo",
      "balance": 10247.32,
      "equity": 10312.45,
      "margin": 0,
      "free_margin": 10312.45,
      "status": "connected",
      "ai_enabled": true,
      "active_model": "passive",
      "last_sync": "2025-01-08T10:30:00Z"
    }
  ]
}
```

#### Connect New Account
```
POST /api/accounts/connect
```

**Request:**
```json
{
  "platform": "MT5",
  "type": "demo",
  "broker": "ICH Markets",
  "account_number": "12345678",
  "server": "ICHMarkets-Demo",
  "password": "investor_password",
  "nickname": "My Trading Account",
  "trade_tag": null,
  "magic_number": null,
  "metaapi_account_id": null
}
```

**Response:**
```json
{
  "success": true,
  "account": {
    "id": "acc_456",
    "status": "connected",
    "balance": 10000.00,
    "...": "..."
  }
}
```

#### Update Account Settings
```
PATCH /api/accounts/:accountId/settings
```

**Request:**
```json
{
  "ai_enabled": true,
  "selected_model": "passive",
  "risk_per_trade": 0.5,
  "london_enabled": true,
  "ny_enabled": false,
  "max_drawdown_percent": 3,
  "...": "..."
}
```

#### Disconnect Account
```
DELETE /api/accounts/:accountId
```

#### Sync Account Data
```
POST /api/accounts/:accountId/sync
```

#### Place Trade Order
```
POST /api/accounts/:accountId/orders
```

**Request:**
```json
{
  "symbol": "GBPUSD",
  "direction": "SELL",
  "order_type": "LIMIT",
  "volume": 0.2,
  "entry_price": 1.2755,
  "stop_loss": 1.2775,
  "take_profit": 1.2690,
  "comment": "MODEL_AGGRESSIVE"
}
```

---

### Trades API

#### Get Recent Trades
```
GET /api/accounts/:accountId/trades
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | integer | Number of trades (default: 50) |
| `offset` | integer | Pagination offset |
| `start_date` | string | Filter by start date (ISO 8601) |
| `end_date` | string | Filter by end date |
| `direction` | string | Filter: "BUY" or "SELL" |
| `outcome` | string | Filter: "TP1", "TP2", "SL", "BE" |
| `session` | string | Filter: "LONDON" or "NY" |

**Response:**
```json
{
  "trades": [
    {
      "id": 1,
      "date": "2024-03-01",
      "time": "10:34",
      "direction": "SELL",
      "entry_price": 1.27834,
      "exit_price": 1.27523,
      "stop_loss": 1.28156,
      "tp1": 1.27534,
      "tp2": 1.27234,
      "pips": 31.1,
      "pnl": 15.55,
      "r_multiple": 1.2,
      "duration": "4h 23m",
      "outcome": "TP1",
      "session": "LONDON",
      "position_size": 0.05,
      "risk_amount": 12.50,
      "balance_after": 10262.87,
      "rules_passed": ["4H Bullish Break", "London Open", "Clean Setup"]
    }
  ],
  "pagination": {
    "total": 147,
    "limit": 50,
    "offset": 0
  },
  "summary": {
    "total_pnl": 847.32,
    "total_pips": 1247.5,
    "avg_r": 0.82
  }
}
```

#### Get Trade Detail
```
GET /api/trades/:tradeId
```

#### Get Trade Statistics
```
GET /api/accounts/:accountId/stats
```

**Response:**
```json
{
  "period": "all_time",
  "balance": 10247.32,
  "starting_balance": 10000.00,
  "net_pnl": 247.32,
  "net_pnl_percent": 2.47,
  "total_trades": 147,
  "winning_trades": 101,
  "losing_trades": 46,
  "win_rate": 68.71,
  "profit_factor": 1.89,
  "avg_win": 15.55,
  "avg_loss": -11.10,
  "max_drawdown": 3.2,
  "sharpe_ratio": 2.34,
  "avg_trade_duration_minutes": 263,
  "trades_today": 3,
  "pnl_today": 41.48
}
```

---

### Models API

#### List Trading Models
```
GET /api/models
```

**Response:**
```json
{
  "models": [
    {
      "id": "aggressive",
      "name": "Aggressive",
      "badge": "High Frequency",
      "description": "Optimized for frequent entries...",
      "characteristics": {
        "entry_style": "Aggressive entry on first confirmation",
        "filters": "Minimal filtering, focus on volume",
        "tp_strategy": "Quick TP1 at 50%, trail to TP2",
        "risk": "0.5-1% per trade, higher frequency",
        "trading": "London & NY sessions",
        "best_for": "Traders seeking higher returns with more activity"
      },
      "six_month_performance": {
        "total_return": 127.4,
        "win_rate": 58.2,
        "total_trades": 1247,
        "trades_per_day": 6.9,
        "max_drawdown": 18.3,
        "sharpe_ratio": 1.67
      },
      "monthly_returns": [
        { "month": "Sep 2024", "return": 24.3, "trades": 198, "win_rate": 59.1, "drawdown": 8.2 }
      ],
      "active_accounts": 234
    },
    {
      "id": "passive",
      "name": "Passive",
      "badge": "Conservative",
      "...": "..."
    }
  ]
}
```

#### Get Model Details
```
GET /api/models/:modelId
```

#### Get Model Backtests
```
GET /api/models/:modelId/backtests
```

#### Copy Model to Account
```
POST /api/models/:modelId/copy
```

**Request:**
```json
{
  "account_id": "acc_123",
  "risk_per_trade": 0.5,
  "use_real_balance": true
}
```

---

### Backtests API

#### List User Backtests
```
GET /api/backtests
```

#### Create Backtest
```
POST /api/backtests
```

**Request:**
```json
{
  "name": "Q1 2024 Test",
  "model": "passive",
  "date_start": "2024-01-01",
  "date_end": "2024-03-31",
  "starting_balance": 10000,
  "symbol": "GBPUSD",
  "settings": {
    "risk_per_trade": 0.5,
    "london_enabled": true,
    "ny_enabled": false,
    "...": "..."
  }
}
```

CSV data can be provided as multipart upload (`csv`) or as a base64 string (`csv_base64`) until broker data providers are wired in.

**Response:**
```json
{
  "id": "bt_789",
  "status": "processing",
  "progress": 0
}
```

Backtests run asynchronously. Poll the status endpoint to see live progress.
Results are persisted in the database and remain available after completion.

#### Backtest Status
```
GET /api/backtests/:backtestId
```

**Response (processing):**
```json
{
  "id": "bt_789",
  "status": "processing",
  "progress": 42
}
```

#### Get Backtest Results
```
GET /api/backtests/:backtestId
```

**Response:**
```json
{
  "id": "bt_789",
  "name": "Q1 2024 Test",
  "model": "passive",
  "status": "completed",
  "date_range": {
    "start": "2024-01-01",
    "end": "2024-03-31"
  },
  "performance": {
    "starting_balance": 10000,
    "ending_balance": 19460,
    "net_pnl": 9460,
    "net_pnl_percent": 94.6,
    "peak_balance": 19890,
    "max_drawdown": 920,
    "max_drawdown_percent": 9.2,
    "recovery_time": "8 days"
  },
  "trade_stats": {
    "total_trades": 487,
    "winning_trades": 333,
    "losing_trades": 127,
    "break_even_trades": 27,
    "win_rate": 68.4,
    "total_buys": 245,
    "buy_win_rate": 66.2,
    "total_sells": 242,
    "sell_win_rate": 70.7
  },
  "profit_metrics": {
    "gross_profit": 14280,
    "gross_loss": 4820,
    "profit_factor": 2.96,
    "avg_win": 42.88,
    "avg_loss": 37.95,
    "win_loss_ratio": 1.13,
    "largest_win": 187.50,
    "largest_loss": 145.00
  },
  "r_metrics": {
    "total_r": 127.8,
    "avg_r": 0.26,
    "best_r": 3.75,
    "worst_r": -1.00,
    "expectancy": 0.26
  },
  "time_analysis": {
    "avg_duration": "4h 12m",
    "shortest_trade": "12m",
    "longest_trade": "18h 45m",
    "trades_per_day": 5.4,
    "most_active_hour": "09:00"
  },
  "session_performance": {
    "london": {
      "trades": 312,
      "wins": 218,
      "win_rate": 69.9,
      "total_pnl": 6280,
      "avg_pnl": 20.13
    },
    "ny": {
      "trades": 175,
      "wins": 115,
      "win_rate": 65.7,
      "total_pnl": 3180,
      "avg_pnl": 18.17
    }
  },
  "equity_data": [
    { "date": "2024-01-01", "balance": 10000, "drawdown": 0, "trade_count": 0 },
    { "date": "2024-01-02", "balance": 10125, "drawdown": 0, "trade_count": 5 }
  ],
  "trades": [
    { "...": "full trade objects" }
  ]
}
```

#### Delete Backtest
```
DELETE /api/backtests/:backtestId
```

#### Export Backtest Report
```
GET /api/backtests/:backtestId/export
```

**Query Parameters:**
- `format`: "pdf" | "csv" | "xlsx"

---

### Notifications API

#### List Notifications
```
GET /api/notifications
```

**Query Parameters:**
- `unread_only`: boolean
- `limit`: integer
- `offset`: integer

**Response:**
```json
{
  "notifications": [
    {
      "id": "notif_1",
      "type": "trade_executed",
      "title": "Trade Executed",
      "message": "GBPUSD buy order filled at 1.0845",
      "time": "2 min ago",
      "unread": true,
      "metadata": {
        "trade_id": "trade_123",
        "account_id": "acc_123"
      }
    }
  ],
  "unread_count": 2
}
```

#### Mark as Read
```
PATCH /api/notifications/:notificationId/read
```

#### Mark All as Read
```
POST /api/notifications/read-all
```

---

### Dashboard API

#### Get Dashboard Summary
```
GET /api/dashboard
```

**Response:**
```json
{
  "stats": {
    "account_balance": 10247.32,
    "balance_change": 847.32,
    "balance_change_percent": 8.27,
    "win_rate": 68.5,
    "win_rate_change": 3.2,
    "total_trades": 147,
    "wins": 89,
    "losses": 58,
    "avg_trade_duration": "4h 23m"
  },
  "ai_status": {
    "active": true,
    "model": "Passive",
    "risk_per_trade": 0.5,
    "todays_trades": 3,
    "todays_pnl": 41.48
  },
  "market_data": {
    "symbol": "GBPUSD",
    "price": 1.27834,
    "change_percent": 0.12,
    "bias": "BULLISH",
    "session": "London"
  },
  "recent_trades": [
    { "...": "last 5 trades" }
  ],
  "equity_chart": [
    { "date": "2024-03-01", "balance": 10000 },
    { "date": "2024-03-02", "balance": 10125 }
  ]
}
```

---

### Market Data API

#### Get Current Price
```
GET /api/market/:symbol
```

**Response:**
```json
{
  "symbol": "GBPUSD",
  "bid": 1.27832,
  "ask": 1.27834,
  "spread": 0.2,
  "change_24h": 0.0015,
  "change_24h_percent": 0.12,
  "high_24h": 1.28145,
  "low_24h": 1.27234,
  "session": "London",
  "bias": "BULLISH",
  "timestamp": "2025-01-08T10:30:00Z"
}
```

#### Get Historical Data (for charts)
```
GET /api/market/:symbol/history
```

**Query Parameters:**
- `timeframe`: "1m" | "5m" | "15m" | "1h" | "4h" | "1d"
- `from`: ISO 8601 timestamp
- `to`: ISO 8601 timestamp

---

## Data Models

### TypeScript Interfaces

All interfaces are provided for frontend-backend contract alignment.

#### Trade
```typescript
interface Trade {
  id: number;
  date: string;                    // "2024-03-01"
  time: string;                    // "10:34"
  direction: "BUY" | "SELL";
  entryPrice: number;
  exitPrice: number;
  stopLoss: number;
  tp1: number;
  tp2: number;
  pips: number;
  pnl: number;
  rMultiple: number;
  duration: string;                // "4h 23m"
  outcome: "TP1" | "TP2" | "TP3" | "SL" | "BE" | "TP1_THEN_BE";
  session: "LONDON" | "NY";
  positionSize: number;
  riskAmount: number;
  balanceAfter: number;
  rulesPassed: string[];
}
```

#### MT5Account
```typescript
interface MT5Account {
  id: string;
  nickname: string;
  accountNumber: string;
  platform: "MT4" | "MT5";
  type: "demo" | "live";
  broker: string;
  server: string;
  balance: number;
  equity: number;
  margin: number;
  freeMargin: number;
  status: "connected" | "disconnected" | "syncing";
  aiEnabled: boolean;
  activeModel: "aggressive" | "passive" | null;
  lastSync: string;
}
```

#### AccountSettings
```typescript
interface AccountSettings {
  // Trading Model
  aiEnabled: boolean;
  selectedModel: "aggressive" | "passive" | null;
  
  // Risk Management
  riskPerTrade: number;            // 0.1 - 5.0 (percentage)
  useRealBalance: boolean;
  balanceOverride: number;
  minLot: number;
  maxLot: number;
  
  // Daily Limits
  enableDrawdownLimit: boolean;
  maxDrawdownPercent: number;
  enableConsecutiveLoss: boolean;
  maxConsecutiveLosses: number;
  enableMaxDailyLosses: boolean;
  maxDailyLosses: number;
  enableProfitTarget: boolean;
  profitTargetPercent: number;
  
  // Stop Loss & Take Profit
  slLookback: number;              // 1-10 candles
  slBuffer: number;                // pips
  enableBreakEven: boolean;
  beBuffer: number;                // pips
  tp1Percent: number;              // 0-100
  tp2Percent: number;              // 0-100
  enableTp3: boolean;
  tp3Percent: number;
  
  // Session
  londonEnabled: boolean;
  nyEnabled: boolean;
  
  // Notifications
  notifyTradeOpen: boolean;
  notifyTpHit: boolean;
  notifySlHit: boolean;
  notifyLimitHit: boolean;
}
```

#### BacktestResult
```typescript
interface BacktestResult {
  id: string;
  name: string;
  model: "aggressive" | "passive";
  dateRange: {
    start: string;
    end: string;
  };
  status: "completed" | "processing" | "failed";
  createdAt: string;
  
  performance: {
    startingBalance: number;
    endingBalance: number;
    netPnl: number;
    netPnlPercent: number;
    peakBalance: number;
    maxDrawdown: number;
    maxDrawdownPercent: number;
    recoveryTime: string;
  };
  
  tradeStats: {
    totalTrades: number;
    winningTrades: number;
    losingTrades: number;
    breakEvenTrades: number;
    winRate: number;
    totalBuys: number;
    buyWinRate: number;
    totalSells: number;
    sellWinRate: number;
  };
  
  profitMetrics: {
    grossProfit: number;
    grossLoss: number;
    profitFactor: number;
    avgWin: number;
    avgLoss: number;
    winLossRatio: number;
    largestWin: number;
    largestLoss: number;
  };
  
  rMetrics: {
    totalR: number;
    avgR: number;
    bestR: number;
    worstR: number;
    expectancy: number;
  };
  
  timeAnalysis: {
    avgDuration: string;
    shortestTrade: string;
    longestTrade: string;
    tradesPerDay: number;
    mostActiveHour: string;
  };
  
  dailyLimits: {
    drawdownLimitHits: number;
    consecutiveLossHits: number;
    maxLossHits: number;
    profitTargetHits: number;
    avgDailyPnl: number;
  };
  
  riskMetrics: {
    sharpeRatio: number;
    sortinoRatio: number;
    calmarRatio: number;
    maxConsecutiveWins: number;
    maxConsecutiveLosses: number;
    avgConsecutiveLosses: number;
  };
  
  sessionPerformance: {
    london: SessionStats;
    ny: SessionStats;
  };
  
  equityData: Array<{
    date: string;
    balance: number;
    drawdown: number;
    tradeCount: number;
  }>;
  
  trades: Trade[];
}

interface SessionStats {
  trades: number;
  wins: number;
  winRate: number;
  totalPnl: number;
  avgPnl: number;
}
```

#### TradingModel
```typescript
interface TradingModel {
  id: "aggressive" | "passive";
  name: string;
  badge: string;
  description: string;
  
  characteristics: {
    entryStyle: string;
    filters: string;
    tpStrategy: string;
    risk: string;
    trading: string;
    bestFor: string;
  };
  
  sixMonthPerformance: {
    totalReturn: number;
    winRate: number;
    totalTrades: number;
    tradesPerDay: number;
    maxDrawdown: number;
    sharpeRatio: number;
  };
  
  monthlyReturns: Array<{
    month: string;
    return: number;
    trades: number;
    winRate: number;
    drawdown: number;
  }>;
  
  settings: {
    riskPerTrade: number;
    minLot: number;
    maxLot: number;
    slBuffer: number;
    tp1Percent: number;
    tp2Percent: number;
    breakEven: boolean;
    maxDrawdown: number;
    consecutiveLosses: number;
    maxDailyLosses: number;
  };
  
  activeAccounts: number;
  backtests: BacktestResult[];
}
```

#### Notification
```typescript
interface Notification {
  id: string;
  type: "trade_executed" | "model_update" | "account_synced" | "limit_hit" | "system";
  title: string;
  message: string;
  time: string;                    // Relative time: "2 min ago"
  unread: boolean;
  metadata?: {
    tradeId?: string;
    accountId?: string;
    [key: string]: any;
  };
}
```

---

## WebSocket Events

Real-time updates for live trading. Connect to: `wss://api.algotrade.ai/ws`

### Connection
```javascript
// Client connects with auth token
ws.send(JSON.stringify({
  type: "auth",
  token: "jwt_token_here"
}));
```

### Events (Server → Client)

#### Trade Opened
```json
{
  "event": "trade:opened",
  "data": {
    "accountId": "acc_123",
    "trade": {
      "id": "trade_456",
      "direction": "BUY",
      "entryPrice": 1.27834,
      "stopLoss": 1.27534,
      "positionSize": 0.05,
      "riskAmount": 12.50
    }
  }
}
```

#### Trade Closed
```json
{
  "event": "trade:closed",
  "data": {
    "accountId": "acc_123",
    "trade": {
      "id": "trade_456",
      "outcome": "TP1",
      "pnl": 15.55,
      "pips": 31.1,
      "rMultiple": 1.2
    }
  }
}
```

#### Account Update
```json
{
  "event": "account:updated",
  "data": {
    "accountId": "acc_123",
    "balance": 10262.87,
    "equity": 10262.87,
    "status": "connected"
  }
}
```

#### Backtest Progress
```json
{
  "event": "backtest:progress",
  "data": {
    "backtestId": "bt_789",
    "progress": 45,
    "currentDate": "2024-02-15"
  }
}
```

#### Backtest Complete
```json
{
  "event": "backtest:completed",
  "data": {
    "backtestId": "bt_789",
    "status": "completed",
    "summary": {
      "netPnl": 9460,
      "winRate": 68.4
    }
  }
}
```

#### Price Update
```json
{
  "event": "price:update",
  "data": {
    "symbol": "GBPUSD",
    "bid": 1.27832,
    "ask": 1.27834,
    "timestamp": "2025-01-08T10:30:00.123Z"
  }
}
```

---

## External Integrations

### MetaApi.cloud (Primary Trade Management)

The backend integrates with MetaTrader via MetaApi.cloud as the **primary** trade management layer. Model signals are generated internally and copied/executed on user-connected accounts.

#### Required Capabilities:
- Account connection (investor password where applicable)
- Balance/equity synchronization
- Trade history retrieval
- Order execution (for live trading)
- Real-time price streaming
- Model tagging via magic number/comment

#### Environment Variables
- `METAAPI_TOKEN` (fallback token when user config is empty)
- `METAAPI_PROVISIONING_URL` (default `https://mt-provisioning-api-v1.metaapi.cloud`)
- `METAAPI_CLIENT_URL` (default `https://mt-client-api-v1.metaapi.cloud`)
- `METAAPI_DEALS_PATH` (default `/users/current/accounts/{account_id}/deals`)
- `METAAPI_TRADE_PATH` (default `/users/current/accounts/{account_id}/trade`)
- `METAAPI_TIMEOUT_SECONDS` (default `15`)

#### Sync Endpoint
- `POST /api/accounts/{id}/sync`
  - Optional JSON body: `{ "sync_trades": true, "start_time": "...", "end_time": "..." }`
  - Syncs balances and optionally ingests trade history.

### Optional Future Integrations
- Manager API (prop firms)
- cTrader Open API (cTrader brokers)

#### Supported Brokers (Phase 1):
- ICH Markets
- FTMO
- Pepperstone
- OANDA
- XM
- Exness

### Stripe Integration

For subscription management:

| Tier | Price | Features |
|------|-------|----------|
| Free | $0 | 2 demo accounts, view-only |
| Demo | $29/mo | 5 demo accounts, backtesting |
| Live | $99/mo | 3 live accounts, unlimited demo, priority support |

---

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "ACCOUNT_NOT_FOUND",
    "message": "The requested account does not exist or you don't have access.",
    "details": {
      "accountId": "acc_invalid"
    }
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Invalid or expired token |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `ACCOUNT_NOT_FOUND` | 404 | MT5 account not found |
| `ACCOUNT_DISCONNECTED` | 400 | Account is not connected |
| `BROKER_ERROR` | 502 | Broker API error |
| `BACKTEST_FAILED` | 500 | Backtest processing failed |
| `SUBSCRIPTION_REQUIRED` | 402 | Feature requires paid subscription |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |

---

## Rate Limiting

| Endpoint Category | Limit |
|-------------------|-------|
| Authentication | 10 req/min |
| Trading Operations | 60 req/min |
| Market Data | 120 req/min |
| Backtests | 5 req/min |
| General API | 100 req/min |

Headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704708000
```

---

## Appendix: Supported Brokers & Servers

```json
{
  "brokers": [
    {
      "name": "ICH Markets",
      "servers": ["ICHMarkets-Demo", "ICHMarkets-Live", "ICHMarkets-Live02"]
    },
    {
      "name": "FTMO",
      "servers": ["FTMO-Demo", "FTMO-Live"]
    },
    {
      "name": "Pepperstone",
      "servers": ["Pepperstone-Demo", "Pepperstone-Edge-Live"]
    },
    {
      "name": "OANDA",
      "servers": ["OANDA-v20-Practice", "OANDA-v20-Live"]
    },
    {
      "name": "XM",
      "servers": ["XMGlobal-Demo", "XMGlobal-Real"]
    },
    {
      "name": "Exness",
      "servers": ["Exness-Demo", "Exness-Real"]
    }
  ]
}
```

---

## Contact

For questions about this specification, contact:
- **Frontend Team Lead:** [Your Name]
- **Project Repository:** [GitHub URL]

---

*This document should be treated as the single source of truth for API contracts between frontend and backend teams.*
