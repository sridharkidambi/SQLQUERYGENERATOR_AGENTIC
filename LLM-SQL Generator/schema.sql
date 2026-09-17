-- =====================================================================
-- ENTERPRISE BROKERAGE PLATFORM — CANONICAL SCHEMA (PostgreSQL 15+)
-- =====================================================================
-- Design notes for the modelling approach used throughout this file:
--
-- 1. SINGLE SOURCE OF TRUTH: every table and every business-meaningful
--    column carries a COMMENT ON ... string. These comments are not
--    decorative — the RAG ingestion pipeline (ingestion_pipeline.py)
--    introspects information_schema + pg_catalog and treats these
--    comments as the canonical business glossary. Do not let the
--    comments drift from reality; a migration that changes a column's
--    meaning must update its comment in the same migration.
--
-- 2. SCHEMA VERSIONING / FUTURE UPDATES: see schema_version at the end
--    of this file. Every change to this schema ships as a numbered
--    migration file (docs/migrations/NNN_description.sql). The
--    ingestion pipeline re-runs automatically on CI merge to main
--    (see RUN.md, "Keeping the model current").
--
-- 3. NAMING CONVENTIONS: snake_case; PK is always <table_singular>_id
--    (BIGINT GENERATED ALWAYS AS IDENTITY); FK columns are named
--    <referenced_table_singular>_id; enums are named <thing>_enum;
--    every mutable table has created_at/updated_at with a shared
--    trigger; every table that can be soft-deleted has is_deleted.
--
-- 4. IDENTIFIERS: internal PKs are surrogate BIGINT identities.
--    External/business identifiers (account numbers, order refs) are
--    separate UNIQUE columns — never expose surrogate keys externally.
-- =====================================================================

CREATE SCHEMA IF NOT EXISTS brokerage;
SET search_path TO brokerage, public;

-- ---------------------------------------------------------------------
-- ENUM TYPES
-- ---------------------------------------------------------------------
CREATE TYPE account_type_enum        AS ENUM ('INDIVIDUAL','JOINT','CORPORATE','NRI','TRUST');
CREATE TYPE account_status_enum      AS ENUM ('PENDING_KYC','ACTIVE','SUSPENDED','DORMANT','CLOSED');
CREATE TYPE kyc_status_enum          AS ENUM ('PENDING','VERIFIED','REJECTED','EXPIRED');
CREATE TYPE address_type_enum        AS ENUM ('REGISTERED','CORRESPONDENCE','OFFICE');
CREATE TYPE instrument_type_enum     AS ENUM ('EQUITY','BOND','ETF','MUTUAL_FUND','FUTURE','OPTION','CURRENCY','COMMODITY');
CREATE TYPE option_type_enum         AS ENUM ('CALL','PUT');
CREATE TYPE exchange_segment_enum    AS ENUM ('NSE_EQ','BSE_EQ','NSE_FO','BSE_FO','MCX_COMM','NSE_CD');
CREATE TYPE order_side_enum          AS ENUM ('BUY','SELL');
CREATE TYPE order_type_enum          AS ENUM ('MARKET','LIMIT','STOP','STOP_LIMIT','ICEBERG');
CREATE TYPE order_validity_enum      AS ENUM ('DAY','IOC','GTC','GTD');
CREATE TYPE order_status_enum        AS ENUM ('PENDING','OPEN','PARTIALLY_FILLED','FILLED','CANCELLED','REJECTED','EXPIRED');
CREATE TYPE transaction_type_enum    AS ENUM ('DEPOSIT','WITHDRAWAL','TRADE_DEBIT','TRADE_CREDIT','FEE','TAX','DIVIDEND','INTEREST','MARGIN_CALL','REVERSAL');
CREATE TYPE transfer_status_enum     AS ENUM ('INITIATED','PROCESSING','COMPLETED','FAILED','REVERSED');
CREATE TYPE corporate_action_type_enum AS ENUM ('DIVIDEND','BONUS','SPLIT','RIGHTS','MERGER','DELISTING','BUYBACK');
CREATE TYPE risk_limit_type_enum     AS ENUM ('DAILY_TURNOVER','POSITION_LIMIT','MARGIN_UTILIZATION','LEVERAGE','SEGMENT_EXPOSURE');
CREATE TYPE notification_channel_enum AS ENUM ('EMAIL','SMS','PUSH','IN_APP');
CREATE TYPE notification_status_enum  AS ENUM ('QUEUED','SENT','FAILED','READ');
CREATE TYPE audit_action_enum        AS ENUM ('INSERT','UPDATE','DELETE');

-- ---------------------------------------------------------------------
-- SHARED TRIGGER FUNCTIONS
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at := now();
  RETURN NEW;
END;
$$;
COMMENT ON FUNCTION set_updated_at() IS 'Stamps updated_at = now() on every UPDATE. Attached to every mutable table.';

CREATE OR REPLACE FUNCTION audit_row_change()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE
  v_action audit_action_enum;
  v_old jsonb;
  v_new jsonb;
BEGIN
  IF TG_OP = 'INSERT' THEN
    v_action := 'INSERT'; v_old := NULL; v_new := to_jsonb(NEW);
  ELSIF TG_OP = 'UPDATE' THEN
    v_action := 'UPDATE'; v_old := to_jsonb(OLD); v_new := to_jsonb(NEW);
  ELSE
    v_action := 'DELETE'; v_old := to_jsonb(OLD); v_new := NULL;
  END IF;

  INSERT INTO audit_log (table_name, row_pk, action, old_data, new_data, changed_by, changed_at)
  VALUES (
    TG_TABLE_NAME,
    COALESCE((v_new->>(TG_ARGV[0])), (v_old->>(TG_ARGV[0]))),
    v_action, v_old, v_new,
    COALESCE(current_setting('app.current_user', true), 'system'),
    now()
  );
  RETURN COALESCE(NEW, OLD);
END;
$$;
COMMENT ON FUNCTION audit_row_change() IS 'Generic row-level audit trigger. First TG_ARGV is the PK column name to record. Writes to audit_log.';

-- =====================================================================
-- 1. CUSTOMERS & KYC
-- =====================================================================
CREATE TABLE customers (
  customer_id       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  customer_code     VARCHAR(20)  NOT NULL UNIQUE,
  full_name         VARCHAR(150) NOT NULL,
  email             VARCHAR(255) NOT NULL,
  phone             VARCHAR(20)  NOT NULL,
  date_of_birth     DATE         NOT NULL,
  pan_number        VARCHAR(10)  NOT NULL UNIQUE,
  is_nri            BOOLEAN      NOT NULL DEFAULT FALSE,
  is_deleted        BOOLEAN      NOT NULL DEFAULT FALSE,
  created_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
  updated_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
  CONSTRAINT chk_customers_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
  CONSTRAINT chk_customers_pan   CHECK (pan_number ~ '^[A-Z]{5}[0-9]{4}[A-Z]{1}$'),
  CONSTRAINT chk_customers_dob   CHECK (date_of_birth <= (CURRENT_DATE - INTERVAL '18 years'))
);
COMMENT ON TABLE customers IS 'One row per natural person or legal entity who owns brokerage accounts. Root identity entity for KYC and compliance.';
COMMENT ON COLUMN customers.customer_code IS 'Human-facing customer reference (external ID), distinct from the internal surrogate key.';
COMMENT ON COLUMN customers.pan_number IS 'Indian Permanent Account Number, primary tax-identity used for regulatory reporting.';
COMMENT ON COLUMN customers.is_nri IS 'True if the customer is a Non-Resident Indian, affecting tax treatment and permitted account types.';

CREATE TRIGGER trg_customers_updated_at BEFORE UPDATE ON customers
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_customers_audit AFTER INSERT OR UPDATE OR DELETE ON customers
  FOR EACH ROW EXECUTE FUNCTION audit_row_change('customer_id');
CREATE INDEX idx_customers_email ON customers (email) WHERE is_deleted = FALSE;
CREATE INDEX idx_customers_phone ON customers (phone) WHERE is_deleted = FALSE;

CREATE TABLE customer_addresses (
  address_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  customer_id    BIGINT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
  address_type   address_type_enum NOT NULL,
  line1          VARCHAR(200) NOT NULL,
  line2          VARCHAR(200),
  city           VARCHAR(100) NOT NULL,
  state          VARCHAR(100) NOT NULL,
  postal_code    VARCHAR(20)  NOT NULL,
  country_code   CHAR(2)      NOT NULL DEFAULT 'IN',
  is_primary     BOOLEAN      NOT NULL DEFAULT FALSE,
  created_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
  updated_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
  UNIQUE (customer_id, address_type)
);
COMMENT ON TABLE customer_addresses IS 'Registered / correspondence / office addresses for a customer, used for KYC and statement delivery.';
CREATE TRIGGER trg_addr_updated_at BEFORE UPDATE ON customer_addresses
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE INDEX idx_addresses_customer ON customer_addresses (customer_id);

CREATE TABLE customer_kyc (
  kyc_id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  customer_id       BIGINT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
  status            kyc_status_enum NOT NULL DEFAULT 'PENDING',
  document_type     VARCHAR(50)  NOT NULL,
  document_number   VARCHAR(100) NOT NULL,
  verified_by       VARCHAR(100),
  verified_at       TIMESTAMPTZ,
  expires_at        DATE,
  rejection_reason  VARCHAR(500),
  created_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
  updated_at        TIMESTAMPTZ  NOT NULL DEFAULT now()
);
COMMENT ON TABLE customer_kyc IS 'KYC verification records per customer. A customer may have multiple document verifications over time; the latest VERIFIED row governs account activation.';
CREATE TRIGGER trg_kyc_updated_at BEFORE UPDATE ON customer_kyc
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE INDEX idx_kyc_customer_status ON customer_kyc (customer_id, status);

-- =====================================================================
-- 2. ACCOUNTS
-- =====================================================================
CREATE TABLE accounts (
  account_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  account_number    VARCHAR(20)  NOT NULL UNIQUE,
  account_type      account_type_enum NOT NULL,
  status            account_status_enum NOT NULL DEFAULT 'PENDING_KYC',
  base_currency     CHAR(3)      NOT NULL DEFAULT 'INR',
  cash_balance      NUMERIC(20,4) NOT NULL DEFAULT 0 CHECK (cash_balance >= -999999999999.9999),
  available_balance NUMERIC(20,4) NOT NULL DEFAULT 0,
  opened_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
  closed_at         TIMESTAMPTZ,
  created_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
  updated_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
  CONSTRAINT chk_accounts_close CHECK (closed_at IS NULL OR closed_at >= opened_at)
);
COMMENT ON TABLE accounts IS 'A tradable brokerage/demat account. Distinct from customers because one customer can hold multiple accounts (individual, joint, corporate) and one account can have multiple holders.';
COMMENT ON COLUMN accounts.cash_balance IS 'Total settled cash balance in base_currency.';
COMMENT ON COLUMN accounts.available_balance IS 'Cash balance minus funds blocked for open orders/margin; the figure usable for new orders.';
CREATE TRIGGER trg_accounts_updated_at BEFORE UPDATE ON accounts
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_accounts_audit AFTER INSERT OR UPDATE OR DELETE ON accounts
  FOR EACH ROW EXECUTE FUNCTION audit_row_change('account_id');
CREATE INDEX idx_accounts_status ON accounts (status);

CREATE TABLE account_holders (
  account_holder_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  account_id        BIGINT NOT NULL REFERENCES accounts(account_id) ON DELETE CASCADE,
  customer_id       BIGINT NOT NULL REFERENCES customers(customer_id) ON DELETE RESTRICT,
  is_primary_holder  BOOLEAN NOT NULL DEFAULT TRUE,
  ownership_pct     NUMERIC(5,2) NOT NULL DEFAULT 100.00 CHECK (ownership_pct > 0 AND ownership_pct <= 100),
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (account_id, customer_id)
);
COMMENT ON TABLE account_holders IS 'Many-to-many bridge between accounts and customers, supporting joint/corporate accounts with fractional ownership.';
CREATE INDEX idx_holders_customer ON account_holders (customer_id);
CREATE UNIQUE INDEX uq_holders_primary ON account_holders (account_id) WHERE is_primary_holder = TRUE;

-- =====================================================================
-- 3. MARKET REFERENCE DATA: EXCHANGES & INSTRUMENTS
-- =====================================================================
CREATE TABLE exchanges (
  exchange_id    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  exchange_code  VARCHAR(10) NOT NULL UNIQUE,
  exchange_name  VARCHAR(100) NOT NULL,
  country_code   CHAR(2) NOT NULL,
  timezone       VARCHAR(50) NOT NULL DEFAULT 'Asia/Kolkata',
  is_active      BOOLEAN NOT NULL DEFAULT TRUE
);
COMMENT ON TABLE exchanges IS 'Trading venues (NSE, BSE, MCX, ...).';

CREATE TABLE instruments (
  instrument_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  isin              CHAR(12) UNIQUE,
  symbol            VARCHAR(30)  NOT NULL,
  instrument_name   VARCHAR(200) NOT NULL,
  instrument_type   instrument_type_enum NOT NULL,
  underlying_instrument_id BIGINT REFERENCES instruments(instrument_id),
  option_type       option_type_enum,
  strike_price      NUMERIC(18,4),
  expiry_date       DATE,
  lot_size          INTEGER NOT NULL DEFAULT 1 CHECK (lot_size > 0),
  tick_size         NUMERIC(10,4) NOT NULL DEFAULT 0.05 CHECK (tick_size > 0),
  is_active         BOOLEAN NOT NULL DEFAULT TRUE,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT chk_derivative_fields CHECK (
    (instrument_type NOT IN ('FUTURE','OPTION')) OR (expiry_date IS NOT NULL)
  ),
  CONSTRAINT chk_option_fields CHECK (
    (instrument_type <> 'OPTION') OR (option_type IS NOT NULL AND strike_price IS NOT NULL)
  )
);
COMMENT ON TABLE instruments IS 'Security master. Equities, bonds, ETFs, mutual funds, and derivatives (which self-reference their underlying via underlying_instrument_id).';
COMMENT ON COLUMN instruments.isin IS 'International Securities Identification Number; globally unique where applicable.';
COMMENT ON COLUMN instruments.lot_size IS 'Minimum tradable quantity multiple; relevant mainly for derivatives.';
CREATE TRIGGER trg_instruments_updated_at BEFORE UPDATE ON instruments
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE INDEX idx_instruments_symbol ON instruments (symbol);
CREATE INDEX idx_instruments_type ON instruments (instrument_type);
CREATE INDEX idx_instruments_underlying ON instruments (underlying_instrument_id) WHERE underlying_instrument_id IS NOT NULL;

CREATE TABLE instrument_exchange_listing (
  listing_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  instrument_id     BIGINT NOT NULL REFERENCES instruments(instrument_id) ON DELETE CASCADE,
  exchange_id       BIGINT NOT NULL REFERENCES exchanges(exchange_id) ON DELETE RESTRICT,
  segment           exchange_segment_enum NOT NULL,
  exchange_symbol   VARCHAR(30) NOT NULL,
  is_tradable       BOOLEAN NOT NULL DEFAULT TRUE,
  UNIQUE (instrument_id, exchange_id, segment)
);
COMMENT ON TABLE instrument_exchange_listing IS 'Where a given instrument is listed/tradable; one instrument may list on multiple exchanges/segments (e.g. NSE_EQ and BSE_EQ).';
CREATE INDEX idx_listing_exchange_symbol ON instrument_exchange_listing (exchange_id, exchange_symbol);

-- =====================================================================
-- 4. WATCHLISTS
-- =====================================================================
CREATE TABLE watchlists (
  watchlist_id  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  account_id    BIGINT NOT NULL REFERENCES accounts(account_id) ON DELETE CASCADE,
  name          VARCHAR(100) NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (account_id, name)
);
COMMENT ON TABLE watchlists IS 'Named instrument lists a customer tracks; purely a UX/personalization construct, no trading effect.';

CREATE TABLE watchlist_items (
  watchlist_item_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  watchlist_id      BIGINT NOT NULL REFERENCES watchlists(watchlist_id) ON DELETE CASCADE,
  instrument_id     BIGINT NOT NULL REFERENCES instruments(instrument_id) ON DELETE CASCADE,
  added_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (watchlist_id, instrument_id)
);
COMMENT ON TABLE watchlist_items IS 'Instruments within a watchlist.';

-- =====================================================================
-- 5. ORDERS, EXECUTIONS, POSITIONS, HOLDINGS
-- =====================================================================
CREATE TABLE orders (
  order_id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  order_ref         VARCHAR(30) NOT NULL UNIQUE,
  account_id        BIGINT NOT NULL REFERENCES accounts(account_id) ON DELETE RESTRICT,
  instrument_id     BIGINT NOT NULL REFERENCES instruments(instrument_id) ON DELETE RESTRICT,
  exchange_id       BIGINT NOT NULL REFERENCES exchanges(exchange_id) ON DELETE RESTRICT,
  side              order_side_enum NOT NULL,
  order_type        order_type_enum NOT NULL,
  validity          order_validity_enum NOT NULL DEFAULT 'DAY',
  quantity          NUMERIC(18,4) NOT NULL CHECK (quantity > 0),
  filled_quantity   NUMERIC(18,4) NOT NULL DEFAULT 0 CHECK (filled_quantity >= 0),
  limit_price       NUMERIC(18,4),
  stop_price        NUMERIC(18,4),
  status            order_status_enum NOT NULL DEFAULT 'PENDING',
  placed_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at        TIMESTAMPTZ,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT chk_order_fill CHECK (filled_quantity <= quantity),
  CONSTRAINT chk_limit_price CHECK (order_type NOT IN ('LIMIT','STOP_LIMIT') OR limit_price IS NOT NULL),
  CONSTRAINT chk_stop_price  CHECK (order_type NOT IN ('STOP','STOP_LIMIT') OR stop_price IS NOT NULL)
);
COMMENT ON TABLE orders IS 'Every order a customer places, in any status. Immutable history of the request; actual fills live in trades.';
COMMENT ON COLUMN orders.filled_quantity IS 'Cumulative quantity executed across all trades for this order.';
CREATE TRIGGER trg_orders_updated_at BEFORE UPDATE ON orders
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_orders_audit AFTER INSERT OR UPDATE OR DELETE ON orders
  FOR EACH ROW EXECUTE FUNCTION audit_row_change('order_id');
CREATE INDEX idx_orders_account_status ON orders (account_id, status);
CREATE INDEX idx_orders_instrument ON orders (instrument_id);
CREATE INDEX idx_orders_placed_at ON orders (placed_at);

CREATE TABLE order_status_history (
  history_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  order_id       BIGINT NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
  old_status     order_status_enum,
  new_status     order_status_enum NOT NULL,
  changed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  reason         VARCHAR(300)
);
COMMENT ON TABLE order_status_history IS 'Append-only audit trail of every order status transition, populated automatically by trg_orders_status_history.';
CREATE INDEX idx_osh_order ON order_status_history (order_id, changed_at);

CREATE OR REPLACE FUNCTION log_order_status_change()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.status IS DISTINCT FROM OLD.status THEN
    INSERT INTO order_status_history (order_id, old_status, new_status, changed_at)
    VALUES (NEW.order_id, OLD.status, NEW.status, now());
  END IF;
  RETURN NEW;
END;
$$;
CREATE TRIGGER trg_orders_status_history AFTER UPDATE OF status ON orders
  FOR EACH ROW EXECUTE FUNCTION log_order_status_change();

-- ---------------------------------------------------------------------
-- COUNTERPARTIES — must precede trades: trades.counterparty_id references
-- it, and plain CREATE TABLE resolves FK targets immediately.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS counterparties (
  counterparty_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  counterparty_code VARCHAR(20) NOT NULL UNIQUE,
  legal_name        VARCHAR(200) NOT NULL,
  counterparty_kind VARCHAR(30) NOT NULL DEFAULT 'CLEARING_MEMBER',
  is_active         BOOLEAN NOT NULL DEFAULT TRUE
);
COMMENT ON TABLE counterparties IS 'Clearing members, exchanges-as-counterparty, or settlement counterparties involved in trade execution/clearing.';

CREATE TABLE trades (
  trade_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  trade_ref       VARCHAR(30) NOT NULL UNIQUE,
  order_id        BIGINT NOT NULL REFERENCES orders(order_id) ON DELETE RESTRICT,
  account_id      BIGINT NOT NULL REFERENCES accounts(account_id) ON DELETE RESTRICT,
  instrument_id   BIGINT NOT NULL REFERENCES instruments(instrument_id) ON DELETE RESTRICT,
  counterparty_id BIGINT REFERENCES counterparties(counterparty_id),
  side            order_side_enum NOT NULL,
  quantity        NUMERIC(18,4) NOT NULL CHECK (quantity > 0),
  price           NUMERIC(18,4) NOT NULL CHECK (price > 0),
  trade_value     NUMERIC(20,4) GENERATED ALWAYS AS (quantity * price) STORED,
  executed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  settlement_date DATE NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE trades IS 'Actual executions ("fills") against an order. One order can have many trades (partial fills).';
COMMENT ON COLUMN trades.trade_value IS 'Computed column: quantity * price, in instrument currency, pre-fees.';
CREATE INDEX idx_trades_account ON trades (account_id, executed_at);
CREATE INDEX idx_trades_instrument ON trades (instrument_id, executed_at);
CREATE INDEX idx_trades_order ON trades (order_id);

CREATE TABLE positions (
  position_id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  account_id       BIGINT NOT NULL REFERENCES accounts(account_id) ON DELETE CASCADE,
  instrument_id    BIGINT NOT NULL REFERENCES instruments(instrument_id) ON DELETE RESTRICT,
  net_quantity     NUMERIC(18,4) NOT NULL DEFAULT 0,
  average_price    NUMERIC(18,4) NOT NULL DEFAULT 0,
  realized_pnl     NUMERIC(20,4) NOT NULL DEFAULT 0,
  updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (account_id, instrument_id)
);
COMMENT ON TABLE positions IS 'Current net holding per account/instrument, maintained by trg_trades_update_position. Positive net_quantity = long, negative = short (derivatives).';
CREATE TRIGGER trg_positions_updated_at BEFORE UPDATE ON positions
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE INDEX idx_positions_account ON positions (account_id);

CREATE OR REPLACE FUNCTION update_position_from_trade()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
DECLARE
  v_sign NUMERIC := CASE WHEN NEW.side = 'BUY' THEN 1 ELSE -1 END;
BEGIN
  INSERT INTO positions (account_id, instrument_id, net_quantity, average_price, updated_at)
  VALUES (NEW.account_id, NEW.instrument_id, v_sign * NEW.quantity, NEW.price, now())
  ON CONFLICT (account_id, instrument_id) DO UPDATE
  SET average_price = CASE
        WHEN sign(positions.net_quantity) = sign(v_sign) OR positions.net_quantity = 0
          THEN ((positions.average_price * ABS(positions.net_quantity)) + (NEW.price * NEW.quantity))
               / NULLIF(ABS(positions.net_quantity) + NEW.quantity, 0)
        ELSE positions.average_price
      END,
      net_quantity = positions.net_quantity + (v_sign * NEW.quantity),
      updated_at = now();
  RETURN NEW;
END;
$$;
COMMENT ON FUNCTION update_position_from_trade() IS 'Maintains a weighted-average-cost position on every trade insert. FIFO lot detail is tracked separately in holdings_lots for tax purposes.';
CREATE TRIGGER trg_trades_update_position AFTER INSERT ON trades
  FOR EACH ROW EXECUTE FUNCTION update_position_from_trade();

CREATE TABLE holdings_lots (
  lot_id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  account_id       BIGINT NOT NULL REFERENCES accounts(account_id) ON DELETE CASCADE,
  instrument_id    BIGINT NOT NULL REFERENCES instruments(instrument_id) ON DELETE RESTRICT,
  acquired_trade_id BIGINT NOT NULL REFERENCES trades(trade_id),
  quantity_remaining NUMERIC(18,4) NOT NULL CHECK (quantity_remaining >= 0),
  cost_price       NUMERIC(18,4) NOT NULL,
  acquired_at      TIMESTAMPTZ NOT NULL,
  is_long_term     BOOLEAN NOT NULL DEFAULT FALSE
);
COMMENT ON TABLE holdings_lots IS 'FIFO tax lots for capital-gains computation. Each BUY trade creates a lot; SELL trades consume lots oldest-first.';
COMMENT ON COLUMN holdings_lots.is_long_term IS 'Recomputed periodically based on holding-period rules for capital-gains tax classification.';
CREATE INDEX idx_lots_account_instrument ON holdings_lots (account_id, instrument_id, acquired_at);

-- =====================================================================
-- 6. LEDGER, FUNDS TRANSFERS, MARGIN, FEES, RISK LIMITS
-- =====================================================================
CREATE TABLE ledger_transactions (
  ledger_txn_id    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  account_id       BIGINT NOT NULL REFERENCES accounts(account_id) ON DELETE RESTRICT,
  txn_type         transaction_type_enum NOT NULL,
  amount           NUMERIC(20,4) NOT NULL,
  currency         CHAR(3) NOT NULL DEFAULT 'INR',
  related_trade_id BIGINT REFERENCES trades(trade_id),
  description      VARCHAR(300),
  txn_date         DATE NOT NULL DEFAULT CURRENT_DATE,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT chk_ledger_amount_sign CHECK (
    (txn_type IN ('DEPOSIT','TRADE_CREDIT','DIVIDEND','INTEREST','REVERSAL') AND amount > 0) OR
    (txn_type IN ('WITHDRAWAL','TRADE_DEBIT','FEE','TAX','MARGIN_CALL') AND amount < 0)
  )
);
COMMENT ON TABLE ledger_transactions IS 'Immutable cash-ledger entries per account. accounts.cash_balance is the running sum of these entries; never update cash_balance directly outside this trigger path.';
CREATE INDEX idx_ledger_account_date ON ledger_transactions (account_id, txn_date);
CREATE TRIGGER trg_ledger_audit AFTER INSERT ON ledger_transactions
  FOR EACH ROW EXECUTE FUNCTION audit_row_change('ledger_txn_id');

CREATE OR REPLACE FUNCTION apply_ledger_to_balance()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  UPDATE accounts
  SET cash_balance = cash_balance + NEW.amount,
      available_balance = available_balance + NEW.amount,
      updated_at = now()
  WHERE account_id = NEW.account_id;
  RETURN NEW;
END;
$$;
CREATE TRIGGER trg_ledger_apply_balance AFTER INSERT ON ledger_transactions
  FOR EACH ROW EXECUTE FUNCTION apply_ledger_to_balance();

CREATE TABLE fund_transfers (
  transfer_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  account_id      BIGINT NOT NULL REFERENCES accounts(account_id) ON DELETE RESTRICT,
  direction       VARCHAR(10) NOT NULL CHECK (direction IN ('IN','OUT')),
  amount          NUMERIC(20,4) NOT NULL CHECK (amount > 0),
  bank_reference  VARCHAR(100),
  status          transfer_status_enum NOT NULL DEFAULT 'INITIATED',
  initiated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at    TIMESTAMPTZ,
  linked_ledger_txn_id BIGINT REFERENCES ledger_transactions(ledger_txn_id)
);
COMMENT ON TABLE fund_transfers IS 'External bank transfer requests (deposits/withdrawals) prior to/around ledger posting. Tracks the bank-side lifecycle; linked_ledger_txn_id ties it to the posted cash entry once COMPLETED.';
CREATE INDEX idx_transfers_account_status ON fund_transfers (account_id, status);

CREATE TABLE margin_accounts (
  margin_account_id  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  account_id         BIGINT NOT NULL UNIQUE REFERENCES accounts(account_id) ON DELETE CASCADE,
  margin_used        NUMERIC(20,4) NOT NULL DEFAULT 0,
  margin_available   NUMERIC(20,4) NOT NULL DEFAULT 0,
  maintenance_margin NUMERIC(20,4) NOT NULL DEFAULT 0,
  leverage_multiple  NUMERIC(6,2)  NOT NULL DEFAULT 1.0 CHECK (leverage_multiple >= 1.0),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE margin_accounts IS 'One-to-one margin/leverage state per account, relevant for derivatives and margin-trading-funded positions.';
CREATE TRIGGER trg_margin_updated_at BEFORE UPDATE ON margin_accounts
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TABLE risk_limits (
  risk_limit_id    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  account_id       BIGINT REFERENCES accounts(account_id) ON DELETE CASCADE,
  limit_type       risk_limit_type_enum NOT NULL,
  limit_value      NUMERIC(20,4) NOT NULL,
  current_value    NUMERIC(20,4) NOT NULL DEFAULT 0,
  is_breached      BOOLEAN GENERATED ALWAYS AS (current_value > limit_value) STORED,
  effective_from   DATE NOT NULL DEFAULT CURRENT_DATE,
  effective_to     DATE,
  UNIQUE (account_id, limit_type, effective_from)
);
COMMENT ON TABLE risk_limits IS 'Per-account (or platform-wide when account_id IS NULL) risk thresholds and their current utilization.';
COMMENT ON COLUMN risk_limits.limit_type IS 'Category of risk limit: MARGIN_UTILIZATION, CONCENTRATION, LEVERAGE, NET_EXPOSURE, etc.';
COMMENT ON COLUMN risk_limits.limit_value IS 'The configured maximum allowed value for this risk limit.';
COMMENT ON COLUMN risk_limits.current_value IS 'The live, current measured value of this risk metric.';
COMMENT ON COLUMN risk_limits.is_breached IS 'TRUE when current_value exceeds limit_value, meaning the risk limit is currently breached.';
COMMENT ON COLUMN risk_limits.account_id IS 'The account this risk limit applies to; NULL means a platform-wide limit.';
CREATE INDEX idx_risk_limits_breached ON risk_limits (is_breached) WHERE is_breached = TRUE;

CREATE TABLE fee_schedules (
  fee_schedule_id  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  segment          exchange_segment_enum NOT NULL,
  fee_name         VARCHAR(100) NOT NULL,
  fee_pct          NUMERIC(8,6),
  fee_flat         NUMERIC(12,4),
  min_fee          NUMERIC(12,4) DEFAULT 0,
  max_fee          NUMERIC(12,4),
  effective_from   DATE NOT NULL,
  effective_to     DATE,
  CONSTRAINT chk_fee_has_value CHECK (fee_pct IS NOT NULL OR fee_flat IS NOT NULL)
);
COMMENT ON TABLE fee_schedules IS 'Brokerage/regulatory fee rules by exchange segment and effective date range (e.g. brokerage %, STT, exchange transaction charges, GST, stamp duty).';
CREATE INDEX idx_fees_segment_effective ON fee_schedules (segment, effective_from);

-- =====================================================================
-- 7. CORPORATE ACTIONS
-- =====================================================================
CREATE TABLE corporate_actions (
  corp_action_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  instrument_id    BIGINT NOT NULL REFERENCES instruments(instrument_id) ON DELETE CASCADE,
  action_type      corporate_action_type_enum NOT NULL,
  ratio_from       NUMERIC(10,4),
  ratio_to         NUMERIC(10,4),
  amount_per_share NUMERIC(18,4),
  record_date      DATE NOT NULL,
  ex_date          DATE NOT NULL,
  payment_date     DATE,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE corporate_actions IS 'Declared corporate actions (dividends, splits, bonuses, mergers, buybacks) affecting an instrument.';
CREATE INDEX idx_corp_actions_instrument ON corporate_actions (instrument_id, ex_date);

CREATE TABLE corporate_action_entitlements (
  entitlement_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  corp_action_id   BIGINT NOT NULL REFERENCES corporate_actions(corp_action_id) ON DELETE CASCADE,
  account_id       BIGINT NOT NULL REFERENCES accounts(account_id) ON DELETE CASCADE,
  entitled_quantity NUMERIC(18,4),
  entitled_amount  NUMERIC(20,4),
  processed        BOOLEAN NOT NULL DEFAULT FALSE,
  processed_at     TIMESTAMPTZ
);
COMMENT ON TABLE corporate_action_entitlements IS 'Per-account entitlement computed from holdings as of record_date for a given corporate action; drives dividend credit / bonus-unit allotment postings.';
CREATE INDEX idx_cae_account ON corporate_action_entitlements (account_id, processed);

-- =====================================================================
-- 8. NOTIFICATIONS & AUDIT
-- =====================================================================
CREATE TABLE notifications (
  notification_id  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  account_id       BIGINT NOT NULL REFERENCES accounts(account_id) ON DELETE CASCADE,
  channel          notification_channel_enum NOT NULL,
  status            notification_status_enum NOT NULL DEFAULT 'QUEUED',
  subject          VARCHAR(200),
  body             TEXT NOT NULL,
  related_order_id BIGINT REFERENCES orders(order_id),
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  sent_at          TIMESTAMPTZ
);
COMMENT ON TABLE notifications IS 'Outbound customer notifications (order fills, margin calls, corporate actions).';
CREATE INDEX idx_notifications_account_status ON notifications (account_id, status);

CREATE TABLE audit_log (
  audit_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  table_name   VARCHAR(100) NOT NULL,
  row_pk       VARCHAR(50),
  action       audit_action_enum NOT NULL,
  old_data     JSONB,
  new_data     JSONB,
  changed_by   VARCHAR(100) NOT NULL DEFAULT 'system',
  changed_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE audit_log IS 'Generic append-only change log populated by audit_row_change() triggers on sensitive tables (customers, accounts, orders, ledger_transactions).';
CREATE INDEX idx_audit_table_pk ON audit_log (table_name, row_pk);
CREATE INDEX idx_audit_changed_at ON audit_log (changed_at);
CREATE INDEX idx_audit_new_data_gin ON audit_log USING GIN (new_data);

-- =====================================================================
-- 10. SCHEMA VERSIONING — provisioning for future updates
-- =====================================================================
CREATE TABLE schema_version (
  version_id    INTEGER PRIMARY KEY,
  description   VARCHAR(300) NOT NULL,
  applied_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  checksum      VARCHAR(64) NOT NULL
);
COMMENT ON TABLE schema_version IS 'Migration ledger. Every DDL change ships as one row here with a checksum of the migration file. The ingestion pipeline reads MAX(version_id) to decide whether re-ingestion into the vector/graph store is required.';

INSERT INTO schema_version (version_id, description, checksum)
VALUES (1, 'Initial brokerage platform schema', 'GENERATED_AT_MIGRATION_TIME');

-- =====================================================================
-- END OF SCHEMA
-- =====================================================================
