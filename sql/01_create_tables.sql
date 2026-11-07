-- ============================================================
-- FINTECH FRAUD RISK ANALYTICS
-- 01_create_tables.sql — schema (PostgreSQL)
-- ============================================================
-- Portfolio simulation only — not a real financial institution's schema
-- or workflow. See README.md "Limitations".
-- ============================================================

DROP TABLE IF EXISTS exception_transactions;
DROP TABLE IF EXISTS fraud_predictions;
DROP TABLE IF EXISTS model_results;
DROP TABLE IF EXISTS transactions;

-- The (cleaned) transaction feed — see data/README.md and
-- notebooks/01-02 for how this is derived from the raw dataset.
CREATE TABLE transactions (
    transaction_id   VARCHAR(12) PRIMARY KEY,
    txn_time         NUMERIC(12, 2) NOT NULL,   -- seconds since first transaction in the dataset (not a timestamp)
    hour_of_day      SMALLINT NOT NULL CHECK (hour_of_day BETWEEN 0 AND 23),
    elapsed_day      SMALLINT NOT NULL,
    amount           NUMERIC(12, 2) NOT NULL CHECK (amount >= 0),
    log_amount       NUMERIC(12, 6) NOT NULL,
    is_fraud         SMALLINT NOT NULL CHECK (is_fraud IN (0, 1))
);

-- One row per model per evaluation run (Phase 12 comparison table).
CREATE TABLE model_results (
    model_name       VARCHAR(50) NOT NULL,
    evaluated_on     VARCHAR(10) NOT NULL,      -- 'validation' or 'test'
    precision_score  NUMERIC(6, 4),
    recall_score     NUMERIC(6, 4),
    f1_score         NUMERIC(6, 4),
    roc_auc          NUMERIC(6, 4),
    pr_auc           NUMERIC(6, 4),
    PRIMARY KEY (model_name, evaluated_on)
);

-- The chosen model's prediction for every transaction in the test set
-- (Phase 15 risk table).
CREATE TABLE fraud_predictions (
    transaction_id     VARCHAR(12) PRIMARY KEY REFERENCES transactions(transaction_id),
    fraud_probability  NUMERIC(6, 4) NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
    risk_level         VARCHAR(10) NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    model_prediction   SMALLINT NOT NULL CHECK (model_prediction IN (0, 1))
);

-- Operational review queue (Phase 18). Simulated case-management state —
-- in this portfolio project it is generated alongside the predictions,
-- not produced by a real review team.
CREATE TABLE exception_transactions (
    transaction_id      VARCHAR(12) PRIMARY KEY REFERENCES transactions(transaction_id),
    transaction_amount  NUMERIC(12, 2) NOT NULL,
    fraud_probability   NUMERIC(6, 4) NOT NULL,
    risk_level          VARCHAR(10) NOT NULL,
    prediction          SMALLINT NOT NULL,
    review_status       VARCHAR(10) NOT NULL DEFAULT 'PENDING'
        CHECK (review_status IN ('PENDING', 'REVIEWED', 'ESCALATED', 'CLEARED'))
);

CREATE INDEX idx_txn_fraud ON transactions(is_fraud);
CREATE INDEX idx_txn_hour ON transactions(hour_of_day);
CREATE INDEX idx_pred_risk ON fraud_predictions(risk_level);
CREATE INDEX idx_exc_status ON exception_transactions(review_status);

-- Example load (psql \copy — run from the sql/ directory, after
-- notebooks/06_model_evaluation.ipynb has produced data/processed/*.csv):
-- \copy transactions FROM '../data/processed/transactions.csv' WITH (FORMAT csv, HEADER true)
-- \copy model_results FROM '../data/processed/model_results.csv' WITH (FORMAT csv, HEADER true)
-- \copy fraud_predictions FROM '../data/processed/fraud_predictions.csv' WITH (FORMAT csv, HEADER true)
-- \copy exception_transactions FROM '../data/processed/exception_transactions.csv' WITH (FORMAT csv, HEADER true)
