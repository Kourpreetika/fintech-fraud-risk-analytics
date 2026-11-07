-- ============================================================
-- FINTECH FRAUD RISK ANALYTICS
-- risk_analysis.sql — analysis queries over the loaded prediction data
-- ============================================================

-- ============================================================
-- 1. FRAUD RATE BY DAY
-- ============================================================
SELECT
    elapsed_day,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY elapsed_day
ORDER BY elapsed_day;


-- ============================================================
-- 2. FRAUD RATE BY HOUR OF DAY
-- ============================================================
SELECT
    hour_of_day,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_transactions,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;


-- ============================================================
-- 3. HIGH-RISK TRANSACTIONS (model output, not the raw label)
-- ============================================================
SELECT
    p.transaction_id,
    t.amount,
    p.fraud_probability,
    p.risk_level,
    p.model_prediction,
    t.is_fraud AS actual_label
FROM fraud_predictions p
JOIN transactions t ON t.transaction_id = p.transaction_id
WHERE p.risk_level IN ('HIGH', 'CRITICAL')
ORDER BY p.fraud_probability DESC;


-- ============================================================
-- 4. AVERAGE TRANSACTION AMOUNT BY RISK LEVEL
-- ============================================================
SELECT
    p.risk_level,
    COUNT(*) AS transaction_count,
    ROUND(AVG(t.amount), 2) AS avg_amount,
    ROUND(SUM(t.amount), 2) AS total_amount
FROM fraud_predictions p
JOIN transactions t ON t.transaction_id = p.transaction_id
GROUP BY p.risk_level
ORDER BY
    CASE p.risk_level WHEN 'LOW' THEN 1 WHEN 'MEDIUM' THEN 2 WHEN 'HIGH' THEN 3 WHEN 'CRITICAL' THEN 4 END;


-- ============================================================
-- 5. MODEL PERFORMANCE COMPARISON (Phase 12 table, from SQL)
-- ============================================================
SELECT model_name, evaluated_on, precision_score, recall_score, f1_score, roc_auc, pr_auc
FROM model_results
ORDER BY evaluated_on, pr_auc DESC;


-- ============================================================
-- 6. CONFUSION MATRIX BREAKDOWN (chosen model, test set)
-- ============================================================
SELECT
    t.is_fraud AS actual,
    p.model_prediction AS predicted,
    COUNT(*) AS transaction_count
FROM fraud_predictions p
JOIN transactions t ON t.transaction_id = p.transaction_id
GROUP BY t.is_fraud, p.model_prediction
ORDER BY actual, predicted;


-- ============================================================
-- 7. OPERATIONAL REVIEW QUEUE STATUS
-- ============================================================
SELECT
    review_status,
    risk_level,
    COUNT(*) AS transaction_count,
    ROUND(SUM(transaction_amount), 2) AS total_amount_at_stake
FROM exception_transactions
GROUP BY review_status, risk_level
ORDER BY
    CASE risk_level WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 WHEN 'LOW' THEN 4 END,
    review_status;
