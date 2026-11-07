"""
Model evaluation (Phase 12), confusion matrix in business language
(Phase 13), threshold optimisation (Phase 14), and risk scoring (Phase 15).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# This project's own configurable risk bands (Phase 15) — NOT an industry
# standard. Chosen so CRITICAL roughly targets the top slice of genuine
# fraud probability mass observed on the validation set; see
# notebooks/06_model_evaluation.ipynb for why these specific cut points.
RISK_BANDS = [
    (0.0, 0.20, "LOW"),
    (0.20, 0.50, "MEDIUM"),
    (0.50, 0.80, "HIGH"),
    (0.80, 1.0001, "CRITICAL"),
]


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Precision/recall/F1 at `threshold`, plus ROC-AUC and PR-AUC (both threshold-independent)."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def comparison_table(results: dict) -> pd.DataFrame:
    """results: {model_name: metrics_dict from compute_metrics}."""
    rows = []
    for name, m in results.items():
        rows.append({
            "Model": name,
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1": round(m["f1"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
            "PR-AUC": round(m["pr_auc"], 4),
        })
    return pd.DataFrame(rows)


def confusion_matrix_business(y_true, y_pred) -> dict:
    """
    Confusion matrix plus a plain-language translation of what each cell
    costs operationally (Phase 13) — the numbers alone don't tell an
    Operations stakeholder anything actionable.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "false_negative_meaning": (
            f"{fn} fraudulent transactions were NOT flagged. This is the costliest "
            "error type — fraud proceeds undetected, and any loss is realized in full."
        ),
        "false_positive_meaning": (
            f"{fp} legitimate transactions WERE flagged. This is a customer-friction "
            "and manual-review-capacity cost, not a direct financial loss."
        ),
    }


def threshold_sweep(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """
    Precision/recall/FP/FN across a range of thresholds, so a threshold is
    chosen by looking at the actual tradeoff curve rather than defaulting
    to 0.5 (Phase 14).
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        })
    return pd.DataFrame(rows)


def risk_tier(prob: float) -> str:
    for low, high, label in RISK_BANDS:
        if low <= prob < high:
            return label
    return "CRITICAL"


def risk_tier_vectorized(probs) -> pd.Series:
    return pd.Series(probs).apply(risk_tier)


def build_risk_table(transaction_ids, y_prob, y_pred) -> pd.DataFrame:
    """The transaction_id / fraud_probability / risk_level / model_prediction table (Phase 15)."""
    return pd.DataFrame({
        "transaction_id": transaction_ids,
        "fraud_probability": np.round(y_prob, 4),
        "risk_level": risk_tier_vectorized(y_prob),
        "model_prediction": y_pred,
    })
