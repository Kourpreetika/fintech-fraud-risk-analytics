"""
Train/test strategy (Phase 8), baseline + advanced models (Phase 9-10),
and class-imbalance handling (Phase 11).
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42


def three_way_split(X, y, test_size=0.2, val_size=0.2, random_state=RANDOM_STATE):
    """
    Stratified train/validation/test split. Fraud is rare enough (0.17%)
    that an un-stratified split risks a test fold with very few — or
    wildly over/under-represented — fraud cases, so every split here is
    stratified on `y`.

    Validation is used for threshold selection (Phase 14); test is held
    out completely until final evaluation (Phase 12-13), so the reported
    metrics are not tuned to the data they're reported on.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=random_state
    )
    relative_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_models(random_state=RANDOM_STATE):
    """
    Three models, in increasing order of complexity — deliberately not
    more than this (Phase 10): the goal is a clean baseline-vs-advanced
    comparison, not an algorithm zoo.

    Class imbalance (Phase 11) is handled via class_weight='balanced' for
    Logistic Regression and Random Forest (both support it natively).
    GradientBoostingClassifier has no class_weight parameter, so it's
    instead fit with per-sample weights (see `balanced_sample_weight`
    below) — same idea, different API.
    """
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, class_weight="balanced",
            n_jobs=-1, random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=random_state
        ),
    }


def balanced_sample_weight(y):
    """Per-sample weights that upweight the minority (fraud) class 1:1 with class_weight='balanced'."""
    return compute_sample_weight(class_weight="balanced", y=y)


def make_scaler():
    """
    Logistic Regression is scale-sensitive (V1-V28 are already
    standardized by the original PCA, but Amount/log_amount/hour_of_day
    are not) — tree-based models (Random Forest, Gradient Boosting) are
    scale-invariant and don't need this.
    """
    return StandardScaler()


def smote_resample(X_train, y_train, random_state=RANDOM_STATE):
    """
    SMOTE oversampling — applied ONLY to the training fold, never before
    the train/validation/test split. Applying it earlier would let
    synthetic points derived from a fraud case leak into validation/test
    neighbors, silently inflating reported performance.
    """
    from imblearn.over_sampling import SMOTE

    sm = SMOTE(random_state=random_state)
    return sm.fit_resample(X_train, y_train)
