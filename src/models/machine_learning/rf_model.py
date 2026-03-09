from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
)


def fit_eval_rf(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    n_estimators: int = 300,
    max_depth=None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features="sqrt",
    class_weight=None,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Fit and evaluate a baseline Random Forest classifier.
    Returns metrics + fitted model.
    """

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_proba)
    test_log_loss = log_loss(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    if verbose:
        print("Test Accuracy:", test_accuracy)
        print("Test ROC AUC:", test_roc_auc)
        print("Test Log Loss:", test_log_loss)
        print("\nConfusion Matrix:\n", cm)
        print("\nClassification Report:\n", report)

    return {
        "test_accuracy": test_accuracy,
        "test_roc_auc": test_roc_auc,
        "test_log_loss": test_log_loss,
        "confusion_matrix": cm,
        "classification_report": report,
        "model": rf_model
    }