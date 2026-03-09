from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
)


def fit_eval_logreg(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    max_iter: int = 2000,
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    class_weight=None,
    verbose: bool = True,
) -> dict:
    """
    Scale -> fit logistic regression -> evaluate.
    Returns metrics + fitted model + scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=max_iter,
        C=C,
        penalty=penalty,
        solver=solver,
        class_weight=class_weight,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

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
        "model": model,
        "scaler": scaler,
    }