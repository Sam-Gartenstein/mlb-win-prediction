from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
)


def fit_eval_xgb(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

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
    }