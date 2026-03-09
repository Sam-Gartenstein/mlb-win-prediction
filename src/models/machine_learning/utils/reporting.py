import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
)


def print_best_model_report(
    results_df: pd.DataFrame,
    model_artifacts: dict,
    *,
    label: str = "Best model"
) -> None:
    """
    Print evaluation metrics for the top-ranked model in a results dataframe.

    Works for both baseline and tuned model artifact dictionaries.

    Expected structure:
    - results_df.iloc[0]["model"] contains the best model name
    - model_artifacts[model_name] contains:
        baseline:
            {
                "fitted": {"model": fitted_model, ...},
                "X_test": ...,
                "y_test": ...
            }
        tuned:
            {
                "best_model": fitted_model,
                "X_test": ...,
                "y_test": ...
            }
    """
    best_row = results_df.iloc[0]
    best_name = best_row["model"]
    artifact = model_artifacts[best_name]

    # Handle baseline vs tuned storage structure
    if "best_model" in artifact:
        model = artifact["best_model"]
    elif "fitted" in artifact and "model" in artifact["fitted"]:
        model = artifact["fitted"]["model"]
    else:
        raise KeyError(
            f"Could not find fitted model for '{best_name}'. "
            "Expected either 'best_model' or artifact['fitted']['model']."
        )

    X_test = artifact["X_test"]
    y_test = artifact["y_test"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"{label}: {best_name}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test ROC AUC:  {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Test Log Loss: {log_loss(y_test, y_proba):.4f}")

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))