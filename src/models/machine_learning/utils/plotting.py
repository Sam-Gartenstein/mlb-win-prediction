import pandas as pd
import matplotlib.pyplot as plt


def plot_model_feature_effects(
    model,
    feature_names,
    *,
    title: str = "Model Feature Effects",
    xlim: tuple[float, float] | None = None,
    decimals: int = 3,
):
    """
    Plot feature effects for different model types.

    - If the model has `coef_`, plot signed coefficients
      (e.g. logistic regression).
    - If the model has `feature_importances_`, plot feature importances
      (e.g. random forest, XGBoost).

    Parameters
    ----------
    model : fitted model
        A fitted sklearn/XGBoost model.
    feature_names : list-like
        Names of the input features.
    title : str
        Plot title.
    xlim : tuple or None
        Optional x-axis limits.
    decimals : int
        Number of decimals for bar labels.
    """
    # Logistic regression / linear models
    if hasattr(model, "coef_"):
        values = pd.Series(model.coef_[0], index=feature_names).sort_values()
        colors = ["tab:red" if v < 0 else "tab:blue" for v in values.values]
        xlabel = "Coefficient (standardized features)"
        zero_line = True

    # Tree-based models
    elif hasattr(model, "feature_importances_"):
        values = pd.Series(model.feature_importances_, index=feature_names).sort_values()
        colors = ["tab:blue"] * len(values)
        xlabel = "Feature importance"
        zero_line = False

    else:
        raise ValueError(
            "Model must have either `coef_` or `feature_importances_`."
        )

    fig_h = max(6, 0.35 * len(values))
    plt.figure(figsize=(12, fig_h))
    plt.barh(values.index, values.values, color=colors)

    if zero_line:
        plt.axvline(0, linewidth=1)

    plt.title(title)
    plt.xlabel(xlabel)

    if xlim is not None:
        plt.xlim(*xlim)

    span = (values.max() - values.min()) if len(values) else 1.0
    offset = 0.01 * span if span != 0 else 0.01

    for y, v in enumerate(values.values):
        if zero_line:
            x = v + offset if v >= 0 else v - offset
            ha = "left" if v >= 0 else "right"
        else:
            x = v + offset
            ha = "left"

        plt.text(x, y, f"{v:.{decimals}f}", va="center", ha=ha, fontsize=9)

    plt.tight_layout()
    plt.show()