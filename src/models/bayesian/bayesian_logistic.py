import numpy as np
import pandas as pd
import pymc as pm
from sklearn.preprocessing import StandardScaler


def fit_bayesian_logistic(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    beta_mu: float | list[float] | np.ndarray = 0.0,
    beta_sigma: float | list[float] | np.ndarray = 1.0,
    intercept_mu: float = 0.0,
    intercept_sigma: float = 2.5,
    standardize: bool = True,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> dict:
    """
    Fit a Bayesian logistic regression in PyMC.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor matrix.
    y : pd.Series or np.ndarray
        Binary outcome (0/1).
    beta_mu : float or array-like
        Prior mean(s) for coefficients. Can be a scalar or one value per feature.
    beta_sigma : float or array-like
        Prior std dev(s) for coefficients. Can be a scalar or one value per feature.
    intercept_mu : float
        Prior mean for intercept.
    intercept_sigma : float
        Prior std dev for intercept.
    standardize : bool
        Whether to standardize X before fitting.
    draws, tune, chains, target_accept, random_seed
        PyMC sampling arguments.

    Returns
    -------
    dict
        Contains model, idata, scaler, X_used, y_used, and feature_names.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    y = np.asarray(y).astype(int)
    feature_names = X.columns.to_list()

    if standardize:
        scaler = StandardScaler()
        X_used = scaler.fit_transform(X)
    else:
        scaler = None
        X_used = X.to_numpy()

    n_features = X_used.shape[1]

    beta_mu_arr = np.asarray(beta_mu)
    beta_sigma_arr = np.asarray(beta_sigma)

    if beta_mu_arr.ndim == 0:
        beta_mu_arr = np.repeat(beta_mu_arr, n_features)
    if beta_sigma_arr.ndim == 0:
        beta_sigma_arr = np.repeat(beta_sigma_arr, n_features)

    if len(beta_mu_arr) != n_features:
        raise ValueError(
            f"beta_mu must have length {n_features}, got {len(beta_mu_arr)}"
        )
    if len(beta_sigma_arr) != n_features:
        raise ValueError(
            f"beta_sigma must have length {n_features}, got {len(beta_sigma_arr)}"
        )

    coords = {"feature": feature_names}

    with pm.Model(coords=coords) as model:
        X_data = pm.Data("X_data", X_used)
        y_data = pm.Data("y_data", y)

        intercept = pm.Normal(
            "intercept",
            mu=intercept_mu,
            sigma=intercept_sigma,
        )

        beta = pm.Normal(
            "beta",
            mu=beta_mu_arr,
            sigma=beta_sigma_arr,
            dims="feature",
        )

        logit_p = intercept + pm.math.dot(X_data, beta)
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))

        pm.Bernoulli("likelihood", p=p, observed=y_data)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )

    return {
        "model": model,
        "idata": idata,
        "scaler": scaler,
        "X_used": X_used,
        "y_used": y,
        "feature_names": feature_names,
    }