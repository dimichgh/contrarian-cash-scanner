"""Readiness model: logistic regression shrunk toward an expert prior, plus AUC."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logit(X: np.ndarray, y: np.ndarray, prior: np.ndarray, lam: float, iters: int = 40):
    """Newton's method on log-loss + lam/2·||w − prior||². lam acts as pseudo-observations."""
    n, k = X.shape
    w = prior.astype(float).copy()
    rate = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
    b = float(np.log(rate / (1 - rate)))
    Xa = np.column_stack([X, np.ones(n)])
    reg = np.zeros((k + 1, k + 1))
    reg[:k, :k] = lam * np.eye(k)
    for _ in range(iters):
        p = sigmoid(X @ w + b)
        r = p - y
        g = np.concatenate([X.T @ r + lam * (w - prior), [r.sum()]])
        H = Xa.T @ (Xa * (p * (1 - p))[:, None]) + reg + 1e-9 * np.eye(k + 1)
        step = np.linalg.solve(H, g)
        w -= step[:k]
        b -= step[k]
        if np.abs(step).max() < 1e-8:
            break
    return w, b


def fit_intercept(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Best intercept for fixed coefficients (used to score the prior on its own)."""
    z = X @ w
    b = 0.0
    for _ in range(50):
        p = sigmoid(z + b)
        step = (p - y).sum() / max((p * (1 - p)).sum(), 1e-9)
        b -= step
        if abs(step) < 1e-9:
            break
    return b


def auc(y, s) -> float:
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    ok = ~(np.isnan(y) | np.isnan(s))
    y, s = y[ok], s[ok]
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(s).rank().to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def predict(m: dict, X: np.ndarray) -> np.ndarray:
    Z = (X - np.array(m["mean"])) / np.array(m["std"])
    return sigmoid(np.nan_to_num(Z) @ np.array(m["coef"]) + m["intercept"])


def readiness(m: dict, prob) -> np.ndarray:
    """Percentile of a probability among all historical training rows: 90 = top-decile setup."""
    q = np.array(m["quantiles"])
    return np.clip(np.searchsorted(q, np.asarray(prob, float), side="right") - 1, 0, 100)
