# src/component/models.py

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

# reuse reward logic from env.py
from .env import compute_reward, RewardWeights


class LinUCB:
    """
    Minimal LinUCB implementation for contextual bandits.
    """
    def __init__(self, alpha: float = 1.5, d: Optional[int] = None):
        self.alpha = alpha
        self.d = d
        self.A: Optional[np.ndarray] = None   # (d x d)
        self.b: Optional[np.ndarray] = None   # (d x 1)
        self.initialized = False

    def _maybe_init(self, d: int) -> None:
        if not self.initialized:
            self.d = d if self.d is None else self.d
            self.A = np.eye(self.d, dtype=float)
            self.b = np.zeros((self.d, 1), dtype=float)
            self.initialized = True

    def select(self, X: np.ndarray) -> int:
        """
        Choose an action index from feature matrix X (K x D).
        """
        K, D = X.shape
        self._maybe_init(D)

        if self.A is None or self.b is None:
            raise RuntimeError("Model not initialized properly.")

        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b

        scores = []
        for k in range(K):
            x = X[k].reshape(-1, 1)
            mu = float(theta.T @ x)                     # mean estimate
            var = float(x.T @ A_inv @ x)
            var = max(var, 0.0)                         # guard tiny negatives
            sigma = float(np.sqrt(var))                 # uncertainty
            scores.append(mu + self.alpha * sigma)

        return int(np.argmax(scores))

    def update(self, x: np.ndarray, reward: float) -> None:
        """
        Update model with selected context x (D,) and observed reward.
        """
        x = x.reshape(-1, 1)
        if self.A is None or self.b is None:
            raise RuntimeError("Model not initialized properly.")

        self.A += x @ x.T
        self.b += reward * x


def run_training_loop(
    model: LinUCB,
    action_groups: list[pd.DataFrame],
    rw: Optional[RewardWeights] = None,
    epochs: int = 100,
    feature_cols: Optional[list[str]] = None,
) -> None:
    """
    Bandit training loop (moved from main.py).

    Parameters
    ----------
    model : LinUCB
    action_groups : list[pd.DataFrame]  # from env.get_actions()
    rw : RewardWeights, optional
    epochs : int
    feature_cols : list[str], optional  # defaults to original 5 columns
    """
    if rw is None:
        rw = RewardWeights()

    if feature_cols is None:
        feature_cols = [
            "Offered_Total",
            "Served_Total",
            "consumption_rate",
            "Production_Cost_Total",
            "Discarded_Cost",
        ]

    for ep in range(epochs):
        epoch_reward = 0.0
        for actions in action_groups:
            if actions.empty:
                continue

            # same behavior as your original loop
            X = actions[feature_cols].to_numpy()
            a = model.select(X)
            reward = compute_reward(actions.iloc[a], rw)
            model.update(X[a], reward)
            epoch_reward += float(reward)

        print(f"Epoch {ep+1}: reward={epoch_reward:.3f}")
