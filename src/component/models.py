import numpy as np
from typing import List, Optional
from component.env import compute_reward, RewardWeights


class LinUCB:
    """
    LinUCB contextual bandit algorithm (NumPy-based).
    """
    def __init__(self, alpha: float = 1.5, d: Optional[int] = None):
        self.alpha = alpha
        self.d = d
        self.A = None
        self.b = None
        self.initialized = False

    def _maybe_init(self, d: int):
        if not self.initialized:
            self.d = d
            self.A = np.eye(self.d)
            self.b = np.zeros((self.d, 1))
            self.initialized = True

    def select(self, X: np.ndarray) -> int:
        """
        Select the best arm index for the given round.
        X: (n_arms x n_features)
        """
        K, D = X.shape
        self._maybe_init(D)

        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b

        scores = []
        for k in range(K):
            x = X[k].reshape(-1, 1)
            mu = float(theta.T @ x)
            sigma = float(np.sqrt(max(x.T @ A_inv @ x, 0.0)))
            scores.append(mu + self.alpha * sigma)

        return int(np.argmax(scores))

    def update(self, x: np.ndarray, reward: float):
        """
        Update LinUCB with chosen arm and observed reward.
        """
        x = x.reshape(-1, 1)
        self.A += x @ x.T
        self.b += reward * x


class Trainer:
    """
    Runs contextual bandit training with round/arm terminology.
    """
    def __init__(self, model: LinUCB, env_rounds: List[np.ndarray],
                 rw: Optional[RewardWeights] = None):
        self.model = model
        self.env_rounds = env_rounds
        self.rw = rw if rw is not None else RewardWeights()

    def run_round(self, round_idx: int) -> float:
        """
        Simulate a single decision round.
        """
        X = self.env_rounds[round_idx]
        chosen_arm = self.model.select(X)
        reward = compute_reward(X[chosen_arm], self.rw)
        self.model.update(X[chosen_arm], reward)
        print(f"  Chosen arm: {chosen_arm}, reward={reward:.4f}")
        return reward

    def train(self):
        """
        Run training over all rounds.
        """
        print("Starting bandit training...\n")
        total_reward = 0.0
        for r in range(len(self.env_rounds)):
            rwd = self.run_round(r)
            total_reward += rwd
            print(f"Round {r+1}/{len(self.env_rounds)} → Reward: {rwd:.4f}")
        print(f"\nTraining complete. Total reward: {total_reward:.4f}")
