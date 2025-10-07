import numpy as np
from typing import List, Optional
from component.env import compute_reward, RewardWeights

# -----------------------------------------------------------
# LINUCB ALGORITHM
# -----------------------------------------------------------
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
        """Select the best arm index for the given round."""
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
        """Update LinUCB with chosen arm and observed reward."""
        x = x.reshape(-1, 1)
        self.A += x @ x.T
        self.b += reward * x


# -----------------------------------------------------------
# TRAINING PIPELINE
# -----------------------------------------------------------
class Trainer:
    """
    Runs contextual bandit training with round/arm terminology.
    """
    import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ============================================================
# LINUCB MODEL
# ============================================================
class LinUCB:
    def __init__(self, n_features: int = None, alpha: float = 1.0):
        self.alpha = alpha
        self.A = None
        self.b = None
        self.theta = None
        self.n_features = n_features

    def init_params(self, d: int):
        """Initialize matrices based on feature dimension d."""
        self.A = np.identity(d)
        self.b = np.zeros((d, 1))
        self.theta = np.zeros((d, 1))
        self.n_features = d

    def select_arm(self, X: np.ndarray) -> Tuple[int, float, float]:
        """Select arm using LinUCB rule."""
        if self.A is None:
            self.init_params(X.shape[1])

        A_inv = np.linalg.inv(self.A)
        p_values = []
        uncertainty_terms = []

        for i in range(X.shape[0]):
            x = X[i].reshape(-1, 1)
            mean = float(np.dot(self.theta.T, x))
            conf = self.alpha * np.sqrt(np.dot(np.dot(x.T, A_inv), x))
            p = mean + conf
            p_values.append(p)
            uncertainty_terms.append(conf)

        # Choose arm with highest UCB
        chosen_arm = int(np.argmax(p_values))
        return chosen_arm, float(p_values[chosen_arm]), float(np.mean(uncertainty_terms))

    def update(self, x: np.ndarray, reward: float):
        """Update model with chosen arm and observed reward."""
        x = x.reshape(-1, 1)
        self.A += np.dot(x, x.T)
        self.b += reward * x
        self.theta = np.linalg.solve(self.A, self.b)


# ============================================================
# TRAINER WITH DIAGNOSTICS
# ============================================================
class Trainer:
    def __init__(self, model: LinUCB, env_rounds: List[np.ndarray], rw, reward_func):
        self.model = model
        self.env_rounds = env_rounds
        self.rw = rw
        self.reward_func = reward_func
        self.history = {
            "round": [], "chosen_arm": [], "reward": [], "regret": [],
            "cumulative_reward": [], "cumulative_regret": [],
            "uncertainty": [], "theta_change": [], "explore_ratio": []
        }

    def train(self, verbose: bool = True):
        print("\n=== Starting LinUCB Bandit Training with Diagnostics ===")
        cumulative_reward = 0.0
        cumulative_regret = 0.0
        prev_theta = None
        exploration_count = 0
        total_rounds = len(self.env_rounds)

        for t, (X, meta) in enumerate(self.env_rounds, start=1):
            # --- Choose arm using LinUCB ---
            chosen_arm, _, avg_uncertainty = self.model.select_arm(X)
            x_chosen = X[chosen_arm]
            reward = self.reward_func(x_chosen, self.rw)

            # --- Compute regret (optimal - actual) ---
            rewards_all = [self.reward_func(x, self.rw) for x in X]
            optimal_reward = np.max(rewards_all)
            regret = optimal_reward - reward

            # --- Track metrics ---
            cumulative_reward += reward
            cumulative_regret += regret

            # θ movement
            prev_theta_norm = np.linalg.norm(prev_theta) if prev_theta is not None else 0
            self.model.update(x_chosen, reward)
            theta_norm = np.linalg.norm(self.model.theta)
            delta_theta = abs(theta_norm - prev_theta_norm)
            prev_theta = self.model.theta.copy()

            # Exploration heuristic: large uncertainty → exploration
            if avg_uncertainty > 0.5:  # threshold heuristic
                exploration_count += 1

            explore_ratio = exploration_count / t

            # --- Log everything ---
            self.history["round"].append(t)
            self.history["chosen_arm"].append(chosen_arm)
            self.history["reward"].append(reward)
            self.history["regret"].append(regret)
            self.history["cumulative_reward"].append(cumulative_reward)
            self.history["cumulative_regret"].append(cumulative_regret)
            self.history["uncertainty"].append(avg_uncertainty)
            self.history["theta_change"].append(delta_theta)
            self.history["explore_ratio"].append(explore_ratio)

            if verbose:
                date = meta.get("date", "Unknown")
                print(f"→ Round {t:02d} | Date: {date} | Arm #{chosen_arm:04d} | "
                      f"Reward={reward:.4f} | Regret={regret:.4f} | "
                      f"Δθ={delta_theta:.4f} | U={avg_uncertainty:.4f}")

        print("\nTraining complete.")
        print(f"Total cumulative reward: {cumulative_reward:.4f}")
        print(f"Total cumulative regret: {cumulative_regret:.4f}")
        print(f"Exploration ratio: {self.history['explore_ratio'][-1]:.2%}")

        return self.history
