import numpy as np
from typing import  Optional
from typing import Tuple

# ============================================================
# LINUCB MODEL
# ============================================================
class LinUCB:
    def __init__(self, n_features: int, alpha: float = 1.0):
        self.alpha = alpha
        self.n_features = n_features
        self.A = np.identity(n_features)
        self.b = np.zeros((n_features, 1))
        self.theta = np.zeros((n_features, 1))

    def reset(self):
        """Resets the model's learned parameters to their initial state."""
        self.A = np.identity(self.n_features)
        self.b = np.zeros((self.n_features, 1))
        self.theta = np.zeros((self.n_features, 1))

    def compute_reward(self, x: np.ndarray) -> float:
        """Compute reward for a selected arm vector."""
        discarded_cost = x[3] if len(x) > 2 else 0.0
        left_over_cost = x[4] if len(x) > 3 else 0.0
        reward = - (left_over_cost + discarded_cost)
        return float(reward)

    def select_arm(self, X: np.ndarray) -> Tuple[int, float, float]:
        """Select arm using LinUCB rule."""
        A_inv = np.linalg.inv(self.A)
        p_values = []
        uncertainty_terms = []

        # Identify available arms (those with non-zero feature vectors)
        available_arms_mask = np.any(X, axis=1)

        for i in range(X.shape[0]):
            if not available_arms_mask[i]:
                p_values.append(-np.inf)  # Assign a very low value to unavailable arms
                uncertainty_terms.append(0.0)
                continue

            x = X[i].reshape(-1, 1)
            mean = float(np.dot(self.theta.T, x))
            conf = self.alpha * np.sqrt(np.dot(np.dot(x.T, A_inv), x))
            p = mean + conf
            p_values.append(float(p))
            uncertainty_terms.append(float(conf))

        # Choose arm with highest UCB
        chosen_arm = int(np.argmax(p_values))
        
        p_value_of_chosen = p_values[chosen_arm]
        if np.isneginf(p_value_of_chosen):
            p_value_of_chosen = 0.0 # Handle case where no arms were available

        return chosen_arm, float(p_value_of_chosen), float(np.mean(uncertainty_terms))

    def update(self, x: np.ndarray, reward: float):
        """Update model with chosen arm and observed reward."""
        x = x.reshape(-1, 1)
        self.A += np.dot(x, x.T)
        self.b += reward * x
        self.theta = np.linalg.solve(self.A, self.b)

    def train(self, env_rounds: list, verbose: bool = True):
        print("\n=== Starting LinUCB Bandit Training with Diagnostics ===")
        history = {
            "round": [], "chosen_arm": [], "reward": [], "regret": [],
            "cumulative_reward": [], "cumulative_regret": [],
            "uncertainty": [], "delta_theta": [], "explore_ratio": []
        }
        cumulative_reward = 0.0
        cumulative_regret = 0.0
        exploration_count = 0

        for t, (X, meta) in enumerate(env_rounds, start=1):
            # --- Snapshot theta BEFORE the update ---
            prev_theta = self.theta.copy() if self.theta is not None else np.zeros((X.shape[1], 1))

            # --- Choose arm using LinUCB ---
            chosen_arm, _, avg_uncertainty = self.select_arm(X)
            x_chosen = X[chosen_arm]
            reward = self.compute_reward(x_chosen)

            # --- Update model and measure per-update change in theta ---
            self.update(x_chosen, reward)
            delta_theta = np.linalg.norm(self.theta - prev_theta)

            # --- Compute regret (optimal reward - chosen reward) ---
            available_arms_mask = np.any(X, axis=1)
            rewards_all = [self.compute_reward(X[i]) for i, available in enumerate(available_arms_mask) if available]
            if not rewards_all:
                optimal_reward = reward
            else:
                optimal_reward = np.max(rewards_all)
            regret = optimal_reward - reward

            # --- Update cumulative metrics ---
            cumulative_reward += reward
            cumulative_regret += regret

            # --- Exploration heuristic: large uncertainty → exploration ---
            if avg_uncertainty > 0.5:  # threshold heuristic
                exploration_count += 1
            explore_ratio = exploration_count / t

            # --- Log metrics ---
            history["round"].append(t)
            history["chosen_arm"].append(chosen_arm)
            history["reward"].append(reward)
            history["regret"].append(regret)
            history["cumulative_reward"].append(cumulative_reward)
            history["cumulative_regret"].append(cumulative_regret)
            history["uncertainty"].append(avg_uncertainty)
            history["delta_theta"].append(delta_theta)
            history["explore_ratio"].append(explore_ratio)

            if verbose:
                date = meta.get("date", "Unknown")
                school = meta.get("schools", "Unknown")
                print(f"→ Round {t:03d} | Date: {date} | School: {school} | "
                      f"Arm #{chosen_arm:03d} | Reward={reward:.4f} | Regret={regret:.4f} | "
                      f"Δθ={delta_theta:.4f} | U={avg_uncertainty:.4f}")

        print("\nTraining complete.")
        print(f"Total cumulative reward: {cumulative_reward:.4f}")
        print(f"Total cumulative regret: {cumulative_regret:.4f}")
        print(f"Exploration ratio: {history['explore_ratio'][-1]:.2%}")

        return history