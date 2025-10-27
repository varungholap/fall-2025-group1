import numpy as np
from typing import Tuple

class LinUCB:
    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros((n_features, 1)) for _ in range(n_arms)]

    def reset(self):
        self.A = [np.identity(self.n_features) for _ in range(self.n_arms)]
        self.b = [np.zeros((self.n_features, 1)) for _ in range(self.n_arms)]

    def compute_reward(self, x: np.ndarray) -> float:
        planned_total = x[0] if len(x) > 0 else 0.0
        served_total = x[1] if len(x) > 1 else 0.0

        if planned_total == 0:
            return 0.0
        if served_total > planned_total:
            return 1.0
        
        reward = served_total / planned_total
        return float(reward)

    def select_arm(self, X: np.ndarray) -> Tuple[int, float, float]:
        """Select arm using the LinUCB rule for disjoint models."""
        p_values = []
        uncertainty_terms = []

        available_arms_mask = np.any(X, axis=1)
        # just do a dot product with available_arms_mask with X
        for i in range(self.n_arms):
            if i >= X.shape[0] or not available_arms_mask[i]:
                p_values.append(-np.inf) 
                uncertainty_terms.append(0.0)
                continue

            A_inv = np.linalg.inv(self.A[i])
            theta_i = np.linalg.solve(self.A[i], self.b[i])
            x = X[i].reshape(-1, 1)
            mean = float(np.dot(theta_i.T, x))
            conf = self.alpha * np.sqrt(np.dot(np.dot(x.T, A_inv), x))
            p = mean + conf
            
            p_values.append(float(p))
            uncertainty_terms.append(float(conf))

        chosen_arm = int(np.argmax(p_values))
        
        p_value_of_chosen = p_values[chosen_arm]
        if np.isneginf(p_value_of_chosen):
            p_value_of_chosen = 0.0 
        return chosen_arm, float(p_value_of_chosen), float(np.mean(uncertainty_terms))

    def update(self, arm_index: int, x: np.ndarray, reward: float):
        x = x.reshape(-1, 1)
        self.A[arm_index] += np.dot(x, x.T)
        self.b[arm_index] += reward * x

    def train(self, env_rounds: list, verbose: bool = True):
        print("\n=== Starting LinUCB Bandit Training with Diagnostics ===")
        history = {
            "round": [], "chosen_arm": [], "reward": [], "regret": [],
            "cumulative_reward": [], "cumulative_regret": [],
            "uncertainty": [], "explore_ratio": []
        }
        cumulative_reward = 0.0
        cumulative_regret = 0.0
        exploration_count = 0

        for t, (X, meta) in enumerate(env_rounds, start=1):
            # --- Choose arm using LinUCB ---
            chosen_arm, _, avg_uncertainty = self.select_arm(X)

            x_chosen = X[chosen_arm]
            reward = self.compute_reward(x_chosen)

            # --- Update model ---
            self.update(chosen_arm, x_chosen, reward)

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
            history["explore_ratio"].append(explore_ratio)

            if verbose:
                date = meta.get("date", "Unknown")
                school = meta.get("schools", "Unknown")
                meal_type = meta.get("meal_type", "Unknown")
                print(f"→ Round {t:03d} | Date: {date} | School: {school} | Meal: {meal_type} | "
                      f"Arm #{chosen_arm:03d} | Reward={reward:.4f} | Regret={regret:.4f} | "
                      f"U={avg_uncertainty:.4f}")

        print("\nTraining complete.")
        print(f"Total cumulative reward: {cumulative_reward:.4f}")
        print(f"Total cumulative regret: {cumulative_regret:.4f}")
        print(f"Exploration ratio: {history['explore_ratio'][-1]:.2%}")

        return history