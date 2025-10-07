import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


def generate_environment_data(n_rounds: int = 10, n_arms: int = 5, seed: Optional[int] = 42):
    """
    Generate synthetic contextual bandit environment using NumPy.
    Each round has several arms with contextual features.
    """
    rng = np.random.default_rng(seed)
    rounds = []

    for _ in range(n_rounds):
        # Features: [Offered_Total, Served_Total, consumption_rate, Production_Cost_Total, Discarded_Cost]
        offered = rng.integers(100, 300, size=n_arms)
        served = offered * rng.uniform(0.5, 1.0, size=n_arms)
        consumption_rate = served / offered
        production_cost = rng.uniform(1.0, 3.0, size=n_arms) * offered
        discarded_cost = rng.uniform(0.0, 1.0, size=n_arms) * (offered - served)

        X = np.stack([offered, served, consumption_rate, production_cost, discarded_cost], axis=1)
        rounds.append(X)

    return rounds


@dataclass
class RewardWeights:
    w_production: float = 0.0
    w_discarded: float = 1.0
    w_consumption: float = 1.0


def compute_reward(x: np.ndarray, rw: RewardWeights) -> float:
    """
    Compute reward for a selected arm vector.
    Columns: [Offered_Total, Served_Total, consumption_rate, Production_Cost_Total, Discarded_Cost]
    """
    production_cost = x[3]
    discarded_cost = x[4]
    consumption_rate = x[2]
    reward = -rw.w_production * production_cost - rw.w_discarded * discarded_cost + rw.w_consumption * consumption_rate
    return float(reward)


def describe_environment(rounds: List[np.ndarray], show_rounds: int = 3):
    """
    Nicely print all arms, features, and their dataset per round.
    Each round is treated as one 'context' (decision situation).
    """
    feature_names = [
        "Offered_Total",
        "Served_Total",
        "consumption_rate",
        "Production_Cost_Total",
        "Discarded_Cost",
    ]

    print(f"\n=== ENVIRONMENT SUMMARY ===")
    print(f"Total rounds (contexts): {len(rounds)}")
    print(f"Each round has {rounds[0].shape[0]} arms and {rounds[0].shape[1]} features\n")

    for i, X in enumerate(rounds[:show_rounds], start=1):
        print(f"--- Round {i} / Context {i} ---")
        df = pd.DataFrame(X, columns=feature_names)
        df.index.name = "Arm_ID"
        print(df.to_string(index=True))
        print("-" * 70)


if __name__ == "__main__":
    env = generate_environment_data(n_rounds=5, n_arms=4)
    describe_environment(env)
