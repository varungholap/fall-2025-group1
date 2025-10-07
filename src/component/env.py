import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

def load_environment_data(file_path: str):
    df = pd.read_csv(file_path, low_memory=False)
    
    # Features: [Offered_Total, Served_Total, consumption_rate, Discarded_Cost]
    df['consumption_rate'] = df['Served_Total'] / df['Offered_Total']
    df['consumption_rate'] = df['consumption_rate'].fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    rounds = []
    for _, group in df.groupby(['Date', 'School_Name']):
        features = group[['Offered_Total', 'Served_Total', 'consumption_rate', 'Discarded_Cost']].values
        rounds.append(features)
        
    return rounds

@dataclass
class RewardWeights:
    w_production: float = 0.0
    w_discarded: float = 1.0
    w_consumption: float = 1.0


def compute_reward(x: np.ndarray, rw: RewardWeights) -> float:
    """
    Compute reward for a selected arm vector.
    Columns: [Offered_Total, Served_Total, consumption_rate, Discarded_Cost]
    """

    discarded_cost = x[3]
    consumption_rate = x[2]
    offered_total = x[0]

    discarded_rate = discarded_cost / (offered_total + 1e-8)
    reward = - rw.w_discarded * discarded_rate + rw.w_consumption * consumption_rate
    return float(reward)


def describe_environment(rounds: List[np.ndarray], show_rounds: int = 3):
    """
    Show action space (arms and features) for each context.
    """
    feature_names = [
        "Offered_Total",
        "Served_Total",
        "consumption_rate",
        "Discarded_Cost",
    ]

    print(f"\n=== ENVIRONMENT SUMMARY ===")
    print(f"Total rounds (contexts): {len(rounds)}")
    if not rounds:
        print("No rounds to describe.")
        return
    print(f"Each round has between {min(len(r) for r in rounds)} and {max(len(r) for r in rounds)} arms (actions) × {rounds[0].shape[1]} features\n")

    for i, X in enumerate(rounds[:show_rounds], start=1):
        print(f"--- Round {i} / Context {i} ---")
        df = pd.DataFrame(X, columns=feature_names)
        df.index.name = "Arm_ID"
        print(df.to_string(index=True))
        print("\nAction space shape:", X.shape)
        print("Each arm (row) = one action, with features:", feature_names)
        print("-" * 70)

if __name__ == "__main__":
    # Load real data
    file_path = "/Users/neerajmagadum/Documents/Capstone Project/fall-2025-group1/data/Production Data.csv"
    env = load_environment_data(file_path)
    describe_environment(env)