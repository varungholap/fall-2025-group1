import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.component.env import get_data, get_features, get_actions, compute_reward, RewardWeights
from src.component.models import LinUCB

def main():
    csv_path = "/Users/neerajmagadum/Documents/Capstone Project/fall-2025-group1/data/Production Data.csv"
    df = get_data(csv_path)
    df, _, _, _ = get_features(df)

    model = LinUCB(alpha=1.5)
    total_reward = 0.0
    rw = RewardWeights()

    action_groups = get_actions(df)

    for ep in range(100):
        epoch_reward = 0.0 
        for actions in action_groups:
            if actions.empty:
                continue
            X = actions[["Offered_Total", "Served_Total", "consumption_rate", "Production_Cost_Total", "Discarded_Cost"]].to_numpy()
            a = model.select(X)
            reward = compute_reward(actions.iloc[a], rw)
            model.update(X[a], reward)
            epoch_reward += reward
        print(f"Epoch {ep+1}: reward={epoch_reward:.3f}")


if __name__ == "__main__":
    main()