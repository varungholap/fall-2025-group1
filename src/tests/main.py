import pandas as pd
import numpy as np
import sys
import os


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.component.env import get_data, get_features, get_actions, compute_reward, RewardWeights
from src.component.models import LinUCB, run_training_loop

def main():
    csv_path = os.path.join(project_root, "data", "Production Data.csv")
    df = get_data(csv_path)
    df, _, _, _ = get_features(df)
    action_groups = get_actions(df)

    model = LinUCB(alpha=1.5)
    rw = RewardWeights()

    run_training_loop(model, action_groups, rw, epochs=5)

if __name__ == "__main__":
    main()