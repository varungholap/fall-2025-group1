import os
import sys
import numpy as np

# Add module path
current_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_path)

from component.env import load_environment_data, describe_environment, RewardWeights , compute_reward
from component.models import LinUCB, Trainer

# -----------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------
if __name__ == "__main__":
    # Load environment
    file_path = os.path.abspath(os.path.join(src_path, '../data/Production Data.csv'))
    env_rounds, metadata = load_environment_data(file_path)

    # Display a preview of rounds
    describe_environment(env_rounds, metadata)

    # Initialize model and reward weights
    model = LinUCB(alpha=0.5)
    rw = RewardWeights(w_production=0, w_discarded=1.0, w_consumption=1.0)

    # Train the contextual bandit
    trainer = Trainer(model=model, env_rounds=list(zip(env_rounds, metadata)), rw=rw, reward_func=compute_reward)
    trainer.train()
