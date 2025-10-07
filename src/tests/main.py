import os
import sys
import numpy as np

current_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_path)
from component.env import load_environment_data, describe_environment, RewardWeights
from component.models import LinUCB, Trainer

# --- Bandit simulation setup ---
file_path = os.path.abspath(os.path.join(src_path, '../data/Production Data.csv'))
env_rounds = load_environment_data(file_path)
describe_environment(env_rounds)

model = LinUCB(alpha=1.5)
rw = RewardWeights(w_production=0, w_discarded=1.0, w_consumption=1.0)

trainer = Trainer(model=model, env_rounds=env_rounds, rw=rw)
trainer.train()
 