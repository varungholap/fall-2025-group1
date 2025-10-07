import os
import sys
import numpy as np
current_dir = os.path.dirname(__file__)
component_path = os.path.join(current_dir, '..', 'component')
sys.path.append(os.path.abspath(component_path))
from env import generate_environment_data, describe_environment, RewardWeights
from models import LinUCB, Trainer

# --- Bandit simulation setup ---
env_rounds = generate_environment_data(n_rounds=10, n_arms=5)
describe_environment(env_rounds)

model = LinUCB(alpha=1.5)
rw = RewardWeights(w_production=0, w_discarded=1.0, w_consumption=1.0)

trainer = Trainer(model=model, env_rounds=env_rounds, rw=rw)
trainer.train()
 