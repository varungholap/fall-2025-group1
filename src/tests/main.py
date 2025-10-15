#%%
import os
import sys
import numpy as np

current_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_path)

from component.env import load_environment_data
from component.models import LinUCB

#%%
# Load environment
file_path = os.path.abspath(os.path.join(src_path, '../data/Production Data.csv'))
env_rounds, metadata = load_environment_data(file_path)

#%%
# Display a preview of rounds
# describe_environment(env_rounds, metadata)

#%%
# Initialize model and reward weights
model = LinUCB(n_features=env_rounds[0].shape[1], alpha=100)

#%%
# Train 
model.train(env_rounds=list(zip(env_rounds, metadata)))

# %%
