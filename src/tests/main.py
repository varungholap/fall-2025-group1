# main.py
import os
import sys
import numpy as np
import pandas as pd

current_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_path)

from component.env import load_environment_data
from component.models import LinUCB, RandomPolicy  # RandomPolicy kept if you still want to run it
from component.visualizations import simulate_random_baseline, generate_all_plots

# Load environment
file_path = os.path.abspath(os.path.join(src_path, '../data/Production Data.csv'))
env_rounds, metadata, unique_dishes = load_environment_data(file_path)

# Initialize model and reward weights
model = LinUCB(n_arms=env_rounds[0].shape[0], n_features=env_rounds[0].shape[1], alpha=0.5)

# Train
training_data = list(zip(env_rounds, metadata))
history = model.train(env_rounds=training_data, verbose=False)  
random_model = RandomPolicy(n_arms=env_rounds[0].shape[0])
random_model.train(training_data)

# Build a Random baseline history (per-round) without modifying RandomPolicy
rnd_hist = simulate_random_baseline(training_data)

# Define the output path for the recommendations CSV
output_file_path = os.path.abspath(os.path.join(src_path, '../reports/recommendations/recommendations.csv'))

# Generate and save recommendations
model.generate_and_display_all_recommendations(env_rounds, metadata, unique_dishes, top_k=3, output_path=output_file_path)

# Generate all plots using the artifacts we already have
generate_all_plots(
    env_rounds=env_rounds,
    metadata=metadata,
    history_linucb=history,
    rnd_hist=rnd_hist,
    recommendations_csv_path=output_file_path
)
