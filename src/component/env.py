#%%

import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple

# -----------------------------------------------------------
# LOAD ENVIRONMENT DATA
# -----------------------------------------------------------

#%%
def load_environment_data(file_path: str) -> Tuple[List[np.ndarray], List[dict]]:
    """Load CSV and create contextual rounds with a fixed number of arms = unique dishes."""
    df = pd.read_csv(file_path, low_memory=False)


    df.replace([np.inf, -np.inf], 0, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["DayOfMonth"] = df["Date"].dt.day.fillna(0).astype(int)
    df["Month"] = df["Date"].dt.month.fillna(0).astype(int)

    # --- Handle missing columns ---
    for col in ["Meal_Type", "Name"]:
        if col not in df.columns:
            df[col] = "Unknown"

    # --- One-hot encode categorical features ---
    df = pd.get_dummies(df, columns=["Meal_Type"], prefix=["meal"])

    # --- Identify all unique dishes (constant arms) ---
    unique_dishes = sorted(df["Name"].dropna().unique().tolist())
    print(f"Total unique dishes (arms): {len(unique_dishes)}")

    # --- Select core features ---
    feature_cols = [
        "Offered_Total", "Served_Total",
        "Discarded_Cost", "Left_Over_Cost", "DayOfMonth", "Month"
    ] + [c for c in df.columns if c.startswith(("meal_", "school_"))]

    # --- Build a complete per-day matrix with all dishes ---
    rounds, metadata = [], []
    for (date, school_name), group in df.groupby(["Date", "School_Name"], dropna=False):
        # map each dish to its row if present
        day_features = []
        for dish in unique_dishes:
            if dish in group["Name"].values:
                row = group[group["Name"] == dish].iloc[0][feature_cols].to_numpy(dtype=float)
            else:
                # dish not available → zero vector
                row = np.zeros(len(feature_cols))
            day_features.append(row)
    
        X = np.vstack(day_features)  # shape = (num_dishes, num_features)
        rounds.append(X)
        metadata.append({
            "date": date,
            "num_arms": len(unique_dishes),
            'schools': school_name,
            "meals": [col for col in group.columns if col.startswith("meal_") and group[col].any()],
            "available_dishes": group["Name"].unique().tolist()
        })

    return rounds, metadata




# # -----------------------------------------------------------
# # DESCRIBE ENVIRONMENT
# # -----------------------------------------------------------
# def describe_environment(rounds: List[np.ndarray], metadata: List[dict], show_rounds: int = 3):
#     """Print an overview of the contextual bandit environment with detailed explanations."""
#     print("\n=== ENVIRONMENT SUMMARY ===")
#     print(f"Total rounds (contexts): {len(rounds)}")
#     print("A 'round' represents a single day of data.")
#     print("An 'arm' represents a unique dish offered on that day (all dishes are potential arms).")
#     print("-" * 80)

#     if not rounds:
#         print("No data found.")
#         return

#     for i, (X, meta) in enumerate(zip(rounds[:show_rounds], metadata[:show_rounds]), start=1):
#         date_str = meta["date"].strftime("%Y-%m-%d") if pd.notna(meta["date"]) else "Unknown"
#         num_arms = meta['num_arms']
#         num_available_arms = len(meta['available_dishes'])
#         school_name = meta.get('schools', "Unknown")
#         num_meals = len(meta['meals'])
#         feature_dim = X.shape[1]
        
#         print(f"\n--- Round {i} | Date: {date_str} ---")
#         print(f"School(s) involved: {school_name}")
#         print(f"Total arms (all possible dishes): {num_arms}")
#         print(f"Available arms this round: {num_available_arms}")
#         print(f"Number of meal types active: {num_meals}")
#         print(f"Feature dimension per arm: {feature_dim}")
        
#         if num_meals > 0 and num_available_arms > 0:
#             avg_items_per_meal = num_available_arms / num_meals
#             print(f"Approx. average items per meal type: {avg_items_per_meal:.1f}")

#         # Count unavailable arms
#         num_unavailable = num_arms - num_available_arms
#         print(f"Number of unavailable arms (zero vectors): {num_unavailable}")

#         print(f"Feature matrix shape (arms x features): {X.shape}")
#         print("First 3 arm features (for preview):")
#         for j, row in enumerate(X[:3]):
#             print(f"  Arm #{j:03d}: {row}")
        
#         print("-" * 80)

# # %%
