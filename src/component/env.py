import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple

# -----------------------------------------------------------
# LOAD ENVIRONMENT DATA
# -----------------------------------------------------------
def load_environment_data(file_path: str) -> Tuple[List[np.ndarray], List[dict]]:
    """Load CSV data, extract numeric/date features, one-hot encode categories, 
    and create contextual rounds for bandit simulation."""
    df = pd.read_csv(file_path, low_memory=False)

    # --- Compute consumption rate safely ---
    df["consumption_rate"] = df["Served_Total"] / df["Offered_Total"].replace(0, np.nan)
    df["consumption_rate"] = df["consumption_rate"].fillna(0)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # --- Parse date and extract DayOfMonth + Month ---
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["DayOfMonth"] = df["Date"].dt.day.fillna(0).astype(int)
    df["Month"] = df["Date"].dt.month.fillna(0).astype(int)

    # --- One-hot encode Meal_Type and School_Name (categorical contexts) ---
    for col in ["Meal_Type", "School_Name"]:
        if col not in df.columns:
            df[col] = "Unknown"
    df = pd.get_dummies(df, columns=["Meal_Type", "School_Name"], prefix=["meal", "school"])

    # --- Select features (core + one-hot) ---
    feature_cols = [
        "Offered_Total", "Served_Total", "consumption_rate",
        "Discarded_Cost", "DayOfMonth", "Month"
    ] + [c for c in df.columns if c.startswith(("meal_", "school_"))]

    # --- Group data by Date (each date = one decision round) ---
    rounds, metadata = [], []
    for date, group in df.groupby("Date", dropna=False):
        X = group[feature_cols].to_numpy(dtype=float)
        rounds.append(X)
        metadata.append({
            "date": date,
            "num_arms": len(group),
            "schools": [col for col in group.columns if col.startswith("school_") and group[col].any()],
            "meals": [col for col in group.columns if col.startswith("meal_") and group[col].any()]
        })

    return rounds, metadata


# -----------------------------------------------------------
# REWARD FUNCTION AND WEIGHTS
# -----------------------------------------------------------
@dataclass
class RewardWeights:
    w_production: float = 0.0
    w_discarded: float = 1.0
    w_consumption: float = 1.0


def compute_reward(x: np.ndarray, rw: RewardWeights) -> float:
    """Compute reward for a selected arm vector.
    Columns: [Offered_Total, Served_Total, consumption_rate, Discarded_Cost, ...]
    """
    #production_cost = x[3] if len(x) > 3 else 0.0
    discarded_cost = x[4] if len(x) > 4 else 0.0
    consumption_rate = x[2] if len(x) > 2 else 0.0

    reward = (
        -rw.w_discarded * discarded_cost
        +rw.w_consumption * consumption_rate
    )
    return float(reward)


# -----------------------------------------------------------
# DESCRIBE ENVIRONMENT
# -----------------------------------------------------------
def describe_environment(rounds: List[np.ndarray], metadata: List[dict], show_rounds: int = 3):
    """Print an overview of the contextual bandit environment."""
    print("\n=== ENVIRONMENT SUMMARY ===")
    print(f"Total rounds (contexts): {len(rounds)}")

    if not rounds:
        print("No data found.")
        return

    for i, (X, meta) in enumerate(zip(rounds[:show_rounds], metadata[:show_rounds]), start=1):
        date_str = meta["date"].strftime("%Y-%m-%d") if pd.notna(meta["date"]) else "Unknown"
        print(f"\n--- Round {i} | Date: {date_str} ---")
        print(f"Number of arms (food items): {meta['num_arms']}")
        print(f"Schools in this round: {meta['schools']}")
        print(f"Meal types: {meta['meals']}")
        print("Feature matrix shape:", X.shape)
        df_preview = pd.DataFrame(X)
        print(df_preview.head())
        print("-" * 80)
