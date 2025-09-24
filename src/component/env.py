from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


def get_data(csv_path: str, dayfirst: bool = True) -> pd.DataFrame:
    """Load and lightly normalize the meal dataframe.
    - Parses dates (day-first by default because your sample was 01-05-2025, Thursday).
    - Renames common variants.
    - Coerces numeric-looking strings to numbers.
    - Builds consumptions/served rates if missing.
    """
    # Specify dtype for column 2 to resolve the DtypeWarning.
    # Reading it as a string is safe because subsequent code handles
    # numeric conversion where necessary.
    df = pd.read_csv(csv_path, dtype={2: str})

    # Defensively rename "School Name" to "School_Name" only if "School_Name" doesn't already exist.
    # This prevents creating duplicate columns if both spellings are in the source file.
    if "School Name" in df.columns and "School_Name" not in df.columns:
        df.rename(columns={"School Name": "School_Name"}, inplace=True)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=dayfirst).dt.date.astype(str)

    # Numeric coercion helper
    def _to_num(col: str):
        if col not in df.columns:
            return
        if pd.api.types.is_numeric_dtype(df[col]):
            return
        df[col] = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_core = [
        "Offered_Total", "Served_Total", "Discarded_Cost", "Production_Cost_Total",
        "Planned_Total", "Left_Over_Total", "Left_Over_Percent_of_Offered", "Subtotal_Cost",
        # Optional numeric fields you gave us
        "offered_safe", "waste_rate", "cost_per_served", "served_rate", "dow", "month",
    ]
    for c in numeric_core:
        _to_num(c)

    # Derived rates
    df["consumption_rate"] = (df["Served_Total"] / df["Offered_Total"].replace(0, np.nan)).fillna(0.0).clip(0, 1)
    if "served_rate" in df.columns:
        df["served_rate"] = df["served_rate"].fillna(df["consumption_rate"])
    else:
        df["served_rate"] = df["consumption_rate"]

    return df


def get_features(df: pd.DataFrame,
                 normalize: bool = True,
                 one_hot_categoricals: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str], Dict[str, float], Dict[str, float]]:
    """Create feature columns and return (df_enriched, X_cols, means, stds).
    - Adds one-hots for categoricals (default: Meal_Type if present).
    - Z-score normalizes numeric columns (keeps means/stds).
    """
    # Default numeric feature set
    base_numeric = [
        "Offered_Total", "Served_Total", "consumption_rate",
        "Production_Cost_Total", "Discarded_Cost", "Left_Over_Total",
        "Left_Over_Percent_of_Offered", "Subtotal_Cost",
    ]
    optional_numeric = [c for c in ["offered_safe", "waste_rate", "cost_per_served", "served_rate", "dow", "month"] if c in df.columns]
    feature_cols = base_numeric + optional_numeric

    # One-hot categoricals
    if one_hot_categoricals is None:
        one_hot_categoricals = [c for c in ["Meal_Type"] if c in df.columns]
    if one_hot_categoricals:
        dummies = [pd.get_dummies(df[c].astype("category"), prefix=c, dummy_na=False) for c in one_hot_categoricals]
        if dummies:
            dummies_df = pd.concat(dummies, axis=1)
            df = pd.concat([df, dummies_df], axis=1)
            feature_cols += list(dummies_df.columns)

    means, stds = {}, {}
    X_cols: List[str] = []
    for c in feature_cols:
        if c not in df.columns:
            continue
        if normalize and pd.api.types.is_numeric_dtype(df[c]):
            m, s = df[c].astype(float).mean(), df[c].astype(float).std(ddof=0)
            means[c] = m
            stds[c] = s if s > 1e-8 else 1.0
            df[c + "_norm"] = (df[c] - means[c]) / stds[c]
            X_cols.append(c + "_norm")
        else:
            X_cols.append(c)

    return df, X_cols, means, stds


def get_actions(df: pd.DataFrame,
                groupby_cols: Tuple[str, str] = ("School_Name", "Date"),
                min_actions: int = 2) -> List[pd.DataFrame]:
    """Group rows into decision rounds by (School_Name, Date) and filter for at least min_actions."""
    rounds = [g for _, g in df.groupby(list(groupby_cols), dropna=False) if len(g) >= min_actions]
    return rounds

@dataclass
class RewardWeights:
    # Higher discarded/production costs should reduce reward; consumption should increase reward
    w_production: float = 1.0
    w_discarded: float = 1.0
    w_consumption: float = 1.0


def compute_reward(row: pd.Series, rw: RewardWeights) -> float:
    """Whiteboard-aligned reward.
    Default: minimize cost & waste; reward consumption.
    R = -w_prod * Production_Cost_Total - w_disc * Discarded_Cost + w_cons * popularity
    """
    prod = float(row.get("Production_Cost_Total", 0.0))
    disc = float(row.get("Discarded_Cost", 0.0))
    pop = float(row.get("served_rate", row.get("consumption_rate", 0.0)))
    return float(-rw.w_production * prod - rw.w_discarded * disc + rw.w_consumption * pop)


# --------------------------
# Class wrapper env (compatible with model.py linUCB)
# --------------------------
class MealEnv:
    REQUIRED_COLS = [
        "School_Name", "Date", "Identifier", "Name",
        "Offered_Total", "Served_Total", "Discarded_Cost", "Production_Cost_Total",
        "Planned_Total", "Left_Over_Total", "Left_Over_Percent_of_Offered", "Subtotal_Cost"
    ]

    OPTIONAL_NUMERIC = [
        "offered_safe", "waste_rate", "cost_per_served", "served_rate", "dow", "month"
    ]
    OPTIONAL_CATEGORICAL = ["Meal_Type"]

    def __init__(
        self,
        df: pd.DataFrame,
        reward_weights: RewardWeights = RewardWeights(),
        min_actions_per_round: int = 2,
        feature_cols: Optional[List[str]] = None,
        normalize: bool = True,
        groupby_cols: Tuple[str, str] = ("School_Name", "Date"),
        one_hot_categoricals: Optional[List[str]] = None,
    ):
        self.df = df.copy()
        self.rw = reward_weights
        self.min_actions = min_actions_per_round
        self.groupby_cols = list(groupby_cols)
        self.feature_cols = feature_cols
        self.normalize = normalize
        self.one_hot_categoricals = one_hot_categoricals

        # Validate required cols
        missing = [c for c in self.REQUIRED_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure derived rates
        self.df["consumption_rate"] = (
            self.df["Served_Total"].astype(float) / self.df["Offered_Total"].replace(0, np.nan)
        ).fillna(0.0).clip(0, 1)
        if "served_rate" in self.df.columns:
            self.df["served_rate"] = self.df["served_rate"].astype(float).fillna(self.df["consumption_rate"])  # keep user's if provided
        else:
            self.df["served_rate"] = self.df["consumption_rate"]

        # Features
        if self.feature_cols is None:
            base_numeric = [
                "Offered_Total", "Served_Total", "consumption_rate",
                "Production_Cost_Total", "Discarded_Cost", "Left_Over_Total",
                "Left_Over_Percent_of_Offered", "Subtotal_Cost",
            ]
            extra_numeric = [c for c in self.OPTIONAL_NUMERIC if c in self.df.columns]
            self.feature_cols = base_numeric + extra_numeric

        # One-hots
        if self.one_hot_categoricals is None:
            self.one_hot_categoricals = [c for c in self.OPTIONAL_CATEGORICAL if c in self.df.columns]
        dummies = []
        if self.one_hot_categoricals:
            dummies = [pd.get_dummies(self.df[c].astype("category"), prefix=c, dummy_na=False) for c in self.one_hot_categoricals]
            dummies_df = pd.concat(dummies, axis=1) if len(dummies) else pd.DataFrame(index=self.df.index)
            self.df = pd.concat([self.df, dummies_df], axis=1)
            self.feature_cols += list(dummies_df.columns)

        # Normalize numeric features per-column (z-score) if requested
        self._means = {}
        self._stds = {}
        self.X_cols = []
        for c in self.feature_cols:
            if c not in self.df.columns:
                continue
            if self.normalize and pd.api.types.is_numeric_dtype(self.df[c]):
                m, s = self.df[c].astype(float).mean(), self.df[c].astype(float).std(ddof=0)
                self._means[c] = m
                self._stds[c] = s if s > 1e-8 else 1.0
                self.df[c + "_norm"] = (self.df[c] - self._means[c]) / self._stds[c]
                self.X_cols.append(c + "_norm")
            else:
                self.X_cols.append(c)

        # Decision rounds
        groups = self.df.groupby(self.groupby_cols, dropna=False)
        self.rounds: List[pd.DataFrame] = [g for _, g in groups if len(g) >= self.min_actions]
        self.n_rounds = len(self.rounds)
        self.ptr = 0

    # Whiteboard-style helpers on the env
    def get_actions(self) -> List[pd.DataFrame]:
        return self.rounds

    def get_features(self) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
        return self.X_cols, self._means, self._stds

    def health_score(self, row: pd.Series) -> float:
        return health_score(row)

    def reset(self, shuffle: bool = True):
        self.ptr = 0
        if shuffle:
            rng = np.random.default_rng()
            rng.shuffle(self.rounds)
        return self._get_observation()

    def _get_observation(self):
        if self.ptr >= self.n_rounds:
            return None
        cur = self.rounds[self.ptr]
        X = cur[self.X_cols].astype(float).to_numpy()  # shape: [K, D]
        action_ids = cur["Identifier"].astype(str).tolist()
        action_names = cur["Name"].astype(str).tolist()
        meta = cur[["School_Name", "Date", "Identifier", "Name"] + self.feature_cols + ["consumption_rate", "Discarded_Cost", "Production_Cost_Total"]]
        return X, action_ids, action_names, meta.reset_index(drop=True)

    def compute_reward(self, meta_row: pd.Series) -> float:
        return compute_reward(meta_row, self.rw)

    def step(self, action_index: int):
        if self.ptr >= self.n_rounds:
            return None, 0.0, True, {}
        cur = self.rounds[self.ptr].reset_index(drop=True)
        chosen_row = cur.iloc[action_index]
        reward = self.compute_reward(chosen_row)
        self.ptr += 1
        obs = self._get_observation()
        done = self.ptr >= self.n_rounds
        info = {
            "chosen_identifier": str(chosen_row["Identifier"]),
            "chosen_name": str(chosen_row["Name"]),
            "school": str(chosen_row["School_Name"]),
            "date": str(chosen_row["Date"]),
            "raw_reward_terms": {
                "production_cost": float(chosen_row.get("Production_Cost_Total", np.nan)),
                "discarded_cost": float(chosen_row.get("Discarded_Cost", np.nan)),
                "consumption_rate": float(chosen_row.get("consumption_rate", 0.0)),
            },
        }
        return obs, reward, done, info

d = get_data("data/Production Data.csv")
print(get_features(d))
print(get_actions(d))
