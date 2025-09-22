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



d = get_data("data/Production Data.csv")
print(get_features(d))
print(get_actions(d))
