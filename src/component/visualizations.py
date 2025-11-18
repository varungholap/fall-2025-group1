import os
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _rolling_mean(values: List[float], window: int = 50) -> List[float]:
    s = pd.Series(values, dtype=float)
    if len(s) == 0:
        return []
    return s.rolling(window=window, min_periods=max(1, window // 5)).mean().tolist()

def _safe_dt(x):
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT

# -----------------------------
# Baseline simulator (no training change needed)
# -----------------------------
def simulate_random_baseline(env_rounds_with_meta: List) -> Dict[str, List[float]]:
    """
    Create per-round history for a Random baseline, matching your RandomPolicy.compute_reward logic.
    This avoids modifying RandomPolicy itself.
    """
    rewards = []
    cum_rewards = []
    total = 0.0
    rng = np.random.default_rng(42)

    for X, _ in env_rounds_with_meta:
        mask = np.any(X, axis=1)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            choice = rng.integers(X.shape[0])
        else:
            choice = int(rng.choice(idxs))
        x = X[choice]

        planned_total = x[0] if len(x) > 0 else 0.0
        served_total = x[1] if len(x) > 1 else 0.0
        reward = 0.0 if planned_total == 0 else float(served_total / planned_total)

        total += reward
        rewards.append(reward)
        cum_rewards.append(total)

    return {
        "reward": rewards,
        "cumulative_reward": cum_rewards
    }

# -----------------------------
# Plot helpers (one figure per chart)
# -----------------------------
def _plot_two_series(x, y1, y2, xlabel, ylabel, label1, label2, outpath, title=None):
    plt.figure()
    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def _plot_one_series(x, y, xlabel, ylabel, outpath, title=None):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def _plot_bar(labels, values, xlabel, ylabel, outpath, title=None, horizontal=False):
    plt.figure()
    if horizontal:
        plt.barh(labels, values)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    else:
        plt.bar(labels, values)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def _plot_boxplot(groups_dict, xlabel, ylabel, outpath, title=None):
    if not groups_dict:
        return
    labels = list(groups_dict.keys())
    data = [groups_dict[k] for k in labels]
    plt.figure()
    plt.boxplot(data, labels=labels, vert=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

# -----------------------------
# Public API: generate all plots
# -----------------------------
def generate_all_plots(
    env_rounds: List[np.ndarray],
    metadata: List[dict],
    history_linucb: Dict[str, List[float]],
    rnd_hist: Dict[str, List[float]],
    recommendations_csv_path: str,
    out_dir: str = None
):
    """
    Build all performance, diagnostics, and recommendation analytics charts using
    the already-computed artifacts from main.
    """
    # reports/plots next to the recommendations.csv by default
    if out_dir is None:
        reports_dir = os.path.dirname(recommendations_csv_path) if recommendations_csv_path else "reports"
        out_dir = os.path.join(reports_dir, "plots")
    _ensure_dir(out_dir)

    # rounds index
    rounds = list(range(1, len(history_linucb.get("round", [])) + 1))

    # A1. Cumulative Reward Over Time: LinUCB vs Random
    if rounds and "cumulative_reward" in history_linucb and "cumulative_reward" in rnd_hist:
        _plot_two_series(
            rounds,
            history_linucb["cumulative_reward"],
            rnd_hist["cumulative_reward"],
            xlabel="Round",
            ylabel="Cumulative Reward",
            label1="LinUCB",
            label2="Random",
            outpath=os.path.join(out_dir, "A1_cumulative_reward.png"),
            title="Cumulative Reward Over Time"
        )

    # A2. Cumulative Regret Over Time (LinUCB)
    if rounds and "cumulative_regret" in history_linucb:
        _plot_one_series(
            rounds,
            history_linucb["cumulative_regret"],
            xlabel="Round",
            ylabel="Cumulative Regret",
            outpath=os.path.join(out_dir, "A2_cumulative_regret.png"),
            title="Cumulative Regret Over Time (LinUCB)"
        )

    # A3. Average Reward per Round (Rolling)
    if rounds and "reward" in history_linucb:
        roll = _rolling_mean(history_linucb["reward"], window=50)
        _plot_one_series(
            rounds[:len(roll)],
            roll,
            xlabel="Round",
            ylabel="Rolling Avg Reward (~50)",
            outpath=os.path.join(out_dir, "A3_rolling_avg_reward.png"),
            title="Rolling Average Reward (LinUCB)"
        )

    # B1. Uncertainty over time
    if rounds and "uncertainty" in history_linucb:
        _plot_one_series(
            rounds,
            history_linucb["uncertainty"],
            xlabel="Round",
            ylabel="Average Uncertainty",
            outpath=os.path.join(out_dir, "B1_uncertainty.png"),
            title="Uncertainty Over Time (LinUCB)"
        )

    # B2. Arm Selection Frequency
    if "chosen_arm" in history_linucb and len(history_linucb["chosen_arm"]) > 0:
        counts = Counter(history_linucb["chosen_arm"])
        topN = 30 if len(counts) > 30 else len(counts)
        idxs, vals = zip(*counts.most_common(topN)) if counts else ([], [])
        _plot_bar(
            labels=[str(i) for i in idxs],
            values=vals,
            xlabel="Arm Index",
            ylabel="Times Chosen",
            outpath=os.path.join(out_dir, "B2_arm_selection_freq.png"),
            title=f"Arm Selection Frequency (Top {topN})"
        )

    # C. Recommendations analytics
    if recommendations_csv_path and os.path.exists(recommendations_csv_path):
        rec = pd.read_csv(recommendations_csv_path)
        if "Date" in rec.columns:
            rec["Date"] = rec["Date"].apply(_safe_dt)

        # C1. Top Recommended Dishes
        if "Recommended_Dish" in rec.columns:
            top_dish_counts = rec["Recommended_Dish"].value_counts().head(20)
            _plot_bar(
                labels=list(top_dish_counts.index),
                values=top_dish_counts.values,
                xlabel="Dish",
                ylabel="Count",
                outpath=os.path.join(out_dir, "C1_top_recommended_dishes.png"),
                title="Top Recommended Dishes",
                horizontal=True
            )

        # C2. Predicted Scores of Top Dishes (boxplot)
        if {"Recommended_Dish", "Predicted_Score"}.issubset(rec.columns):
            top10 = set(rec["Recommended_Dish"].value_counts().head(10).index)
            groups = defaultdict(list)
            for _, row in rec.iterrows():
                name = row.get("Recommended_Dish")
                if name in top10:
                    groups[name].append(float(row.get("Predicted_Score", 0.0)))
            _plot_boxplot(
                groups,
                xlabel="Dish (Top 10 by frequency)",
                ylabel="Predicted Score",
                outpath=os.path.join(out_dir, "C2_predicted_scores_boxplot.png"),
                title="Predicted Score Distribution (Top Dishes)"
            )



        # C3. Top dishes per School (one chart per school)
        if {"School", "Recommended_Dish"}.issubset(rec.columns):
            top_by_school = (
                rec.groupby(["School", "Recommended_Dish"]).size()
                .reset_index(name="count")
                .sort_values(["School", "count"], ascending=[True, False])
            )
            for school, grp in top_by_school.groupby("School"):
                top = grp.head(10)
                _plot_bar(
                    labels=top["Recommended_Dish"].tolist(),
                    values=top["count"].tolist(),
                    xlabel="Dish",
                    ylabel="Count",
                    outpath=os.path.join(out_dir, f"C3_top_dishes_{str(school).replace(' ', '_')}.png"),
                    title=f"Top Recommended Dishes — {school}",
                    horizontal=True
                )

        # C4. Predicted Score distribution by School (boxplot for top-10 by count)
        if {"School", "Predicted_Score"}.issubset(rec.columns):
            score_by_school = defaultdict(list)
            for _, row in rec.iterrows():
                score_by_school[row.get("School", "Unknown")].append(float(row.get("Predicted_Score", 0.0)))
            sizes = {k: len(v) for k, v in score_by_school.items()}
            top_schools = sorted(sizes, key=sizes.get, reverse=True)[:10]
            score_by_school_top = {k: score_by_school[k] for k in top_schools}
            _plot_boxplot(
                score_by_school_top,
                xlabel="School (Top 10 by recommendations)",
                ylabel="Predicted Score",
                outpath=os.path.join(out_dir, "C4_score_by_school.png"),
                title="Predicted Score Distribution by School"
            )

        # C5. Diversity: unique dishes per month
        if "_month" in rec.columns and "Recommended_Dish" in rec.columns:
            uniq_per_month = rec.groupby("_month")["Recommended_Dish"].nunique().reset_index()
            _plot_one_series(
                x=range(len(uniq_per_month)),
                y=uniq_per_month["Recommended_Dish"].tolist(),
                xlabel="Month Index (sorted)",
                ylabel="Unique Dishes Recommended",
                outpath=os.path.join(out_dir, "C5_unique_dishes_per_month.png"),
                title="Diversity of Recommendations Over Time"
            )

        # C6. Rank analysis per dish (top 10 dishes)
        if {"Recommended_Dish", "Rank"}.issubset(rec.columns):
            top10_names = set(rec["Recommended_Dish"].value_counts().head(10).index)
            rank_counts = (
                rec.groupby(["Recommended_Dish", "Rank"]).size()
                .reset_index(name="count")
            )
            rank_counts = rank_counts[rank_counts["Recommended_Dish"].isin(top10_names)]
            for dish, grp in rank_counts.groupby("Recommended_Dish"):
                ranks = sorted(grp["Rank"].unique())
                counts = grp.set_index("Rank")["count"].reindex(ranks, fill_value=0)
                _plot_bar(
                    labels=[str(r) for r in ranks],
                    values=counts.values.tolist(),
                    xlabel="Rank",
                    ylabel="Count",
                    outpath=os.path.join(out_dir, f"C6_rank_mix_{dish}.png"),
                    title=f"Rank Mix for {dish}"
                )

    print(f"Charts saved to: {out_dir}")
