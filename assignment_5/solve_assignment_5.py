from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR.parent / "assignment_4" / "ratings_small.csv"
MIN_USER_INTERACTIONS = 10
MIN_ITEM_INTERACTIONS = 5


def compute_gini(values: pd.Series) -> float:
    array = np.sort(values.to_numpy(dtype=float))
    if len(array) == 0:
        return 0.0
    if np.allclose(array.sum(), 0.0):
        return 0.0
    index = np.arange(1, len(array) + 1)
    return float((2 * np.sum(index * array)) / (len(array) * np.sum(array)) - (len(array) + 1) / len(array))


def interaction_filter(
    ratings: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pd.DataFrame:
    user_counts = ratings["userId"].value_counts()
    item_counts = ratings["movieId"].value_counts()
    keep_users = user_counts[user_counts >= min_user_interactions].index
    keep_items = item_counts[item_counts >= min_item_interactions].index
    return ratings[
        ratings["userId"].isin(keep_users) & ratings["movieId"].isin(keep_items)
    ].copy()


def load_and_preprocess() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    ratings = pd.read_csv(SOURCE_FILE)
    initial_rows = len(ratings)
    duplicate_rows = int(ratings.duplicated().sum())

    ratings = ratings.drop_duplicates().copy()
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], errors="coerce")

    missing_before = ratings.isna().sum().to_dict()
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()
    ratings["timestamp"] = ratings["timestamp"].fillna(ratings["timestamp"].min())

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)

    filtered = interaction_filter(
        ratings,
        min_user_interactions=MIN_USER_INTERACTIONS,
        min_item_interactions=MIN_ITEM_INTERACTIONS,
    )

    filtered["rating_year"] = filtered["timestamp"].dt.year
    filtered["rating_month"] = filtered["timestamp"].dt.month
    filtered["day_name"] = filtered["timestamp"].dt.day_name()
    filtered["is_weekend"] = filtered["timestamp"].dt.dayofweek >= 5
    filtered["rating_bucket"] = pd.cut(
        filtered["rating"],
        bins=[0, 2, 3, 4, 5],
        labels=["Low (<=2)", "Mid (2-3]", "Good (3-4]", "High (4-5]"],
        include_lowest=True,
    )

    user_stats = (
        filtered.groupby("userId")
        .agg(
            user_rating_count=("rating", "size"),
            user_mean_rating=("rating", "mean"),
            user_rating_std=("rating", "std"),
        )
        .fillna(0)
    )
    item_stats = (
        filtered.groupby("movieId")
        .agg(
            item_rating_count=("rating", "size"),
            item_mean_rating=("rating", "mean"),
            item_rating_std=("rating", "std"),
        )
        .fillna(0)
    )

    filtered = filtered.merge(user_stats, on="userId", how="left")
    filtered = filtered.merge(item_stats, on="movieId", how="left")

    filtered["user_activity_segment"] = pd.qcut(
        filtered["user_rating_count"].rank(method="first"),
        q=min(3, filtered["userId"].nunique()),
        labels=["Low activity", "Medium activity", "High activity"][: min(3, filtered["userId"].nunique())],
    )
    filtered["item_popularity_segment"] = pd.qcut(
        filtered["item_rating_count"].rank(method="first"),
        q=min(3, filtered["movieId"].nunique()),
        labels=["Tail", "Middle", "Head"][: min(3, filtered["movieId"].nunique())],
    )

    preprocessing_summary = {
        "initial_rows": initial_rows,
        "rows_after_dedup": len(ratings),
        "duplicates_removed": duplicate_rows,
        "missing_userId": int(missing_before.get("userId", 0)),
        "missing_movieId": int(missing_before.get("movieId", 0)),
        "missing_rating": int(missing_before.get("rating", 0)),
        "missing_timestamp": int(missing_before.get("timestamp", 0)),
        "rows_after_filtering": len(filtered),
        "users_after_filtering": int(filtered["userId"].nunique()),
        "items_after_filtering": int(filtered["movieId"].nunique()),
    }

    return ratings, filtered, preprocessing_summary


def compute_matrix_metrics(filtered: pd.DataFrame) -> dict[str, float]:
    user_item = filtered.pivot_table(index="userId", columns="movieId", values="rating", aggfunc="mean")
    observed = int(user_item.notna().sum().sum())
    total_possible = int(user_item.shape[0] * user_item.shape[1])
    sparsity = 1 - (observed / total_possible) if total_possible else 0.0

    user_activity = filtered.groupby("userId").size()
    item_popularity = filtered.groupby("movieId").size()

    return {
        "observed_interactions": observed,
        "total_possible_interactions": total_possible,
        "sparsity": float(sparsity),
        "user_activity_gini": compute_gini(user_activity),
        "item_popularity_gini": compute_gini(item_popularity),
    }


def build_groupwise_tables(filtered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    activity_summary = (
        filtered.groupby("user_activity_segment", observed=False)
        .agg(
            users=("userId", "nunique"),
            avg_rating=("rating", "mean"),
            avg_user_interactions=("user_rating_count", "mean"),
            avg_item_popularity_seen=("item_rating_count", "mean"),
        )
        .reset_index()
    )

    popularity_summary = (
        filtered.groupby("item_popularity_segment", observed=False)
        .agg(
            items=("movieId", "nunique"),
            avg_rating=("rating", "mean"),
            avg_item_interactions=("item_rating_count", "mean"),
            avg_user_activity=("user_rating_count", "mean"),
        )
        .reset_index()
    )

    return activity_summary, popularity_summary


def classify_head_tail(filtered: pd.DataFrame) -> pd.DataFrame:
    item_counts = (
        filtered.groupby("movieId")
        .size()
        .sort_values(ascending=False)
        .rename("interaction_count")
        .reset_index()
    )
    item_counts["interaction_share"] = item_counts["interaction_count"] / item_counts["interaction_count"].sum()
    item_counts["cumulative_share"] = item_counts["interaction_share"].cumsum()
    item_counts["long_tail_group"] = np.where(item_counts["cumulative_share"] <= 0.8, "Head", "Tail")
    return item_counts


def create_visualizations(filtered: pd.DataFrame, metrics: dict[str, float], long_tail_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.histplot(filtered["rating"], bins=9, kde=True, color="#2f6f80", ax=axes[0, 0])
    axes[0, 0].set_title("Rating Distribution")
    axes[0, 0].set_xlabel("Rating")

    user_activity = filtered.groupby("userId").size().sort_values(ascending=False)
    sns.barplot(x=user_activity.index.astype(str), y=user_activity.values, color="#8fb339", ax=axes[0, 1])
    axes[0, 1].set_title("User Activity")
    axes[0, 1].set_xlabel("User ID")
    axes[0, 1].set_ylabel("Number of ratings")

    item_popularity = filtered.groupby("movieId").size().sort_values(ascending=False).head(15)
    sns.barplot(x=item_popularity.index.astype(str), y=item_popularity.values, color="#f2a541", ax=axes[1, 0])
    axes[1, 0].set_title("Top 15 Item Popularity")
    axes[1, 0].set_xlabel("Movie ID")
    axes[1, 0].set_ylabel("Number of ratings")
    axes[1, 0].tick_params(axis="x", rotation=45)

    gini_df = pd.DataFrame(
        {
            "measure": ["User activity Gini", "Item popularity Gini"],
            "value": [metrics["user_activity_gini"], metrics["item_popularity_gini"]],
        }
    )
    sns.barplot(data=gini_df, x="measure", y="value", hue="measure", palette=["#d95d39", "#5c80bc"], legend=False, ax=axes[1, 1])
    axes[1, 1].set_title("Data Inequality via Gini Index")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Gini value")
    axes[1, 1].set_ylim(0, 1)

    fig.tight_layout()
    plt.savefig(BASE_DIR / "eda_overview.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    user_item = filtered.pivot_table(index="userId", columns="movieId", values="rating", aggfunc="mean")
    sns.heatmap(user_item.notna(), cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"User-Item Interaction Matrix\nSparsity = {metrics['sparsity']:.2%}")
    ax.set_xlabel("Movie ID")
    ax.set_ylabel("User ID")
    fig.tight_layout()
    plt.savefig(BASE_DIR / "user_item_sparsity_heatmap.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.countplot(data=filtered, x="rating_year", color="#6a994e", ax=ax)
    ax.set_title("Ratings by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Ratings count")
    fig.tight_layout()
    plt.savefig(BASE_DIR / "ratings_by_year.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=long_tail_df.head(25), x="movieId", y="interaction_count", hue="long_tail_group", dodge=False, ax=ax)
    ax.set_title("Long-Tail Distribution of Item Popularity")
    ax.set_xlabel("Movie ID")
    ax.set_ylabel("Interaction count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    plt.savefig(BASE_DIR / "long_tail_distribution.png", dpi=200)
    plt.close(fig)


def save_outputs(
    original: pd.DataFrame,
    filtered: pd.DataFrame,
    preprocessing_summary: dict[str, float],
    metrics: dict[str, float],
    activity_summary: pd.DataFrame,
    popularity_summary: pd.DataFrame,
    long_tail_df: pd.DataFrame,
) -> None:
    original.to_csv(BASE_DIR / "ratings_deduplicated.csv", index=False)
    filtered.to_csv(BASE_DIR / "ratings_filtered_featured.csv", index=False)
    activity_summary.to_csv(BASE_DIR / "groupwise_user_activity_summary.csv", index=False)
    popularity_summary.to_csv(BASE_DIR / "groupwise_item_popularity_summary.csv", index=False)
    long_tail_df.to_csv(BASE_DIR / "long_tail_head_tail_items.csv", index=False)

    summary_lines = [
        "Assignment 5: Explore Non-Personalized Data and Features",
        "",
        f"Source dataset: {SOURCE_FILE}",
        f"Initial interactions: {preprocessing_summary['initial_rows']}",
        f"Duplicates removed: {preprocessing_summary['duplicates_removed']}",
        f"Rows after filtering: {preprocessing_summary['rows_after_filtering']}",
        f"Users after filtering: {preprocessing_summary['users_after_filtering']}",
        f"Items after filtering: {preprocessing_summary['items_after_filtering']}",
        f"Sparsity: {metrics['sparsity']:.4f}",
        f"User activity Gini index: {metrics['user_activity_gini']:.4f}",
        f"Item popularity Gini index: {metrics['item_popularity_gini']:.4f}",
        f"Head items: {int((long_tail_df['long_tail_group'] == 'Head').sum())}",
        f"Tail items: {int((long_tail_df['long_tail_group'] == 'Tail').sum())}",
    ]
    (BASE_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    report_lines = [
        "# Assignment 5 Report",
        "",
        "## Objective",
        "Explore the ratings dataset to understand data quality, interaction structure, sparsity, inequality, and long-tail behavior relevant to recommender-system design.",
        "",
        "## Preprocessing",
        f"- Loaded `{SOURCE_FILE.name}` from `assignment_4`.",
        "- Removed duplicate rows.",
        "- Converted timestamps to datetime format.",
        "- Dropped rows with missing user, item, or rating values and filled missing timestamps with the earliest available timestamp.",
        f"- Applied a one-pass filter to remove users with fewer than {MIN_USER_INTERACTIONS} ratings and items with fewer than {MIN_ITEM_INTERACTIONS} ratings based on original interaction counts.",
        "",
        "## Feature Engineering",
        "- Extracted rating year, month, day name, and weekend flag from the timestamp.",
        "- Created rating buckets for low, mid, good, and high ratings.",
        "- Added user-level features: rating count, mean rating, rating standard deviation.",
        "- Added item-level features: rating count, mean rating, rating standard deviation.",
        "- Derived user activity segments and item popularity segments for group-wise comparison.",
        "",
        "## Key Findings",
        f"- Filtered dataset contains {preprocessing_summary['rows_after_filtering']} interactions, {preprocessing_summary['users_after_filtering']} users, and {preprocessing_summary['items_after_filtering']} items.",
        f"- User-item matrix sparsity is `{metrics['sparsity']:.2%}`, indicating a sparse interaction space.",
        f"- User activity Gini index is `{metrics['user_activity_gini']:.4f}`, showing how unevenly user activity is distributed.",
        f"- Item popularity Gini index is `{metrics['item_popularity_gini']:.4f}`, showing how concentrated interactions are among a smaller set of movies.",
        f"- Long-tail split identifies {int((long_tail_df['long_tail_group'] == 'Head').sum())} head items and {int((long_tail_df['long_tail_group'] == 'Tail').sum())} tail items using the 80% cumulative interaction rule.",
        "",
        "## Output Files",
        "- `solve_assignment_5.py`: main solution script.",
        "- `ratings_filtered_featured.csv`: cleaned and feature-engineered interaction data.",
        "- `groupwise_user_activity_summary.csv`: user-segment comparison table.",
        "- `groupwise_item_popularity_summary.csv`: item-segment comparison table.",
        "- `long_tail_head_tail_items.csv`: head vs tail classification.",
        "- `eda_overview.png`, `user_item_sparsity_heatmap.png`, `ratings_by_year.png`, `long_tail_distribution.png`: generated plots.",
    ]
    (BASE_DIR / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    original, filtered, preprocessing_summary = load_and_preprocess()
    metrics = compute_matrix_metrics(filtered)
    activity_summary, popularity_summary = build_groupwise_tables(filtered)
    long_tail_df = classify_head_tail(filtered)
    create_visualizations(filtered, metrics, long_tail_df)
    save_outputs(
        original,
        filtered,
        preprocessing_summary,
        metrics,
        activity_summary,
        popularity_summary,
        long_tail_df,
    )

    print("Assignment 5 solution completed.")
    print(f"Filtered interactions: {preprocessing_summary['rows_after_filtering']}")
    print(f"Users: {preprocessing_summary['users_after_filtering']}, Items: {preprocessing_summary['items_after_filtering']}")
    print(f"Sparsity: {metrics['sparsity']:.2%}")
    print(f"User Gini: {metrics['user_activity_gini']:.4f}")
    print(f"Item Gini: {metrics['item_popularity_gini']:.4f}")


if __name__ == "__main__":
    main()
