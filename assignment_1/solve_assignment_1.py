from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


MOVIES_RAW = [
    {"title": "The Dark Knight", "category": "Action", "year": 2008, "rating": 4.7, "ratings_count": 2900000, "tags": "hero crime thriller dc", "source": "Public movie metadata"},
    {"title": "Mad Max: Fury Road", "category": "Action", "year": 2015, "rating": 4.4, "ratings_count": 1100000, "tags": "post-apocalyptic chase desert spectacle", "source": "Public movie metadata"},
    {"title": "Dune: Part Two", "category": "Sci-Fi", "year": 2024, "rating": 4.5, "ratings_count": 650000, "tags": "space epic prophecy sandworms", "source": "Public movie metadata"},
    {"title": "Interstellar", "category": "Sci-Fi", "year": 2014, "rating": 4.6, "ratings_count": 2300000, "tags": "space time family nasa", "source": "Public movie metadata"},
    {"title": "Everything Everywhere All at Once", "category": "Comedy", "year": 2022, "rating": 4.3, "ratings_count": 720000, "tags": "multiverse absurd heartfelt action", "source": "Public movie metadata"},
    {"title": "Barbie", "category": "Comedy", "year": 2023, "rating": 4.1, "ratings_count": 590000, "tags": "satire fantasy colorful doll", "source": "Public movie metadata"},
    {"title": "Poor Things", "category": "Comedy", "year": 2023, "rating": 4.0, "ratings_count": 310000, "tags": "surreal period feminist quirky", "source": "Public movie metadata"},
    {"title": "Inside Out 2", "category": "Comedy", "year": 2024, "rating": 4.2, "ratings_count": 240000, "tags": "animation family emotions sequel", "source": "Public movie metadata"},
    {"title": "Top Gun: Maverick", "category": "Adventure", "year": 2022, "rating": 4.3, "ratings_count": 690000, "tags": "aviation legacy adrenaline drama", "source": "Public movie metadata"},
    {"title": "Avatar: The Way of Water", "category": "Adventure", "year": 2022, "rating": 4.0, "ratings_count": 830000, "tags": "ocean spectacle franchise alien", "source": "Public movie metadata"},
    {"title": "Spider-Man: Across the Spider-Verse", "category": "Adventure", "year": 2023, "rating": 4.5, "ratings_count": 480000, "tags": "animation multiverse superhero vibrant", "source": "Public movie metadata"},
    {"title": "Mission: Impossible - Dead Reckoning", "category": "Adventure", "year": 2023, "rating": 4.0, "ratings_count": 320000, "tags": "spy stunts chase action", "source": "Public movie metadata"},
    {"title": "Deadpool & Wolverine", "category": "Action", "year": 2024, "rating": 4.1, "ratings_count": 410000, "tags": "marvel comedy antihero multiverse", "source": "Public movie metadata"},
    {"title": "John Wick: Chapter 4", "category": "Action", "year": 2023, "rating": 4.2, "ratings_count": 540000, "tags": "assassin stylized gun-fu revenge", "source": "Public movie metadata"},
    {"title": "The Super Mario Bros. Movie", "category": "Comedy", "year": 2023, "rating": 3.8, "ratings_count": 260000, "tags": "animation family game adaptation", "source": "Public movie metadata"},
    {"title": "Guardians of the Galaxy Vol. 3", "category": "Sci-Fi", "year": 2023, "rating": 4.2, "ratings_count": 430000, "tags": "marvel found-family space humor", "source": "Public movie metadata"},
    {"title": "Oppenheimer", "category": "Drama", "year": 2023, "rating": 4.4, "ratings_count": 950000, "tags": "biography history science thriller", "source": "Public movie metadata"},
    {"title": "Parasite", "category": "Drama", "year": 2019, "rating": 4.6, "ratings_count": 1050000, "tags": "satire class suspense korean", "source": "Public movie metadata"},
    {"title": "The Dark Knight", "category": "Action", "year": 2008, "rating": 4.7, "ratings_count": 2900000, "tags": "hero crime thriller dc", "source": "Public movie metadata"},
    {"title": "Unknown Indie Film", "category": "Drama", "year": 2024, "rating": None, "ratings_count": 3, "tags": "festival character study", "source": "Classroom placeholder"},
]


def prepare_dataset() -> pd.DataFrame:
    raw_df = pd.DataFrame(MOVIES_RAW)
    raw_df.to_csv(BASE_DIR / "raw_movie_catalog.csv", index=False)

    cleaned_df = raw_df.drop_duplicates(subset=["title", "year"]).copy()
    cleaned_df["rating"] = cleaned_df["rating"].fillna(cleaned_df["rating"].median())
    cleaned_df = cleaned_df.loc[cleaned_df["ratings_count"] >= 5].copy()
    cleaned_df["is_popular"] = (cleaned_df["rating"] >= 4.2) & (cleaned_df["ratings_count"] >= 250000)
    cleaned_df["is_recent"] = cleaned_df["year"] >= 2024
    cleaned_df["tag_count"] = cleaned_df["tags"].str.split().str.len()
    cleaned_df = cleaned_df.sort_values(["category", "rating", "ratings_count"], ascending=[True, False, False])
    cleaned_df.to_csv(BASE_DIR / "cleaned_movie_catalog.csv", index=False)
    return cleaned_df


def query_results(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    results = {
        "q1_comedy_and_rating_ge_4": df.loc[(df["category"] == "Comedy") & (df["rating"] >= 4.0)],
        "q2_is_popular_or_year_gt_2020": df.loc[df["is_popular"] | (df["year"] > 2020)],
        "q3_action_or_adventure_and_rating_ge_3_5_and_year_gt_2018": df.loc[
            (df["category"].isin(["Action", "Adventure"])) & (df["rating"] >= 3.5) & (df["year"] > 2018)
        ],
        "q4_scifi_and_rating_ge_4": df.loc[(df["category"] == "Sci-Fi") & (df["rating"] >= 4.0)],
        "q5_popular_and_recent": df.loc[df["is_popular"] & df["is_recent"]],
    }
    for name, result in results.items():
        result.to_csv(BASE_DIR / f"{name}.csv", index=False)
    return results


def evaluate_queries(df: pd.DataFrame, results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    relevant_mask = (df["rating"] >= 4.0) & (df["ratings_count"] >= 300000)
    evaluation_rows = []
    total_items = len(df)
    total_relevant = int(relevant_mask.sum())

    for name, result in results.items():
        if result.empty:
            precision = 0.0
            recall = 0.0
        else:
            relevant_retrieved = int(((result["rating"] >= 4.0) & (result["ratings_count"] >= 300000)).sum())
            precision = relevant_retrieved / len(result)
            recall = relevant_retrieved / total_relevant if total_relevant else 0.0

        evaluation_rows.append(
            {
                "query": name,
                "retrieved_items": len(result),
                "precision": round(precision, 3),
                "coverage": round(len(result) / total_items, 3),
                "recall_against_high_quality_items": round(recall, 3),
            }
        )

    evaluation_df = pd.DataFrame(evaluation_rows).sort_values("retrieved_items", ascending=False)
    evaluation_df.to_csv(BASE_DIR / "query_evaluation.csv", index=False)
    return evaluation_df


def build_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    rule_based = pd.concat(
        [
            df.loc[(df["category"] == "Sci-Fi") & (df["rating"] >= 4.0)].assign(rule="Top-rated Sci-Fi"),
            df.loc[df["is_popular"] & df["is_recent"]].assign(rule="Popular and recent"),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["title", "year", "rule"])

    rule_based = rule_based.sort_values(["rule", "rating", "ratings_count"], ascending=[True, False, False])
    rule_based.to_csv(BASE_DIR / "boolean_recommendations.csv", index=False)
    return rule_based


def compare_with_baselines(df: pd.DataFrame, boolean_recs: pd.DataFrame) -> pd.DataFrame:
    top_n = max(1, min(5, len(boolean_recs)))
    most_popular = df.sort_values(["ratings_count", "rating"], ascending=[False, False]).head(top_n)
    most_recent = df.sort_values(["year", "rating"], ascending=[False, False]).head(top_n)

    comparison = pd.DataFrame(
        [
            {
                "method": "Boolean rules",
                "items_considered": len(boolean_recs),
                "avg_rating": round(boolean_recs["rating"].mean(), 3),
                "avg_year": round(boolean_recs["year"].mean(), 1),
            },
            {
                "method": "Most popular baseline",
                "items_considered": len(most_popular),
                "avg_rating": round(most_popular["rating"].mean(), 3),
                "avg_year": round(most_popular["year"].mean(), 1),
            },
            {
                "method": "Most recent baseline",
                "items_considered": len(most_recent),
                "avg_rating": round(most_recent["rating"].mean(), 3),
                "avg_year": round(most_recent["year"].mean(), 1),
            },
        ]
    )
    comparison.to_csv(BASE_DIR / "baseline_comparison.csv", index=False)
    return comparison


def create_visualizations(results: dict[str, pd.DataFrame], evaluation_df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8")

    label_map = {
        "q1_comedy_and_rating_ge_4": "Q1 Comedy && rating >= 4.0",
        "q2_is_popular_or_year_gt_2020": "Q2 Popular OR year > 2020",
        "q3_action_or_adventure_and_rating_ge_3_5_and_year_gt_2018": "Q3 Action/Adventure nested query",
        "q4_scifi_and_rating_ge_4": "Q4 Sci-Fi && rating >= 4.0",
        "q5_popular_and_recent": "Q5 Popular && recent",
    }

    query_counts = pd.Series({label_map[name]: len(result) for name, result in results.items()}).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(query_counts.index, query_counts.values, color="#2f6f80")
    ax.set_title("Retrieved Items per Boolean Query")
    ax.set_xlabel("Item count")
    ax.set_ylabel("Query")
    ax.set_xlim(0, max(query_counts.values) + 2)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.2, bar.get_y() + bar.get_height() / 2, f"{int(width)}", va="center", fontsize=9)
    fig.tight_layout()
    plt.savefig(BASE_DIR / "task_2_query_counts.png", dpi=200)
    plt.close()

    eval_plot = evaluation_df.copy()
    eval_plot["query"] = eval_plot["query"].map(label_map)
    eval_plot = eval_plot.set_index("query")[["precision", "coverage"]]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    eval_plot.plot(kind="bar", ax=ax, color=["#8fb339", "#f2a541"], width=0.72)
    ax.set_title("Precision and Coverage of Boolean Retrieval")
    ax.set_ylabel("Score")
    ax.set_xlabel("Query")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=15)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
    fig.tight_layout()
    plt.savefig(BASE_DIR / "task_4_precision_coverage.png", dpi=200)
    plt.close()


def write_summary(df: pd.DataFrame, evaluation_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    lines = [
        "Assignment 1 Solution Summary",
        "",
        f"Cleaned dataset size: {len(df)} movies",
        f"Popular items: {int(df['is_popular'].sum())}",
        f"Recent items: {int(df['is_recent'].sum())}",
        "",
        "Best precision query:",
        evaluation_df.sort_values(["precision", "retrieved_items"], ascending=[False, False]).head(1).to_string(index=False),
        "",
        "Baseline comparison:",
        comparison_df.to_string(index=False),
    ]
    (BASE_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = prepare_dataset()
    results = query_results(df)
    evaluation_df = evaluate_queries(df, results)
    boolean_recs = build_recommendations(df)
    comparison_df = compare_with_baselines(df, boolean_recs)
    create_visualizations(results, evaluation_df)
    write_summary(df, evaluation_df, comparison_df)


if __name__ == "__main__":
    main()
