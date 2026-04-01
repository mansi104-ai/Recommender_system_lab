from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
RATINGS_FILE = BASE_DIR / "ratings_small.csv"
RANDOM_STATE = 42
TOP_K = 5
NEIGHBOR_COUNT = 5


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        fill_value=0.0,
    )


def jaccard_similarity(binary_matrix: pd.DataFrame) -> np.ndarray:
    matrix_values = binary_matrix.to_numpy(dtype=np.int8)
    intersection = matrix_values @ matrix_values.T
    row_sums = matrix_values.sum(axis=1, keepdims=True)
    union = row_sums + row_sums.T - intersection
    similarity = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union != 0,
    )
    return similarity


def hybrid_similarity(cosine_sim: np.ndarray, jaccard_sim: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * cosine_sim + (1.0 - alpha) * jaccard_sim


def recommend_movies(
    user_id: int,
    train_matrix: pd.DataFrame,
    similarity_matrix: np.ndarray,
    k: int = TOP_K,
    neighbor_count: int = NEIGHBOR_COUNT,
) -> list[int]:
    if user_id not in train_matrix.index:
        return []

    user_index = train_matrix.index.get_loc(user_id)
    similarity_scores = similarity_matrix[user_index].copy()
    similarity_scores[user_index] = -np.inf
    neighbor_indices = np.argsort(similarity_scores)[::-1][:neighbor_count]

    neighbor_ids = train_matrix.index[neighbor_indices]
    user_ratings = train_matrix.loc[user_id]
    unseen_movies = user_ratings[user_ratings == 0].index
    movie_scores: dict[int, float] = {}

    for movie_id in unseen_movies:
        weighted_sum = 0.0
        similarity_sum = 0.0

        for neighbor_index, neighbor_id in zip(neighbor_indices, neighbor_ids):
            neighbor_rating = train_matrix.at[neighbor_id, movie_id]
            if neighbor_rating <= 0:
                continue

            similarity = similarity_matrix[user_index, neighbor_index]
            if similarity <= 0:
                continue

            weighted_sum += similarity * neighbor_rating
            similarity_sum += similarity

        if similarity_sum > 0:
            movie_scores[movie_id] = weighted_sum / similarity_sum

    ranked_movies = sorted(movie_scores.items(), key=lambda item: item[1], reverse=True)
    return [movie_id for movie_id, _ in ranked_movies[:k]]


def precision_at_k(
    user_id: int,
    train_matrix: pd.DataFrame,
    test_matrix: pd.DataFrame,
    similarity_matrix: np.ndarray,
    k: int = TOP_K,
) -> float:
    if user_id not in test_matrix.index:
        return 0.0

    recommended_movies = recommend_movies(user_id, train_matrix, similarity_matrix, k=k)
    if not recommended_movies:
        return 0.0

    relevant_movies = test_matrix.loc[user_id]
    relevant_movies = relevant_movies[relevant_movies >= 3.0].index

    if len(relevant_movies) == 0:
        return 0.0

    hits = len(set(recommended_movies) & set(relevant_movies))
    return hits / k


def evaluate_similarity(
    user_ids: list[int],
    train_matrix: pd.DataFrame,
    test_matrix: pd.DataFrame,
    similarity_matrix: np.ndarray,
    k: int = TOP_K,
) -> float:
    scores = [
        precision_at_k(user_id, train_matrix, test_matrix, similarity_matrix, k=k)
        for user_id in user_ids
    ]
    return float(np.mean(scores)) if scores else 0.0


def main() -> None:
    ratings = pd.read_csv(RATINGS_FILE)
    train_ratings, test_ratings = train_test_split(
        ratings,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    train_matrix = build_user_item_matrix(train_ratings)
    test_matrix = (
        test_ratings.pivot_table(index="userId", columns="movieId", values="rating")
        .reindex(index=train_matrix.index, columns=train_matrix.columns, fill_value=0.0)
        .fillna(0.0)
    )

    cosine_sim = cosine_similarity(train_matrix)
    binary_train_matrix = (train_matrix > 0).astype(int)
    jaccard_sim = jaccard_similarity(binary_train_matrix)

    sample_users = train_matrix.index[: min(20, len(train_matrix.index))].tolist()
    alphas = np.linspace(0.0, 1.0, 6)
    results = []

    print("Alpha optimization results:")
    for alpha in alphas:
        similarity_matrix = hybrid_similarity(cosine_sim, jaccard_sim, alpha)
        average_precision = evaluate_similarity(sample_users, train_matrix, test_matrix, similarity_matrix)
        results.append((alpha, average_precision))
        print(f"alpha={alpha:.2f}, precision@{TOP_K}={average_precision:.4f}")

    best_alpha, best_precision = max(results, key=lambda item: item[1])
    print(f"\nBest alpha: {best_alpha:.2f} with precision@{TOP_K}={best_precision:.4f}")

    n_components = max(1, min(10, min(train_matrix.shape) - 1))
    reduced_train_matrix = PCA(n_components=n_components, random_state=RANDOM_STATE).fit_transform(train_matrix)
    cosine_sim_pca = cosine_similarity(reduced_train_matrix)

    comparison_methods = {
        "Cosine": cosine_sim,
        "Jaccard": jaccard_sim,
        "Hybrid": hybrid_similarity(cosine_sim, jaccard_sim, best_alpha),
        "Cosine + PCA": cosine_sim_pca,
    }

    print("\nMethod comparison:")
    for method_name, similarity_matrix in comparison_methods.items():
        average_precision = evaluate_similarity(sample_users, train_matrix, test_matrix, similarity_matrix)
        print(f"{method_name}: precision@{TOP_K}={average_precision:.4f}")


if __name__ == "__main__":
    main()
