from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer, StandardScaler


matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parent
PLOT_PATH = BASE_DIR / "task_2_accuracy_comparison.png"
SIMILARITY_PATH = BASE_DIR / "task_2_cosine_similarity_matrix.png"


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    similarity = np.dot(a, b) / denominator
    similarity = np.clip(similarity, -1.0, 1.0)
    return 1.0 - similarity


def jaccard_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    intersection = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    similarity = intersection / union if union else 0.0
    return 1.0 - similarity


def knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test_row: np.ndarray,
    k: int,
    distance_func,
) -> int:
    distances = np.array([distance_func(x_test_row, row) for row in x_train])
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_indices]
    return Counter(nearest_labels).most_common(1)[0][0]


def plot_accuracies(k_values: list[int], cosine_scores: list[float], jaccard_scores: list[float]) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, cosine_scores, marker="o", label="Cosine")
    plt.plot(k_values, jaccard_scores, marker="s", label="Jaccard")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy for Cosine vs Jaccard Distance")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()


def plot_similarity_matrix(similarity_matrix: np.ndarray) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(similarity_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Cosine similarity")
    plt.title("Training-set Cosine Similarity Matrix")
    plt.tight_layout()
    plt.savefig(SIMILARITY_PATH)
    plt.close()


def main() -> None:
    wine = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(
        wine.data,
        wine.target,
        test_size=0.2,
        random_state=42,
        stratify=wine.target,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    binarizer = Binarizer(threshold=0.0)
    x_train_binary = binarizer.fit_transform(x_train_scaled)
    x_test_binary = binarizer.transform(x_test_scaled)

    k_values = list(range(1, 7))
    cosine_scores = []
    jaccard_scores = []

    for k in k_values:
        cosine_predictions = [
            knn_predict(x_train_scaled, y_train, row, k, cosine_distance)
            for row in x_test_scaled
        ]
        jaccard_predictions = [
            knn_predict(x_train_binary, y_train, row, k, jaccard_distance)
            for row in x_test_binary
        ]

        cosine_scores.append(accuracy_score(y_test, cosine_predictions))
        jaccard_scores.append(accuracy_score(y_test, jaccard_predictions))

    similarity_matrix = 1 - pairwise_distances(x_train_scaled, metric="cosine")

    plot_accuracies(k_values, cosine_scores, jaccard_scores)
    plot_similarity_matrix(similarity_matrix)

    for k, cosine_score, jaccard_score in zip(k_values, cosine_scores, jaccard_scores):
        print(
            f"k={k}: cosine_accuracy={cosine_score:.3f}, "
            f"jaccard_accuracy={jaccard_score:.3f}"
        )

    print(f"Saved accuracy plot to {PLOT_PATH.name}")
    print(f"Saved similarity matrix to {SIMILARITY_PATH.name}")


if __name__ == "__main__":
    main()
