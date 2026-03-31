import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


np.set_printoptions(precision=3, suppress=True)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = np.dot(a, b) / (norm_a * norm_b)
    similarity = np.clip(similarity, -1.0, 1.0)
    return 1.0 - similarity


def main() -> None:
    iris = load_iris()
    scaled_features = StandardScaler().fit_transform(iris.data)
    cosine_matrix = pairwise_distances(scaled_features, metric=cosine_distance)

    print("Cosine distance matrix (first 5 samples):")
    print(cosine_matrix[:5, :5])


if __name__ == "__main__":
    main()
