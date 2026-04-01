import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import Binarizer, StandardScaler


np.set_printoptions(precision=3, suppress=True)


def jaccard_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    intersection = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 0.0
    similarity = intersection / union
    return 1.0 - similarity


def main() -> None:
    iris = load_iris()
    features = iris.data

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    binary_features = Binarizer(threshold=0.0).fit_transform(scaled_features)
    jaccard_matrix = pairwise_distances(binary_features, metric=jaccard_distance)

    print("Jaccard distance matrix (first 5 samples):")
    print(jaccard_matrix[:5, :5])


if __name__ == "__main__":
    main()
