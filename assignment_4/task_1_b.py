from sklearn.datasets import load_iris
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.metrics import pairwise_distances
import numpy as np

# Set output to 3 decimal places
np.set_printoptions(precision=3, suppress=True)

# Load dataset
iris_dataset = load_iris()
x = iris_dataset.data
y = iris_dataset.target

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

def cosine_distance(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    similarity = dot_product / (norm_a * norm_b)
    return 1 - similarity   # distance

if __name__ == "__main__":
    # Cosine (on continuous scaled data)
    cosine_dist = pairwise_distances(x_scaled, metric=cosine_distance)
    print("\nCosine Distance (first 5 samples):\n", cosine_dist[:5, :5])