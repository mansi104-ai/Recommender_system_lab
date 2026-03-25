from sklearn.datasets import load_iris
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.metrics import pairwise_distances
import numpy as np
iris_dataset = load_iris()

x= iris_dataset.data
y = iris_dataset.target

scaler= StandardScaler()
x_scaled = scaler.fit_transform(x)

binarizer = Binarizer(threshold=0.0)
X_binary = binarizer.fit_transform(x_scaled)

#Compute jaccard similarity

def jaccard_binary(x, y):
    intersection = np.logical_and(x, y)
    union = np.logical_or(x, y)
    
    if union.sum() == 0:
        return 0.0   
    similarity = intersection.sum() / float(union.sum())
    return 1 - similarity

if __name__ == "__main__":
  jaccard_distance = pairwise_distances(X_binary, metric=jaccard_binary)
  print(jaccard_distance)
