import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.metrics import accuracy_score, pairwise_distances
import matplotlib.pyplot as plt
from collections import Counter

# Cosine distance
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Jaccard distance
def jaccard_distance(a, b):
    a, b = a.astype(bool), b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1 - intersection / union if union != 0 else 0

# KNN
def knn_predict(X_train, y_train, x_test, k, distance_func):
    distances = [distance_func(x_test, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    return Counter(k_labels).most_common(1)[0][0]

# Load data
wine = load_wine()
X, y = wine.data, wine.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale for cosine
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

binarizer = Binarizer(threshold=0.0)
X_train_bin = binarizer.fit_transform(X_train_scaled)
X_test_bin = binarizer.transform(X_test_scaled)

k_values = range(1, 7)
cos_acc = []
jac_acc = []

for k in k_values:
    # Cosine
    y_pred_cos = [knn_predict(X_train_scaled, y_train, x, k, cosine_distance) for x in X_test_scaled]
    cos_acc.append(accuracy_score(y_test, y_pred_cos))

    # Jaccard
    y_pred_jac = [knn_predict(X_train_bin, y_train, x, k, jaccard_distance) for x in X_test_bin]
    jac_acc.append(accuracy_score(y_test, y_pred_jac))
plt.plot(k_values, cos_acc, label="Cosine")
plt.plot(k_values, jac_acc, label="Jaccard")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.legend()
plt.title("K vs Accuracy")
plt.show()

sim_matrix = 1 - pairwise_distances(X_train_scaled, metric='cosine')
plt.imshow(sim_matrix)
plt.colorbar()
plt.title("Cosine Similarity Matrix")
plt.show()