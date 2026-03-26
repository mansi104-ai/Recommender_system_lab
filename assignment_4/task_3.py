import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
ratings = pd.read_csv("ratings_small.csv")

train, test = train_test_split(ratings, test_size=0.2, random_state=42)

# Create user-item matrices
train_matrix = train.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
test_matrix = test.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)


cos_sim = cosine_similarity(train_matrix)

binary_matrix = (train_matrix > 0).astype(int)

def jaccard_similarity(matrix):
    intersection = np.dot(matrix, matrix.T)
    row_sums = matrix.sum(axis=1).values.reshape(-1, 1)
    union = row_sums + row_sums.T - intersection
    return intersection / (union + 1e-9)

jac_sim = jaccard_similarity(binary_matrix)

def hybrid_similarity(alpha):
    return alpha * cos_sim + (1 - alpha) * jac_sim

def recommend(user_id, sim_matrix, k=5):
    if user_id not in train_matrix.index:
        return []

    user_index = train_matrix.index.get_loc(user_id)

    # Get similar users
    sim_scores = list(enumerate(sim_matrix[user_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    similar_users = [train_matrix.index[i[0]] for i in sim_scores]

    user_ratings = train_matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings == 0].index

    scores = {}

    for item in unrated_items:
        score = 0
        for i, sim_user in enumerate(similar_users):
            sim_user_index = train_matrix.index.get_loc(sim_user)
            similarity = sim_matrix[user_index][sim_user_index]
            score += similarity * train_matrix.loc[sim_user, item]

        scores[item] = score

    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [i[0] for i in recommended]

def precision_at_k(user_id, sim_matrix, k=5):
    if user_id not in test_matrix.index:
        return 0

    recommended = recommend(user_id, sim_matrix, k)

    actual = test_matrix.loc[user_id]
    relevant = actual[actual >= 3].index

    if len(relevant) == 0:
        return 0

    hits = len(set(recommended) & set(relevant))
    return hits / k
alphas = np.linspace(0, 1, 6)
results = []

sample_users = train_matrix.index[:20]

print("Alpha Optimization Results:\n")

for alpha in alphas:
    sim_matrix = hybrid_similarity(alpha)
    precision_scores = []

    for user in sample_users:
        precision_scores.append(precision_at_k(user, sim_matrix))

    avg_precision = np.mean(precision_scores)
    results.append((alpha, avg_precision))

    print(f"Alpha: {alpha:.2f} -> Precision@K: {avg_precision:.4f}")

best_alpha = max(results, key=lambda x: x[1])
print("\nBest Alpha:", best_alpha)


n_components = min(10, min(train_matrix.shape) - 1)

pca = PCA(n_components=n_components)
train_reduced = pca.fit_transform(train_matrix)

cos_sim_pca = cosine_similarity(train_reduced)

print(f"\nPCA applied with {n_components} components")
print("\nComparison of Methods:")

methods = {
    "Cosine": cos_sim,
    "Jaccard": jac_sim,
    "Hybrid": hybrid_similarity(best_alpha[0])
}

for name, sim_matrix in methods.items():
    scores = []
    for user in sample_users:
        scores.append(precision_at_k(user, sim_matrix))
    print(f"{name}: Avg Precision@K = {np.mean(scores):.4f}")