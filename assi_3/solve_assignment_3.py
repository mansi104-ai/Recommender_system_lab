from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = (
    BASE_DIR.parent
    / "venv"
    / "Lib"
    / "site-packages"
    / "gensim"
    / "test"
    / "test_data"
    / "lee_background.cor"
)

DEMO_QUERY = "india pakistan militant kashmir parliament security"
RELEVANT_KEYWORDS = {"india", "pakistan", "kashmir", "militant", "parliament", "security"}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def title_from_text(text: str, index: int) -> str:
    words = normalize(text).split()[:8]
    return f"Doc {index:02d}: {' '.join(words)}"


def load_articles(limit: int = 80) -> pd.DataFrame:
    lines = [normalize(line) for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    articles = pd.DataFrame({"content": lines[:limit]})
    articles["doc_id"] = np.arange(len(articles))
    articles["title"] = [title_from_text(text, i) for i, text in enumerate(articles["content"], start=1)]
    articles["keyword_relevant"] = articles["content"].str.lower().apply(
        lambda text: sum(keyword in text for keyword in RELEVANT_KEYWORDS) >= 2
    )
    return articles[["doc_id", "title", "content", "keyword_relevant"]]


def top_terms_for_centers(vectorizer: TfidfVectorizer, centers: np.ndarray, top_n: int = 4) -> list[str]:
    vocab = np.array(vectorizer.get_feature_names_out())
    labels = []
    for center in centers:
        terms = vocab[np.argsort(center)[-top_n:][::-1]]
        labels.append(", ".join(terms))
    return labels


def add_clusters(articles: pd.DataFrame, tfidf_matrix, vectorizer: TfidfVectorizer, n_clusters: int = 5) -> pd.DataFrame:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    articles = articles.copy()
    articles["cluster_id"] = kmeans.fit_predict(tfidf_matrix)
    cluster_labels = top_terms_for_centers(vectorizer, kmeans.cluster_centers_)
    articles["cluster_label"] = articles["cluster_id"].map(dict(enumerate(cluster_labels)))
    return articles


def vectorize_corpus(articles: pd.DataFrame) -> tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(articles["content"])
    return vectorizer, tfidf_matrix


def rank_documents(query: str, vectorizer: TfidfVectorizer, doc_matrix, top_k: int = 10) -> pd.DataFrame:
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, doc_matrix).ravel()
    order = np.argsort(scores)[::-1][:top_k]
    return pd.DataFrame({"doc_id": order, "content_score": scores[order]})


def expand_query_with_terms(base_query: str, articles: pd.DataFrame, vectorizer: TfidfVectorizer, doc_ids: list[int], top_terms: int = 4) -> str:
    selected = articles.loc[articles["doc_id"].isin(doc_ids), "content"]
    if selected.empty:
        return base_query
    temp_matrix = vectorizer.transform(selected)
    term_weights = np.asarray(temp_matrix.mean(axis=0)).ravel()
    vocab = np.array(vectorizer.get_feature_names_out())
    new_terms = [term for term in vocab[np.argsort(term_weights)[-top_terms:][::-1]] if term not in base_query]
    return f"{base_query} {' '.join(new_terms)}".strip()


def build_feedback_result(
    articles: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    doc_matrix,
    query: str,
    relevant_ids: list[int],
    non_relevant_ids: list[int],
    top_k: int = 10,
) -> tuple[pd.DataFrame, np.ndarray]:
    query_vec = vectorizer.transform([query]).toarray()
    relevant_vec = np.asarray(doc_matrix[relevant_ids].mean(axis=0)) if relevant_ids else np.zeros((1, doc_matrix.shape[1]))
    non_relevant_vec = np.asarray(doc_matrix[non_relevant_ids].mean(axis=0)) if non_relevant_ids else np.zeros((1, doc_matrix.shape[1]))
    updated_query = query_vec + 0.8 * relevant_vec - 0.2 * non_relevant_vec
    scores = cosine_similarity(updated_query, doc_matrix).ravel()
    order = np.argsort(scores)[::-1][:top_k]
    result = pd.DataFrame({"doc_id": order, "feedback_score": scores[order]})
    return result, updated_query


def build_synthetic_feedback_matrix(articles: pd.DataFrame, vectorizer: TfidfVectorizer, doc_matrix) -> pd.DataFrame:
    seed_queries = [
        "bushfire weather sydney rain",
        "india pakistan militant security",
        "argentina economy debt president",
        "hospital health labour midwives",
        "roads christmas holiday deaths",
    ]
    matrix_rows = []
    for user_id, query in enumerate(seed_queries, start=1):
        ranked = rank_documents(query, vectorizer, doc_matrix, top_k=8)
        liked_ids = ranked.head(4)["doc_id"].tolist()
        row = {"user_id": user_id}
        for doc_id in articles["doc_id"]:
            row[f"doc_{doc_id}"] = 1 if doc_id in liked_ids else 0
        matrix_rows.append(row)
    return pd.DataFrame(matrix_rows)


def hybrid_recommendations(
    content_ranked: pd.DataFrame,
    articles: pd.DataFrame,
    feedback_matrix: pd.DataFrame,
    relevant_ids: list[int],
    top_k: int = 10,
) -> pd.DataFrame:
    target_profile = np.array(
        [1 if doc_id in relevant_ids else 0 for doc_id in articles["doc_id"]],
        dtype=float,
    ).reshape(1, -1)
    historical_profiles = feedback_matrix.drop(columns=["user_id"]).to_numpy(dtype=float)
    user_sim = cosine_similarity(target_profile, historical_profiles).ravel()
    collaborative_scores = historical_profiles.T @ user_sim
    if collaborative_scores.max() > 0:
        collaborative_scores = collaborative_scores / collaborative_scores.max()

    merged = articles[["doc_id", "title", "keyword_relevant"]].merge(content_ranked, on="doc_id", how="left").fillna(0)
    merged["collaborative_score"] = collaborative_scores[merged["doc_id"].to_numpy()]
    merged["hybrid_score"] = 0.7 * merged["content_score"] + 0.3 * merged["collaborative_score"]
    return merged.sort_values("hybrid_score", ascending=False).head(top_k)


def prf_results(articles: pd.DataFrame, vectorizer: TfidfVectorizer, doc_matrix, query: str) -> tuple[pd.DataFrame, str]:
    initial = rank_documents(query, vectorizer, doc_matrix, top_k=10)
    expanded_query = expand_query_with_terms(query, articles, vectorizer, initial.head(5)["doc_id"].tolist())
    refined = rank_documents(expanded_query, vectorizer, doc_matrix, top_k=10)
    comparison = initial.rename(columns={"content_score": "initial_score"}).merge(
        refined.rename(columns={"content_score": "prf_score"}), on="doc_id", how="outer"
    )
    return comparison.sort_values(["prf_score", "initial_score"], ascending=False).fillna(0), expanded_query


def cluster_prf_results(articles: pd.DataFrame, vectorizer: TfidfVectorizer, doc_matrix, query: str) -> tuple[pd.DataFrame, str]:
    initial = rank_documents(query, vectorizer, doc_matrix, top_k=8)
    top_docs = initial["doc_id"].tolist()
    top_vectors = doc_matrix[top_docs]
    cluster_model = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = cluster_model.fit_predict(top_vectors)
    query_vec = vectorizer.transform([query]).toarray()
    centroid_scores = cosine_similarity(query_vec, cluster_model.cluster_centers_).ravel()
    selected_cluster = int(np.argmax(centroid_scores))
    selected_ids = [doc_id for doc_id, label in zip(top_docs, cluster_labels) if label == selected_cluster]
    expanded_query = expand_query_with_terms(query, articles, vectorizer, selected_ids)
    refined = rank_documents(expanded_query, vectorizer, doc_matrix, top_k=10)
    return refined, expanded_query


def rocchio_rank(query: str, vectorizer: TfidfVectorizer, doc_matrix, relevant_ids: list[int], non_relevant_ids: list[int], alpha: float, beta: float, gamma: float, top_k: int = 10) -> pd.DataFrame:
    query_vec = vectorizer.transform([query]).toarray()
    relevant_vec = np.asarray(doc_matrix[relevant_ids].mean(axis=0)) if relevant_ids else np.zeros((1, doc_matrix.shape[1]))
    non_relevant_vec = np.asarray(doc_matrix[non_relevant_ids].mean(axis=0)) if non_relevant_ids else np.zeros((1, doc_matrix.shape[1]))
    updated_query = alpha * query_vec + beta * relevant_vec - gamma * non_relevant_vec
    scores = cosine_similarity(updated_query, doc_matrix).ravel()
    order = np.argsort(scores)[::-1][:top_k]
    return pd.DataFrame({"doc_id": order, "score": scores[order], "alpha": alpha, "beta": beta, "gamma": gamma})


def rocchio_with_lsa(query: str, vectorizer: TfidfVectorizer, doc_matrix, relevant_ids: list[int], non_relevant_ids: list[int], n_components: int = 50) -> pd.DataFrame:
    svd = TruncatedSVD(n_components=min(n_components, doc_matrix.shape[1] - 1), random_state=42)
    reduced_docs = svd.fit_transform(doc_matrix)
    query_vec = svd.transform(vectorizer.transform([query]))
    relevant_vec = reduced_docs[relevant_ids].mean(axis=0) if relevant_ids else np.zeros(reduced_docs.shape[1])
    non_relevant_vec = reduced_docs[non_relevant_ids].mean(axis=0) if non_relevant_ids else np.zeros(reduced_docs.shape[1])
    updated = 1.0 * query_vec + 0.8 * relevant_vec - 0.2 * non_relevant_vec
    scores = cosine_similarity(updated.reshape(1, -1), reduced_docs).ravel()
    order = np.argsort(scores)[::-1][:10]
    return pd.DataFrame({"doc_id": order, "lsa_rocchio_score": scores[order]})


def precision_at_k(result_doc_ids: list[int], articles: pd.DataFrame, k: int = 5) -> float:
    top_ids = result_doc_ids[:k]
    if not top_ids:
        return 0.0
    relevant = articles.set_index("doc_id").loc[top_ids, "keyword_relevant"]
    return float(relevant.mean())


def save_outputs(
    articles: pd.DataFrame,
    task1_initial: pd.DataFrame,
    task1_feedback: pd.DataFrame,
    task1_hybrid: pd.DataFrame,
    prf_comparison: pd.DataFrame,
    cluster_prf: pd.DataFrame,
    rocchio_df: pd.DataFrame,
    lsa_rocchio_df: pd.DataFrame,
    expanded_query: str,
    cluster_expanded_query: str,
) -> None:
    articles.to_csv(BASE_DIR / "articles_dataset.csv", index=False)
    task1_initial.to_csv(BASE_DIR / "task_1_initial_recommendations.csv", index=False)
    task1_feedback.to_csv(BASE_DIR / "task_1_feedback_recommendations.csv", index=False)
    task1_hybrid.to_csv(BASE_DIR / "task_1_hybrid_recommendations.csv", index=False)
    prf_comparison.to_csv(BASE_DIR / "task_2_prf_comparison.csv", index=False)
    cluster_prf.to_csv(BASE_DIR / "task_2_cluster_prf.csv", index=False)
    rocchio_df.to_csv(BASE_DIR / "task_3_rocchio_results.csv", index=False)
    lsa_rocchio_df.to_csv(BASE_DIR / "task_3_rocchio_lsa_results.csv", index=False)

    summary_lines = [
        "Assignment 3 Solution Summary",
        "",
        f"Corpus size: {len(articles)} documents",
        f"Demo query: {DEMO_QUERY}",
        f"PRF expanded query: {expanded_query}",
        f"Cluster-aware PRF expanded query: {cluster_expanded_query}",
    ]
    (BASE_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    metrics = pd.DataFrame(
        [
            {"method": "Initial TF-IDF", "precision_at_5": precision_at_k(task1_initial["doc_id"].tolist(), articles)},
            {"method": "Explicit feedback", "precision_at_5": precision_at_k(task1_feedback["doc_id"].tolist(), articles)},
            {"method": "Hybrid feedback", "precision_at_5": precision_at_k(task1_hybrid["doc_id"].tolist(), articles)},
            {"method": "Pseudo relevance feedback", "precision_at_5": precision_at_k(prf_comparison.sort_values('prf_score', ascending=False)["doc_id"].tolist(), articles)},
            {"method": "Cluster PRF", "precision_at_5": precision_at_k(cluster_prf["doc_id"].tolist(), articles)},
            {"method": "Rocchio", "precision_at_5": precision_at_k(rocchio_df["doc_id"].tolist(), articles)},
            {"method": "Rocchio with LSA", "precision_at_5": precision_at_k(lsa_rocchio_df["doc_id"].tolist(), articles)},
        ]
    )
    metrics.to_csv(BASE_DIR / "evaluation_metrics.csv", index=False)

    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(9, 5))
    plt.bar(metrics["method"], metrics["precision_at_5"], color="#3b7a57")
    plt.title("Precision@5 Across Relevance Feedback Methods")
    plt.ylabel("Precision@5")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "evaluation_precision_at_5.png", dpi=200)
    plt.close()


def interactive_demo(articles: pd.DataFrame, initial_df: pd.DataFrame) -> tuple[list[int], list[int]]:
    print("Top recommendations for the query:")
    print(initial_df[["doc_id", "content_score"]].to_string(index=False))
    relevant = input("Enter comma-separated relevant doc_ids (or press Enter for demo defaults): ").strip()
    non_relevant = input("Enter comma-separated non-relevant doc_ids (or press Enter for demo defaults): ").strip()

    if relevant:
        relevant_ids = [int(item) for item in relevant.split(",") if item.strip()]
    else:
        relevant_ids = initial_df.head(3)["doc_id"].tolist()

    if non_relevant:
        non_relevant_ids = [int(item) for item in non_relevant.split(",") if item.strip()]
    else:
        non_relevant_ids = initial_df.tail(2)["doc_id"].tolist()

    return relevant_ids, non_relevant_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Assignment 3 recommender system solution")
    parser.add_argument("--interactive", action="store_true", help="Collect relevance feedback from the terminal")
    args = parser.parse_args()

    articles = load_articles()
    vectorizer, doc_matrix = vectorize_corpus(articles)
    articles = add_clusters(articles, doc_matrix, vectorizer)

    initial = rank_documents(DEMO_QUERY, vectorizer, doc_matrix, top_k=10)
    task1_initial = articles[["doc_id", "title", "cluster_label", "keyword_relevant"]].merge(initial, on="doc_id")

    if args.interactive:
        relevant_ids, non_relevant_ids = interactive_demo(articles, task1_initial)
    else:
        relevant_ids = task1_initial.head(3)["doc_id"].tolist()
        non_relevant_ids = task1_initial.tail(2)["doc_id"].tolist()

    feedback_ranked, _ = build_feedback_result(
        articles,
        vectorizer,
        doc_matrix,
        DEMO_QUERY,
        relevant_ids,
        non_relevant_ids,
        top_k=10,
    )
    task1_feedback = articles[["doc_id", "title", "cluster_label", "keyword_relevant"]].merge(feedback_ranked, on="doc_id")

    feedback_matrix = build_synthetic_feedback_matrix(articles, vectorizer, doc_matrix)
    task1_hybrid = hybrid_recommendations(task1_initial[["doc_id", "content_score"]], articles, feedback_matrix, relevant_ids, top_k=10)

    prf_comparison, expanded_query = prf_results(articles, vectorizer, doc_matrix, DEMO_QUERY)
    cluster_prf, cluster_expanded_query = cluster_prf_results(articles, vectorizer, doc_matrix, DEMO_QUERY)
    cluster_prf = articles[["doc_id", "title", "cluster_label", "keyword_relevant"]].merge(cluster_prf, on="doc_id")

    rocchio_runs = [
        rocchio_rank(DEMO_QUERY, vectorizer, doc_matrix, relevant_ids, non_relevant_ids, alpha, beta, gamma, top_k=5)
        for alpha, beta, gamma in [(1.0, 0.6, 0.1), (1.0, 0.8, 0.2), (1.2, 0.8, 0.2)]
    ]
    rocchio_df = pd.concat(rocchio_runs, ignore_index=True)
    rocchio_df = articles[["doc_id", "title", "cluster_label", "keyword_relevant"]].merge(rocchio_df, on="doc_id")
    lsa_rocchio_df = articles[["doc_id", "title", "cluster_label", "keyword_relevant"]].merge(
        rocchio_with_lsa(DEMO_QUERY, vectorizer, doc_matrix, relevant_ids, non_relevant_ids),
        on="doc_id",
    )

    save_outputs(
        articles,
        task1_initial,
        task1_feedback,
        task1_hybrid,
        prf_comparison,
        cluster_prf,
        rocchio_df,
        lsa_rocchio_df,
        expanded_query,
        cluster_expanded_query,
    )


if __name__ == "__main__":
    main()
