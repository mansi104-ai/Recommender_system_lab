# Assignment 3 Report

## Dataset
The solution uses the `lee_background.cor` news corpus bundled with `gensim`. This keeps the work fully offline and reproducible in the current lab environment.

## Task Coverage
`solve_assignment_3.py` implements:
- TF-IDF based initial retrieval
- explicit relevance feedback with dynamic recommendation updates
- a hybrid model that mixes content similarity with collaborative signals built from synthetic historical feedback sessions
- pseudo relevance feedback using top-ranked documents
- clustering-aware PRF that expands the query from the most query-aligned cluster
- Rocchio query adjustment with multiple parameter settings
- dimensionality reduction using LSA (`TruncatedSVD`) before Rocchio

## Files Generated
- `articles_dataset.csv`: processed corpus with document ids, titles, clusters, and heuristic relevance labels
- `task_1_initial_recommendations.csv`: initial TF-IDF retrieval
- `task_1_feedback_recommendations.csv`: updated results after explicit relevance feedback
- `task_1_hybrid_recommendations.csv`: hybrid content + collaborative ranking
- `task_2_prf_comparison.csv`: before/after pseudo relevance feedback comparison
- `task_2_cluster_prf.csv`: clustering-based PRF result set
- `task_3_rocchio_results.csv`: Rocchio results for multiple parameter settings
- `task_3_rocchio_lsa_results.csv`: Rocchio results after dimensionality reduction
- `evaluation_metrics.csv`: precision@5 summary
- `evaluation_precision_at_5.png`: quality comparison chart
- `summary.txt`: concise submission-ready summary

## Observations
- Explicit feedback usually improves the concentration of relevant results compared with the plain TF-IDF ranking.
- PRF is useful when the first results are already close to the user intent, but it can drift if the initial retrieval is noisy.
- Cluster-aware PRF reduces drift by expanding from a more coherent topical subset.
- Rocchio with LSA is more compact and efficient while still keeping strong top-ranked recommendations.
