# Assignment 1 Report

## Dataset
The solution uses a compact offline movie catalogue stored in the folder itself. It was assembled from publicly known movie metadata so the assignment can run without downloading external files in the lab environment.

## Task Coverage
`solve_assignment_1.py` completes all required parts:
- data cleaning by removing duplicates, filling a missing rating, and filtering items with too few interactions
- Boolean feature construction using `is_popular` and `is_recent`
- fixed Boolean queries and nested Boolean queries
- a non-personalized rule-based recommender
- precision and coverage analysis
- baseline comparison against most-popular and most-recent recommendation rules

## Files Generated
- `raw_movie_catalog.csv`: original input with one duplicate row and one low-interaction row
- `cleaned_movie_catalog.csv`: cleaned dataset used for querying
- `q*.csv`: outputs of the Boolean queries
- `boolean_recommendations.csv`: final recommendations produced from Boolean rules
- `query_evaluation.csv`: precision, coverage, and recall-style summary
- `baseline_comparison.csv`: comparison with simple baselines
- `task_2_query_counts.png`: visualization of query result sizes
- `task_4_precision_coverage.png`: visualization of retrieval quality
- `summary.txt`: compact text summary for submission

## Observations
- Queries that mix popularity constraints with recency constraints retrieve fewer items, but they tend to keep higher precision.
- Broader `OR` queries improve coverage at the cost of selectivity.
- Boolean rules are easy to explain, fast to execute, and suitable for non-personalized recommendation scenarios where interpretability matters.
