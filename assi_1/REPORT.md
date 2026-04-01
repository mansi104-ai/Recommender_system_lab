# Assignment 5 Report

## Objective
Explore the ratings dataset to understand data quality, interaction structure, sparsity, inequality, and long-tail behavior relevant to recommender-system design.

## Preprocessing
- Loaded `ratings_small.csv` from `assi_4`.
- Removed duplicate rows.
- Converted timestamps to datetime format.
- Dropped rows with missing user, item, or rating values and filled missing timestamps with the earliest available timestamp.
- Applied a one-pass filter to remove users with fewer than 10 ratings and items with fewer than 5 ratings based on original interaction counts.

## Feature Engineering
- Extracted rating year, month, day name, and weekend flag from the timestamp.
- Created rating buckets for low, mid, good, and high ratings.
- Added user-level features: rating count, mean rating, rating standard deviation.
- Added item-level features: rating count, mean rating, rating standard deviation.
- Derived user activity segments and item popularity segments for group-wise comparison.

## Key Findings
- Filtered dataset contains 82 interactions, 11 users, and 14 items.
- User-item matrix sparsity is `46.75%`, indicating a sparse interaction space.
- User activity Gini index is `0.2905`, showing how unevenly user activity is distributed.
- Item popularity Gini index is `0.0976`, showing how concentrated interactions are among a smaller set of movies.
- Long-tail split identifies 10 head items and 4 tail items using the 80% cumulative interaction rule.

## Output Files
- `solve_assignment_1.py`: main solution script.
- `ratings_filtered_featured.csv`: cleaned and feature-engineered interaction data.
- `groupwise_user_activity_summary.csv`: user-segment comparison table.
- `groupwise_item_popularity_summary.csv`: item-segment comparison table.
- `long_tail_head_tail_items.csv`: head vs tail classification.
- `eda_overview.png`, `user_item_sparsity_heatmap.png`, `ratings_by_year.png`, `long_tail_distribution.png`: generated plots.