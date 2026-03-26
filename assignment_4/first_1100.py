import pandas as pd
ratings = pd.read_csv("rating.csv")
ratings_small = ratings.head(1100)
ratings_small.to_csv("ratings_small.csv", index=False)
print("Saved 1100 reviews successfully!")