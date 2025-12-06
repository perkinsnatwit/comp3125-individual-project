import pandas as pd
import numpy as np

df = pd.read_csv('data/car_price_dataset_large.csv')
print("Dataset loaded successfully.")
print("First 5 rows of the dataset:")

# Sort rows by 'Price_USD' in descending order
df_sorted = df.sort_values(by='Price_USD', ascending=False)
print("Top 5 most expensive cars:")
print(df_sorted.head())
print("\nBottom 5 least expensive cars:")
print(df_sorted.tail())