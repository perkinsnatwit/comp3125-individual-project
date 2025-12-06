import pandas as pd
import numpy as np

df = pd.read_csv('data/car_price_dataset_large.csv')
print("Dataset loaded successfully.")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Convert categorical columns to numbers (same as in loader.py)
df = pd.get_dummies(df, drop_first=True)

# Calculate correlations
correlations = df.corr()['Price_USD'].sort_values(ascending=False)
print("\nFeature correlations with Price_USD:")
print(correlations)

# Get mean, median, mode of Price_USD
mean_price = df['Price_USD'].mean()
median_price = df['Price_USD'].median()
mode_price = df['Price_USD'].mode()[0]

print(f"\nPrice_USD statistics:")
print(f"Mean: {mean_price}")
print(f"Median: {median_price}")
print(f"Mode: {mode_price}")

# Get std deviation of Price_USD
std_price = df['Price_USD'].std()
print(f"Standard Deviation: {std_price}")