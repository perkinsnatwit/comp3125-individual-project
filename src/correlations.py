import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_correlations(csv_path):
    # Load the data
    df = pd.read_csv(csv_path)
    print("\nLoaded: " + csv_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Separate features and target
    target_column = 'Price_USD'
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Get only numeric columns (no one-hot encoding)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Keep only numeric columns for correlation analysis
    df_numeric = df[numeric_cols].copy()
    
    # Calculate correlation with Price_USD
    correlations = df_numeric.corr()[target_column].drop(target_column)
    
    # Sort by absolute correlation value
    correlations_sorted = correlations.abs().sort_values(ascending=False)
    
    print(f"\n{'Feature':<30} {'Correlation':>15}")
    print("-" * 47)
    
    for feature, abs_corr in correlations_sorted.items():
        actual_corr = correlations[feature]
        print(f"{feature:<30} {actual_corr:>15.6f}")
    
    return correlations


def plot_correlations(csv_path, save_path=None):
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Separate features and target
    target_column = 'Price_USD'
    
    # Get only numeric columns (no one-hot encoding)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df[numeric_cols].copy()
    
    # Calculate correlation
    correlations = df_numeric.corr()[target_column].drop(target_column)
    
    # Sort by absolute value
    correlations_sorted = correlations.reindex(
        correlations.abs().sort_values(ascending=True).index
    )
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if x > 0 else 'red' for x in correlations_sorted.values]
    correlations_sorted.plot(kind='barh', ax=ax, color=colors, edgecolor='black')
    
    ax.set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Correlations with {target_column}', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def calculate_fuel_efficiency_power_correlation(csv_path):
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Check if required columns exist
    if 'Mileage_kmpl' not in df.columns or 'Max_Power_bhp' not in df.columns:
        raise ValueError("Required columns 'Mileage_kmpl' or 'Max_Power_bhp' not found in dataset")
    
    # Calculate correlation
    correlation = df['Mileage_kmpl'].corr(df['Max_Power_bhp'])
    
    print(f"\n{'='*50}")
    print("CORRELATION: Fuel Efficiency vs Engine Power")
    print(f"{'='*50}")
    print(f"{'Mileage_kmpl (Fuel Efficiency)':30} vs {'Max_Power_bhp (Engine Power)':20}")
    print(f"Correlation Coefficient: {correlation:>15.6f}")
    print(f"{'='*50}")
    
    return correlation


if __name__ == '__main__':
    # Calculate correlations for the initial dataset
    csv_path = 'data/car_price_dataset_medium.csv'
    
    print("=" * 67)
    print("CORRELATION ANALYSIS: Features vs Price_USD")
    print("=" * 67)
    
    correlations = calculate_correlations(csv_path)
    
    # Calculate fuel efficiency vs engine power correlation
    fuel_power_corr = calculate_fuel_efficiency_power_correlation(csv_path)
    
    # Generate correlation plot
    plot_path = 'models/correlations_plot.png'
    plot_correlations(csv_path, save_path=plot_path)
