import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_market_segments():
    """Analyze and compare vehicle specifications across price categories"""
    
    # Load munged data with engineered features
    df = pd.read_csv('data/car_price_dataset_munged.csv')
    
    # Define price categories
    categories = ['Budget', 'Mid-range', 'Premium', 'Luxury']
    
    # Initialize results dictionary
    results = {}
    
    # Analyze each price category
    for category in categories:
        subset = df[df['Price_Category'] == category]
        
        results[category] = {
            'Count': len(subset),
            'Avg Price': subset['Price_USD'].mean(),
            'Avg Power (bhp)': subset['Max_Power_bhp'].mean(),
            'Avg Mileage (kmpl)': subset['Mileage_kmpl'].mean(),
            'Avg Engine Size (CC)': subset['Engine_CC'].mean(),
            'Automatic %': (subset['Is_Automatic'].sum() / len(subset)) * 100,
            'Avg Vehicle Age': subset['Vehicle_Age'].mean(),
            'Avg Usage Intensity': subset['Usage_Intensity'].mean(),
            'Eco-Friendly %': (subset['Fuel_Type_Category'] == 'Eco-Friendly').sum() / len(subset) * 100,
            'Avg Seats': subset['Seats'].mean(),
            'Diesel %': (subset['Fuel_Type'] == 'Diesel').sum() / len(subset) * 100,
            'Petrol %': (subset['Fuel_Type'] == 'Petrol').sum() / len(subset) * 100,
            'Hybrid %': (subset['Fuel_Type'] == 'Hybrid').sum() / len(subset) * 100,
        }
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).T
    
    # Key insights
    print("="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    budget_avg = results['Budget']['Avg Price']
    luxury_avg = results['Luxury']['Avg Price']
    price_multiplier = luxury_avg / budget_avg
    
    print(f"\n1. PRICE ANALYSIS:")
    print(f"   Budget vehicles average:      ${budget_avg:>12,.2f}")
    print(f"   Luxury vehicles average:      ${luxury_avg:>12,.2f}")
    print(f"   Luxury/Budget ratio:          {price_multiplier:>12.2f}x")
    
    power_diff = results['Luxury']['Avg Power (bhp)'] - results['Budget']['Avg Power (bhp)']
    power_pct = (power_diff / results['Budget']['Avg Power (bhp)']) * 100
    print(f"\n2. ENGINE POWER:")
    print(f"   Budget:                       {results['Budget']['Avg Power (bhp)']:>12.1f} bhp")
    print(f"   Luxury:                       {results['Luxury']['Avg Power (bhp)']:>12.1f} bhp")
    print(f"   Difference:                   {power_diff:>12.1f} bhp (+{power_pct:.1f}%)")
    
    mileage_diff = results['Luxury']['Avg Mileage (kmpl)'] - results['Budget']['Avg Mileage (kmpl)']
    mileage_pct = (mileage_diff / results['Budget']['Avg Mileage (kmpl)']) * 100
    print(f"\n3. FUEL EFFICIENCY:")
    print(f"   Budget:                       {results['Budget']['Avg Mileage (kmpl)']:>12.2f} kmpl")
    print(f"   Luxury:                       {results['Luxury']['Avg Mileage (kmpl)']:>12.2f} kmpl")
    print(f"   Difference:                   {mileage_diff:>12.2f} kmpl ({mileage_pct:+.1f}%)")
    
    engine_diff = results['Luxury']['Avg Engine Size (CC)'] - results['Budget']['Avg Engine Size (CC)']
    print(f"\n4. ENGINE SIZE:")
    print(f"   Budget:                       {results['Budget']['Avg Engine Size (CC)']:>12.0f} CC")
    print(f"   Luxury:                       {results['Luxury']['Avg Engine Size (CC)']:>12.0f} CC")
    print(f"   Difference:                   {engine_diff:>12.0f} CC")
    
    auto_diff = results['Luxury']['Automatic %'] - results['Budget']['Automatic %']
    print(f"\n5. TRANSMISSION TYPE:")
    print(f"   Budget Automatic:             {results['Budget']['Automatic %']:>12.1f}%")
    print(f"   Luxury Automatic:             {results['Luxury']['Automatic %']:>12.1f}%")
    print(f"   Difference:                   {auto_diff:>12.1f}%")
    
    eco_diff = results['Luxury']['Eco-Friendly %'] - results['Budget']['Eco-Friendly %']
    print(f"\n6. FUEL TYPE PREFERENCE:")
    print(f"   Budget Eco-Friendly:          {results['Budget']['Eco-Friendly %']:>12.1f}%")
    print(f"   Luxury Eco-Friendly:          {results['Luxury']['Eco-Friendly %']:>12.1f}%")
    print(f"   Difference:                   {eco_diff:>12.1f}%")
    
    print(f"\n7. AVERAGE VEHICLE AGE:")
    print(f"   Budget:                       {results['Budget']['Avg Vehicle Age']:>12.1f} years")
    print(f"   Luxury:                       {results['Luxury']['Avg Vehicle Age']:>12.1f} years")
    
    print(f"\n8. AVERAGE SEATING:")
    print(f"   Budget:                       {results['Budget']['Avg Seats']:>12.1f} seats")
    print(f"   Luxury:                       {results['Luxury']['Avg Seats']:>12.1f} seats")
    
    # Create visualizations
    create_visualizations(df, results_df, categories)
    
    return results_df

def create_visualizations(df, results_df, categories):
    """Create comparison charts"""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Budget vs. Luxury Vehicle Market Analysis', fontsize=16, fontweight='bold')
    
    # 1. Average Price by Category
    ax = axes[0, 0]
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
    prices = results_df['Avg Price']
    ax.bar(categories, prices, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Price (USD)', fontweight='bold')
    ax.set_title('Average Price by Category')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(prices):
        ax.text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Engine Power by Category
    ax = axes[0, 1]
    power = results_df['Avg Power (bhp)']
    ax.bar(categories, power, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Power (bhp)', fontweight='bold')
    ax.set_title('Average Engine Power by Category')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(power):
        ax.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Fuel Efficiency by Category
    ax = axes[0, 2]
    mileage = results_df['Avg Mileage (kmpl)']
    ax.bar(categories, mileage, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Mileage (kmpl)', fontweight='bold')
    ax.set_title('Average Fuel Efficiency by Category')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(mileage):
        ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Engine Size by Category
    ax = axes[1, 0]
    engine = results_df['Avg Engine Size (CC)']
    ax.bar(categories, engine, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Engine Size (CC)', fontweight='bold')
    ax.set_title('Average Engine Displacement by Category')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(engine):
        ax.text(i, v, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Transmission Type Distribution
    ax = axes[1, 1]
    auto_pct = results_df['Automatic %']
    ax.bar(categories, auto_pct, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Automatic (%)', fontweight='bold')
    ax.set_title('Automatic Transmission Adoption by Category')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim([0, 100])
    for i, v in enumerate(auto_pct):
        ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 6. Eco-Friendly Fuel Type Distribution
    ax = axes[1, 2]
    eco_pct = results_df['Eco-Friendly %']
    ax.bar(categories, eco_pct, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Eco-Friendly (%)', fontweight='bold')
    ax.set_title('Eco-Friendly Vehicle Adoption by Category')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim([0, 100])
    for i, v in enumerate(eco_pct):
        ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    models_dir = Path(__file__).parent.parent / "models"
    plt.savefig(models_dir / 'budget_vs_luxury_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    results = analyze_market_segments()
