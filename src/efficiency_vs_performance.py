import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path

def analyze_efficiency_vs_performance():
    """Analyze the trade-off between fuel efficiency and engine power"""
    
    # Load munged data with engineered features
    df = pd.read_csv('data/car_price_dataset_munged.csv')
    
    # Calculate correlations
    mileage_power_corr, mileage_power_pval = pearsonr(df['Mileage_kmpl'], df['Max_Power_bhp'])
    
    print(f"p-value: {mileage_power_pval:.2e}")
    
    # Analyze by price category
    categories = ['Budget', 'Mid-range', 'Premium', 'Luxury']
    category_analysis = {}
    
    for category in categories:
        subset = df[df['Price_Category'] == category]
        
        category_analysis[category] = {
            'Count': len(subset),
            'Avg Efficiency_to_Power': subset['Efficiency_to_Power'].mean(),
            'Avg Power_Per_CC': subset['Power_Per_CC'].mean(),
            'Avg Mileage (kmpl)': subset['Mileage_kmpl'].mean(),
            'Avg Power (bhp)': subset['Max_Power_bhp'].mean(),
            'High Power %': (subset['Max_Power_bhp'] > subset['Max_Power_bhp'].median()).sum() / len(subset) * 100,
            'High Efficiency %': (subset['Mileage_kmpl'] > subset['Mileage_kmpl'].median()).sum() / len(subset) * 100,
        }
    
    category_df = pd.DataFrame(category_analysis).T
    
    # Identify vehicle segments based on efficiency and power
    
    efficiency_median = df['Efficiency_to_Power'].median()
    power_median = df['Power_Per_CC'].median()
    
    # Create quadrants
    eco_efficient = df[(df['Efficiency_to_Power'] > efficiency_median) & (df['Power_Per_CC'] < power_median)]
    performance_focused = df[(df['Efficiency_to_Power'] < efficiency_median) & (df['Power_Per_CC'] > power_median)]
    balanced = df[(df['Efficiency_to_Power'] > efficiency_median) & (df['Power_Per_CC'] > power_median)]
    underpowered = df[(df['Efficiency_to_Power'] < efficiency_median) & (df['Power_Per_CC'] < power_median)]
    
    segments = {
        'Eco-Efficient (High Efficiency, Low Power)': eco_efficient,
        'Performance-Focused (Low Efficiency, High Power)': performance_focused,
        'Balanced (High Efficiency, High Power)': balanced,
        'Underpowered (Low Efficiency, Low Power)': underpowered
    }
    
    for segment_name, segment_data in segments.items():
        pass
    
    # Create visualizations
    create_visualizations(df, categories, category_analysis, segments)
    
    return category_df, df

def create_visualizations(df, categories, category_analysis, segments):
    """Create comprehensive comparison charts"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Fuel Efficiency vs Performance Trade-off Analysis', fontsize=16, fontweight='bold')
    
    # 1. Main scatter plot: Efficiency vs Power
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    scatter = ax1.scatter(df['Efficiency_to_Power'], df['Power_Per_CC'], 
                         c=df['Mileage_kmpl'], cmap='RdYlGn', alpha=0.6, s=80, edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Efficiency to Power Ratio\n(Higher = More Efficient)', fontweight='bold')
    ax1.set_ylabel('Power Per CC\n(Higher = More Powerful)', fontweight='bold')
    ax1.set_title('Efficiency vs Performance Trade-off', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Mileage (kmpl)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Efficiency by Category
    ax2 = fig.add_subplot(gs[0, 2])
    category_eff = [category_analysis[cat]['Avg Efficiency_to_Power'] for cat in categories]
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
    ax2.bar(categories, category_eff, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Avg Efficiency Ratio', fontweight='bold')
    ax2.set_title('Efficiency by Category')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Power by Category
    ax3 = fig.add_subplot(gs[1, 2])
    category_power = [category_analysis[cat]['Avg Power_Per_CC'] for cat in categories]
    ax3.bar(categories, category_power, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Avg Power Per CC', fontweight='bold')
    ax3.set_title('Power by Category')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Vehicle Count by Segment
    ax4 = fig.add_subplot(gs[2, 0])
    segment_names = list(segments.keys())
    segment_counts = []
    for seg_name in segment_names:
        seg_data = segments[seg_name]
        segment_counts.append(len(seg_data))
    
    seg_labels = ['Eco-Eff', 'Perf', 'Balanced', 'Under']
    ax4.bar(seg_labels, segment_counts, alpha=0.7, edgecolor='black', color=['#2ecc71', '#c0392b', '#3498db', '#95a5a6'])
    ax4.set_ylabel('Vehicle Count', fontweight='bold')
    ax4.set_title('Vehicle Count by Segment')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Segment Distribution
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.pie(segment_counts, labels=seg_labels, autopct='%1.1f%%', startangle=90,
           colors=['#2ecc71', '#c0392b', '#3498db', '#95a5a6'])
    ax5.set_title('Vehicle Distribution by Segment')
    
    # 6. Average Mileage by Power Level
    ax6 = fig.add_subplot(gs[2, 2])
    power_quartiles = pd.qcut(df['Max_Power_bhp'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    mileage_by_power = df.groupby(power_quartiles)['Mileage_kmpl'].mean()
    ax6.plot(range(len(mileage_by_power)), mileage_by_power.values, marker='o', linewidth=2, markersize=8, color='#e74c3c')
    ax6.set_xticks(range(len(mileage_by_power)))
    ax6.set_xticklabels(['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    ax6.set_ylabel('Avg Mileage (kmpl)', fontweight='bold')
    ax6.set_title('Efficiency Declines with Power')
    ax6.grid(True, alpha=0.3)
    
    models_dir = Path(__file__).parent.parent / "models"
    plt.savefig(models_dir / 'efficiency_vs_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    category_df, full_df = analyze_efficiency_vs_performance()
