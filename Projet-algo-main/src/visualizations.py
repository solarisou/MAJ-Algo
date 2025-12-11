"""
Visualization module for Knapsack Algorithm Benchmark Results.
Generates comprehensive graphs for algorithm analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime


def load_results(results_path: str) -> pd.DataFrame:
    """Load benchmark results from CSV file."""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    df = pd.read_csv(results_path)
    
    # Normalize column names to expected format
    column_mapping = {
        'time': 'time_seconds',
        'n': 'n_items',
        'value': 'value_found',
    }
    
    df = df.rename(columns=column_mapping)
    return df


def create_output_directory(output_dir: str) -> str:
    """Create output directory for graphs."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_execution_time_comparison(df: pd.DataFrame, output_dir: str):
    """
    Graph 1: Bar chart comparing execution times across algorithms.
    Groups by instance size category.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get unique algorithms and instances
    algorithms = df['algorithm'].unique()
    instances = df['instance'].unique()
    
    x = np.arange(len(instances))
    width = 0.8 / len(algorithms)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(algorithms)))
    
    for i, algo in enumerate(algorithms):
        algo_data = df[df['algorithm'] == algo]
        times = []
        for inst in instances:
            inst_data = algo_data[algo_data['instance'] == inst]
            if len(inst_data) > 0:
                times.append(inst_data['time_seconds'].values[0])
            else:
                times.append(0)
        
        bars = ax.bar(x + i * width, times, width, label=algo, color=colors[i])
    
    ax.set_xlabel('Instance', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Execution Time Comparison by Algorithm', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(instances, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.set_yscale('log')  # Log scale for better visualization
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_execution_time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("âœ“[OK] Generated: 01_execution_time_comparison.png")


def plot_gap_comparison(df: pd.DataFrame, output_dir: str):
    """
    Graph 2: Bar chart showing gap to optimal solution for each algorithm.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate average gap per algorithm
    avg_gaps = df.groupby('algorithm')['gap_percent'].mean().sort_values()
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(avg_gaps)))
    
    bars = ax.barh(avg_gaps.index, avg_gaps.values, color=colors)
    
    # Add value labels
    for bar, gap in zip(bars, avg_gaps.values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{gap:.2f}%', va='center', fontsize=10)
    
    ax.set_xlabel('Average Gap to Optimal (%)', fontsize=12)
    ax.set_ylabel('Algorithm', fontsize=12)
    ax.set_title('Solution Quality: Average Gap to Optimal', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Highlight optimal solutions (gap = 0)
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Optimal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_gap_to_optimal.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("âœ“[OK] Generated: 02_gap_to_optimal.png")


def plot_scalability_analysis(df: pd.DataFrame, output_dir: str):
    """
    Graph 3: Line plot showing how execution time scales with problem size.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    algorithms = df['algorithm'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(algorithms)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*', 'X', 'P']
    
    for i, algo in enumerate(algorithms):
        algo_data = df[df['algorithm'] == algo].sort_values('n_items')
        ax.plot(algo_data['n_items'], algo_data['time_seconds'], 
                marker=markers[i % len(markers)], 
                color=colors[i],
                label=algo,
                linewidth=2,
                markersize=8)
    
    ax.set_xlabel('Number of Items (n)', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title('Scalability Analysis: Time vs Problem Size', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_scalability_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("âœ“[OK] Generated: 03_scalability_analysis.png")


def plot_quality_vs_time_tradeoff(df: pd.DataFrame, output_dir: str):
    """
    Graph 4: Scatter plot showing quality-time trade-off.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate average time and gap per algorithm
    summary = df.groupby('algorithm').agg({
        'time_seconds': 'mean',
        'gap_percent': 'mean'
    }).reset_index()
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(summary)))
    
    scatter = ax.scatter(summary['time_seconds'], summary['gap_percent'],
                        c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add labels for each point
    for i, row in summary.iterrows():
        ax.annotate(row['algorithm'], 
                   (row['time_seconds'], row['gap_percent']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Average Execution Time (seconds)', fontsize=12)
    ax.set_ylabel('Average Gap to Optimal (%)', fontsize=12)
    ax.set_title('Quality vs Time Trade-off', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add quadrant lines
    ax.axhline(y=summary['gap_percent'].median(), color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=summary['time_seconds'].median(), color='red', linestyle='--', alpha=0.5)
    
    # Add annotations for quadrants
    ax.text(0.02, 0.98, 'Fast & Accurate\n(Best)', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', color='green', fontweight='bold')
    ax.text(0.98, 0.02, 'Slow & Inaccurate\n(Worst)', transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_quality_vs_time_tradeoff.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("âœ“[OK] Generated: 04_quality_vs_time_tradeoff.png")


def plot_performance_heatmap(df: pd.DataFrame, output_dir: str):
    """
    Graph 5: Heatmap of algorithm performance across instances.
    """
    # Create pivot table for gap
    pivot_gap = df.pivot_table(values='gap_percent', index='algorithm', columns='instance')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    im = ax.imshow(pivot_gap.values, cmap='RdYlGn_r', aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Gap to Optimal (%)', rotation=-90, va="bottom", fontsize=11)
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot_gap.columns)))
    ax.set_yticks(np.arange(len(pivot_gap.index)))
    ax.set_xticklabels(pivot_gap.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot_gap.index)
    
    # Add value annotations
    for i in range(len(pivot_gap.index)):
        for j in range(len(pivot_gap.columns)):
            value = pivot_gap.values[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.1f}',
                              ha="center", va="center", color="black", fontsize=8)
    
    ax.set_xlabel('Instance', fontsize=12)
    ax.set_ylabel('Algorithm', fontsize=12)
    ax.set_title('Performance Heatmap: Gap to Optimal (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("âœ“[OK] Generated: 05_performance_heatmap.png")


def plot_time_heatmap(df: pd.DataFrame, output_dir: str):
    """
    Graph 6: Heatmap of execution times across instances.
    """
    # Create pivot table for time
    pivot_time = df.pivot_table(values='time_seconds', index='algorithm', columns='instance')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use log scale for time
    log_times = np.log10(pivot_time.values + 1e-6)
    
    im = ax.imshow(log_times, cmap='YlOrRd', aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('log10(Time in seconds)', rotation=-90, va="bottom", fontsize=11)
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot_time.columns)))
    ax.set_yticks(np.arange(len(pivot_time.index)))
    ax.set_xticklabels(pivot_time.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot_time.index)
    
    ax.set_xlabel('Instance', fontsize=12)
    ax.set_ylabel('Algorithm', fontsize=12)
    ax.set_title('Execution Time Heatmap (log scale)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_time_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("âœ“[OK] Generated: 06_time_heatmap.png")


def plot_boxplot_by_algorithm(df: pd.DataFrame, output_dir: str):
    """
    Graph 7: Box plots showing distribution of gaps per algorithm.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    algorithms = df['algorithm'].unique()
    data = [df[df['algorithm'] == algo]['gap_percent'].values for algo in algorithms]
    
    bp = ax.boxplot(data, labels=algorithms, patch_artist=True)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(algorithms)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Gap to Optimal (%)', fontsize=12)
    ax.set_title('Distribution of Solution Quality by Algorithm', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_boxplot_gap_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("âœ“[OK] Generated: 07_boxplot_gap_distribution.png")


def plot_boxplot_time_by_algorithm(df: pd.DataFrame, output_dir: str):
    """
    Graph 8: Box plots showing distribution of execution times per algorithm.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    algorithms = df['algorithm'].unique()
    data = [df[df['algorithm'] == algo]['time_seconds'].values for algo in algorithms]
    
    bp = ax.boxplot(data, labels=algorithms, patch_artist=True)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(algorithms)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title('Distribution of Execution Times by Algorithm', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_boxplot_time_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("âœ“[OK] Generated: 08_boxplot_time_distribution.png")


def plot_algorithm_ranking(df: pd.DataFrame, output_dir: str):
    """
    Graph 9: Radar chart showing multi-criteria ranking of algorithms.
    """
    # Calculate metrics for each algorithm
    summary = df.groupby('algorithm').agg({
        'time_seconds': 'mean',
        'gap_percent': 'mean',
        'value_found': 'mean'
    }).reset_index()
    
    # Normalize metrics (0-1 scale, higher is better)
    # For time: lower is better, so invert
    summary['time_score'] = 1 - (summary['time_seconds'] - summary['time_seconds'].min()) / \
                            (summary['time_seconds'].max() - summary['time_seconds'].min() + 1e-10)
    
    # For gap: lower is better, so invert
    summary['quality_score'] = 1 - (summary['gap_percent'] - summary['gap_percent'].min()) / \
                               (summary['gap_percent'].max() - summary['gap_percent'].min() + 1e-10)
    
    # Stability: based on variance of gap (lower variance = more stable)
    stability = df.groupby('algorithm')['gap_percent'].std().reset_index()
    stability.columns = ['algorithm', 'gap_std']
    summary = summary.merge(stability, on='algorithm')
    summary['stability_score'] = 1 - (summary['gap_std'] - summary['gap_std'].min()) / \
                                  (summary['gap_std'].max() - summary['gap_std'].min() + 1e-10)
    
    # Create overall score
    summary['overall_score'] = (summary['time_score'] + summary['quality_score'] + summary['stability_score']) / 3
    summary = summary.sort_values('overall_score', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(summary))
    width = 0.2
    
    ax.bar(x - width, summary['quality_score'], width, label='Quality Score', color='green', alpha=0.7)
    ax.bar(x, summary['time_score'], width, label='Speed Score', color='blue', alpha=0.7)
    ax.bar(x + width, summary['stability_score'], width, label='Stability Score', color='orange', alpha=0.7)
    
    # Add overall score line
    ax.plot(x, summary['overall_score'], 'r-o', label='Overall Score', linewidth=2, markersize=8)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Normalized Score (0-1)', fontsize=12)
    ax.set_title('Multi-Criteria Algorithm Ranking', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary['algorithm'], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_algorithm_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("âœ“[OK] Generated: 09_algorithm_ranking.png")


def plot_category_comparison(df: pd.DataFrame, output_dir: str):
    """
    Graph 10: Grouped bar chart comparing algorithms by instance category.
    """
    # Extract category from instance name
    df['category'] = df['instance'].apply(lambda x: x.split('_')[0] if '_' in x else 'unknown')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    categories = df['category'].unique()
    algorithms = df['algorithm'].unique()
    
    # Time comparison
    ax1 = axes[0]
    x = np.arange(len(categories))
    width = 0.8 / len(algorithms)
    
    for i, algo in enumerate(algorithms):
        means = []
        for cat in categories:
            cat_data = df[(df['algorithm'] == algo) & (df['category'] == cat)]
            means.append(cat_data['time_seconds'].mean() if len(cat_data) > 0 else 0)
        ax1.bar(x + i * width, means, width, label=algo)
    
    ax1.set_xlabel('Category', fontsize=12)
    ax1.set_ylabel('Average Time (seconds)', fontsize=12)
    ax1.set_title('Execution Time by Category', fontsize=13, fontweight='bold')
    ax1.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax1.set_xticklabels(categories)
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3)
    
    # Gap comparison
    ax2 = axes[1]
    for i, algo in enumerate(algorithms):
        means = []
        for cat in categories:
            cat_data = df[(df['algorithm'] == algo) & (df['category'] == cat)]
            means.append(cat_data['gap_percent'].mean() if len(cat_data) > 0 else 0)
        ax2.bar(x + i * width, means, width, label=algo)
    
    ax2.set_xlabel('Category', fontsize=12)
    ax2.set_ylabel('Average Gap (%)', fontsize=12)
    ax2.set_title('Solution Quality by Category', fontsize=13, fontweight='bold')
    ax2.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax2.set_xticklabels(categories)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '10_category_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("âœ“[OK] Generated: 10_category_comparison.png")


def plot_convergence_summary(df: pd.DataFrame, output_dir: str):
    """
    Graph 11: Summary pie charts showing algorithm success rates.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart 1: Algorithms finding optimal solution
    optimal_count = (df['gap_percent'] == 0).groupby(df['algorithm']).sum()
    total_count = df.groupby('algorithm').size()
    optimal_rate = (optimal_count / total_count * 100).sort_values(ascending=False)
    
    ax1 = axes[0]
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(optimal_rate)))
    wedges, texts, autotexts = ax1.pie(optimal_rate.values, labels=optimal_rate.index,
                                        autopct='%1.1f%%', colors=colors,
                                        explode=[0.05] * len(optimal_rate))
    ax1.set_title('Optimal Solution Rate by Algorithm', fontsize=12, fontweight='bold')
    
    # Pie chart 2: Distribution of solutions by gap range
    ax2 = axes[1]
    gap_ranges = pd.cut(df['gap_percent'], bins=[-1, 0, 1, 5, 10, 100],
                       labels=['Optimal (0%)', '< 1%', '1-5%', '5-10%', '> 10%'])
    gap_distribution = gap_ranges.value_counts()
    
    colors2 = ['#2ecc71', '#27ae60', '#f1c40f', '#e67e22', '#e74c3c']
    wedges2, texts2, autotexts2 = ax2.pie(gap_distribution.values, labels=gap_distribution.index,
                                          autopct='%1.1f%%', colors=colors2,
                                          explode=[0.05] * len(gap_distribution))
    ax2.set_title('Solution Distribution by Gap Range', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '11_solution_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("âœ“[OK] Generated: 11_solution_summary.png")


def generate_summary_report(df: pd.DataFrame, output_dir: str):
    """Generate a text summary report of the benchmark results."""
    report_path = os.path.join(output_dir, 'ANALYSIS_SUMMARY.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("KNAPSACK ALGORITHM BENCHMARK - ANALYSIS SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # Overall statistics
        f.write("1. OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total runs: {len(df)}\n")
        f.write(f"Algorithms tested: {df['algorithm'].nunique()}\n")
        f.write(f"Instances tested: {df['instance'].nunique()}\n")
        f.write(f"Average gap to optimal: {df['gap_percent'].mean():.2f}%\n")
        f.write(f"Average execution time: {df['time_seconds'].mean():.4f}s\n\n")
        
        # Best algorithm by quality
        f.write("2. BEST ALGORITHMS BY QUALITY (Lowest Gap)\n")
        f.write("-" * 40 + "\n")
        quality_rank = df.groupby('algorithm')['gap_percent'].mean().sort_values()
        for i, (algo, gap) in enumerate(quality_rank.head(5).items(), 1):
            f.write(f"  {i}. {algo}: {gap:.2f}% average gap\n")
        f.write("\n")
        
        # Fastest algorithms
        f.write("3. FASTEST ALGORITHMS (Average Time)\n")
        f.write("-" * 40 + "\n")
        speed_rank = df.groupby('algorithm')['time_seconds'].mean().sort_values()
        for i, (algo, time) in enumerate(speed_rank.head(5).items(), 1):
            f.write(f"  {i}. {algo}: {time:.4f}s average\n")
        f.write("\n")
        
        # Algorithms finding optimal
        f.write("4. OPTIMAL SOLUTION RATE\n")
        f.write("-" * 40 + "\n")
        optimal_count = (df['gap_percent'] == 0).groupby(df['algorithm']).sum()
        total_count = df.groupby('algorithm').size()
        optimal_rate = (optimal_count / total_count * 100).sort_values(ascending=False)
        for algo, rate in optimal_rate.items():
            f.write(f"  {algo}: {rate:.1f}% optimal\n")
        f.write("\n")
        
        # Per-algorithm detailed stats
        f.write("5. DETAILED ALGORITHM STATISTICS\n")
        f.write("-" * 40 + "\n")
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            f.write(f"\n  {algo}:\n")
            f.write(f"    - Avg Gap: {algo_data['gap_percent'].mean():.2f}%\n")
            f.write(f"    - Min Gap: {algo_data['gap_percent'].min():.2f}%\n")
            f.write(f"    - Max Gap: {algo_data['gap_percent'].max():.2f}%\n")
            f.write(f"    - Avg Time: {algo_data['time_seconds'].mean():.4f}s\n")
            f.write(f"    - Max Time: {algo_data['time_seconds'].max():.4f}s\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"âœ“[OK] Generated: ANALYSIS_SUMMARY.txt")


def generate_all_visualizations(results_path: str, output_dir: str = None):
    """
    Main function to generate all visualizations.
    
    Args:
        results_path: Path to the CSV file with benchmark results
        output_dir: Directory to save the graphs (default: results/graphs/)
    """
    print("\n" + "=" * 60)
    print("GENERATING BENCHMARK VISUALIZATIONS")
    print("=" * 60 + "\n")
    
    # Load data
    print(f"Loading results from: {results_path}")
    df = load_results(results_path)
    print(f"Loaded {len(df)} records\n")
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(results_path), 'graphs')
    create_output_directory(output_dir)
    print(f"Output directory: {output_dir}\n")
    
    # Generate all plots
    print("Generating visualizations...")
    print("-" * 40)
    
    try:
        plot_execution_time_comparison(df, output_dir)
        plot_gap_comparison(df, output_dir)
        plot_scalability_analysis(df, output_dir)
        plot_quality_vs_time_tradeoff(df, output_dir)
        plot_performance_heatmap(df, output_dir)
        plot_time_heatmap(df, output_dir)
        plot_boxplot_by_algorithm(df, output_dir)
        plot_boxplot_time_by_algorithm(df, output_dir)
        plot_algorithm_ranking(df, output_dir)
        plot_category_comparison(df, output_dir)
        plot_convergence_summary(df, output_dir)
        generate_summary_report(df, output_dir)
        
        print("\n" + "=" * 60)
        print(f"âœ… All visualizations generated successfully!")
        print(f"ðŸ“ Output directory: {output_dir}")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\nâŒ Error generating visualizations: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    # Default path
    default_results = "results/benchmark_results.csv"
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = default_results
    
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    generate_all_visualizations(results_path, output_dir)

