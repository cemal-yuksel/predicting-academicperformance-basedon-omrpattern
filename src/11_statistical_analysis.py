"""
Statistical Analysis for Transfer Learning Classification Results
Performs Friedman test and post-hoc Wilcoxon signed-rank tests
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_all_results(results_dir):
    """Load combined results"""
    combined_file = Path(results_dir) / 'all_results_combined.csv'
    df = pd.read_csv(combined_file)
    
    # Rename columns to match expected names
    df = df.rename(columns={
        'Transfer Learning Method': 'model',
        'Classification Method': 'classifier',
        'Accuracy': 'accuracy',
        'Recall': 'recall',
        'Precision': 'precision',
        'F1 Score': 'f1'
    })
    
    print(f"✓ Loaded {len(df)} results from {len(df['model'].unique())} models")
    return df

def perform_friedman_test(df):
    """Perform Friedman test across all models for each classifier"""
    print("\n" + "="*80)
    print("🔬 FRIEDMAN TEST - Comparing Transfer Learning Models")
    print("="*80)
    
    # Pivot: rows=classifiers, columns=models, values=accuracy
    pivot_df = df.pivot(index='classifier', columns='model', values='accuracy')
    
    # Remove classifiers with missing values
    pivot_df = pivot_df.dropna()
    
    print(f"\n📊 Testing {len(pivot_df)} classifiers across {len(pivot_df.columns)} models")
    
    # Friedman test
    statistic, p_value = stats.friedmanchisquare(*[pivot_df[col].values for col in pivot_df.columns])
    
    print(f"\n{'Test Statistic':<25}: χ²({len(pivot_df.columns)-1}) = {statistic:.4f}")
    print(f"{'P-value':<25}: {p_value:.2e}")
    
    if p_value < 0.001:
        print(f"{'Result':<25}: *** Highly significant (p < .001)")
    elif p_value < 0.01:
        print(f"{'Result':<25}: ** Significant (p < .01)")
    elif p_value < 0.05:
        print(f"{'Result':<25}: * Significant (p < .05)")
    else:
        print(f"{'Result':<25}: Not significant (p >= .05)")
    
    print("\n💡 Interpretation: Significant differences exist between transfer learning models")
    
    return statistic, p_value, pivot_df

def model_rankings(df):
    """Calculate and display model rankings"""
    print("\n" + "="*80)
    print("🏆 TRANSFER LEARNING MODEL RANKINGS")
    print("="*80)
    
    # Group by model and calculate statistics
    model_stats = df.groupby('model')['accuracy'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(6)
    
    # Sort by mean accuracy
    model_stats = model_stats.sort_values('mean', ascending=False)
    
    print(f"\n{'Rank':<6}{'Model':<25}{'Mean Acc':<12}{'Std':<10}{'Median':<10}{'Min':<10}{'Max':<10}")
    print("-"*80)
    
    for rank, (model, row) in enumerate(model_stats.iterrows(), 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{emoji} {rank:<4}{model:<25}{row['mean']:.6f}  {row['std']:.6f}  {row['median']:.6f}  {row['min']:.6f}  {row['max']:.6f}")
    
    return model_stats

def classifier_rankings(df):
    """Calculate and display classifier rankings"""
    print("\n" + "="*80)
    print("🤖 CLASSIFICATION ALGORITHM RANKINGS")
    print("="*80)
    
    # Group by classifier and calculate statistics
    clf_stats = df.groupby('classifier')['accuracy'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).round(6)
    
    # Sort by mean accuracy
    clf_stats = clf_stats.sort_values('mean', ascending=False)
    
    print(f"\n{'Rank':<6}{'Classifier':<35}{'Mean Acc':<12}{'Std':<10}{'Count':<8}")
    print("-"*80)
    
    for rank, (clf, row) in enumerate(clf_stats.iterrows(), 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{emoji} {rank:<4}{clf:<35}{row['mean']:.6f}  {row['std']:.6f}  {int(row['count']):<8}")
    
    return clf_stats

def post_hoc_wilcoxon(pivot_df, alpha=0.05):
    """Perform post-hoc Wilcoxon signed-rank tests with Bonferroni correction"""
    print("\n" + "="*80)
    print("📈 POST-HOC ANALYSIS - Pairwise Wilcoxon Signed-Rank Tests")
    print("="*80)
    
    models = list(pivot_df.columns)
    n_comparisons = len(models) * (len(models) - 1) // 2
    bonferroni_alpha = alpha / n_comparisons
    
    print(f"\n🔢 Total comparisons: {n_comparisons}")
    print(f"🎯 Bonferroni corrected α: {bonferroni_alpha:.6f}")
    
    # Sort models by mean accuracy
    model_means = pivot_df.mean().sort_values(ascending=False)
    top_models = model_means.head(5).index.tolist()
    
    print(f"\n🏆 Top 5 models: {', '.join(top_models)}")
    print("\nComparing top 5 models (pairwise):")
    print(f"{'Model 1':<25} vs {'Model 2':<25} {'Statistic':<12} {'P-value':<12} {'Significant'}")
    print("-"*90)
    
    significant_pairs = []
    
    for i, model1 in enumerate(top_models):
        for model2 in top_models[i+1:]:
            stat, p_val = stats.wilcoxon(pivot_df[model1], pivot_df[model2])
            is_sig = p_val < bonferroni_alpha
            sig_mark = "***" if is_sig else ""
            
            print(f"{model1:<25} vs {model2:<25} {stat:<12.2f} {p_val:<12.6f} {sig_mark}")
            
            if is_sig:
                significant_pairs.append((model1, model2, p_val))
    
    if significant_pairs:
        print(f"\n✓ Found {len(significant_pairs)} significant pairwise differences")
    else:
        print("\n✗ No significant pairwise differences after Bonferroni correction")
    
    return significant_pairs

def generate_summary_report(df, model_stats, clf_stats, friedman_stat, friedman_p, output_dir):
    """Generate comprehensive summary report"""
    print("\n" + "="*80)
    print("📝 GENERATING SUMMARY REPORT")
    print("="*80)
    
    report_path = Path(output_dir) / 'statistical_analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("STATISTICAL ANALYSIS REPORT\n")
        f.write("Transfer Learning for OMR Classification\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("EXPERIMENT OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Transfer learning models: {len(df['model'].unique())}\n")
        f.write(f"Classification algorithms: {len(df['classifier'].unique())}\n")
        f.write(f"Overall mean accuracy: {df['accuracy'].mean():.6f}\n")
        f.write(f"Overall std accuracy: {df['accuracy'].std():.6f}\n")
        f.write(f"Best result: {df['accuracy'].max():.6f}\n")
        f.write(f"Worst result: {df['accuracy'].min():.6f}\n\n")
        
        # Friedman test
        f.write("FRIEDMAN TEST RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"Test statistic: χ²({len(model_stats)-1}) = {friedman_stat:.4f}\n")
        f.write(f"P-value: {friedman_p:.2e}\n")
        f.write(f"Conclusion: {'Significant' if friedman_p < 0.05 else 'Not significant'}\n\n")
        
        # Top models
        f.write("TOP 10 TRANSFER LEARNING MODELS (by mean accuracy)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<6}{'Model':<25}{'Mean':<12}{'Std':<10}{'Median':<10}\n")
        f.write("-"*80 + "\n")
        for rank, (model, row) in enumerate(model_stats.head(10).iterrows(), 1):
            f.write(f"{rank:<6}{model:<25}{row['mean']:.6f}  {row['std']:.6f}  {row['median']:.6f}\n")
        f.write("\n")
        
        # Top classifiers
        f.write("TOP 10 CLASSIFICATION ALGORITHMS (by mean accuracy)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<6}{'Classifier':<35}{'Mean':<12}{'Std':<10}\n")
        f.write("-"*80 + "\n")
        for rank, (clf, row) in enumerate(clf_stats.head(10).iterrows(), 1):
            f.write(f"{rank:<6}{clf:<35}{row['mean']:.6f}  {row['std']:.6f}\n")
        f.write("\n")
        
        # Best combinations
        f.write("TOP 10 MODEL-CLASSIFIER COMBINATIONS\n")
        f.write("-"*80 + "\n")
        top_combos = df.nlargest(10, 'accuracy')[['model', 'classifier', 'accuracy']]
        f.write(f"{'Rank':<6}{'Model':<25}{'Classifier':<35}{'Accuracy':<12}\n")
        f.write("-"*80 + "\n")
        for rank, (_, row) in enumerate(top_combos.iterrows(), 1):
            f.write(f"{rank:<6}{row['model']:<25}{row['classifier']:<35}{row['accuracy']:.6f}\n")
    
    print(f"\n✓ Report saved to: {report_path}")
    return report_path

def main():
    # Setup
    results_dir = Path('outputs/paper_results')
    
    print("="*80)
    print("🔬 STATISTICAL ANALYSIS - TRANSFER LEARNING CLASSIFICATION")
    print("="*80)
    
    # Load data
    df = load_all_results(results_dir)
    
    # Model rankings
    model_stats = model_rankings(df)
    
    # Classifier rankings
    clf_stats = classifier_rankings(df)
    
    # Friedman test
    friedman_stat, friedman_p, pivot_df = perform_friedman_test(df)
    
    # Post-hoc tests
    significant_pairs = post_hoc_wilcoxon(pivot_df)
    
    # Generate report
    report_path = generate_summary_report(df, model_stats, clf_stats, 
                                         friedman_stat, friedman_p, results_dir)
    
    print("\n" + "="*80)
    print("✅ STATISTICAL ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📊 Key Findings:")
    print(f"   • Best model: {model_stats.index[0]} ({model_stats.iloc[0]['mean']:.4f})")
    print(f"   • Best classifier: {clf_stats.index[0]} ({clf_stats.iloc[0]['mean']:.4f})")
    print(f"   • Friedman test: χ² = {friedman_stat:.4f}, p = {friedman_p:.2e}")
    print(f"\n📝 Full report: {report_path}")

if __name__ == '__main__':
    main()
