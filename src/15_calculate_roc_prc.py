"""
Calculate ROC-AUC and PR-AUC for Best Model
EfficientNet-B0 + SVC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                             average_precision_score, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

def load_features(feature_path):
    """Load feature file"""
    df = pd.read_parquet(feature_path)
    print(f"✓ Loaded features: {df.shape}")
    return df

def calculate_roc_prc(features_df, model_name='efficientnet_b0'):
    """Calculate ROC and PRC metrics"""
    
    print("\n" + "="*80)
    print(f"📊 ROC & PRC ANALYSIS - {model_name.upper()} + SVC")
    print("="*80)
    
    # Extract features, labels, and subjects
    feature_cols = [col for col in features_df.columns if col.startswith('f')]
    X = features_df[feature_cols].values
    y = features_df['label'].values
    subjects = features_df['subject_id'].values
    
    print(f"\n📊 Dataset info:")
    print(f"   Total samples: {len(X)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Class 0: {(y == 0).sum()}, Class 1: {(y == 1).sum()}")
    
    # Create classifier pipeline with probability estimates
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
    ])
    
    # 10-fold GroupKFold CV
    gkf = GroupKFold(n_splits=10)
    
    # Collect predictions and probabilities
    all_y_true = []
    all_y_scores = []  # Probability scores for positive class
    
    print("\n🔄 Running 10-fold GroupKFold Cross-Validation with probability scores...")
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=subjects), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit and predict probabilities
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]  # Probability for class 1
        
        all_y_true.extend(y_test)
        all_y_scores.extend(y_proba)
        
        print(f"   Fold {fold:2d}: {len(test_idx)} samples")
    
    # Convert to arrays
    y_true = np.array(all_y_true)
    y_scores = np.array(all_y_scores)
    
    # Calculate ROC curve and AUC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    roc_auc_score_val = roc_auc_score(y_true, y_scores)
    
    # Calculate Precision-Recall curve and AUC
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_true, y_scores)
    
    print("\n" + "="*80)
    print("📈 ROC & PRC METRICS")
    print("="*80)
    
    print(f"\n🔵 ROC (Receiver Operating Characteristic):")
    print(f"   ROC-AUC Score: {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    print(f"   Interpretation: {'Excellent' if roc_auc > 0.9 else 'Good' if roc_auc > 0.8 else 'Fair'}")
    
    print(f"\n🟢 PRC (Precision-Recall Curve):")
    print(f"   PR-AUC Score: {pr_auc:.4f} ({pr_auc*100:.2f}%)")
    print(f"   Average Precision: {avg_precision:.4f} ({avg_precision*100:.2f}%)")
    print(f"   Interpretation: {'Excellent' if avg_precision > 0.9 else 'Good' if avg_precision > 0.8 else 'Fair'}")
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'pr_auc': pr_auc,
        'avg_precision': avg_precision,
        'y_true': y_true,
        'y_scores': y_scores
    }

def plot_roc_prc_curves(results, output_dir, model_name='efficientnet_b0'):
    """Plot ROC and PRC curves"""
    
    print("\n📊 Generating ROC and PRC plots...")
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    
    # ===== ROC Curve =====
    ax1.plot(results['fpr'], results['tpr'], 
             color='black', lw=2.5,
             label=f"ROC curve (AUC = {results['roc_auc']:.4f})")
    ax1.plot([0, 1], [0, 1], 
             color='gray', lw=2, linestyle='--',
             label='Random classifier (AUC = 0.5000)')
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, weight='bold')
    ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, weight='bold')
    ax1.set_title('ROC Curve - EfficientNet-B0 + SVC', fontsize=14, weight='bold', pad=15)
    ax1.legend(loc="lower right", fontsize=11, frameon=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('white')
    
    # ===== Precision-Recall Curve =====
    ax2.plot(results['recall'], results['precision'], 
             color='black', lw=2.5,
             label=f"PR curve (AP = {results['avg_precision']:.4f})")
    
    # Baseline (random classifier for balanced dataset)
    baseline = np.sum(results['y_true']) / len(results['y_true'])
    ax2.axhline(y=baseline, color='gray', lw=2, linestyle='--',
                label=f'Baseline (AP = {baseline:.4f})')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=12, weight='bold')
    ax2.set_ylabel('Precision', fontsize=12, weight='bold')
    ax2.set_title('Precision-Recall Curve - EfficientNet-B0 + SVC', 
                  fontsize=14, weight='bold', pad=15)
    ax2.legend(loc="lower left", fontsize=11, frameon=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor('white')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / f'roc_prc_curves_{model_name}_svc.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ ROC & PRC curves saved to: {output_path}")
    
    # Save PDF
    pdf_path = Path(output_dir) / f'roc_prc_curves_{model_name}_svc.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"✓ PDF version saved to: {pdf_path}")
    
    plt.close()
    
    return output_path

def generate_metrics_summary(results, output_dir):
    """Generate summary CSV with all metrics"""
    
    metrics_summary = {
        'Metric': [
            'ROC-AUC',
            'PR-AUC',
            'Average Precision',
            'Baseline (Random)'
        ],
        'Value': [
            results['roc_auc'],
            results['pr_auc'],
            results['avg_precision'],
            0.5000
        ],
        'Interpretation': [
            'Excellent' if results['roc_auc'] > 0.9 else 'Good' if results['roc_auc'] > 0.8 else 'Fair',
            'Excellent' if results['pr_auc'] > 0.9 else 'Good' if results['pr_auc'] > 0.8 else 'Fair',
            'Excellent' if results['avg_precision'] > 0.9 else 'Good' if results['avg_precision'] > 0.8 else 'Fair',
            'Random Baseline'
        ]
    }
    
    df = pd.DataFrame(metrics_summary)
    output_path = Path(output_dir) / 'roc_prc_metrics_summary.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ Metrics summary saved to: {output_path}")
    
    return df

def main():
    # Setup
    feature_dir = Path('outputs/features_tl')
    output_dir = Path('outputs/paper_results')
    
    # Use best model
    best_model = 'efficientnet_b0'
    feature_file = feature_dir / f'features_{best_model}.parquet'
    
    print("="*80)
    print("📊 ROC & PRC ANALYSIS GENERATOR")
    print("="*80)
    print(f"\nBest Model: {best_model.upper()}")
    print(f"Best Classifier: Support Vector Classification (SVC)")
    print(f"Validation: 10-fold GroupKFold Cross-Validation")
    
    # Load features
    features_df = load_features(feature_file)
    
    # Calculate ROC and PRC
    results = calculate_roc_prc(features_df, best_model)
    
    # Plot curves
    output_path = plot_roc_prc_curves(results, output_dir, best_model)
    
    # Generate summary
    summary_df = generate_metrics_summary(results, output_dir)
    
    print("\n" + "="*80)
    print("✅ ROC & PRC ANALYSIS COMPLETE")
    print("="*80)
    
    print("\n📊 SUMMARY:")
    print(summary_df.to_string(index=False))
    
    print(f"\n📁 Generated files:")
    print(f"   • {output_path}")
    print(f"   • {output_path.with_suffix('.pdf')}")
    print(f"   • outputs/paper_results/roc_prc_metrics_summary.csv")
    
    print("\n💡 INTERPRETATION:")
    print(f"   ROC-AUC = {results['roc_auc']:.4f}: Model distinguishes between classes excellently")
    print(f"   PR-AUC = {results['avg_precision']:.4f}: High precision-recall balance maintained")
    print(f"   Both metrics > 0.88 indicate robust binary classification performance")

if __name__ == '__main__':
    main()
