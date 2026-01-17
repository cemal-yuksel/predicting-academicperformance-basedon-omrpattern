"""
Individual-Level Performance Analysis
Evaluates model performance for each student separately
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
import warnings
warnings.filterwarnings('ignore')

def load_features(feature_path):
    """Load feature file"""
    df = pd.read_parquet(feature_path)
    print(f"✓ Loaded features: {df.shape}")
    return df

def analyze_individual_performance(features_df, model_name='efficientnet_b0'):
    """Analyze performance for each individual student"""
    
    print("\n" + "="*80)
    print(f"👤 INDIVIDUAL PERFORMANCE ANALYSIS - {model_name.upper()}")
    print("="*80)
    
    # Extract features, labels, and subjects
    feature_cols = [col for col in features_df.columns if col.startswith('f')]
    X = features_df[feature_cols].values
    y = features_df['label'].values
    subjects = features_df['subject_id'].values
    
    print(f"\n📊 Dataset info:")
    print(f"   Total samples: {len(X)}")
    print(f"   Unique subjects: {len(np.unique(subjects))}")
    print(f"   Features: {len(feature_cols)}")
    
    # Create classifier pipeline (using best model: SVC)
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
    ])
    
    # Leave-One-Group-Out (each subject is left out once)
    logo = LeaveOneGroupOut()
    
    results = []
    
    print("\n🔄 Evaluating individual performance...")
    print(f"{'Subject':<12}{'Samples':<10}{'Accuracy':<12}{'Recall':<12}{'Precision':<12}{'F1 Score':<12}")
    print("-"*80)
    
    for train_idx, test_idx in logo.split(X, y, groups=subjects):
        # Get test subject
        test_subject = subjects[test_idx][0]
        
        # Train and test
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
        prec = precision_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
        
        results.append({
            'Person': test_subject,
            'Samples': len(test_idx),
            'Accuracy': acc,
            'Recall': rec,
            'Precision': prec,
            'F1 Score': f1,
            'True_Label': y_test[0]  # All samples from same subject have same label
        })
        
        print(f"{test_subject:<12}{len(test_idx):<10}{acc:<12.6f}{rec:<12.6f}{prec:<12.6f}{f1:<12.6f}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Sort by accuracy
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    return results_df

def generate_summary(results_df, output_dir):
    """Generate summary statistics and save results"""
    
    print("\n" + "="*80)
    print("📊 SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\n{'Metric':<20}{'Mean':<12}{'Std':<12}{'Min':<12}{'Max':<12}")
    print("-"*80)
    
    for metric in ['Accuracy', 'Recall', 'Precision', 'F1 Score']:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        min_val = results_df[metric].min()
        max_val = results_df[metric].max()
        print(f"{metric:<20}{mean_val:<12.6f}{std_val:<12.6f}{min_val:<12.6f}{max_val:<12.6f}")
    
    # Top 10 and Bottom 10
    print("\n🏆 TOP 10 INDIVIDUALS (by accuracy):")
    print(f"{'Rank':<6}{'Person':<12}{'Accuracy':<12}{'Recall':<12}{'Precision':<12}{'F1':<12}")
    print("-"*80)
    for idx, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"{idx:<6}{row['Person']:<12}{row['Accuracy']:<12.6f}{row['Recall']:<12.6f}"
              f"{row['Precision']:<12.6f}{row['F1 Score']:<12.6f}")
    
    print("\n⚠️ BOTTOM 10 INDIVIDUALS (by accuracy):")
    print(f"{'Rank':<6}{'Person':<12}{'Accuracy':<12}{'Recall':<12}{'Precision':<12}{'F1':<12}")
    print("-"*80)
    for idx, (_, row) in enumerate(results_df.tail(10).iterrows(), 1):
        print(f"{idx:<6}{row['Person']:<12}{row['Accuracy']:<12.6f}{row['Recall']:<12.6f}"
              f"{row['Precision']:<12.6f}{row['F1 Score']:<12.6f}")
    
    # Save to CSV
    output_path = Path(output_dir) / 'table3_individual_performance.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    return output_path

def main():
    # Setup
    feature_dir = Path('outputs/features_tl')
    output_dir = Path('outputs/paper_results')
    
    # Use best model (EfficientNet-B0)
    best_model = 'efficientnet_b0'
    feature_file = feature_dir / f'features_{best_model}.parquet'
    
    print("="*80)
    print("👤 INDIVIDUAL-LEVEL PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"\nUsing best model: {best_model.upper()}")
    print(f"Classifier: Support Vector Classification (SVC)")
    print(f"Strategy: Leave-One-Subject-Out Cross-Validation")
    
    # Load features
    features_df = load_features(feature_file)
    
    # Analyze individual performance
    results_df = analyze_individual_performance(features_df, best_model)
    
    # Generate summary and save
    output_path = generate_summary(results_df, output_dir)
    
    print("\n" + "="*80)
    print("✅ INDIVIDUAL ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📄 Table 3 generated: {output_path}")
    print(f"📊 Analyzed {len(results_df)} individuals")
    print(f"🎯 Mean accuracy: {results_df['Accuracy'].mean():.4f}")
    print(f"📈 Accuracy range: {results_df['Accuracy'].min():.4f} - {results_df['Accuracy'].max():.4f}")

if __name__ == '__main__':
    main()
