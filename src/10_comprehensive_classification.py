"""
25 Farklı Classification Algoritması ile Kapsamlı Test.
Her transfer learning feature set'i üzerinde 25 classifier test eder.

Classifiers:
- Linear: LogisticRegression, Ridge, LinearSVC, SGD, Perceptron
- SVM: SVC, NuSVC
- Tree-based: DecisionTree, ExtraTree, RandomForest, ExtraTrees
- Ensemble: Bagging, Boosting (Ada, Gradient, HistGradient), Voting
- Naive Bayes: Gaussian, Bernoulli
- NN: MLPClassifier
- Neighbors: KNN, NearestCentroid
- Discriminant: LDA, QDA
- Others: PassiveAggressive

Kullanım:
    python src/10_comprehensive_classification.py --feature_dir outputs/features_tl
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_validate, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Linear models
from sklearn.linear_model import (
    LogisticRegression, LogisticRegressionCV,
    RidgeClassifier, RidgeClassifierCV,
    SGDClassifier, PassiveAggressiveClassifier, Perceptron
)

# SVM
from sklearn.svm import SVC, LinearSVC, NuSVC

# Trees and Ensembles
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    BaggingClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    VotingClassifier
)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB, BernoulliNB

# Neural Network
from sklearn.neural_network import MLPClassifier

# Neighbors
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

# Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from tqdm import tqdm


# ============================================================================
# CLASSIFIER FACTORY
# ============================================================================

def get_all_classifiers():
    """
    25 farklı classification algoritmasını döndürür.
    
    Returns:
        dict: {classifier_name: sklearn_pipeline}
    """
    
    classifiers = {}
    
    # Linear Models (scaled)
    classifiers['LogisticRegression'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    classifiers['LogisticRegressionCV'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegressionCV(max_iter=1000, random_state=42, cv=5))
    ])
    
    classifiers['RidgeClassifier'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RidgeClassifier(random_state=42))
    ])
    
    classifiers['RidgeClassifierCV'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RidgeClassifierCV(cv=5))
    ])
    
    classifiers['LinearSVC'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearSVC(max_iter=2000, random_state=42, dual=False))
    ])
    
    classifiers['SGDClassifier'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SGDClassifier(max_iter=1000, random_state=42))
    ])
    
    classifiers['PassiveAggressiveClassifier'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', PassiveAggressiveClassifier(max_iter=1000, random_state=42))
    ])
    
    classifiers['Perceptron'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', Perceptron(max_iter=1000, random_state=42))
    ])
    
    # SVM (scaled)
    classifiers['SVC'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', random_state=42, probability=True))
    ])
    
    classifiers['NuSVC'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', NuSVC(nu=0.5, random_state=42, probability=True))
    ])
    
    # Tree-based (no scaling needed)
    classifiers['DecisionTreeClassifier'] = Pipeline([
        ('clf', DecisionTreeClassifier(random_state=42))
    ])
    
    classifiers['ExtraTreeClassifier'] = Pipeline([
        ('clf', ExtraTreeClassifier(random_state=42))
    ])
    
    classifiers['RandomForestClassifier'] = Pipeline([
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    classifiers['ExtraTreesClassifier'] = Pipeline([
        ('clf', ExtraTreesClassifier(n_estimators=100, random_state=42))
    ])
    
    # Ensemble methods
    classifiers['BaggingClassifier'] = Pipeline([
        ('clf', BaggingClassifier(n_estimators=10, random_state=42))
    ])
    
    classifiers['AdaBoostClassifier'] = Pipeline([
        ('clf', AdaBoostClassifier(n_estimators=50, random_state=42))
    ])
    
    classifiers['GradientBoostingClassifier'] = Pipeline([
        ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
    
    classifiers['HistGradientBoostingClassifier'] = Pipeline([
        ('clf', HistGradientBoostingClassifier(max_iter=100, random_state=42))
    ])
    
    # Naive Bayes
    classifiers['GaussianNB'] = Pipeline([
        ('clf', GaussianNB())
    ])
    
    classifiers['BernoulliNB'] = Pipeline([
        ('clf', BernoulliNB())
    ])
    
    # Neural Network (scaled)
    classifiers['MLPClassifier'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, 
                             random_state=42, early_stopping=True))
    ])
    
    # Neighbors (scaled)
    classifiers['KNeighborsClassifier'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ])
    
    classifiers['NearestCentroid'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', NearestCentroid())
    ])
    
    # Discriminant Analysis (scaled)
    classifiers['LinearDiscriminantAnalysis'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearDiscriminantAnalysis())
    ])
    
    classifiers['QuadraticDiscriminantAnalysis'] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', QuadraticDiscriminantAnalysis())
    ])
    
    # Voting Classifier (ensemble of top 3) - REMOVED: too slow for large datasets
    # classifiers['VotingClassifier'] = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('clf', VotingClassifier(estimators=[
    #         ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    #         ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    #         ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    #     ], voting='soft'))
    # ])
    
    return classifiers


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_classifiers(X, y, groups, classifiers, n_splits=10):
    """
    Tüm classifiers'ı GroupKFold CV ile değerlendirir.
    
    Args:
        X: Feature matrix
        y: Labels
        groups: Subject IDs
        classifiers: Dict of classifiers
        n_splits: CV fold sayısı
        
    Returns:
        pd.DataFrame: Results with metrics
    """
    
    cv = GroupKFold(n_splits=n_splits)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    
    results = []
    
    for clf_name, clf in tqdm(classifiers.items(), desc="Evaluating classifiers"):
        try:
            # Cross-validate
            cv_results = cross_validate(
                clf, X, y, groups=groups, cv=cv,
                scoring=scoring, n_jobs=1, error_score='raise'
            )
            
            # Aggregate results
            result = {
                'Classification Method': clf_name,
                'Accuracy': np.mean(cv_results['test_accuracy']),
                'Recall': np.mean(cv_results['test_recall']),
                'Precision': np.mean(cv_results['test_precision']),
                'F1 Score': np.mean(cv_results['test_f1']),
                'Accuracy_std': np.std(cv_results['test_accuracy']),
                'Recall_std': np.std(cv_results['test_recall']),
                'Precision_std': np.std(cv_results['test_precision']),
                'F1_std': np.std(cv_results['test_f1'])
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"\n⚠️ Error with {clf_name}: {e}")
            results.append({
                'Classification Method': clf_name,
                'Accuracy': 0.0,
                'Recall': 0.0,
                'Precision': 0.0,
                'F1 Score': 0.0,
                'error': str(e)
            })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    return df_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Classification with 25 Algorithms')
    parser.add_argument('--feature_dir', type=str, default='outputs/features_tl',
                       help='Directory containing feature parquet files')
    parser.add_argument('--output_dir', type=str, default='outputs/paper_results',
                       help='Output directory for results')
    parser.add_argument('--n_splits', type=int, default=10,
                       help='Number of CV folds')
    parser.add_argument('--models', type=str, default='all',
                       help='Comma-separated feature file names or "all"')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔬 COMPREHENSIVE CLASSIFICATION - 25 ALGORITHMS")
    print("=" * 80)
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get feature files
    feature_dir = Path(args.feature_dir)
    if not feature_dir.exists():
        print(f"❌ Feature directory not found: {feature_dir}")
        print("   Run 09_multiple_transfer_learning.py first!")
        return
    
    feature_files = list(feature_dir.glob('features_*.parquet'))
    
    if args.models != 'all':
        model_names = [m.strip() for m in args.models.split(',')]
        feature_files = [f for f in feature_files if any(m in f.stem for m in model_names)]
    
    print(f"\n📂 Found {len(feature_files)} feature files")
    
    # Get classifiers
    classifiers = get_all_classifiers()
    print(f"🤖 Testing {len(classifiers)} classification algorithms")
    print(f"🎯 Using {args.n_splits}-fold GroupKFold CV")
    
    # Process each feature file
    all_results = []
    
    for feature_file in feature_files:
        model_name = feature_file.stem.replace('features_', '')
        
        print(f"\n{'='*80}")
        print(f"📊 Processing: {model_name}")
        print(f"{'='*80}")
        
        # Load features
        df = pd.read_parquet(feature_file)
        
        # Extract X, y, groups
        feature_cols = [c for c in df.columns if c.startswith('f')]
        X = df[feature_cols].values
        y = df['label'].values
        groups = df['subject_id'].values
        
        print(f"   Features: {X.shape}")
        print(f"   Labels: {len(np.unique(y))} classes")
        print(f"   Subjects: {len(np.unique(groups))} unique")
        
        # Evaluate
        print(f"\n   🔄 Evaluating {len(classifiers)} classifiers...")
        start_time = time.time()
        
        df_results = evaluate_classifiers(X, y, groups, classifiers, n_splits=args.n_splits)
        
        elapsed = time.time() - start_time
        print(f"   ✅ Complete in {elapsed:.2f}s")
        
        # Add model name
        df_results['Transfer Learning Method'] = model_name
        
        # Save individual results
        output_path = output_dir / f'results_{model_name}.csv'
        df_results.to_csv(output_path, index=False)
        print(f"   💾 Saved: {output_path.name}")
        
        # Show top 5
        print(f"\n   🏆 Top 5 Classifiers:")
        for i, row in df_results.head(5).iterrows():
            print(f"      {i+1}. {row['Classification Method']:<40} "
                  f"Acc: {row['Accuracy']:.6f}  "
                  f"F1: {row['F1 Score']:.6f}")
        
        all_results.append(df_results)
    
    # Combine all results
    print(f"\n{'='*80}")
    print("📊 GENERATING SUMMARY")
    print(f"{'='*80}")
    
    df_all = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    combined_path = output_dir / 'all_results_combined.csv'
    df_all.to_csv(combined_path, index=False)
    print(f"✅ Combined results saved: {combined_path}")
    
    # Table 1: Best classifier per TL model
    print(f"\n📋 Table 1: Best Classifier per Transfer Learning Model")
    df_table1 = df_all.loc[df_all.groupby('Transfer Learning Method')['Accuracy'].idxmax()]
    df_table1 = df_table1[['Transfer Learning Method', 'Classification Method', 'Accuracy']].sort_values('Accuracy', ascending=False)
    
    table1_path = output_dir / 'table1_best_per_model.csv'
    df_table1.to_csv(table1_path, index=False)
    print(f"💾 Saved: {table1_path}")
    
    # Table 2: All classifiers for best TL model
    best_model = df_table1.iloc[0]['Transfer Learning Method']
    print(f"\n📋 Table 2: All Classifiers for Best Model ({best_model})")
    df_table2 = df_all[df_all['Transfer Learning Method'] == best_model]
    df_table2 = df_table2[['Classification Method', 'Accuracy', 'Recall', 'Precision', 'F1 Score']].sort_values('Accuracy', ascending=False)
    
    table2_path = output_dir / f'table2_all_classifiers_{best_model}.csv'
    df_table2.to_csv(table2_path, index=False)
    print(f"💾 Saved: {table2_path}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("✨ SUMMARY")
    print(f"{'='*80}")
    print(f"📊 Total experiments: {len(df_all)}")
    print(f"🤖 Transfer Learning models: {len(df_all['Transfer Learning Method'].unique())}")
    print(f"🔬 Classification algorithms: {len(df_all['Classification Method'].unique())}")
    print(f"\n🏆 BEST OVERALL:")
    best_row = df_all.loc[df_all['Accuracy'].idxmax()]
    print(f"   TL Model: {best_row['Transfer Learning Method']}")
    print(f"   Classifier: {best_row['Classification Method']}")
    print(f"   Accuracy: {best_row['Accuracy']:.6f}")
    print(f"   F1 Score: {best_row['F1 Score']:.6f}")
    
    print(f"\n📁 All results saved in: {output_dir}")
    print(f"\n🎯 Next step: Generate paper-style tables")
    print(f"   python src/11_generate_paper_tables.py")


if __name__ == "__main__":
    main()
