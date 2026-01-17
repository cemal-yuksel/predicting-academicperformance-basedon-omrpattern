"""
Generate Confusion Matrix for Best Model
EfficientNet-B0 + SVC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_features(feature_path):
    """Load feature file"""
    df = pd.read_parquet(feature_path)
    print(f"✓ Loaded features: {df.shape}")
    return df

def generate_confusion_matrix(features_df, model_name='efficientnet_b0'):
    """Generate confusion matrix using 10-fold GroupKFold CV"""
    
    print("\n" + "="*80)
    print(f"📊 CONFUSION MATRIX GENERATION - {model_name.upper()} + SVC")
    print("="*80)
    
    # Extract features, labels, and subjects
    feature_cols = [col for col in features_df.columns if col.startswith('f')]
    X = features_df[feature_cols].values
    y = features_df['label'].values
    subjects = features_df['subject_id'].values
    
    print(f"\n📊 Dataset info:")
    print(f"   Total samples: {len(X)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Class 0 (Low GPA): {(y == 0).sum()}")
    print(f"   Class 1 (High GPA): {(y == 1).sum()}")
    
    # Create classifier pipeline (best model: SVC)
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
    ])
    
    # 10-fold GroupKFold CV
    gkf = GroupKFold(n_splits=10)
    
    # Collect all predictions
    all_y_true = []
    all_y_pred = []
    
    print("\n🔄 Running 10-fold GroupKFold Cross-Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=subjects), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        print(f"   Fold {fold:2d}: {len(test_idx)} samples")
    
    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    
    print(f"\n✓ Confusion Matrix Generated:")
    print(f"   True Negatives (TN): {cm[0,0]}")
    print(f"   False Positives (FP): {cm[0,1]}")
    print(f"   False Negatives (FN): {cm[1,0]}")
    print(f"   True Positives (TP): {cm[1,1]}")
    
    # Calculate metrics
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    return cm, all_y_true, all_y_pred

def plot_confusion_matrix(cm, output_dir, model_name='efficientnet_b0'):
    """Plot confusion matrix in grayscale (black & white style)"""
    
    print("\n📊 Generating Confusion Matrix Plot...")
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot with grayscale colormap (black to white)
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Greys',  # Grayscale: white to black
                cbar_kws={'label': ''},
                linewidths=2,
                linecolor='black',
                square=True,
                annot_kws={'size': 24, 'weight': 'bold'},
                vmin=0,
                ax=ax)
    
    # Set labels
    ax.set_xlabel('Predicted Label', fontsize=14, weight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontsize=14, weight='bold', labelpad=10)
    ax.set_title('Confusion Matrix for EfficientNet-B0 Features with SVC Classifier', 
                 fontsize=16, weight='bold', pad=20)
    
    # Set tick labels
    ax.set_xticklabels(['Predicted: 0\n(Low GPA)', 'Predicted: 1\n(High GPA)'], 
                       fontsize=12, weight='bold')
    ax.set_yticklabels(['Actual: 0\n(Low GPA)', 'Actual: 1\n(High GPA)'], 
                       fontsize=12, weight='bold', rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / f'confusion_matrix_{model_name}_svc.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Confusion matrix saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = Path(output_dir) / f'confusion_matrix_{model_name}_svc.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"✓ PDF version saved to: {pdf_path}")
    
    plt.close()
    
    return output_path

def main():
    # Setup
    feature_dir = Path('outputs/features_tl')
    output_dir = Path('outputs/paper_results')
    
    # Use best model (EfficientNet-B0)
    best_model = 'efficientnet_b0'
    feature_file = feature_dir / f'features_{best_model}.parquet'
    
    print("="*80)
    print("📊 CONFUSION MATRIX GENERATOR")
    print("="*80)
    print(f"\nBest Model: {best_model.upper()}")
    print(f"Best Classifier: Support Vector Classification (SVC)")
    print(f"Validation: 10-fold GroupKFold Cross-Validation")
    
    # Load features
    features_df = load_features(feature_file)
    
    # Generate confusion matrix
    cm, y_true, y_pred = generate_confusion_matrix(features_df, best_model)
    
    # Plot confusion matrix
    output_path = plot_confusion_matrix(cm, output_dir, best_model)
    
    print("\n" + "="*80)
    print("✅ CONFUSION MATRIX GENERATION COMPLETE")
    print("="*80)
    print(f"\n📊 Total predictions: {len(y_true)}")
    print(f"📈 Correct predictions: {(y_true == y_pred).sum()}")
    print(f"❌ Incorrect predictions: {(y_true != y_pred).sum()}")
    print(f"🎯 Overall accuracy: {(y_true == y_pred).mean():.4f}")
    print(f"\n📁 Output files:")
    print(f"   • {output_path}")
    print(f"   • {output_path.with_suffix('.pdf')}")

if __name__ == '__main__':
    main()
