"""
Kapsamlı ML Pipeline: Model eğitimi, subject-wise CV ve değerlendirme.
- Klasik ML modelleri (Logistic, SVC, RF, GB, MLP)
- GroupKFold ile subject-wise CV (veri sızıntısı önleme)
- Tüm metrikler (Acc, Prec, Rec, F1, ROC-AUC, PR-AUC)
- Confusion matrices
- Model karşılaştırma

Kullanım:
    python src/06_train_evaluate.py --features outputs/features_resnet50.parquet
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time
from collections import defaultdict

# Scikit-learn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

import matplotlib.pyplot as plt
from tqdm import tqdm

# Pastel görselleştirme
from config_viz import (
    create_figure,
    create_pastel_barplot,
    save_figure,
    get_pastel_color,
    PASTEL_COLORS,
    DARK_GRAY
)


# ============================================================================
# MODEL TANIMLARI
# ============================================================================

def get_models():
    """
    Klasik ML modellerini pipeline olarak döndürür.
    
    Returns:
        dict: Model adı -> sklearn pipeline
    """
    models = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        
        'LinearSVC': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LinearSVC(max_iter=2000, random_state=42, dual=False))
        ]),
        
        'RandomForest': Pipeline([
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        
        'GradientBoosting': Pipeline([
            ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]),
        
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, 
                                 random_state=42, early_stopping=True))
        ])
    }
    
    return models


# ============================================================================
# SUBJECT-WISE CROSS VALIDATION
# ============================================================================

def subject_wise_cv_evaluate(X, y, groups, model, model_name, n_splits=5):
    """
    GroupKFold ile subject-wise cross validation yapar ve değerlendirir.
    
    Args:
        X: Feature matrix
        y: Labels
        groups: Subject IDs (for GroupKFold)
        model: sklearn pipeline
        model_name: Model adı
        n_splits: Fold sayısı
        
    Returns:
        dict: Metrikler ve tahminler
    """
    print(f"\n{'='*60}")
    print(f"🔄 Model: {model_name}")
    print(f"{'='*60}")
    
    cv = GroupKFold(n_splits=n_splits)
    
    results = {
        'model_name': model_name,
        'fold_metrics': [],
        'all_y_true': [],
        'all_y_pred': [],
        'all_y_proba': [],
        'confusion_matrices': []
    }
    
    fold_num = 0
    for train_idx, test_idx in tqdm(cv.split(X, y, groups), 
                                     total=n_splits, 
                                     desc=f"Training {model_name}"):
        
        fold_num += 1
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Veri sızıntısı kontrolü (assert)
        train_subjects = set(groups[train_idx])
        test_subjects = set(groups[test_idx])
        assert len(train_subjects & test_subjects) == 0, \
            f"Veri sızıntısı! Ortak subject'ler: {train_subjects & test_subjects}"
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Probability estimation
        # LinearSVC için CalibratedClassifierCV kullan
        if hasattr(model.named_steps['clf'], 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model.named_steps['clf'], 'decision_function'):
            # Calibrate decision function
            calibrated = CalibratedClassifierCV(model.named_steps['clf'], cv=3)
            if 'scaler' in model.named_steps:
                X_train_scaled = model.named_steps['scaler'].transform(X_train)
                X_test_scaled = model.named_steps['scaler'].transform(X_test)
                calibrated.fit(X_train_scaled, y_train)
                y_proba = calibrated.predict_proba(X_test_scaled)[:, 1]
            else:
                calibrated.fit(X_train, y_train)
                y_proba = calibrated.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred.astype(float)  # Fallback
        
        # Metrics
        fold_metrics = {
            'fold': fold_num,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_train_subjects': len(train_subjects),
            'n_test_subjects': len(test_subjects)
        }
        
        results['fold_metrics'].append(fold_metrics)
        results['all_y_true'].extend(y_test)
        results['all_y_pred'].extend(y_pred)
        results['all_y_proba'].extend(y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrices'].append(cm)
    
    # Ortalama metrikler
    results['mean_metrics'] = {
        'accuracy': np.mean([f['accuracy'] for f in results['fold_metrics']]),
        'precision': np.mean([f['precision'] for f in results['fold_metrics']]),
        'recall': np.mean([f['recall'] for f in results['fold_metrics']]),
        'f1': np.mean([f['f1'] for f in results['fold_metrics']]),
        'roc_auc': np.mean([f['roc_auc'] for f in results['fold_metrics']]),
        'pr_auc': np.mean([f['pr_auc'] for f in results['fold_metrics']])
    }
    
    # Std
    results['std_metrics'] = {
        'accuracy': np.std([f['accuracy'] for f in results['fold_metrics']]),
        'precision': np.std([f['precision'] for f in results['fold_metrics']]),
        'recall': np.std([f['recall'] for f in results['fold_metrics']]),
        'f1': np.std([f['f1'] for f in results['fold_metrics']]),
        'roc_auc': np.std([f['roc_auc'] for f in results['fold_metrics']]),
        'pr_auc': np.std([f['pr_auc'] for f in results['fold_metrics']])
    }
    
    # Print özet
    print(f"\n📊 Ortalama Metrikler ({model_name}):")
    for metric, value in results['mean_metrics'].items():
        std = results['std_metrics'][metric]
        print(f"   {metric:12s}: {value:.4f} ± {std:.4f}")
    
    return results


# ============================================================================
# SONUÇLARI KAYDETME
# ============================================================================

def save_results_csv(all_results, output_path='outputs/reports/cv_results.csv'):
    """
    Tüm model sonuçlarını CSV'ye kaydeder.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for result in all_results:
        model_name = result['model_name']
        mean_metrics = result['mean_metrics']
        std_metrics = result['std_metrics']
        
        row = {
            'model': model_name,
            'accuracy_mean': mean_metrics['accuracy'],
            'accuracy_std': std_metrics['accuracy'],
            'precision_mean': mean_metrics['precision'],
            'precision_std': std_metrics['precision'],
            'recall_mean': mean_metrics['recall'],
            'recall_std': std_metrics['recall'],
            'f1_mean': mean_metrics['f1'],
            'f1_std': std_metrics['f1'],
            'roc_auc_mean': mean_metrics['roc_auc'],
            'roc_auc_std': std_metrics['roc_auc'],
            'pr_auc_mean': mean_metrics['pr_auc'],
            'pr_auc_std': std_metrics['pr_auc']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('pr_auc_mean', ascending=False)
    df.to_csv(output_path, index=False)
    
    print(f"\n💾 Sonuçlar kaydedildi: {output_path}")
    return df


def save_detailed_report(all_results, output_path='outputs/reports/detailed_report.txt'):
    """
    Detaylı rapor dosyası oluşturur.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SUBJECT-WISE CROSS VALIDATION - DETAYLI RAPOR\n")
        f.write("="*80 + "\n\n")
        
        for result in all_results:
            model_name = result['model_name']
            f.write(f"\n{'='*80}\n")
            f.write(f"MODEL: {model_name}\n")
            f.write(f"{'='*80}\n\n")
            
            # Fold detayları
            f.write("Fold Detayları:\n")
            f.write("-"*80 + "\n")
            for fold_metric in result['fold_metrics']:
                f.write(f"Fold {fold_metric['fold']}:\n")
                f.write(f"  Train: {fold_metric['n_train']} örnek, {fold_metric['n_train_subjects']} subject\n")
                f.write(f"  Test:  {fold_metric['n_test']} örnek, {fold_metric['n_test_subjects']} subject\n")
                f.write(f"  Accuracy:  {fold_metric['accuracy']:.4f}\n")
                f.write(f"  Precision: {fold_metric['precision']:.4f}\n")
                f.write(f"  Recall:    {fold_metric['recall']:.4f}\n")
                f.write(f"  F1:        {fold_metric['f1']:.4f}\n")
                f.write(f"  ROC-AUC:   {fold_metric['roc_auc']:.4f}\n")
                f.write(f"  PR-AUC:    {fold_metric['pr_auc']:.4f}\n\n")
            
            # Ortalama metrikler
            f.write("\nOrtalama Metrikler:\n")
            f.write("-"*80 + "\n")
            mean = result['mean_metrics']
            std = result['std_metrics']
            f.write(f"Accuracy:  {mean['accuracy']:.4f} ± {std['accuracy']:.4f}\n")
            f.write(f"Precision: {mean['precision']:.4f} ± {std['precision']:.4f}\n")
            f.write(f"Recall:    {mean['recall']:.4f} ± {std['recall']:.4f}\n")
            f.write(f"F1:        {mean['f1']:.4f} ± {std['f1']:.4f}\n")
            f.write(f"ROC-AUC:   {mean['roc_auc']:.4f} ± {std['roc_auc']:.4f}\n")
            f.write(f"PR-AUC:    {mean['pr_auc']:.4f} ± {std['pr_auc']:.4f}\n\n")
            
            # Toplam confusion matrix
            total_cm = np.sum(result['confusion_matrices'], axis=0)
            f.write("\nToplam Confusion Matrix:\n")
            f.write("-"*80 + "\n")
            f.write(f"              Predicted 0    Predicted 1\n")
            f.write(f"Actual 0      {total_cm[0,0]:6d}         {total_cm[0,1]:6d}\n")
            f.write(f"Actual 1      {total_cm[1,0]:6d}         {total_cm[1,1]:6d}\n\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"📄 Detaylı rapor kaydedildi: {output_path}")


# ============================================================================
# GÖRSELLEŞTİRMELER
# ============================================================================

def visualize_model_comparison(results_df, output_dir='outputs/figures'):
    """
    Model karşılaştırma bar chart'ları.
    """
    print("\n📊 Model Karşılaştırma Görselleri")
    print("-" * 60)
    
    output_dir = Path(output_dir)
    
    # PR-AUC sıralaması
    create_pastel_barplot(
        data=results_df['pr_auc_mean'].values,
        labels=results_df['model'].values,
        title='Model Performans Karşılaştırması (PR-AUC)',
        xlabel='PR-AUC Skoru',
        ylabel='Model',
        filepath=output_dir / 'model_comparison_pr_auc.png',
        horizontal=True,
        sort_descending=True
    )
    
    # ROC-AUC sıralaması
    create_pastel_barplot(
        data=results_df['roc_auc_mean'].values,
        labels=results_df['model'].values,
        title='Model Performans Karşılaştırması (ROC-AUC)',
        xlabel='ROC-AUC Skoru',
        ylabel='Model',
        filepath=output_dir / 'model_comparison_roc_auc.png',
        horizontal=True,
        sort_descending=True
    )
    
    # F1 Score
    create_pastel_barplot(
        data=results_df['f1_mean'].values,
        labels=results_df['model'].values,
        title='Model Performans Karşılaştırması (F1 Score)',
        xlabel='F1 Score',
        ylabel='Model',
        filepath=output_dir / 'model_comparison_f1.png',
        horizontal=True,
        sort_descending=True
    )
    
    print("✅ Karşılaştırma görselleri oluşturuldu")


def visualize_metrics_heatmap(all_results, output_dir='outputs/figures'):
    """
    Tüm metrikleri heatmap olarak gösterir.
    """
    print("\n📊 Metrik Heatmap")
    print("-" * 60)
    
    output_dir = Path(output_dir)
    
    # Metrik matrisini oluştur
    models = [r['model_name'] for r in all_results]
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    
    matrix = np.zeros((len(models), len(metrics)))
    for i, result in enumerate(all_results):
        for j, metric in enumerate(metrics):
            matrix[i, j] = result['mean_metrics'][metric]
    
    fig, ax = create_figure(figsize=(10, 6), title='Model-Metrik Heatmap', grid=False)
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Eksenleri ayarla
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=10)
    ax.set_yticklabels(models, fontsize=10)
    
    # Değerleri yaz
    for i in range(len(models)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Skor', fontsize=10)
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'metrics_heatmap.png')
    plt.close()
    
    print("✅ Heatmap oluşturuldu")


def visualize_confusion_matrices(all_results, output_dir='outputs/figures'):
    """
    Her model için confusion matrix görselleştirmesi.
    """
    print("\n📊 Confusion Matrices")
    print("-" * 60)
    
    output_dir = Path(output_dir)
    
    n_models = len(all_results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='white')
    fig.suptitle('Confusion Matrices (Toplam - Tüm Fold\'lar)', 
                fontsize=16, fontweight='bold', color=DARK_GRAY, y=0.98)
    
    axes = axes.flatten()
    
    for i, result in enumerate(all_results):
        if i >= len(axes):
            break
        
        ax = axes[i]
        model_name = result['model_name']
        
        # Toplam confusion matrix
        cm = np.sum(result['confusion_matrices'], axis=0)
        
        # Normalize (percentage)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Heatmap
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=100)
        
        # Değerleri yaz
        for row in range(2):
            for col in range(2):
                text = ax.text(col, row, 
                             f'{cm[row, col]}\n({cm_norm[row, col]:.1f}%)',
                             ha="center", va="center", 
                             color="black" if cm_norm[row, col] < 50 else "white",
                             fontsize=10, fontweight='bold')
        
        ax.set_title(model_name, fontsize=11, fontweight='bold', color=DARK_GRAY)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1'], fontsize=9)
        ax.set_yticklabels(['True 0', 'True 1'], fontsize=9)
    
    # Boş subplot'ları gizle
    for i in range(len(all_results), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'confusion_matrices_all.png')
    plt.close()
    
    print("✅ Confusion matrices oluşturuldu")


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def main():
    """Ana çalıştırma fonksiyonu."""
    
    # Argümanlar
    parser = argparse.ArgumentParser(description='ML Training and Evaluation')
    parser.add_argument('--features', type=str, 
                       default='outputs/features_resnet50.parquet',
                       help='Feature dosya yolu')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='CV fold sayısı')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 ML PIPELINE: SUBJECT-WISE CV İLE MODEL EĞİTİMİ VE DEĞERLENDİRME")
    print("=" * 80)
    
    # Features yükle
    features_path = Path(args.features)
    if not features_path.exists():
        print(f"❌ HATA: {features_path} bulunamadı!")
        return
    
    print(f"\n📂 Features yükleniyor: {features_path}")
    df = pd.read_parquet(features_path)
    print(f"✅ Yüklendi: {df.shape}")
    
    # X, y, groups ayır
    feature_cols = [c for c in df.columns if c.startswith('f')]
    X = df[feature_cols].values
    y = df['label'].values
    groups = df['subject_id'].values
    
    print(f"\n📊 Veri Özeti:")
    print(f"   Features: {X.shape}")
    print(f"   Labels: {y.shape}")
    print(f"   Benzersiz subject: {len(np.unique(groups))}")
    print(f"   Label dağılımı: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
    
    # Modelleri yükle
    models = get_models()
    print(f"\n🤖 Yüklenecek modeller: {list(models.keys())}")
    
    # Tüm modelleri eğit ve değerlendir
    all_results = []
    
    for model_name, model in models.items():
        result = subject_wise_cv_evaluate(
            X, y, groups, model, model_name, n_splits=args.n_splits
        )
        all_results.append(result)
    
    # Sonuçları kaydet
    print("\n" + "=" * 80)
    print("💾 SONUÇLAR KAYDEDİLİYOR")
    print("=" * 80)
    
    results_df = save_results_csv(all_results)
    save_detailed_report(all_results)
    
    # Görselleştirmeler
    print("\n" + "=" * 80)
    print("📊 GÖRSELLEŞTİRMELER OLUŞTURULUYOR")
    print("=" * 80)
    
    visualize_model_comparison(results_df)
    visualize_metrics_heatmap(all_results)
    visualize_confusion_matrices(all_results)
    
    # Özet
    print("\n" + "=" * 80)
    print("✨ TÜM İŞLEMLER TAMAMLANDI!")
    print("=" * 80)
    print(f"\n📊 En İyi Model (PR-AUC): {results_df.iloc[0]['model']} "
          f"({results_df.iloc[0]['pr_auc_mean']:.4f})")
    print(f"📊 En İyi Model (ROC-AUC): "
          f"{results_df.sort_values('roc_auc_mean', ascending=False).iloc[0]['model']} "
          f"({results_df.sort_values('roc_auc_mean', ascending=False).iloc[0]['roc_auc_mean']:.4f})")
    
    print(f"\n📁 Çıktılar:")
    print(f"   - outputs/reports/cv_results.csv")
    print(f"   - outputs/reports/detailed_report.txt")
    print(f"   - outputs/figures/model_comparison_*.png")
    print(f"   - outputs/figures/metrics_heatmap.png")
    print(f"   - outputs/figures/confusion_matrices_all.png")
    
    print("\n🎯 Sonraki adım: ROC/PRC eğrileri")
    print("   python src/07_plot_curves.py")


if __name__ == "__main__":
    main()
