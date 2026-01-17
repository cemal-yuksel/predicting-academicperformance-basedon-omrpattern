"""
Veri Sızıntısı (Data Leakage) Analizi ve Alternatif CV Stratejileri.
- StratifiedKFold vs GroupKFold karşılaştırması
- Data leakage etkisini gösterme
- LeaveOneGroupOut (LOSO) deneme

Kullanım:
    python src/08_leakage_analysis.py --features outputs/features_resnet50.parquet
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold, GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

import matplotlib.pyplot as plt
from tqdm import tqdm

from config_viz import (
    create_figure,
    save_figure,
    create_pastel_barplot,
    get_pastel_color,
    DARK_GRAY
)


def evaluate_with_strategy(X, y, groups, model, strategy_name, cv_splitter):
    """
    Verilen CV stratejisi ile modeli değerlendir.
    
    Args:
        X: Features
        y: Labels
        groups: Subject IDs
        model: sklearn pipeline
        strategy_name: Strateji adı
        cv_splitter: CV splitter (StratifiedKFold, GroupKFold, LOSO)
        
    Returns:
        dict: Metrikler
    """
    print(f"\n{'='*60}")
    print(f"🧪 Strateji: {strategy_name}")
    print(f"{'='*60}")
    
    metrics = []
    leakage_detected = []
    
    n_splits = cv_splitter.get_n_splits(X, y, groups) if hasattr(cv_splitter, 'get_n_splits') else len(set(groups))
    
    fold_num = 0
    for train_idx, test_idx in tqdm(cv_splitter.split(X, y, groups), 
                                     total=n_splits,
                                     desc=f"Evaluating {strategy_name}"):
        
        fold_num += 1
        
        # Split
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Veri sızıntısı kontrolü
        train_subjects = set(groups[train_idx])
        test_subjects = set(groups[test_idx])
        overlap = train_subjects & test_subjects
        
        if len(overlap) > 0:
            leakage_detected.append({
                'fold': fold_num,
                'overlap_subjects': list(overlap),
                'n_overlap': len(overlap)
            })
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Probability
        if hasattr(model.named_steps['clf'], 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred.astype(float)
        
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
            'n_test_subjects': len(test_subjects),
            'leakage': len(overlap) > 0
        }
        
        metrics.append(fold_metrics)
    
    # Özet
    mean_metrics = {
        'strategy': strategy_name,
        'n_folds': len(metrics),
        'accuracy': np.mean([m['accuracy'] for m in metrics]),
        'precision': np.mean([m['precision'] for m in metrics]),
        'recall': np.mean([m['recall'] for m in metrics]),
        'f1': np.mean([m['f1'] for m in metrics]),
        'roc_auc': np.mean([m['roc_auc'] for m in metrics]),
        'pr_auc': np.mean([m['pr_auc'] for m in metrics]),
        'n_leakage_folds': sum([m['leakage'] for m in metrics]),
        'leakage_details': leakage_detected
    }
    
    # Print
    print(f"\n📊 Sonuçlar:")
    print(f"   Fold sayısı: {mean_metrics['n_folds']}")
    print(f"   Accuracy:  {mean_metrics['accuracy']:.4f}")
    print(f"   Precision: {mean_metrics['precision']:.4f}")
    print(f"   Recall:    {mean_metrics['recall']:.4f}")
    print(f"   F1:        {mean_metrics['f1']:.4f}")
    print(f"   ROC-AUC:   {mean_metrics['roc_auc']:.4f}")
    print(f"   PR-AUC:    {mean_metrics['pr_auc']:.4f}")
    print(f"   Veri sızıntısı: {mean_metrics['n_leakage_folds']} fold")
    
    if len(leakage_detected) > 0:
        print(f"\n⚠️  VERİ SIZINTISI TESPİT EDİLDİ!")
        for leak in leakage_detected:
            print(f"   Fold {leak['fold']}: {leak['n_overlap']} ortak subject")
    
    return mean_metrics, metrics


def compare_strategies(results, output_path='outputs/reports/leakage_comparison.txt'):
    """
    Stratejileri karşılaştır ve rapora kaydet.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("VERİ SIZINTISI ANALİZİ - STRATEJİ KARŞILAŞTIRMASI\n")
        f.write("="*80 + "\n\n")
        
        f.write("AMAÇ:\n")
        f.write("-"*80 + "\n")
        f.write("Subject-wise CV ile standart CV arasındaki farkı göstermek.\n")
        f.write("Aynı subject'in train ve test setinde bulunması veri sızıntısına yol açar.\n\n")
        
        f.write("STRATEJİLER:\n")
        f.write("-"*80 + "\n\n")
        
        for result in results:
            strategy = result['strategy']
            f.write(f"{strategy}:\n")
            f.write(f"  Fold sayısı:      {result['n_folds']}\n")
            f.write(f"  Accuracy:         {result['accuracy']:.4f}\n")
            f.write(f"  Precision:        {result['precision']:.4f}\n")
            f.write(f"  Recall:           {result['recall']:.4f}\n")
            f.write(f"  F1:               {result['f1']:.4f}\n")
            f.write(f"  ROC-AUC:          {result['roc_auc']:.4f}\n")
            f.write(f"  PR-AUC:           {result['pr_auc']:.4f}\n")
            f.write(f"  Veri sızıntısı:   {result['n_leakage_folds']} fold\n\n")
            
            if result['n_leakage_folds'] > 0:
                f.write(f"  ⚠️  VERİ SIZINTISI:\n")
                for leak in result['leakage_details']:
                    f.write(f"    Fold {leak['fold']}: {leak['n_overlap']} ortak subject\n")
                f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SONUÇ:\n")
        f.write("-"*80 + "\n")
        
        # StratifiedKFold vs GroupKFold
        stratified = [r for r in results if 'Stratified' in r['strategy']][0]
        grouped = [r for r in results if 'Group' in r['strategy'] and 'LOSO' not in r['strategy']][0]
        
        acc_diff = stratified['accuracy'] - grouped['accuracy']
        f1_diff = stratified['f1'] - grouped['f1']
        
        f.write(f"\nStratifiedKFold vs GroupKFold:\n")
        f.write(f"  Accuracy farkı: {acc_diff:+.4f} (StratifiedKFold lehine)\n")
        f.write(f"  F1 farkı:       {f1_diff:+.4f} (StratifiedKFold lehine)\n\n")
        
        if acc_diff > 0:
            f.write("StratifiedKFold daha yüksek performans gösteriyor çünkü:\n")
            f.write("- Aynı subject'in train ve test'te olması modele 'ipucu' veriyor\n")
            f.write("- Bu gerçek dünya performansını AŞIRI İYİMSER tahmin ediyor\n")
            f.write("- Yeni subject'lerde (production'da) performans düşecektir\n\n")
        
        f.write("GroupKFold kullanılmalıdır çünkü:\n")
        f.write("- Her subject train VEYA test'te olur, ikisinde birden olmaz\n")
        f.write("- Gerçek dünya senaryosunu simüle eder (yeni hastalar/öğrenciler)\n")
        f.write("- Model genelleme yeteneği doğru değerlendirilir\n\n")
        
        # LOSO varsa
        loso_results = [r for r in results if 'LOSO' in r['strategy']]
        if len(loso_results) > 0:
            loso = loso_results[0]
            f.write(f"\nLeaveOneGroupOut (LOSO):\n")
            f.write(f"  Fold sayısı: {loso['n_folds']} (her subject 1 fold)\n")
            f.write(f"  Accuracy:    {loso['accuracy']:.4f}\n")
            f.write(f"  F1:          {loso['f1']:.4f}\n\n")
            f.write("LOSO en katı CV stratejisidir:\n")
            f.write("- Her seferinde 1 subject test, geri kalanı train\n")
            f.write("- Çok fazla fold (21 fold) nedeniyle hesaplama pahalı\n")
            f.write("- Küçük test setleri nedeniyle varyans yüksek\n")
            f.write("- GroupKFold (5 fold) genelde yeterli ve daha pratiktir\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"\n📄 Karşılaştırma raporu: {output_path}")


def visualize_comparison(results, output_dir='outputs/figures'):
    """
    Stratejileri görsel olarak karşılaştır.
    """
    print("\n📊 Karşılaştırma Görselleri")
    print("-" * 60)
    
    output_dir = Path(output_dir)
    
    strategies = [r['strategy'] for r in results]
    
    # Accuracy karşılaştırma
    accuracies = [r['accuracy'] for r in results]
    colors = [get_pastel_color(i) for i in range(len(results))]
    
    fig, ax = create_figure(figsize=(10, 5), 
                           title='CV Stratejileri Karşılaştırması - Accuracy',
                           grid=True)
    
    ax.set_xlabel('Accuracy', fontsize=11)
    ax.set_ylabel('Strateji', fontsize=11)
    
    bars = ax.barh(strategies, accuracies, color=colors, edgecolor=DARK_GRAY, linewidth=1.5)
    
    # Veri sızıntısı olanları işaretle
    for i, result in enumerate(results):
        if result['n_leakage_folds'] > 0:
            ax.text(accuracies[i] + 0.01, i, '⚠️ LEAKAGE', 
                   va='center', fontsize=10, color='red', fontweight='bold')
    
    # Değerleri yaz
    for i, v in enumerate(accuracies):
        ax.text(v - 0.01, i, f'{v:.4f}', 
               va='center', ha='right', fontsize=10, 
               color='white', fontweight='bold')
    
    ax.set_xlim([min(accuracies) - 0.05, max(accuracies) + 0.05])
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'leakage_comparison_accuracy.png')
    plt.close()
    
    # F1 karşılaştırma
    f1_scores = [r['f1'] for r in results]
    
    fig, ax = create_figure(figsize=(10, 5), 
                           title='CV Stratejileri Karşılaştırması - F1 Score',
                           grid=True)
    
    ax.set_xlabel('F1 Score', fontsize=11)
    ax.set_ylabel('Strateji', fontsize=11)
    
    bars = ax.barh(strategies, f1_scores, color=colors, edgecolor=DARK_GRAY, linewidth=1.5)
    
    # Veri sızıntısı olanları işaretle
    for i, result in enumerate(results):
        if result['n_leakage_folds'] > 0:
            ax.text(f1_scores[i] + 0.01, i, '⚠️ LEAKAGE', 
                   va='center', fontsize=10, color='red', fontweight='bold')
    
    # Değerleri yaz
    for i, v in enumerate(f1_scores):
        ax.text(v - 0.01, i, f'{v:.4f}', 
               va='center', ha='right', fontsize=10, 
               color='white', fontweight='bold')
    
    ax.set_xlim([min(f1_scores) - 0.05, max(f1_scores) + 0.05])
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'leakage_comparison_f1.png')
    plt.close()
    
    print("✅ Karşılaştırma görselleri oluşturuldu")


def main():
    """Ana çalıştırma fonksiyonu."""
    
    # Argümanlar
    parser = argparse.ArgumentParser(description='Data Leakage Analysis')
    parser.add_argument('--features', type=str, 
                       default='outputs/features_resnet50.parquet',
                       help='Feature dosya yolu')
    parser.add_argument('--include_loso', action='store_true',
                       help='LOSO da dahil et (çok uzun sürer)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 VERİ SIZINTISI ANALİZİ - STRATEJİ KARŞILAŞTIRMASI")
    print("=" * 80)
    
    # Features yükle
    features_path = Path(args.features)
    if not features_path.exists():
        print(f"❌ HATA: {features_path} bulunamadı!")
        return
    
    print(f"\n📂 Features yükleniyor: {features_path}")
    df = pd.read_parquet(features_path)
    print(f"✅ Yüklendi: {df.shape}")
    
    # X, y, groups
    feature_cols = [c for c in df.columns if c.startswith('f')]
    X = df[feature_cols].values
    y = df['label'].values
    groups = df['subject_id'].values
    
    print(f"\n📊 Veri:")
    print(f"   Features: {X.shape}")
    print(f"   Benzersiz subject: {len(np.unique(groups))}")
    
    # Model (en iyi model: GradientBoosting)
    model = Pipeline([
        ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
    
    print(f"\n🤖 Model: GradientBoosting")
    
    # Stratejiler
    results = []
    
    # 1. StratifiedKFold (VERİ SIZINTISI VAR!)
    print("\n" + "="*80)
    print("1️⃣ StratifiedKFold - VERİ SIZINTISI RİSKİ")
    print("="*80)
    stratified_result, _ = evaluate_with_strategy(
        X, y, groups, model, 
        'StratifiedKFold (5 fold)',
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    results.append(stratified_result)
    
    # 2. GroupKFold (VERİ SIZINTISI YOK!)
    print("\n" + "="*80)
    print("2️⃣ GroupKFold - VERİ SIZINTISI YOK ✅")
    print("="*80)
    grouped_result, _ = evaluate_with_strategy(
        X, y, groups, model,
        'GroupKFold (5 fold)',
        GroupKFold(n_splits=5)
    )
    results.append(grouped_result)
    
    # 3. LOSO (opsiyonel, uzun sürer)
    if args.include_loso:
        print("\n" + "="*80)
        print("3️⃣ LeaveOneGroupOut (LOSO) - En Katı Strateji")
        print("="*80)
        loso_result, _ = evaluate_with_strategy(
            X, y, groups, model,
            'LeaveOneGroupOut (21 fold)',
            LeaveOneGroupOut()
        )
        results.append(loso_result)
    else:
        print("\n💡 LOSO atlandı (uzun sürer). --include_loso ile eklenebilir.")
    
    # Karşılaştırma
    print("\n" + "="*80)
    print("📋 KARŞILAŞTIRMA VE RAPORLAMA")
    print("="*80)
    
    compare_strategies(results)
    visualize_comparison(results)
    
    # Özet
    print("\n" + "="*80)
    print("✨ VERİ SIZINTISI ANALİZİ TAMAMLANDI!")
    print("="*80)
    
    stratified = results[0]
    grouped = results[1]
    
    print(f"\n📊 Sonuç:")
    print(f"   StratifiedKFold Accuracy: {stratified['accuracy']:.4f} ⚠️ LEAKAGE")
    print(f"   GroupKFold Accuracy:      {grouped['accuracy']:.4f} ✅ DOĞRU")
    print(f"   Fark:                     {stratified['accuracy'] - grouped['accuracy']:+.4f}")
    
    print(f"\n💡 Yorum:")
    if stratified['accuracy'] > grouped['accuracy']:
        print(f"   StratifiedKFold daha yüksek çünkü aynı subject'i train+test'te görüyor.")
        print(f"   Bu gerçek performansı AŞIRI İYİMSER tahmin eder!")
    print(f"   GroupKFold gerçek dünya senaryosunu doğru simüle eder.")
    
    print(f"\n📁 Çıktılar:")
    print(f"   - outputs/reports/leakage_comparison.txt")
    print(f"   - outputs/figures/leakage_comparison_accuracy.png")
    print(f"   - outputs/figures/leakage_comparison_f1.png")
    
    print(f"\n🎉 TÜM PIPELINE TAMAMLANDI!")


if __name__ == "__main__":
    main()
