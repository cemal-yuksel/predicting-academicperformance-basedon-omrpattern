"""
Metadata görselleştirme script'i.
annotations.csv verilerini pastel renklerle görselleştirir.

Kullanım:
    python src/03_visualize_metadata.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Pastel görselleştirme modülünü import et
from config_viz import (
    create_figure,
    create_pastel_barplot,
    create_info_box,
    save_figure,
    get_pastel_color,
    PASTEL_COLORS,
    DARK_GRAY
)


def visualize_label_distribution(df, output_dir='outputs/figures'):
    """
    Label dağılımını pastel bar chart ile görselleştirir.
    """
    print("\n📊 Label Dağılımı Görselleştirmesi")
    print("-" * 60)
    
    output_dir = Path(output_dir)
    
    # Label sayılarını hesapla
    label_counts = df['label'].value_counts().sort_index()
    labels = ['Label 0 (Doğru)', 'Label 1 (Yanlış)']
    values = [label_counts[0], label_counts[1]]
    
    # Bar chart oluştur
    fig, ax = create_pastel_barplot(
        data=values,
        labels=labels,
        title='Label Dağılımı (Dengeli Veri Seti)',
        xlabel='Label Türü',
        ylabel='Örnek Sayısı',
        filepath=output_dir / 'metadata_label_distribution.png',
        horizontal=False,
        sort_descending=False
    )
    plt.close()
    
    print("✅ Label dağılımı görseli oluşturuldu")


def visualize_subject_distribution(df, output_dir='outputs/figures'):
    """
    Subject başına örnek sayısını görselleştirir.
    """
    print("\n📊 Subject Başına Örnek Dağılımı")
    print("-" * 60)
    
    output_dir = Path(output_dir)
    
    # Subject başına örnek sayısı
    subject_counts = df['subject_id'].value_counts().sort_index()
    
    # Tüm subject'leri göster
    fig, ax = create_figure(
        figsize=(12, 6),
        title='Subject Başına Örnek Sayısı (Her Subject: 100 Satır)'
    )
    
    subjects = subject_counts.index.tolist()
    counts = subject_counts.values
    colors = [get_pastel_color(i) for i in range(len(subjects))]
    
    bars = ax.bar(subjects, counts, color=colors, edgecolor=DARK_GRAY, linewidth=0.5)
    
    # Ortalama çizgisi
    mean_count = counts.mean()
    ax.axhline(mean_count, color=DARK_GRAY, linestyle='--', linewidth=1.5, 
              label=f'Ortalama: {mean_count:.0f}', alpha=0.6)
    
    ax.set_xlabel('Subject ID', fontsize=11, color=DARK_GRAY)
    ax.set_ylabel('Örnek Sayısı', fontsize=11, color=DARK_GRAY)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.legend(framealpha=0.8)
    
    # Her bar'ın üzerine değer yaz
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=8, color=DARK_GRAY)
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'metadata_subject_distribution.png')
    plt.close()
    
    print("✅ Subject dağılımı görseli oluşturuldu")


def visualize_subject_label_distribution(df, output_dir='outputs/figures'):
    """
    Her subject'in label dağılımını stacked bar chart ile gösterir.
    """
    print("\n📊 Subject Başına Label Dağılımı")
    print("-" * 60)
    
    output_dir = Path(output_dir)
    
    # Her subject için label sayılarını hesapla
    subject_label_counts = df.groupby(['subject_id', 'label']).size().unstack(fill_value=0)
    
    fig, ax = create_figure(
        figsize=(12, 6),
        title='Subject Başına Label Dağılımı (Dengeli: 50-50)'
    )
    
    subjects = subject_label_counts.index.tolist()
    label_0_counts = subject_label_counts[0].values
    label_1_counts = subject_label_counts[1].values
    
    x = np.arange(len(subjects))
    width = 0.6
    
    # Stacked bar chart
    bar1 = ax.bar(x, label_0_counts, width, 
                  label='Label 0 (Doğru)', 
                  color=PASTEL_COLORS['blue'], 
                  edgecolor=DARK_GRAY, 
                  linewidth=0.5)
    
    bar2 = ax.bar(x, label_1_counts, width, 
                  bottom=label_0_counts,
                  label='Label 1 (Yanlış)', 
                  color=PASTEL_COLORS['pink'], 
                  edgecolor=DARK_GRAY, 
                  linewidth=0.5)
    
    ax.set_xlabel('Subject ID', fontsize=11, color=DARK_GRAY)
    ax.set_ylabel('Satır Sayısı', fontsize=11, color=DARK_GRAY)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha='right')
    ax.legend(framealpha=0.8)
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'metadata_subject_label_stacked.png')
    plt.close()
    
    print("✅ Subject-label stacked chart oluşturuldu")


def create_dataset_info_box(df, output_dir='outputs/figures'):
    """
    Veri seti istatistiklerini bilgi kutusu olarak gösterir.
    """
    print("\n📊 Veri Seti Bilgi Kutusu")
    print("-" * 60)
    
    output_dir = Path(output_dir)
    
    # İstatistikleri hesapla
    n_samples = len(df)
    n_subjects = df['subject_id'].nunique()
    n_lines_per_subject = n_samples / n_subjects
    label_0_count = (df['label'] == 0).sum()
    label_1_count = (df['label'] == 1).sum()
    label_0_pct = (label_0_count / n_samples) * 100
    label_1_pct = (label_1_count / n_samples) * 100
    
    # Bilgi metinleri
    info_lines = [
        f"Toplam Satır Sayısı: {n_samples:,}",
        f"Benzersiz Birey Sayısı: {n_subjects}",
        f"Birey Başına Satır: {n_lines_per_subject:.0f}",
        "",
        f"Label 0 (Doğru): {label_0_count:,} ({label_0_pct:.1f}%)",
        f"Label 1 (Yanlış): {label_1_count:,} ({label_1_pct:.1f}%)",
        "",
        "✓ Dengeli Veri Seti",
        "✓ Subject-wise CV Hazır"
    ]
    
    fig = create_info_box(
        text_lines=info_lines,
        title="📊 OMR Veri Seti Özeti",
        filepath=output_dir / 'metadata_dataset_info.png',
        box_color=PASTEL_COLORS['baby_blue']
    )
    plt.close()
    
    print("✅ Veri seti bilgi kutusu oluşturuldu")


def visualize_line_number_distribution(df, output_dir='outputs/figures'):
    """
    Satır numaralarının dağılımını histogram ile gösterir.
    """
    print("\n📊 Satır Numarası Dağılımı")
    print("-" * 60)
    
    output_dir = Path(output_dir)
    
    fig, ax = create_figure(
        figsize=(10, 6),
        title='Satır Numarası Dağılımı (0-50)'
    )
    
    # Histogram
    line_numbers = df['line_number'].values
    n, bins, patches = ax.hist(line_numbers, bins=51, 
                                color=PASTEL_COLORS['lavender'], 
                                edgecolor=DARK_GRAY, 
                                linewidth=0.5,
                                alpha=0.8)
    
    # Her patch'e farklı pastel renk (gradient etkisi)
    for i, patch in enumerate(patches):
        color = get_pastel_color(i % 8)
        patch.set_facecolor(color)
    
    ax.set_xlabel('Satır Numarası', fontsize=11, color=DARK_GRAY)
    ax.set_ylabel('Frekans', fontsize=11, color=DARK_GRAY)
    
    # İstatistikler ekle
    mean_line = line_numbers.mean()
    median_line = np.median(line_numbers)
    ax.axvline(mean_line, color=DARK_GRAY, linestyle='--', 
              linewidth=2, label=f'Ortalama: {mean_line:.1f}', alpha=0.7)
    ax.axvline(median_line, color=DARK_GRAY, linestyle=':', 
              linewidth=2, label=f'Medyan: {median_line:.0f}', alpha=0.7)
    ax.legend(framealpha=0.8)
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'metadata_line_number_histogram.png')
    plt.close()
    
    print("✅ Satır numarası histogramı oluşturuldu")


def create_summary_figure(df, output_dir='outputs/figures'):
    """
    Tüm istatistikleri tek bir özet figüründe gösterir.
    """
    print("\n📊 Özet Figür Oluşturuluyor")
    print("-" * 60)
    
    output_dir = Path(output_dir)
    
    # 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
    fig.suptitle('📊 OMR Veri Seti Metadata Özeti', 
                fontsize=16, fontweight='bold', color=DARK_GRAY, y=0.98)
    
    # 1. Label dağılımı (sol üst)
    ax1 = axes[0, 0]
    label_counts = df['label'].value_counts().sort_index()
    labels_text = ['Label 0\n(Doğru)', 'Label 1\n(Yanlış)']
    colors = [PASTEL_COLORS['blue'], PASTEL_COLORS['pink']]
    bars = ax1.bar(labels_text, label_counts.values, color=colors, 
                   edgecolor=DARK_GRAY, linewidth=1.5)
    ax1.set_title('Label Dağılımı', fontsize=12, fontweight='bold', color=DARK_GRAY)
    ax1.set_ylabel('Örnek Sayısı', fontsize=10)
    ax1.grid(True, alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}\n({height/len(df)*100:.1f}%)',
               ha='center', va='bottom', fontsize=9, color=DARK_GRAY)
    
    # 2. Subject başına örnek sayısı (sağ üst)
    ax2 = axes[0, 1]
    subject_counts = df['subject_id'].value_counts().sort_index()
    ax2.bar(range(len(subject_counts)), subject_counts.values,
           color=[get_pastel_color(i) for i in range(len(subject_counts))],
           edgecolor=DARK_GRAY, linewidth=0.5)
    ax2.set_title('Subject Başına Satır Sayısı', fontsize=12, fontweight='bold', color=DARK_GRAY)
    ax2.set_xlabel('Subject Index', fontsize=10)
    ax2.set_ylabel('Satır Sayısı', fontsize=10)
    ax2.grid(True, alpha=0.3)
    mean_count = subject_counts.mean()
    ax2.axhline(mean_count, color=DARK_GRAY, linestyle='--', linewidth=1.5, alpha=0.6)
    ax2.text(len(subject_counts)-1, mean_count, f' Ort: {mean_count:.0f}',
            va='center', fontsize=9, color=DARK_GRAY)
    
    # 3. Satır numarası dağılımı (sol alt)
    ax3 = axes[1, 0]
    line_numbers = df['line_number'].values
    n, bins, patches = ax3.hist(line_numbers, bins=25, 
                                color=PASTEL_COLORS['green'], 
                                edgecolor=DARK_GRAY, 
                                linewidth=0.5, alpha=0.8)
    ax3.set_title('Satır Numarası Dağılımı', fontsize=12, fontweight='bold', color=DARK_GRAY)
    ax3.set_xlabel('Satır Numarası', fontsize=10)
    ax3.set_ylabel('Frekans', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. İstatistik bilgileri (sağ alt)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    📊 VERİ SETİ İSTATİSTİKLERİ
    
    Toplam Satır: {len(df):,}
    Benzersiz Birey: {df['subject_id'].nunique()}
    Birey Başına Satır: {len(df)/df['subject_id'].nunique():.0f}
    
    Label 0 (Doğru): {(df['label']==0).sum():,} ({(df['label']==0).sum()/len(df)*100:.1f}%)
    Label 1 (Yanlış): {(df['label']==1).sum():,} ({(df['label']==1).sum()/len(df)*100:.1f}%)
    
    Satır No Min: {df['line_number'].min()}
    Satır No Max: {df['line_number'].max()}
    Satır No Ort: {df['line_number'].mean():.1f}
    
    ✓ Dengeli Veri Seti
    ✓ GroupKFold CV Hazır
    ✓ Veri Sızıntısı Önlendi
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top',
            family='monospace', color=DARK_GRAY,
            bbox=dict(boxstyle='round', facecolor=PASTEL_COLORS['baby_blue'], 
                     alpha=0.3, edgecolor=DARK_GRAY, linewidth=1.5))
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'metadata_summary_all.png')
    plt.close()
    
    print("✅ Özet figür oluşturuldu")


def main():
    """Ana çalıştırma fonksiyonu."""
    print("=" * 60)
    print("📊 METADATA GÖRSELLEŞTİRME")
    print("=" * 60)
    
    # CSV'yi oku
    csv_path = Path('data/meta/annotations.csv')
    
    if not csv_path.exists():
        print(f"❌ HATA: {csv_path} dosyası bulunamadı!")
        print("Önce src/create_annotations.py çalıştırın.")
        return
    
    df = pd.read_csv(csv_path)
    print(f"✅ CSV okundu: {len(df)} satır, {df['subject_id'].nunique()} birey")
    
    # Görselleri oluştur
    output_dir = Path('outputs/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_label_distribution(df, output_dir)
    visualize_subject_distribution(df, output_dir)
    visualize_subject_label_distribution(df, output_dir)
    create_dataset_info_box(df, output_dir)
    visualize_line_number_distribution(df, output_dir)
    create_summary_figure(df, output_dir)
    
    print("\n" + "=" * 60)
    print("✨ Tüm metadata görselleri oluşturuldu!")
    print(f"📁 Klasör: {output_dir}")
    print("=" * 60)
    
    # Oluşturulan dosyaları listele
    created_files = [
        'metadata_label_distribution.png',
        'metadata_subject_distribution.png',
        'metadata_subject_label_stacked.png',
        'metadata_dataset_info.png',
        'metadata_line_number_histogram.png',
        'metadata_summary_all.png'
    ]
    
    print("\n📋 Oluşturulan Görseller:")
    for i, filename in enumerate(created_files, 1):
        print(f"   {i}. {filename}")


if __name__ == "__main__":
    main()
