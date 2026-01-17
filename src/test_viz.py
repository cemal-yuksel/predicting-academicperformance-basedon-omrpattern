"""
Pastel görselleştirme konfigürasyonunu test eden script.
Kullanım: python src/test_viz.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# config_viz modülünü import et (otomatik stil aktive olur)
from config_viz import (
    create_figure, 
    create_pastel_barplot,
    create_info_box,
    create_comparison_bars,
    PASTEL_PALETTE,
    PASTEL_COLORS
)


def test_basic_plot():
    """Temel pastel plot testi."""
    print("\n📊 Test 1: Temel pastel plot")
    
    fig, ax = create_figure(title="Pastel Renk Testi")
    
    # Örnek veri
    x = np.linspace(0, 10, 100)
    for i, color in enumerate(PASTEL_PALETTE[:5]):
        y = np.sin(x + i * 0.5)
        ax.plot(x, y, color=color, linewidth=2, label=f'Seri {i+1}')
    
    ax.set_xlabel('X Değerleri')
    ax.set_ylabel('Y Değerleri')
    ax.legend()
    
    output_path = Path('outputs/figures/test_pastel_plot.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Kaydedildi: {output_path}")
    plt.close()


def test_bar_chart():
    """Pastel bar chart testi."""
    print("\n📊 Test 2: Pastel bar chart")
    
    data = [0.85, 0.92, 0.78, 0.88, 0.95]
    labels = ['Model A', 'Model B', 'Model C', 'Model D', 'Model E']
    
    fig, ax = create_pastel_barplot(
        data=data,
        labels=labels,
        title='Model Performans Karşılaştırması',
        xlabel='Modeller',
        ylabel='PR-AUC Skoru',
        filepath='outputs/figures/test_bar_chart.png',
        horizontal=False,
        sort_descending=True
    )
    plt.close()
    print("✅ Bar chart oluşturuldu")


def test_horizontal_bar():
    """Yatay bar chart testi."""
    print("\n📊 Test 3: Yatay bar chart")
    
    data = [250, 180, 320, 210, 195]
    labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    
    fig, ax = create_pastel_barplot(
        data=data,
        labels=labels,
        title='Sınıf Dağılımı',
        xlabel='Örnek Sayısı',
        ylabel='Sınıflar',
        filepath='outputs/figures/test_horizontal_bar.png',
        horizontal=True,
        sort_descending=False
    )
    plt.close()
    print("✅ Yatay bar chart oluşturuldu")


def test_info_box():
    """Bilgi kutusu testi."""
    print("\n📊 Test 4: Bilgi kutusu")
    
    info_lines = [
        "Feature Boyutu: 2048",
        "Toplam Örnek: 1500",
        "Train Örnekleri: 1200",
        "Test Örnekleri: 300",
        "Birey Sayısı: 30",
    ]
    
    fig = create_info_box(
        text_lines=info_lines,
        title="Veri Seti Özeti",
        filepath='outputs/figures/test_info_box.png'
    )
    plt.close()
    print("✅ Bilgi kutusu oluşturuldu")


def test_comparison():
    """Karşılaştırmalı bar chart testi."""
    print("\n📊 Test 5: Karşılaştırmalı bar chart")
    
    data_dict = {
        'Yanlış CV (StratifiedKFold)': [0.95, 0.93, 0.96, 0.94],
        'Doğru CV (GroupKFold)': [0.82, 0.78, 0.84, 0.80]
    }
    
    fig, ax = create_comparison_bars(
        data_dict=data_dict,
        title='Veri Sızıntısı Etkisi',
        ylabel='ROC-AUC Skoru',
        filepath='outputs/figures/test_comparison.png'
    )
    plt.close()
    print("✅ Karşılaştırma grafiği oluşturuldu")


def test_color_palette():
    """Renk paleti görselleştirmesi."""
    print("\n📊 Test 6: Renk paleti")
    
    fig, ax = create_figure(title='Pastel Renk Paleti', figsize=(12, 4))
    
    n_colors = len(PASTEL_PALETTE)
    for i, color in enumerate(PASTEL_PALETTE):
        ax.bar(i, 1, color=color, edgecolor='#333333', linewidth=1.5, width=0.9)
        ax.text(i, 0.5, color, ha='center', va='center', 
               fontsize=9, color='#333333', rotation=90)
    
    ax.set_xlim(-0.5, n_colors - 0.5)
    ax.set_ylim(0, 1.2)
    ax.set_xticks(range(n_colors))
    ax.set_xticklabels([f'Renk {i+1}' for i in range(n_colors)], rotation=45, ha='right')
    ax.set_ylabel('Renk Örneği')
    ax.set_yticks([])
    
    output_path = Path('outputs/figures/test_color_palette.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Kaydedildi: {output_path}")
    plt.close()


def main():
    """Tüm testleri çalıştır."""
    print("=" * 60)
    print("🎨 Pastel Görselleştirme Testleri Başlıyor...")
    print("=" * 60)
    
    try:
        test_basic_plot()
        test_bar_chart()
        test_horizontal_bar()
        test_info_box()
        test_comparison()
        test_color_palette()
        
        print("\n" + "=" * 60)
        print("✨ Tüm testler başarıyla tamamlandı!")
        print("📁 Görseller: outputs/figures/test_*.png")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
