"""
Satır görüntülerini ön işleme script'i.
Grayscale dönüşüm, normalize, resize işlemleri yapar.

Kullanım:
    python src/04_preprocess_images.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Pastel görselleştirme
from config_viz import (
    create_figure,
    save_figure,
    PASTEL_COLORS,
    DARK_GRAY
)


# ============================================================================
# ÖN İŞLEME PARAMETRELERİ
# ============================================================================

TARGET_SIZE = (224, 224)  # CNN için standart boyut (ResNet, VGG vs. için)
NORMALIZE_RANGE = (0, 1)  # [0, 1] aralığına normalize


# ============================================================================
# ÖN İŞLEME FONKSİYONLARI
# ============================================================================

def load_image(image_path):
    """
    Görüntüyü yükler.
    
    Args:
        image_path (str): Görüntü dosya yolu
        
    Returns:
        numpy.ndarray: BGR formatında görüntü (OpenCV default)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")
    return img


def preprocess_image(img, target_size=TARGET_SIZE, to_rgb=True):
    """
    Görüntüyü ön işler: grayscale, normalize, resize.
    
    Args:
        img (numpy.ndarray): Orijinal görüntü
        target_size (tuple): Hedef boyut (height, width)
        to_rgb (bool): RGB'ye çevir mi (CNN için)
        
    Returns:
        tuple: (processed_img, grayscale_img, resized_img)
            - processed_img: RGB 224x224 [0,1] normalized
            - grayscale_img: Grayscale orijinal boyut
            - resized_img: Grayscale 224x224
    """
    # 1. Grayscale'e çevir
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 2. Resize (224x224)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    
    # 3. Normalize [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # 4. RGB'ye çevir (3 kanal - CNN'ler için)
    if to_rgb:
        # Grayscale'i 3 kanala kopyala
        rgb = np.stack([normalized] * 3, axis=-1)
    else:
        rgb = normalized
    
    return rgb, gray, resized


def preprocess_for_classical_ml(img):
    """
    Klasik ML için ön işleme: flatten vektör.
    
    Args:
        img (numpy.ndarray): 224x224 grayscale görüntü
        
    Returns:
        numpy.ndarray: Flatten edilmiş vektör (50176,)
    """
    # Normalize
    normalized = img.astype(np.float32) / 255.0
    # Flatten
    flattened = normalized.flatten()
    return flattened


def process_dataset(annotations_path, output_dir='data/processed', 
                   save_processed=False, sample_size=None):
    """
    Tüm veri setini işler ve istatistikler toplar.
    
    Args:
        annotations_path (str): annotations.csv yolu
        output_dir (str): İşlenmiş görüntülerin kaydedileceği klasör
        save_processed (bool): İşlenmiş görüntüleri diske kaydet mi
        sample_size (int): Kaç örnek işlensin (None = hepsi)
        
    Returns:
        dict: İstatistikler ve örnek veriler
    """
    print("\n📊 Veri Seti Ön İşleme Başlıyor")
    print("=" * 60)
    
    # Annotations'ı oku
    df = pd.read_csv(annotations_path)
    
    if sample_size:
        df = df.head(sample_size)
        print(f"ℹ️  Örnek boyutu: {sample_size}")
    
    print(f"✅ {len(df)} görüntü işlenecek")
    
    # İstatistikler
    stats = {
        'original_shapes': [],
        'processed_shapes': [],
        'mean_pixel_values': [],
        'std_pixel_values': [],
        'sample_images': []
    }
    
    # Output klasörü
    if save_processed:
        output_path = Path(output_dir) / 'preprocessed_lines'
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 İşlenmiş görüntüler: {output_path}")
    
    # İşleme döngüsü
    processed_count = 0
    error_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="İşleniyor"):
        try:
            # Görüntüyü yükle
            img_path = Path(row['line_image_path'])
            img = load_image(img_path)
            
            # Ön işle
            processed_rgb, gray, resized = preprocess_image(img)
            
            # İstatistik topla
            stats['original_shapes'].append(img.shape)
            stats['processed_shapes'].append(processed_rgb.shape)
            stats['mean_pixel_values'].append(np.mean(processed_rgb))
            stats['std_pixel_values'].append(np.std(processed_rgb))
            
            # İlk 5 örneği sakla
            if len(stats['sample_images']) < 5:
                stats['sample_images'].append({
                    'original': img,
                    'grayscale': gray,
                    'resized': resized,
                    'processed': processed_rgb,
                    'subject_id': row['subject_id'],
                    'label': row['label'],
                    'line_number': row['line_number']
                })
            
            # Kaydet (opsiyonel)
            if save_processed:
                filename = f"{row['subject_id']}_L{row['label']}_line{row['line_number']:03d}.npy"
                save_path = output_path / filename
                np.save(save_path, processed_rgb)
            
            processed_count += 1
            
        except Exception as e:
            print(f"\n⚠️  Hata [{img_path}]: {e}")
            error_count += 1
            continue
    
    print(f"\n✅ İşleme tamamlandı: {processed_count} başarılı, {error_count} hata")
    
    # İstatistikleri hesapla
    stats['total_processed'] = processed_count
    stats['total_errors'] = error_count
    stats['mean_pixel_overall'] = np.mean(stats['mean_pixel_values'])
    stats['std_pixel_overall'] = np.mean(stats['std_pixel_values'])
    
    return stats


def visualize_preprocessing_steps(sample_data, output_path='outputs/figures/preprocess_steps.png'):
    """
    Ön işleme adımlarını görselleştirir (before/after).
    
    Args:
        sample_data (dict): Örnek görüntü verisi
        output_path (str): Çıktı dosya yolu
    """
    print("\n📊 Ön İşleme Adımları Görselleştiriliyor")
    print("-" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor='white')
    fig.suptitle('🖼️ Görüntü Ön İşleme Adımları', 
                fontsize=16, fontweight='bold', color=DARK_GRAY, y=0.98)
    
    # İlk satır: İlk örnek
    sample1 = sample_data['sample_images'][0]
    
    # Orijinal
    axes[0, 0].imshow(cv2.cvtColor(sample1['original'], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('1. Orijinal Görüntü', fontsize=11, fontweight='bold', color=DARK_GRAY)
    axes[0, 0].axis('off')
    axes[0, 0].text(0.5, -0.1, f"Boyut: {sample1['original'].shape}", 
                   transform=axes[0, 0].transAxes, ha='center', fontsize=9, color=DARK_GRAY)
    
    # Grayscale
    axes[0, 1].imshow(sample1['grayscale'], cmap='gray')
    axes[0, 1].set_title('2. Grayscale Dönüşüm', fontsize=11, fontweight='bold', color=DARK_GRAY)
    axes[0, 1].axis('off')
    axes[0, 1].text(0.5, -0.1, f"Boyut: {sample1['grayscale'].shape}", 
                   transform=axes[0, 1].transAxes, ha='center', fontsize=9, color=DARK_GRAY)
    
    # Resized + Normalized
    axes[0, 2].imshow(sample1['resized'], cmap='gray')
    axes[0, 2].set_title('3. Resize (224x224) + Normalize', fontsize=11, fontweight='bold', color=DARK_GRAY)
    axes[0, 2].axis('off')
    axes[0, 2].text(0.5, -0.1, f"Boyut: {sample1['resized'].shape}\nDeğer aralığı: [0, 1]", 
                   transform=axes[0, 2].transAxes, ha='center', fontsize=9, color=DARK_GRAY)
    
    # İkinci satır: İkinci örnek
    sample2 = sample_data['sample_images'][1]
    
    axes[1, 0].imshow(cv2.cvtColor(sample2['original'], cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"Örnek 2 - Orijinal", fontsize=10, color=DARK_GRAY)
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, -0.1, f"Label: {sample2['label']}", 
                   transform=axes[1, 0].transAxes, ha='center', fontsize=9, color=DARK_GRAY)
    
    axes[1, 1].imshow(sample2['grayscale'], cmap='gray')
    axes[1, 1].set_title(f"Grayscale", fontsize=10, color=DARK_GRAY)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(sample2['resized'], cmap='gray')
    axes[1, 2].set_title(f"Resize + Normalize", fontsize=10, color=DARK_GRAY)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Kaydet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Görselleştirme kaydedildi: {output_path}")


def visualize_sample_grid(sample_data, output_path='outputs/figures/preprocess_sample_grid.png'):
    """
    5 örnek görüntüyü grid halinde gösterir.
    """
    print("\n📊 Örnek Grid Oluşturuluyor")
    print("-" * 60)
    
    n_samples = len(sample_data['sample_images'])
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, n_samples*3), facecolor='white')
    fig.suptitle('🖼️ Örnek Satır Görüntüleri - Ön İşleme Pipeline', 
                fontsize=16, fontweight='bold', color=DARK_GRAY, y=0.995)
    
    for i, sample in enumerate(sample_data['sample_images']):
        # Orijinal
        axes[i, 0].imshow(cv2.cvtColor(sample['original'], cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Orijinal\n{sample['subject_id']}, Label {sample['label']}", 
                            fontsize=9, color=DARK_GRAY)
        axes[i, 0].axis('off')
        
        # Grayscale
        axes[i, 1].imshow(sample['grayscale'], cmap='gray')
        axes[i, 1].set_title(f"Grayscale\n{sample['grayscale'].shape}", 
                            fontsize=9, color=DARK_GRAY)
        axes[i, 1].axis('off')
        
        # Resized
        axes[i, 2].imshow(sample['resized'], cmap='gray')
        axes[i, 2].set_title(f"Resized 224x224\nNormalized [0,1]", 
                            fontsize=9, color=DARK_GRAY)
        axes[i, 2].axis('off')
        
        # Processed (RGB 3 kanal)
        axes[i, 3].imshow(sample['processed'][:,:,0], cmap='gray')
        axes[i, 3].set_title(f"CNN Ready\n{sample['processed'].shape}", 
                            fontsize=9, color=DARK_GRAY)
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    # Kaydet
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Grid görseli kaydedildi: {output_path}")


def visualize_statistics(stats, output_path='outputs/figures/preprocess_statistics.png'):
    """
    Ön işleme istatistiklerini görselleştirir.
    """
    print("\n📊 İstatistikler Görselleştiriliyor")
    print("-" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='white')
    fig.suptitle('📊 Ön İşleme İstatistikleri', 
                fontsize=16, fontweight='bold', color=DARK_GRAY, y=0.98)
    
    # 1. Piksel değerleri dağılımı (histogram)
    ax1 = axes[0, 0]
    ax1.hist(stats['mean_pixel_values'], bins=30, 
            color=PASTEL_COLORS['blue'], edgecolor=DARK_GRAY, linewidth=0.5, alpha=0.8)
    ax1.set_title('Ortalama Piksel Değerleri Dağılımı', fontsize=11, fontweight='bold', color=DARK_GRAY)
    ax1.set_xlabel('Ortalama Piksel Değeri [0,1]', fontsize=10)
    ax1.set_ylabel('Frekans', fontsize=10)
    ax1.axvline(stats['mean_pixel_overall'], color='red', linestyle='--', linewidth=2, 
               label=f"Genel Ort: {stats['mean_pixel_overall']:.3f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Standart sapma dağılımı
    ax2 = axes[0, 1]
    ax2.hist(stats['std_pixel_values'], bins=30, 
            color=PASTEL_COLORS['pink'], edgecolor=DARK_GRAY, linewidth=0.5, alpha=0.8)
    ax2.set_title('Piksel Standart Sapması Dağılımı', fontsize=11, fontweight='bold', color=DARK_GRAY)
    ax2.set_xlabel('Standart Sapma', fontsize=10)
    ax2.set_ylabel('Frekans', fontsize=10)
    ax2.axvline(stats['std_pixel_overall'], color='red', linestyle='--', linewidth=2,
               label=f"Genel Ort: {stats['std_pixel_overall']:.3f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Orijinal boyutlar (unique shapes)
    ax3 = axes[1, 0]
    unique_shapes = list(set([str(s) for s in stats['original_shapes']]))
    shape_counts = [stats['original_shapes'].count(eval(s)) for s in unique_shapes]
    if len(unique_shapes) > 10:
        # Çok fazla farklı boyut varsa sadece en yaygın 10'unu göster
        sorted_pairs = sorted(zip(shape_counts, unique_shapes), reverse=True)[:10]
        shape_counts, unique_shapes = zip(*sorted_pairs)
    
    ax3.barh(range(len(unique_shapes)), shape_counts, 
            color=[PASTEL_COLORS['green']] * len(unique_shapes),
            edgecolor=DARK_GRAY, linewidth=0.5)
    ax3.set_yticks(range(len(unique_shapes)))
    ax3.set_yticklabels(unique_shapes, fontsize=8)
    ax3.set_title('Orijinal Görüntü Boyutları', fontsize=11, fontweight='bold', color=DARK_GRAY)
    ax3.set_xlabel('Sayı', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. İşleme özeti (text)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    ÖN İŞLEME ÖZETİ
    
    Toplam İşlenen: {stats['total_processed']:,}
    Hata Sayısı: {stats['total_errors']}
    
    Hedef Boyut: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}
    Normalize Aralığı: [0, 1]
    Renk Uzayı: Grayscale → RGB (3 kanal)
    
    Genel İstatistikler:
    • Ort. Piksel: {stats['mean_pixel_overall']:.4f}
    • Ort. Std: {stats['std_pixel_overall']:.4f}
    
    ✓ CNN için hazır (224x224x3)
    ✓ Normalize edildi [0,1]
    ✓ Tutarlı veri formatı
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top',
            family='monospace', color=DARK_GRAY,
            bbox=dict(boxstyle='round', facecolor=PASTEL_COLORS['lavender'], 
                     alpha=0.3, edgecolor=DARK_GRAY, linewidth=1.5))
    
    plt.tight_layout()
    
    # Kaydet
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ İstatistik görseli kaydedildi: {output_path}")


def save_statistics_report(stats, output_path='outputs/reports/preprocessing_stats.txt'):
    """
    İstatistikleri metin dosyası olarak kaydeder.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ÖN İŞLEME İSTATİSTİKLERİ\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Toplam İşlenen Görüntü: {stats['total_processed']:,}\n")
        f.write(f"Hata Sayısı: {stats['total_errors']}\n\n")
        
        f.write(f"Hedef Boyut: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}\n")
        f.write(f"Normalize Aralığı: [0, 1]\n")
        f.write(f"Çıktı Formatı: (224, 224, 3) - RGB\n\n")
        
        f.write(f"Piksel Değeri İstatistikleri:\n")
        f.write(f"  Genel Ortalama: {stats['mean_pixel_overall']:.6f}\n")
        f.write(f"  Genel Std: {stats['std_pixel_overall']:.6f}\n")
        f.write(f"  Min Ortalama: {min(stats['mean_pixel_values']):.6f}\n")
        f.write(f"  Max Ortalama: {max(stats['mean_pixel_values']):.6f}\n\n")
        
        f.write(f"Orijinal Boyut Çeşitliliği:\n")
        unique_shapes = list(set([str(s) for s in stats['original_shapes']]))
        f.write(f"  Farklı Boyut Sayısı: {len(unique_shapes)}\n")
        if len(unique_shapes) <= 10:
            for shape in unique_shapes:
                count = stats['original_shapes'].count(eval(shape))
                f.write(f"    {shape}: {count} görüntü\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"📄 İstatistik raporu kaydedildi: {output_path}")


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def main():
    """Ana çalıştırma fonksiyonu."""
    print("=" * 60)
    print("🖼️  GÖRÜNTÜ ÖN İŞLEME PIPELINE")
    print("=" * 60)
    
    # Annotations yolu
    annotations_path = Path('data/meta/annotations.csv')
    
    if not annotations_path.exists():
        print(f"❌ HATA: {annotations_path} dosyası bulunamadı!")
        return
    
    # Veri setini işle (sadece ilk 100 örnek - hızlı test için)
    # Tüm veri seti için sample_size=None yapın
    stats = process_dataset(
        annotations_path=annotations_path,
        save_processed=False,  # Disk alanı tasarrufu için kapalı
        sample_size=100  # İlk 100 görüntü
    )
    
    # Görselleştirmeler
    visualize_preprocessing_steps(stats)
    visualize_sample_grid(stats)
    visualize_statistics(stats)
    
    # Rapor kaydet
    save_statistics_report(stats)
    
    print("\n" + "=" * 60)
    print("✨ Ön işleme tamamlandı!")
    print("📁 Görseller: outputs/figures/preprocess_*.png")
    print("📄 Rapor: outputs/reports/preprocessing_stats.txt")
    print("=" * 60)
    
    print("\n💡 Not: Tüm veri setini işlemek için sample_size=None yapın")
    print("   (Bu işlem birkaç dakika sürebilir)")


if __name__ == "__main__":
    main()
