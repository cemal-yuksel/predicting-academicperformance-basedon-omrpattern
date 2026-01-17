"""
Feature extraction script - Frozen CNN ile transfer learning.
ResNet50 (veya başka bir CNN) kullanarak görüntülerden feature vektörleri çıkarır.

Kullanım:
    python src/05_feature_extraction.py --model resnet50 --batch_size 32
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import time

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

# Pastel görselleştirme
from config_viz import (
    create_info_box,
    create_figure,
    save_figure,
    PASTEL_COLORS,
    DARK_GRAY
)


# ============================================================================
# DATASET CLASS
# ============================================================================

class OMRLineDataset(Dataset):
    """
    OMR satır görüntüleri için PyTorch Dataset.
    """
    def __init__(self, annotations_df, transform=None):
        """
        Args:
            annotations_df (pd.DataFrame): annotations.csv verisi
            transform: PyTorch transforms
        """
        self.df = annotations_df.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Görüntüyü yükle
        img_path = row['line_image_path']
        img = cv2.imread(str(img_path))
        
        if img is None:
            raise ValueError(f"Görüntü okunamadı: {img_path}")
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Transform uygula
        if self.transform:
            img = self.transform(img)
        
        return {
            'image': img,
            'subject_id': row['subject_id'],
            'label': row['label'],
            'line_number': row['line_number'],
            'index': idx
        }


# ============================================================================
# MODEL SETUP
# ============================================================================

def get_feature_extractor(model_name='resnet50', pretrained=True):
    """
    Pretrained CNN modelini yükler ve feature extractor olarak hazırlar.
    
    Args:
        model_name (str): Model adı ('resnet50', 'resnet18', 'vgg16', etc.)
        pretrained (bool): Pretrained weights kullan mı
        
    Returns:
        tuple: (model, feature_dim)
    """
    print(f"\n🧠 Model Yükleniyor: {model_name}")
    print("-" * 60)
    
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        # Son FC layer'ı kaldır
        model = nn.Sequential(*list(model.children())[:-1])
        feature_dim = 2048
        
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model = nn.Sequential(*list(model.children())[:-1])
        feature_dim = 512
        
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        # Classifier'ı kaldır, sadece features
        model = model.features
        feature_dim = 512 * 7 * 7  # VGG16 için
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        # Son classifier'ı kaldır
        model = nn.Sequential(*list(model.children())[:-1])
        feature_dim = 1280
        
    else:
        raise ValueError(f"Desteklenmeyen model: {model_name}")
    
    # Model'i eval moduna al ve freeze et
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"✅ Model yüklendi: {model_name}")
    print(f"📊 Feature boyutu: {feature_dim}")
    
    return model, feature_dim


def get_transforms():
    """
    PyTorch transforms (ImageNet normalization).
    
    Returns:
        transforms.Compose
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]     # ImageNet std
        )
    ])


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(model, dataloader, device, feature_dim):
    """
    Tüm veri setinden feature'ları çıkarır.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: torch.device
        feature_dim: Feature vektör boyutu
        
    Returns:
        dict: Features ve metadata
    """
    print("\n🔍 Feature Extraction Başlıyor")
    print("-" * 60)
    
    model = model.to(device)
    
    # Sonuçları saklamak için listeler
    all_features = []
    all_subject_ids = []
    all_labels = []
    all_line_numbers = []
    all_indices = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # Batch'i device'a taşı
            images = batch['image'].to(device)
            
            # Forward pass
            features = model(images)
            
            # Flatten (eğer gerekliyse)
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
            
            # CPU'ya al ve numpy'a çevir
            features = features.cpu().numpy()
            
            # Sonuçları sakla
            all_features.append(features)
            all_subject_ids.extend(batch['subject_id'])
            all_labels.extend(batch['label'].numpy())
            all_line_numbers.extend(batch['line_number'].numpy())
            all_indices.extend(batch['index'].numpy())
    
    elapsed_time = time.time() - start_time
    
    # Tüm feature'ları birleştir
    all_features = np.vstack(all_features)
    
    print(f"\n✅ Feature extraction tamamlandı!")
    print(f"   Süre: {elapsed_time:.2f} saniye")
    print(f"   Hız: {len(all_features) / elapsed_time:.1f} görüntü/saniye")
    print(f"   Feature shape: {all_features.shape}")
    
    return {
        'features': all_features,
        'subject_ids': all_subject_ids,
        'labels': np.array(all_labels),
        'line_numbers': np.array(all_line_numbers),
        'indices': np.array(all_indices),
        'elapsed_time': elapsed_time
    }


def save_features(features_dict, model_name, output_dir='outputs'):
    """
    Feature'ları parquet formatında kaydeder.
    
    Args:
        features_dict (dict): Extract edilen feature'lar
        model_name (str): Model adı
        output_dir (str): Çıktı klasörü
        
    Returns:
        Path: Kaydedilen dosya yolu
    """
    print("\n💾 Features Kaydediliyor")
    print("-" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # DataFrame oluştur
    feature_cols = [f'f{i}' for i in range(features_dict['features'].shape[1])]
    
    df = pd.DataFrame(features_dict['features'], columns=feature_cols)
    df.insert(0, 'subject_id', features_dict['subject_ids'])
    df.insert(1, 'label', features_dict['labels'])
    df.insert(2, 'line_number', features_dict['line_numbers'])
    df.insert(3, 'index', features_dict['indices'])
    
    # Parquet olarak kaydet
    output_path = output_dir / f'features_{model_name}.parquet'
    df.to_parquet(output_path, index=False)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"✅ Features kaydedildi: {output_path}")
    print(f"📊 Dosya boyutu: {file_size_mb:.2f} MB")
    print(f"📋 Shape: {df.shape}")
    
    return output_path


def visualize_feature_info(features_dict, model_name, output_dir='outputs/figures'):
    """
    Feature extraction bilgilerini görselleştirir.
    """
    print("\n📊 Feature Info Görselleştiriliyor")
    print("-" * 60)
    
    output_dir = Path(output_dir)
    
    n_samples = len(features_dict['features'])
    feature_dim = features_dict['features'].shape[1]
    elapsed_time = features_dict['elapsed_time']
    speed = n_samples / elapsed_time
    
    # Bilgi kutusu
    info_lines = [
        f"Model: {model_name}",
        f"Feature Boyutu: {feature_dim}",
        "",
        f"Toplam Görüntü: {n_samples:,}",
        f"İşleme Süresi: {elapsed_time:.1f} saniye",
        f"Hız: {speed:.1f} görüntü/saniye",
        "",
        f"Feature Shape: ({n_samples}, {feature_dim})",
        "",
        "✓ Frozen CNN",
        "✓ Transfer Learning",
        "✓ Batch Processing"
    ]
    
    fig = create_info_box(
        text_lines=info_lines,
        title=f"🧠 Feature Extraction - {model_name}",
        filepath=output_dir / f'feature_extraction_info_{model_name}.png',
        box_color=PASTEL_COLORS['lavender']
    )
    
    print(f"✅ Info box kaydedildi")


def visualize_feature_statistics(features_dict, model_name, output_dir='outputs/figures'):
    """
    Feature istatistiklerini görselleştirir.
    """
    print("\n📊 Feature İstatistikleri Görselleştiriliyor")
    print("-" * 60)
    
    output_dir = Path(output_dir)
    
    features = features_dict['features']
    
    # İstatistikler hesapla
    feature_means = np.mean(features, axis=0)
    feature_stds = np.std(features, axis=0)
    feature_mins = np.min(features, axis=0)
    feature_maxs = np.max(features, axis=0)
    
    # Görselleştir
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
    fig.suptitle(f'📊 Feature İstatistikleri - {model_name}', 
                fontsize=16, fontweight='bold', color=DARK_GRAY, y=0.98)
    
    # 1. Feature ortalamaları
    ax1 = axes[0, 0]
    ax1.hist(feature_means, bins=50, color=PASTEL_COLORS['blue'], 
            edgecolor=DARK_GRAY, linewidth=0.5, alpha=0.8)
    ax1.set_title('Feature Ortalamaları Dağılımı', fontsize=11, fontweight='bold', color=DARK_GRAY)
    ax1.set_xlabel('Ortalama Değer', fontsize=10)
    ax1.set_ylabel('Frekans', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Feature standart sapmaları
    ax2 = axes[0, 1]
    ax2.hist(feature_stds, bins=50, color=PASTEL_COLORS['pink'], 
            edgecolor=DARK_GRAY, linewidth=0.5, alpha=0.8)
    ax2.set_title('Feature Standart Sapmaları', fontsize=11, fontweight='bold', color=DARK_GRAY)
    ax2.set_xlabel('Standart Sapma', fontsize=10)
    ax2.set_ylabel('Frekans', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature aralıkları (max - min)
    ax3 = axes[1, 0]
    feature_ranges = feature_maxs - feature_mins
    ax3.hist(feature_ranges, bins=50, color=PASTEL_COLORS['green'], 
            edgecolor=DARK_GRAY, linewidth=0.5, alpha=0.8)
    ax3.set_title('Feature Aralıkları (Max - Min)', fontsize=11, fontweight='bold', color=DARK_GRAY)
    ax3.set_xlabel('Aralık', fontsize=10)
    ax3.set_ylabel('Frekans', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. İstatistik özeti
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    FEATURE İSTATİSTİKLERİ
    
    Feature Boyutu: {features.shape[1]:,}
    Örnek Sayısı: {features.shape[0]:,}
    
    Genel İstatistikler:
    • Ort. Mean: {np.mean(feature_means):.4f}
    • Ort. Std: {np.mean(feature_stds):.4f}
    • Min Değer: {np.min(features):.4f}
    • Max Değer: {np.max(features):.4f}
    
    Feature Özellikleri:
    • Normalize: ImageNet
    • Frozen: Evet
    • Pretrained: ImageNet
    
    ✓ Transfer Learning
    ✓ ML için hazır
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top',
            family='monospace', color=DARK_GRAY,
            bbox=dict(boxstyle='round', facecolor=PASTEL_COLORS['baby_blue'], 
                     alpha=0.3, edgecolor=DARK_GRAY, linewidth=1.5))
    
    plt.tight_layout()
    save_figure(fig, output_dir / f'feature_statistics_{model_name}.png')
    plt.close()
    
    print(f"✅ İstatistik görseli kaydedildi")


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def main():
    """Ana çalıştırma fonksiyonu."""
    
    # Argüman parse
    parser = argparse.ArgumentParser(description='Feature Extraction with CNN')
    parser.add_argument('--model', type=str, default='resnet50', 
                       choices=['resnet50', 'resnet18', 'vgg16', 'efficientnet_b0'],
                       help='CNN model adı')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, 
                       help='DataLoader worker sayısı')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 FEATURE EXTRACTION - FROZEN CNN")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Batch Size: {args.batch_size}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Annotations'ı yükle
    annotations_path = Path('data/meta/annotations.csv')
    if not annotations_path.exists():
        print(f"❌ HATA: {annotations_path} bulunamadı!")
        return
    
    df = pd.read_csv(annotations_path)
    print(f"\n✅ Annotations yüklendi: {len(df)} görüntü")
    
    # Dataset ve DataLoader
    transform = get_transforms()
    dataset = OMRLineDataset(df, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # Sırayı koru
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    print(f"✅ DataLoader hazır: {len(dataloader)} batch")
    
    # Model yükle
    model, feature_dim = get_feature_extractor(args.model, pretrained=True)
    
    # Feature extraction
    features_dict = extract_features(model, dataloader, device, feature_dim)
    
    # Kaydet
    output_path = save_features(features_dict, args.model, output_dir='outputs')
    
    # Görselleştir
    visualize_feature_info(features_dict, args.model)
    visualize_feature_statistics(features_dict, args.model)
    
    print("\n" + "=" * 60)
    print("✨ Feature extraction tamamlandı!")
    print(f"📁 Features: {output_path}")
    print(f"📊 Görseller: outputs/figures/feature_*.png")
    print("=" * 60)
    
    print("\n🎯 Sonraki adım: Klasik ML modelleri eğitimi")
    print("   python src/06_train_models.py")


if __name__ == "__main__":
    main()
