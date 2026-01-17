"""
Çoklu Transfer Learning Modelleri ile Feature Extraction.
Makaledeki 19 farklı TL modelini test eder ve karşılaştırır.

Modeller:
- AlexNet, VGG16, VGG19
- ResNet50, ResNet101, ResNet152
- DenseNet169, DenseNet201
- InceptionV3
- EfficientNet serisi (B0-B7, V2-B2, V2-B3, V2-S, V2-L)

Kullanım:
    python src/09_multiple_transfer_learning.py --models all
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2

import matplotlib.pyplot as plt
from tqdm import tqdm

from config_viz import create_pastel_barplot, save_figure, PASTEL_COLORS


# ============================================================================
# DATASET
# ============================================================================

class OMRLineDataset(Dataset):
    """OMR line images dataset for PyTorch."""
    
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['line_image_path']
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, idx


# ============================================================================
# MODEL FACTORY
# ============================================================================

def get_model_and_dim(model_name):
    """
    Transfer learning modelini ve feature dimension'ını döndürür.
    
    Returns:
        tuple: (model, feature_dim)
    """
    
    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print(f"\n📥 Loading model: {model_name}")
    
    # Model selection
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        feature_dim = 4096  # FC6 output
        # Remove last layer
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        feature_dim = 4096
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
        feature_dim = 4096
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        feature_dim = 2048
        model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
        feature_dim = 2048
        model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
        feature_dim = 2048
        model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        
    elif model_name == 'densenet169':
        model = models.densenet169(pretrained=True)
        feature_dim = 1664
        model = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        
    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=True)
        feature_dim = 1920
        model = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        
    elif model_name == 'inceptionv3':
        model = models.inception_v3(pretrained=True, transform_input=False, aux_logits=False)
        feature_dim = 2048
        model.fc = nn.Identity()
        # Inception needs 299x299 input
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        feature_dim = 1280
        model.classifier = nn.Identity()
        
    elif model_name == 'efficientnet_b1':
        model = models.efficientnet_b1(pretrained=True)
        feature_dim = 1280
        model.classifier = nn.Identity()
        
    elif model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(pretrained=True)
        feature_dim = 1408
        model.classifier = nn.Identity()
        
    elif model_name == 'efficientnet_b5':
        model = models.efficientnet_b5(pretrained=True)
        feature_dim = 2048
        model.classifier = nn.Identity()
        
    elif model_name == 'efficientnet_b6':
        model = models.efficientnet_b6(pretrained=True)
        feature_dim = 2304
        model.classifier = nn.Identity()
        
    elif model_name == 'efficientnet_b7':
        model = models.efficientnet_b7(pretrained=True)
        feature_dim = 2560
        model.classifier = nn.Identity()
        
    elif model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(pretrained=True)
        feature_dim = 1280
        model.classifier = nn.Identity()
        
    elif model_name == 'efficientnet_v2_m':
        model = models.efficientnet_v2_m(pretrained=True)
        feature_dim = 1280
        model.classifier = nn.Identity()
        
    elif model_name == 'efficientnet_v2_l':
        model = models.efficientnet_v2_l(pretrained=True)
        feature_dim = 1280
        model.classifier = nn.Identity()
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    
    print(f"✅ Model loaded: {model_name} (feature_dim={feature_dim})")
    
    return model, transform, feature_dim


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(model, dataloader, device='cpu'):
    """Extract features using the model."""
    
    features_list = []
    indices_list = []
    
    with torch.no_grad():
        for batch_img, batch_idx in tqdm(dataloader, desc="Extracting features"):
            batch_img = batch_img.to(device)
            
            # Forward pass
            features = model(batch_img)
            
            features_list.append(features.cpu().numpy())
            indices_list.append(batch_idx.numpy())
    
    # Concatenate
    features = np.vstack(features_list)
    indices = np.concatenate(indices_list)
    
    return features, indices


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multiple Transfer Learning Feature Extraction')
    parser.add_argument('--annotations', type=str, default='data/meta/annotations.csv',
                       help='Annotations CSV path')
    parser.add_argument('--output_dir', type=str, default='outputs/features_tl',
                       help='Output directory for features')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for DataLoader')
    parser.add_argument('--models', type=str, default='all',
                       help='Comma-separated model names or "all"')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"{'🧠 MULTIPLE TRANSFER LEARNING FEATURE EXTRACTION':^80}")
    print("=" * 80)
    print(f"""
    ╔════════════════════════════════════════════════════════════════════╗
    ║  🎯 MISSION: Extract features from 19 different TL architectures  ║
    ║  📊 Dataset: 2100 OMR line images                                  ║
    ║  🔬 Method: Frozen pretrained CNNs                                 ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load annotations
    print(f"\n📂 Loading annotations: {args.annotations}")
    df = pd.read_csv(args.annotations)
    print(f"✅ Loaded: {len(df)} samples")
    
    # Model list
    all_models = [
        'alexnet', 'vgg16', 'vgg19',
        'resnet50', 'resnet101', 'resnet152',
        'densenet169', 'densenet201',
        'inceptionv3',
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
        'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
        'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'
    ]
    
    if args.models == 'all':
        model_names = all_models
    else:
        model_names = [m.strip() for m in args.models.split(',')]
    
    print(f"\n🤖 Models to process: {len(model_names)}")
    print(f"┌{'─'*78}┐")
    for i, name in enumerate(model_names, 1):
        status = "⏳" if i == 1 else "⏸️"
        print(f"│ {status} {i:2d}. {name:<70} │")
    print(f"└{'─'*78}┘")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n💻 Device: {device}")
    
    # Process each model
    results = []
    total_models = len(model_names)
    
    for model_idx, model_name in enumerate(model_names, 1):
        print(f"\n{'='*80}")
        print(f"📦 MODEL {model_idx}/{total_models}: {model_name.upper()}")
        print(f"{'='*80}")
        print(f"Progress: [{'█' * model_idx}{'░' * (total_models - model_idx)}] {model_idx}/{total_models} ({100*model_idx/total_models:.1f}%)")
        
        try:
            # Get model
            model, transform, feature_dim = get_model_and_dim(model_name)
            model = model.to(device)
            
            # Dataset and DataLoader
            dataset = OMRLineDataset(df, transform=transform)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                                   shuffle=False, num_workers=0)
            
            # Extract features
            start_time = time.time()
            print(f"\n⏳ Extracting features...")
            print(f"├─ Batch size: {args.batch_size}")
            print(f"├─ Total batches: {len(dataloader)}")
            print(f"└─ Expected time: ~{len(dataloader) * 0.5:.0f}s")
            
            features, indices = extract_features(model, dataloader, device)
            elapsed_time = time.time() - start_time
            
            # Create feature DataFrame
            feature_cols = [f'f{i:04d}' for i in range(feature_dim)]
            df_features = pd.DataFrame(features, columns=feature_cols)
            
            # Add metadata
            df_features['subject_id'] = df.iloc[indices]['subject_id'].values
            df_features['label'] = df.iloc[indices]['label'].values
            df_features['line_image_path'] = df.iloc[indices]['line_image_path'].values
            df_features['line_number'] = df.iloc[indices]['line_number'].values
            
            # Reorder columns
            cols = ['subject_id', 'label', 'line_image_path', 'line_number'] + feature_cols
            df_features = df_features[cols]
            
            # Save
            output_path = output_dir / f'features_{model_name}.parquet'
            df_features.to_parquet(output_path, index=False)
            
            file_size_mb = output_path.stat().st_size / (1024**2)
            
            print(f"\n✅ SUCCESS!")
            print(f"┌{'─'*78}┐")
            print(f"│ {'Feature Extraction Complete':<76} │")
            print(f"├{'─'*78}┤")
            print(f"│ 📊 Shape:        {str(features.shape):<60} │")
            print(f"│ ⏱️  Time:         {elapsed_time:.2f}s ({len(df)/elapsed_time:.1f} img/s){' '*(60-len(f'{elapsed_time:.2f}s ({len(df)/elapsed_time:.1f} img/s)'))} │")
            print(f"│ 💾 File Size:    {file_size_mb:.2f} MB{' '*(60-len(f'{file_size_mb:.2f} MB'))} │")
            print(f"│ 📁 Output:       {output_path.name:<60} │")
            print(f"└{'─'*78}┘")
            
            results.append({
                'model': model_name,
                'feature_dim': feature_dim,
                'n_samples': len(df),
                'time_sec': elapsed_time,
                'images_per_sec': len(df)/elapsed_time,
                'file_size_mb': file_size_mb,
                'output_path': str(output_path)
            })
            
            # Clean up
            del model
            del dataset
            del dataloader
            torch.cuda.empty_cache() if device == 'cuda' else None
            
        except Exception as e:
            print(f"\n❌ ERROR!")
            print(f"├─ Model: {model_name}")
            print(f"└─ Error: {e}")
            results.append({
                'model': model_name,
                'error': str(e)
            })
        
        # Show overall progress
        if model_idx < total_models:
            remaining = total_models - model_idx
            print(f"\n{'─'*80}")
            print(f"📈 Overall Progress: {model_idx}/{total_models} complete")
            print(f"⏭️  Remaining: {remaining} models")
            print(f"{'─'*80}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"{'🎉 EXTRACTION SUMMARY':^80}")
    print(f"{'='*80}")
    
    df_results = pd.DataFrame(results)
    
    # Save summary
    summary_path = output_dir / 'extraction_summary.csv'
    df_results.to_csv(summary_path, index=False)
    
    successful = len([r for r in results if 'error' not in r])
    failed = len([r for r in results if 'error' in r])
    
    print(f"\n📊 Results:")
    print(f"   ✅ Successful: {successful}/{total_models}")
    print(f"   ❌ Failed: {failed}/{total_models}")
    print(f"   📄 Summary: {summary_path.name}")
    
    # Display table
    if 'error' not in df_results.columns or df_results['error'].isna().all():
        print(f"\n{'─'*80}")
        print(f"{'Model':<25} {'Dim':<8} {'Time':<10} {'Speed':<15} {'Size':<10}")
        print(f"{'─'*80}")
        for _, row in df_results.iterrows():
            print(f"{row['model']:<25} {row['feature_dim']:<8} "
                  f"{row['time_sec']:<10.1f}s {row['images_per_sec']:<15.1f}img/s "
                  f"{row['file_size_mb']:<10.1f}MB")
        print(f"{'─'*80}")
        
        # Statistics
        total_time = df_results['time_sec'].sum()
        avg_speed = df_results['images_per_sec'].mean()
        total_size = df_results['file_size_mb'].sum()
        
        print(f"\n📈 Statistics:")
        print(f"   ⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"   ⚡ Avg speed: {avg_speed:.1f} images/second")
        print(f"   💾 Total size: {total_size:.1f} MB")
    
    print(f"\n🎯 All features saved in: {output_dir}")
    print(f"\n{'='*80}")
    print(f"✨ NEXT STEP: Comprehensive Classification")
    print(f"{'='*80}")
    print(f"   python src/10_comprehensive_classification.py")


if __name__ == "__main__":
    main()
