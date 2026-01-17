"""
Metadata doğrulama ve analiz script'i.
annotations.csv şemasını kontrol eder ve veri seti istatistiklerini raporlar.

Kullanım:
    python src/02_metadata.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter


# ============================================================================
# ŞEMA TANIMLARI
# ============================================================================

REQUIRED_COLUMNS = [
    'subject_id',
    'label',
    'line_image_path',
    'line_number'
]

OPTIONAL_COLUMNS = [
    'form_id',
    'scan_date',
    'notes'
]

NUMERIC_COLUMNS = ['label', 'line_number']
VALID_LABELS = {0, 1}


# ============================================================================
# DOĞRULAMA FONKSİYONLARI
# ============================================================================

def check_column_existence(df):
    """Zorunlu kolonların varlığını kontrol eder."""
    print("\n🔍 Kolon Varlığı Kontrolü")
    print("-" * 60)
    
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    
    if missing_columns:
        print(f"❌ HATA: Eksik kolonlar bulundu: {missing_columns}")
        print(f"   Zorunlu kolonlar: {REQUIRED_COLUMNS}")
        return False
    
    print("✅ Tüm zorunlu kolonlar mevcut")
    
    # Opsiyonel kolonları kontrol et
    available_optional = [col for col in OPTIONAL_COLUMNS if col in df.columns]
    if available_optional:
        print(f"ℹ️  Opsiyonel kolonlar: {available_optional}")
    
    return True


def check_data_types(df):
    """Veri tiplerini kontrol eder."""
    print("\n🔍 Veri Tipi Kontrolü")
    print("-" * 60)
    
    errors = []
    
    # subject_id string olmalı
    if df['subject_id'].dtype not in ['object', 'string']:
        errors.append("subject_id string tipinde olmalı")
    
    # Numerik kolonlar
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    errors.append(f"{col} kolonunda numeric olmayan değerler var")
            except Exception as e:
                errors.append(f"{col} kolonunda dönüşüm hatası: {e}")
    
    if errors:
        print("❌ Veri tipi hataları:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    print("✅ Veri tipleri uygun")
    return True


def check_missing_values(df):
    """Eksik değerleri kontrol eder."""
    print("\n🔍 Eksik Değer Kontrolü")
    print("-" * 60)
    
    errors = []
    
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                errors.append(f"{col}: {missing_count} eksik değer")
    
    if errors:
        print("❌ Eksik değerler bulundu:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    print("✅ Zorunlu kolonlarda eksik değer yok")
    return True


def check_label_values(df):
    """Label değerlerini kontrol eder."""
    print("\n🔍 Label Değerleri Kontrolü")
    print("-" * 60)
    
    unique_labels = set(df['label'].dropna().unique())
    invalid_labels = unique_labels - VALID_LABELS
    
    if invalid_labels:
        print(f"❌ HATA: Geçersiz label değerleri bulundu: {invalid_labels}")
        print(f"   Geçerli değerler: {VALID_LABELS}")
        return False
    
    print(f"✅ Label değerleri geçerli: {sorted(unique_labels)}")
    
    # Label dağılımı
    label_counts = df['label'].value_counts().sort_index()
    print("\nLabel Dağılımı:")
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   Label {label}: {count} ({percentage:.1f}%)")
    
    return True


def check_line_numbers(df):
    """Satır numaralarını kontrol eder."""
    print("\n🔍 Satır Numarası Kontrolü")
    print("-" * 60)
    
    errors = []
    
    # Negatif satır numarası kontrolü
    negative = df[df['line_number'] < 0]
    if len(negative) > 0:
        errors.append(f"line_number < 0: {len(negative)} satır")
    
    if errors:
        print("❌ Satır numarası hataları:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    print("✅ Satır numaraları geçerli")
    
    # İstatistikler
    print(f"\nSatır Numarası İstatistikleri:")
    print(f"   Min: {df['line_number'].min()}")
    print(f"   Max: {df['line_number'].max()}")
    print(f"   Ortalama: {df['line_number'].mean():.1f}")
    
    # Her subject için satır sayısı
    lines_per_subject = df.groupby('subject_id')['line_number'].nunique()
    print(f"\nSubject başına benzersiz satır sayısı:")
    print(f"   Min: {lines_per_subject.min()}")
    print(f"   Max: {lines_per_subject.max()}")
    print(f"   Ortalama: {lines_per_subject.mean():.1f}")
    
    return True


def check_file_paths(df, base_dir=None):
    """Dosya yollarının varlığını kontrol eder."""
    print("\n🔍 Dosya Yolu Kontrolü")
    print("-" * 60)
    
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)
    
    missing_files = []
    existing_count = 0
    
    for idx, row in df.iterrows():
        file_path = Path(row['line_image_path'])
        
        # Göreceli yol ise base_dir ile birleştir
        if not file_path.is_absolute():
            file_path = base_dir / file_path
        
        if not file_path.exists():
            missing_files.append(row['line_image_path'])
        else:
            existing_count += 1
    
    if missing_files:
        print(f"⚠️  UYARI: {len(missing_files)} dosya bulunamadı")
        print(f"   Mevcut: {existing_count}/{len(df)}")
        if len(missing_files) <= 5:
            print("   Eksik dosyalar:")
            for f in missing_files:
                print(f"      - {f}")
        else:
            print(f"   İlk 5 eksik dosya:")
            for f in missing_files[:5]:
                print(f"      - {f}")
        return False
    
    print(f"✅ Tüm dosyalar mevcut ({existing_count}/{len(df)})")
    return True


def analyze_subject_distribution(df):
    """Subject ID dağılımını analiz eder."""
    print("\n📊 Subject ID Analizi")
    print("-" * 60)
    
    n_subjects = df['subject_id'].nunique()
    n_samples = len(df)
    samples_per_subject = n_samples / n_subjects
    
    print(f"Toplam Örnek Sayısı: {n_samples}")
    print(f"Benzersiz Birey Sayısı (subject_id): {n_subjects}")
    print(f"Birey Başına Ortalama Örnek: {samples_per_subject:.2f}")
    
    # Her subject için örnek sayısı
    subject_counts = df['subject_id'].value_counts()
    print(f"\nÖrnek Sayısı Dağılımı:")
    print(f"   Min: {subject_counts.min()} örnek")
    print(f"   Max: {subject_counts.max()} örnek")
    print(f"   Medyan: {subject_counts.median():.0f} örnek")
    
    # Subject başına label dağılımı
    print(f"\n📌 Subject Başına Label Dağılımı:")
    
    # Her subject için label sayılarını hesapla
    for subject_id in df['subject_id'].unique()[:5]:  # İlk 5 subject örneği
        subject_data = df[df['subject_id'] == subject_id]
        label_counts = subject_data['label'].value_counts().to_dict()
        print(f"   {subject_id}: {label_counts}")
    
    return n_subjects, n_samples


def print_sample_data(df, n=5):
    """İlk n satırı yazdırır."""
    print(f"\n📋 İlk {n} Satır Önizleme")
    print("-" * 60)
    print(df.head(n).to_string())


def save_summary_report(df, output_path):
    """Özet rapor dosyası oluşturur."""
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("METADATA DOĞRULAMA RAPORU")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Genel bilgiler
    report_lines.append("GENEL BİLGİLER")
    report_lines.append(f"Toplam Satır: {len(df)}")
    report_lines.append(f"Toplam Kolon: {len(df.columns)}")
    report_lines.append(f"Kolonlar: {', '.join(df.columns.tolist())}")
    report_lines.append("")
    
    # Subject analizi
    n_subjects = df['subject_id'].nunique()
    report_lines.append("SUBJECT ANALİZİ")
    report_lines.append(f"Benzersiz Birey Sayısı: {n_subjects}")
    report_lines.append(f"Birey Başına Ortalama Örnek: {len(df)/n_subjects:.2f}")
    report_lines.append("")
    
    # Label dağılımı
    report_lines.append("LABEL DAĞILIMI")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        report_lines.append(f"Label {label}: {count} ({percentage:.1f}%)")
    report_lines.append("")
    
    # Satır numarası istatistikleri
    report_lines.append("SATIR NUMARASI İSTATİSTİKLERİ")
    report_lines.append(f"Min: {df['line_number'].min()}")
    report_lines.append(f"Max: {df['line_number'].max()}")
    report_lines.append(f"Ortalama: {df['line_number'].mean():.1f}")
    report_lines.append("")
    
    # İlk 10 subject
    report_lines.append("İLK 10 SUBJECT")
    for i, (subject_id, count) in enumerate(df['subject_id'].value_counts().head(10).items()):
        report_lines.append(f"{i+1}. {subject_id}: {count} örnek")
    report_lines.append("")
    
    report_lines.append("=" * 60)
    
    # Dosyaya yaz
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n📄 Özet rapor kaydedildi: {output_path}")


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def validate_metadata(csv_path, base_dir=None, save_report=True):
    """
    Metadata dosyasını doğrular ve analiz eder.
    
    Args:
        csv_path (str): annotations.csv dosya yolu
        base_dir (str): Proje ana dizini (dosya yolu kontrolü için)
        save_report (bool): Özet rapor oluşturulsun mu
        
    Returns:
        tuple: (df, is_valid)
    """
    csv_path = Path(csv_path)
    
    print("=" * 60)
    print("🔍 METADATA DOĞRULAMA BAŞLIYOR")
    print("=" * 60)
    print(f"Dosya: {csv_path}")
    
    # CSV'yi oku
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ CSV başarıyla okundu: {len(df)} satır")
    except Exception as e:
        print(f"❌ HATA: CSV okunamadı: {e}")
        return None, False
    
    # Doğrulama kontrollerini sırayla yap
    all_checks_passed = True
    
    all_checks_passed &= check_column_existence(df)
    all_checks_passed &= check_data_types(df)
    all_checks_passed &= check_missing_values(df)
    all_checks_passed &= check_label_values(df)
    all_checks_passed &= check_line_numbers(df)
    
    # Dosya kontrolü (hata olsa da devam et, sadece uyarı)
    check_file_paths(df, base_dir)
    
    # Subject analizi
    analyze_subject_distribution(df)
    
    # Önizleme
    print_sample_data(df)
    
    # Rapor kaydet
    if save_report:
        report_path = Path('outputs/reports/metadata_validation.txt')
        save_summary_report(df, report_path)
    
    # Sonuç
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✅ TÜM DOĞRULAMA KONTROLLERI BAŞARILI!")
        print("✨ Metadata GroupKFold CV için hazır")
    else:
        print("❌ DOĞRULAMA HATALARI VAR!")
        print("⚠️  Lütfen yukarıdaki hataları düzeltin")
    print("=" * 60)
    
    return df, all_checks_passed


# ============================================================================
# SCRIPT ÇALIŞTIRMA
# ============================================================================

def main():
    """Ana çalıştırma fonksiyonu."""
    
    # CSV yolu
    csv_path = Path('data/meta/annotations.csv')
    
    if not csv_path.exists():
        print(f"❌ HATA: {csv_path} dosyası bulunamadı!")
        print(f"\nÖrnek annotations.csv oluşturmak için:")
        print(f"   1. data/meta/ klasörüne annotations.csv dosyası ekleyin")
        print(f"   2. Zorunlu kolonlar: {', '.join(REQUIRED_COLUMNS)}")
        print(f"\nŞema dokümantasyonu: data/meta/annotations_schema.md")
        return
    
    # Doğrulama
    df, is_valid = validate_metadata(
        csv_path=csv_path,
        base_dir=Path.cwd(),
        save_report=True
    )
    
    if is_valid:
        print("\n🎉 Metadata hazır! Sonraki adıma geçebilirsiniz.")
    else:
        print("\n⚠️  Lütfen hataları düzeltip tekrar çalıştırın.")


if __name__ == "__main__":
    main()
