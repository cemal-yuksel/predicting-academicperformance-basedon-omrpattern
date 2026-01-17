"""
Mevcut satır görüntülerinden annotations.csv otomatik oluşturur.
Dosya formatı: {subject_id}_{label}_{line_number}.jpg
Örnek: 10_D_5.jpg -> subject_id=10, label='D', line_number=5

Kullanım: python src/create_annotations.py
"""
import os
import re
from pathlib import Path
import pandas as pd


def parse_filename(filename):
    """
    Dosya adından subject_id, label ve line_number çıkarır.
    
    Format: {subject_id}_{label}_{line_number}.jpg
    Örnek: 10_D_5.jpg -> (10, 'D', 5)
    """
    pattern = r'^(\d+)_([DY])_(\d+)\.jpg$'
    match = re.match(pattern, filename)
    
    if match:
        subject_id = match.group(1)
        label_str = match.group(2)
        line_number = int(match.group(3))
        
        # D=0 (Doğru cevap yok), Y=1 (Yanl doğru cevap var)
        label = 0 if label_str == 'D' else 1
        
        return subject_id, label, line_number
    else:
        return None, None, None


def create_annotations_from_images(data_dir='data', output_path='data/meta/annotations.csv'):
    """
    data/ klasöründeki tüm .jpg dosyalarından annotations.csv oluşturur.
    
    Not: Bu satır görüntüleri zaten kesilmiş formda olduğu için,
    bounding box koordinatları placeholder olarak eklenecek.
    """
    data_path = Path(data_dir)
    
    print("🔍 Görüntü dosyaları taranıyor...")
    
    records = []
    
    # data/ klasöründeki tüm jpg dosyalarını tara
    jpg_files = list(data_path.glob('*.jpg'))
    
    print(f"📁 {len(jpg_files)} adet .jpg dosyası bulundu")
    
    for img_file in jpg_files:
        filename = img_file.name
        subject_id, label, line_number = parse_filename(filename)
        
        if subject_id is not None:
            # Relative path
            relative_path = f"data/{filename}"
            
            # Bu satır görüntüleri için placeholder koordinatlar
            # (Gerçek formdan kesilen satırlar olduğu için bounding box yok)
            record = {
                'subject_id': f"S{subject_id.zfill(2)}",  # S01, S02, ... formatı
                'label': label,
                'line_image_path': relative_path,
                'line_number': line_number,
                'notes': 'Cropped line image'
            }
            
            records.append(record)
        else:
            print(f"⚠️  Uyarı: {filename} parse edilemedi")
    
    # DataFrame oluştur
    df = pd.DataFrame(records)
    
    # Subject ID ve line number'a göre sırala
    df = df.sort_values(['subject_id', 'line_number']).reset_index(drop=True)
    
    # Özet istatistikler
    print("\n📊 Veri Seti Özeti:")
    print(f"   Toplam satır: {len(df)}")
    print(f"   Benzersiz birey sayısı: {df['subject_id'].nunique()}")
    print(f"   Label 0 (D): {(df['label'] == 0).sum()}")
    print(f"   Label 1 (Y): {(df['label'] == 1).sum()}")
    
    # Birey başına satır sayısı
    lines_per_subject = df.groupby('subject_id').size()
    print(f"\n   Birey başına satır sayısı:")
    print(f"      Min: {lines_per_subject.min()}")
    print(f"      Max: {lines_per_subject.max()}")
    print(f"      Ortalama: {lines_per_subject.mean():.1f}")
    
    # CSV olarak kaydet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Annotations dosyası oluşturuldu: {output_path}")
    print(f"📋 İlk 10 satır:")
    print(df.head(10).to_string())
    
    return df


def main():
    """Ana çalıştırma fonksiyonu."""
    print("=" * 60)
    print("📝 ANNOTATIONS.CSV OLUŞTURUCU")
    print("=" * 60)
    
    df = create_annotations_from_images(
        data_dir='data',
        output_path='data/meta/annotations.csv'
    )
    
    print("\n" + "=" * 60)
    print("✨ İşlem tamamlandı!")
    print("🔍 Doğrulama için: python src/02_metadata.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
