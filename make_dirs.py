"""
Proje klasör yapısını oluşturan script.
Kullanım: python make_dirs.py
"""
import os
from pathlib import Path


def create_project_structure():
    """Proje için gerekli tüm klasörleri oluşturur."""
    
    # Ana dizin (script'in çalıştığı yer)
    base_dir = Path(__file__).parent
    
    # Oluşturulacak klasörler
    directories = [
        # Veri klasörleri
        "data/raw",
        "data/processed",
        "data/processed/lines",
        "data/meta",
        
        # Çıktı klasörleri
        "outputs/figures",
        "outputs/reports",
        "outputs/models",
        
        # Kaynak kod klasörü
        "src",
        
        # Notebook klasörü (opsiyonel)
        "notebooks",
    ]
    
    print("🚀 Proje klasör yapısı oluşturuluyor...\n")
    
    created_count = 0
    existing_count = 0
    
    for directory in directories:
        dir_path = base_dir / directory
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Oluşturuldu: {directory}")
            created_count += 1
        else:
            print(f"ℹ️  Zaten mevcut: {directory}")
            existing_count += 1
    
    print(f"\n📊 Özet:")
    print(f"   - Yeni oluşturulan: {created_count}")
    print(f"   - Zaten mevcut: {existing_count}")
    print(f"   - Toplam: {len(directories)}")
    print(f"\n✨ Klasör yapısı hazır!")


if __name__ == "__main__":
    create_project_structure()
