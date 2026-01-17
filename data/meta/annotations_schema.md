# Annotations CSV Şema Dokümantasyonu

## 📋 Genel Bakış

Bu dosya, `annotations.csv` dosyasının şemasını ve veri yapısını tanımlar. Bu şema, **subject-wise cross-validation** ile veri sızıntısını önlemek için kritik öneme sahiptir.

## 🔑 Zorunlu Kolonlar

| Kolon Adı | Veri Tipi | Açıklama | Örnek |
|-----------|-----------|----------|-------|
| `subject_id` | string | Birey kimliği (veri sızıntısı önleme için) | "S001", "P_042" |
| `label` | int | Sınıf etiketi (0 veya 1) | 0, 1 |
| `raw_image_path` | string | Ham form görüntüsünün yolu | "data/raw/form_001.png" |
| `ul_x` | int | İlk satırın sol üst köşe X koordinatı | 120 |
| `ul_y` | int | İlk satırın sol üst köşe Y koordinatı | 80 |
| `lr_x` | int | İlk satırın sağ alt köşe X koordinatı | 850 |
| `lr_y` | int | İlk satırın sağ alt köşe Y koordinatı | 120 |

## 🧩 Opsiyonel Kolonlar

| Kolon Adı | Veri Tipi | Açıklama | Örnek |
|-----------|-----------|----------|-------|
| `form_id` | string | Form kimliği | "F001" |
| `scan_date` | string | Tarama tarihi | "2025-01-15" |
| `notes` | string | Notlar | "Kaliteli tarama" |

## 🚨 Kritik Kurallar

### 1. Subject ID Zorunluluğu

`subject_id` kolonunun **mutlaka bulunması gerekir** çünkü:
- GroupKFold cross-validation için kullanılır
- Aynı bireyin form/satırları asla hem train hem test'e düşmemelidir
- Veri sızıntısını önlemenin temelidir

**Kural:** Her `subject_id` benzersiz bir bireyi temsil eder. Aynı bireyin tüm formları/satırları aynı `subject_id`'ye sahip olmalıdır.

### 2. Label Değerleri

- **Binary sınıflandırma**: Sadece `0` veya `1` değerleri kabul edilir
- **Eksik değer**: `label` kolonunda `NaN` veya boş değer olmamalıdır

### 3. Bounding Box Koordinatları

- `ul_x < lr_x` (sol < sağ)
- `ul_y < lr_y` (üst < alt)
- Tüm koordinatlar pozitif tam sayı olmalıdır
- Koordinatlar, görüntü boyutları içinde olmalıdır

### 4. Dosya Yolu Kontrolü

- `raw_image_path` kolonundaki her dosya fiziksel olarak var olmalıdır
- Göreceli (relative) veya mutlak (absolute) yol kullanılabilir
- Desteklenen formatlar: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`

## 📊 Örnek CSV Yapısı

```csv
subject_id,label,raw_image_path,ul_x,ul_y,lr_x,lr_y,form_id
S001,0,data/raw/X_Y_1.png,120,80,850,120,F001
S001,1,data/raw/X_Y_2.png,115,85,845,125,F002
S002,0,data/raw/X_D_1.png,130,75,860,118,F003
S002,0,data/raw/X_D_2.png,125,82,855,122,F004
S003,1,data/raw/Z_Y_1.png,118,78,848,115,F005
```

## 🔍 Subject ID Çıkarım Kuralları

Eğer `subject_id` kolonunu manuel oluşturmuyorsanız, dosya adından otomatik çıkarım yapılabilir:

### Örnek 1: Standart Format (X_Y_1, X_D_1)
```
Dosya adı: X_Y_1.png
Subject ID: X (ilk karakter veya segment)
```

### Örnek 2: Alt çizgi ayrımı
```
Dosya adı: Subject001_Form01_20250115.png
Subject ID: Subject001 (ilk segment)
```

### Örnek 3: Regex ile özelleştirilmiş çıkarım
```python
import re
filename = "P042_F05_line01.png"
subject_id = re.match(r"^([A-Z]\d+)", filename).group(1)  # P042
```

**Önerilen yaklaşım:** Subject ID'leri manuel olarak `annotations.csv`'ye ekleyin. Bu, hataları minimize eder.

## ✅ Doğrulama Kontrolleri

`src/02_metadata.py` script'i aşağıdaki kontrolleri yapar:

1. **Kolon varlığı**: Tüm zorunlu kolonlar var mı?
2. **Veri tipi**: Her kolon doğru veri tipinde mi?
3. **Eksik değer**: Zorunlu kolonlarda eksik değer var mı?
4. **Label değerleri**: Sadece 0 ve 1 var mı?
5. **Koordinat mantığı**: ul < lr kontrolü
6. **Dosya varlığı**: Tüm `raw_image_path` dosyaları mevcut mu?
7. **Subject ID sayısı**: Kaç benzersiz birey var?
8. **Sınıf dengesi**: Her subject için label dağılımı

## 📈 Veri Seti İstatistikleri

Script çalıştırıldığında şu bilgiler raporlanır:

- Toplam örnek sayısı
- Benzersiz birey sayısı (`subject_id`)
- Birey başına ortalama örnek sayısı
- Label dağılımı (0/1 oranları)
- Birey başına label dağılımı (heterojen mi homojen mi?)

## 🔬 Veri Sızıntısı Senaryoları

### ❌ Yanlış (Veri Sızıntısı Var)

```python
# Aynı bireyin formları train ve test'e karışmış
Train: [S001_form1, S002_form1, S003_form1]
Test:  [S001_form2, S004_form1, S005_form1]
# S001 hem train hem test'te! ❌
```

### ✅ Doğru (Subject-wise CV)

```python
# Aynı birey asla hem train hem test'te olmuyor
Train: [S001_form1, S001_form2, S002_form1, S002_form2]
Test:  [S003_form1, S003_form2, S004_form1, S004_form2]
# Her subject sadece bir grupta ✅
```

## 🛠️ Troubleshooting

### Problem: "subject_id kolonunu bulamıyorum"
**Çözüm:** CSV'de kolon adını tam olarak `subject_id` yazın (küçük harf, alt çizgi).

### Problem: "Label değerleri 0 ve 1 dışında"
**Çözüm:** Label sütununu kontrol edin, sadece 0 veya 1 olmalı.

### Problem: "Dosya yolu bulunamadı"
**Çözüm:** `raw_image_path` kolonundaki yolların doğru ve dosyaların mevcut olduğunu kontrol edin.

### Problem: "Koordinatlar tutarsız (ul >= lr)"
**Çözüm:** Bounding box koordinatlarını kontrol edin, sol üst < sağ alt olmalı.

## 📝 Notlar

- Bu şema, GroupKFold CV için optimize edilmiştir
- Her değişiklikte `src/02_metadata.py` ile doğrulama yapın
- Subject ID'ler birey bazlı olmalı, form bazlı değil

---

**Son güncelleme:** 17 Ocak 2026  
**İlgili script:** `src/02_metadata.py`
