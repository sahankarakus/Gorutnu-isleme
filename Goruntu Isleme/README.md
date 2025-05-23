# Görüntü İşleme Uygulaması

Bu uygulama, Python ile geliştirilmiş kapsamlı bir görüntü işleme aracıdır. PyQt6 tabanlı modern bir arayüz sunar ve çeşitli görüntü işleme fonksiyonlarını destekler.

## Özellikler

- Görüntü yükleme ve kaydetme
- Gri tonlamaya dönüştürme
- Negatif görüntü oluşturma
- Parlaklık ve kontrast ayarları
- Yatay ve dikey çevirme
- 90 derece döndürme

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Uygulamayı çalıştırın:
```bash
python image_processor.py
```

## Kullanım

1. "Görüntü Yükle" butonuna tıklayarak bir görüntü seçin
2. Sol panelde orijinal görüntüyü, sağ panelde işlenmiş görüntüyü göreceksiniz
3. Sağ taraftaki kontrol panelinden istediğiniz işlemi seçin
4. İşlenmiş görüntüyü kaydetmek için "Görüntüyü Kaydet" butonunu kullanın

## Gereksinimler

- Python 3.8 veya üstü
- NumPy
- OpenCV
- PyQt6
- SciPy
- Pillow

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. 