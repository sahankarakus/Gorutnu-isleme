import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QSlider, QMessageBox, QTabWidget, QGroupBox,
                           QScrollArea, QSpinBox, QDoubleSpinBox, QComboBox,
                           QCheckBox, QDialog, QDialogButtonBox)
from PyQt6.QtCore import Qt, QPoint, QRect
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor, QPainter, QPen, QBrush
from scipy.fft import fft2, ifft2, fftshift
from PIL import Image, ImageEnhance
import os
import matplotlib
matplotlib.use('Qt5Agg')  # Qt5Agg backend'i kullan
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class CropWidget(QLabel):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())
        w, h = pixmap.width(), pixmap.height()
        # Başlangıçta ortada bir dikdörtgen
        self.rect = QRect(w//4, h//4, w//2, h//2)
        self.dragging = False
        self.drag_handle = None
        self.handle_size = 12
        self.setMouseTracking(True)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        # Yarı saydam overlay
        overlay_color = QColor(0, 0, 0, 120)
        painter.setBrush(QBrush(overlay_color))
        painter.setPen(Qt.PenStyle.NoPen)
        # Dış alanları boya
        r = self.rect
        w, h = self.width(), self.height()
        painter.drawRect(0, 0, w, r.top())
        painter.drawRect(0, r.bottom(), w, h - r.bottom())
        painter.drawRect(0, r.top(), r.left(), r.height())
        painter.drawRect(r.right(), r.top(), w - r.right(), r.height())
        # Seçili alan çerçevesi
        painter.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(r)
        # 4 köşe tutamacı
        for pt in self.handle_points():
            painter.setBrush(QBrush(QColor(255,255,255)))
            painter.setPen(QPen(Qt.GlobalColor.red, 2))
            painter.drawEllipse(pt, self.handle_size//2, self.handle_size//2)

    def handle_points(self):
        r = self.rect
        return [r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()]

    def mousePressEvent(self, event):
        pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
        for i, pt in enumerate(self.handle_points()):
            if (pt - pos).manhattanLength() < self.handle_size:
                self.dragging = True
                self.drag_handle = i
                return
        if self.rect.contains(pos):
            self.dragging = True
            self.drag_handle = 'move'
            self.drag_offset = pos - self.rect.topLeft()

    def mouseMoveEvent(self, event):
        if not self.dragging:
            return
        pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
        r = self.rect
        if self.drag_handle == 0:  # topLeft
            r.setTopLeft(pos)
        elif self.drag_handle == 1:  # topRight
            r.setTopRight(pos)
        elif self.drag_handle == 2:  # bottomLeft
            r.setBottomLeft(pos)
        elif self.drag_handle == 3:  # bottomRight
            r.setBottomRight(pos)
        elif self.drag_handle == 'move':
            size = r.size()
            new_topleft = pos - self.drag_offset
            r = QRect(new_topleft, size)
        # Sınırları kontrol et
        r = r.normalized()
        r.setLeft(max(0, r.left()))
        r.setTop(max(0, r.top()))
        r.setRight(min(self.width()-1, r.right()))
        r.setBottom(min(self.height()-1, r.bottom()))
        self.rect = r
        self.update()

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.drag_handle = None

    def get_crop_rect(self):
        return self.rect.normalized()

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Görüntü İşleme Uygulaması")
        self.setGeometry(100, 100, 1400, 800)

        # Koyu tema için arka plan ve yazı rengi ayarları
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        self.setPalette(dark_palette)

        # Stil ayarları
        self.setStyleSheet("""
            QGroupBox {
                border: 2px solid #3A3A3A;
                border-radius: 5px;
                margin-top: 1em;
                padding-top: 10px;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                color: white;
            }
            QTabWidget::pane {
                border: 1px solid #3A3A3A;
                background: #353535;
            }
            QTabBar::tab {
                background: #2A2A2A;
                color: white;
                padding: 8px;
                border: 1px solid #3A3A3A;
            }
            QTabBar::tab:selected {
                background: #404040;
                border-bottom: none;
            }
            QPushButton {
                background-color: #2A2A2A;
                border: 1px solid #3A3A3A;
                border-radius: 3px;
                padding: 5px;
                color: white;
            }
            QPushButton:hover {
                background-color: #404040;
            }
            QPushButton:pressed {
                background-color: #505050;
            }
            QSlider::groove:horizontal {
                border: 1px solid #3A3A3A;
                height: 8px;
                background: #2A2A2A;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #4A4A4A;
                border: 1px solid #5A5A5A;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
            QComboBox {
                background-color: #2A2A2A;
                border: 1px solid #3A3A3A;
                border-radius: 3px;
                padding: 5px;
                color: white;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #2A2A2A;
                border: 1px solid #3A3A3A;
                border-radius: 3px;
                padding: 5px;
                color: white;
            }
        """)

        # Ana widget ve layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Sol panel (görüntüler ve dosya işlemleri)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Görüntü paneli
        images_widget = QWidget()
        images_layout = QHBoxLayout(images_widget)
        
        # Orijinal görüntü
        original_group = QGroupBox("Orijinal Görüntü")
        original_layout = QVBoxLayout(original_group)
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(400, 400)
        original_layout.addWidget(self.original_label)
        images_layout.addWidget(original_group)
        
        # İşlenmiş görüntü
        processed_group = QGroupBox("İşlenmiş Görüntü")
        processed_layout = QVBoxLayout(processed_group)
        self.processed_label = QLabel()
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_label.setMinimumSize(400, 400)
        processed_layout.addWidget(self.processed_label)
        images_layout.addWidget(processed_group)
        
        left_layout.addWidget(images_widget)
        
        # Dosya işlem butonları
        file_buttons = QWidget()
        file_layout = QHBoxLayout(file_buttons)
        
        btn_load = QPushButton("Görüntü Yükle")
        btn_load.clicked.connect(self.load_image)
        file_layout.addWidget(btn_load)
        
        btn_save = QPushButton("Görüntüyü Kaydet")
        btn_save.clicked.connect(self.save_image)
        file_layout.addWidget(btn_save)

        btn_reset = QPushButton("Orijinale Dön")
        btn_reset.clicked.connect(self.reset_image)
        file_layout.addWidget(btn_reset)
        
        left_layout.addWidget(file_buttons)
        main_layout.addWidget(left_panel)

        # Sağ panel (sekmeli kontrol paneli)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Sekmeli widget
        self.tab_widget = QTabWidget()
        
        # Sekmeleri oluştur
        self.create_basic_tab()      # Temel İşlemler
        self.create_color_tab()      # Renk İşlemleri
        self.create_geometric_tab()  # Geometrik Dönüşümler
        self.create_filter_tab()     # Filtreleme İşlemleri
        self.create_frequency_tab()  # Frekans Alanı İşlemleri
        self.create_crop_tab()       # Kırpma
        self.create_edge_tab()       # Kenar Bulma
        self.create_morph_tab()      # Morfolojik İşlemler
        self.create_segment_tab()    # Segmentasyon
        self.create_advanced_filter_tab() # Gelişmiş Filtreler
        
        # Scroll Area içine tab widget'ı ekle
        scroll = QScrollArea()
        scroll.setWidget(self.tab_widget)
        scroll.setWidgetResizable(True)
        right_layout.addWidget(scroll)
        
        # Sağ paneli ana layout'a ekle
        main_layout.addWidget(right_panel, 1)

        # Görüntü değişkenleri
        self.original_image = None
        self.processed_image = None
        self.perspective_points = []
        self.is_selecting_points = False
        self.is_cropping = False
        self.crop_points = []
        self.crop_rect = None

    def create_basic_tab(self):
        basic_tab = QWidget()
        layout = QVBoxLayout(basic_tab)

        # Parlaklık ve Kontrast grubu
        adjust_group = QGroupBox("Parlaklık ve Kontrast")
        adjust_layout = QVBoxLayout(adjust_group)
        
        adjust_layout.addWidget(QLabel("Parlaklık"))
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.adjust_brightness)
        adjust_layout.addWidget(self.brightness_slider)
        
        adjust_layout.addWidget(QLabel("Kontrast"))
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setMinimum(-100)
        self.contrast_slider.setMaximum(100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(self.adjust_contrast)
        adjust_layout.addWidget(self.contrast_slider)
        
        layout.addWidget(adjust_group)

        # Eşikleme grubu
        threshold_group = QGroupBox("Eşikleme")
        threshold_layout = QVBoxLayout(threshold_group)
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(255)
        self.threshold_slider.setValue(127)
        self.threshold_slider.valueChanged.connect(self.apply_threshold)
        threshold_layout.addWidget(QLabel("Eşik Değeri"))
        threshold_layout.addWidget(self.threshold_slider)
        
        layout.addWidget(threshold_group)

        # Histogram grubu
        histogram_group = QGroupBox("Histogram İşlemleri")
        histogram_layout = QVBoxLayout(histogram_group)
        
        btn_show_hist = QPushButton("Histogram Göster")
        btn_show_hist.clicked.connect(self.show_histogram)
        histogram_layout.addWidget(btn_show_hist)
        
        btn_equalize = QPushButton("Histogram Eşitleme")
        btn_equalize.clicked.connect(self.equalize_histogram)
        histogram_layout.addWidget(btn_equalize)
        
        layout.addWidget(histogram_group)
        
        layout.addStretch()
        self.tab_widget.addTab(basic_tab, "Temel İşlemler")

    def create_color_tab(self):
        color_tab = QWidget()
        layout = QVBoxLayout(color_tab)

        # Renk İşlemleri grubu
        color_group = QGroupBox("Renk İşlemleri")
        color_layout = QVBoxLayout(color_group)
        
        btn_grayscale = QPushButton("Gri Tonlama")
        btn_grayscale.clicked.connect(self.convert_to_grayscale)
        color_layout.addWidget(btn_grayscale)
        
        btn_negative = QPushButton("Negatif")
        btn_negative.clicked.connect(self.convert_to_negative)
        color_layout.addWidget(btn_negative)
        
        btn_channels = QPushButton("RGB Kanallara Ayır")
        btn_channels.clicked.connect(self.split_channels)
        color_layout.addWidget(btn_channels)
        
        layout.addWidget(color_group)
        layout.addStretch()
        self.tab_widget.addTab(color_tab, "Renk İşlemleri")

    def create_geometric_tab(self):
        geometric_tab = QWidget()
        layout = QVBoxLayout(geometric_tab)

        # Temel Dönüşümler grubu
        transform_group = QGroupBox("Temel Dönüşümler")
        transform_layout = QVBoxLayout(transform_group)
        
        btn_flip_h = QPushButton("Yatay Çevir")
        btn_flip_h.clicked.connect(lambda: self.flip_image(1))
        transform_layout.addWidget(btn_flip_h)
        
        btn_flip_v = QPushButton("Dikey Çevir")
        btn_flip_v.clicked.connect(lambda: self.flip_image(0))
        transform_layout.addWidget(btn_flip_v)
        
        btn_rotate = QPushButton("90° Döndür")
        btn_rotate.clicked.connect(self.rotate_image)
        transform_layout.addWidget(btn_rotate)
        
        layout.addWidget(transform_group)

        # İleri Dönüşümler grubu
        advanced_group = QGroupBox("İleri Dönüşümler")
        advanced_layout = QVBoxLayout(advanced_group)
        
        # Taşıma
        translation_layout = QHBoxLayout()
        self.tx_spin = QSpinBox()
        self.tx_spin.setRange(-1000, 1000)
        self.ty_spin = QSpinBox()
        self.ty_spin.setRange(-1000, 1000)
        translation_layout.addWidget(QLabel("X:"))
        translation_layout.addWidget(self.tx_spin)
        translation_layout.addWidget(QLabel("Y:"))
        translation_layout.addWidget(self.ty_spin)
        btn_translate = QPushButton("Taşı")
        btn_translate.clicked.connect(self.translate_image)
        advanced_layout.addWidget(QLabel("Taşıma:"))
        advanced_layout.addLayout(translation_layout)
        advanced_layout.addWidget(btn_translate)
        
        # Ölçekleme
        scale_layout = QHBoxLayout()
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 5.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(0.1)
        scale_layout.addWidget(QLabel("Ölçek:"))
        scale_layout.addWidget(self.scale_spin)
        btn_scale = QPushButton("Ölçekle")
        btn_scale.clicked.connect(self.scale_image)
        advanced_layout.addLayout(scale_layout)
        advanced_layout.addWidget(btn_scale)
        
        # Eğme
        shear_layout = QHBoxLayout()
        self.shear_spin = QDoubleSpinBox()
        self.shear_spin.setRange(-2.0, 2.0)
        self.shear_spin.setSingleStep(0.1)
        shear_layout.addWidget(QLabel("Eğme Faktörü:"))
        shear_layout.addWidget(self.shear_spin)
        btn_shear = QPushButton("Eğ")
        btn_shear.clicked.connect(self.shear_image)
        advanced_layout.addLayout(shear_layout)
        advanced_layout.addWidget(btn_shear)
        
        # Perspektif Düzeltme
        btn_perspective = QPushButton("Perspektif Düzeltme")
        btn_perspective.clicked.connect(self.start_perspective_correction)
        advanced_layout.addWidget(btn_perspective)
        
        layout.addWidget(advanced_group)
        layout.addStretch()
        self.tab_widget.addTab(geometric_tab, "Geometrik İşlemler")

    def create_filter_tab(self):
        filter_tab = QWidget()
        layout = QVBoxLayout(filter_tab)

        # Temel Filtreler grubu
        basic_filter_group = QGroupBox("Temel Filtreler")
        basic_filter_layout = QVBoxLayout(basic_filter_group)
        
        # Filtre boyutu seçimi
        self.kernel_size = QComboBox()
        self.kernel_size.addItems(['3x3', '5x5', '7x7'])
        basic_filter_layout.addWidget(QLabel("Filtre Boyutu:"))
        basic_filter_layout.addWidget(self.kernel_size)
        
        btn_average = QPushButton("Ortalama Filtre")
        btn_average.clicked.connect(self.apply_average_filter)
        basic_filter_layout.addWidget(btn_average)
        
        btn_median = QPushButton("Medyan Filtre")
        btn_median.clicked.connect(self.apply_median_filter)
        basic_filter_layout.addWidget(btn_median)
        
        btn_gaussian = QPushButton("Gauss Filtre")
        btn_gaussian.clicked.connect(self.apply_gaussian_filter)
        basic_filter_layout.addWidget(btn_gaussian)
        
        layout.addWidget(basic_filter_group)

        # İleri Filtreler grubu
        advanced_filter_group = QGroupBox("İleri Filtreler")
        advanced_filter_layout = QVBoxLayout(advanced_filter_group)
        
        btn_conservative = QPushButton("Konservatif Filtre")
        btn_conservative.clicked.connect(self.apply_conservative_filter)
        advanced_filter_layout.addWidget(btn_conservative)
        
        btn_crimmins = QPushButton("Crimmins Speckle")
        btn_crimmins.clicked.connect(self.apply_crimmins)
        advanced_filter_layout.addWidget(btn_crimmins)
        
        layout.addWidget(advanced_filter_group)
        layout.addStretch()
        self.tab_widget.addTab(filter_tab, "Filtreleme")

    def create_frequency_tab(self):
        frequency_tab = QWidget()
        layout = QVBoxLayout(frequency_tab)

        # Fourier Dönüşümleri grubu
        fourier_group = QGroupBox("Fourier Dönüşümleri")
        fourier_layout = QVBoxLayout(fourier_group)
        
        btn_lpf = QPushButton("Alçak Geçiren Filtre")
        btn_lpf.clicked.connect(lambda: self.apply_frequency_filter("lpf"))
        fourier_layout.addWidget(btn_lpf)
        
        btn_hpf = QPushButton("Yüksek Geçiren Filtre")
        btn_hpf.clicked.connect(lambda: self.apply_frequency_filter("hpf"))
        fourier_layout.addWidget(btn_hpf)
        
        layout.addWidget(fourier_group)

        # Band Filtreleri grubu
        band_group = QGroupBox("Band Filtreleri")
        band_layout = QVBoxLayout(band_group)
        
        btn_band_pass = QPushButton("Band Geçiren Filtre")
        btn_band_pass.clicked.connect(lambda: self.apply_frequency_filter("band_pass"))
        band_layout.addWidget(btn_band_pass)
        
        btn_band_stop = QPushButton("Band Durduran Filtre")
        btn_band_stop.clicked.connect(lambda: self.apply_frequency_filter("band_stop"))
        band_layout.addWidget(btn_band_stop)
        
        layout.addWidget(band_group)

        # Özel Filtreler grubu
        special_group = QGroupBox("Özel Filtreler")
        special_layout = QVBoxLayout(special_group)
        
        btn_butterworth = QPushButton("Butterworth Filtre")
        btn_butterworth.clicked.connect(self.apply_butterworth)
        special_layout.addWidget(btn_butterworth)
        
        btn_gaussian = QPushButton("Gaussian Filtre")
        btn_gaussian.clicked.connect(lambda: self.apply_frequency_filter("gaussian"))
        special_layout.addWidget(btn_gaussian)
        
        btn_homomorphic = QPushButton("Homomorfik Filtre")
        btn_homomorphic.clicked.connect(self.apply_homomorphic)
        special_layout.addWidget(btn_homomorphic)
        
        layout.addWidget(special_group)
        layout.addStretch()
        self.tab_widget.addTab(frequency_tab, "Frekans İşlemleri")

    def create_crop_tab(self):
        crop_tab = QWidget()
        layout = QVBoxLayout(crop_tab)
        btn_crop = QPushButton("Kırp")
        btn_crop.clicked.connect(self.start_crop)
        layout.addWidget(btn_crop)
        layout.addStretch()
        self.tab_widget.addTab(crop_tab, "Kırpma")

    def create_edge_tab(self):
        edge_tab = QWidget()
        layout = QVBoxLayout(edge_tab)
        btn_sobel = QPushButton("Sobel")
        btn_sobel.clicked.connect(self.apply_sobel)
        layout.addWidget(btn_sobel)
        btn_prewitt = QPushButton("Prewitt")
        btn_prewitt.clicked.connect(self.apply_prewitt)
        layout.addWidget(btn_prewitt)
        btn_roberts = QPushButton("Roberts Cross")
        btn_roberts.clicked.connect(self.apply_roberts)
        layout.addWidget(btn_roberts)
        btn_compass = QPushButton("Compass")
        btn_compass.clicked.connect(self.apply_compass)
        layout.addWidget(btn_compass)
        btn_canny = QPushButton("Canny")
        btn_canny.clicked.connect(self.apply_canny)
        layout.addWidget(btn_canny)
        btn_laplace = QPushButton("Laplace")
        btn_laplace.clicked.connect(self.apply_laplace)
        layout.addWidget(btn_laplace)
        btn_gabor = QPushButton("Gabor")
        btn_gabor.clicked.connect(self.apply_gabor)
        layout.addWidget(btn_gabor)
        btn_hough = QPushButton("Hough Dönüşümü")
        btn_hough.clicked.connect(self.apply_hough)
        layout.addWidget(btn_hough)
        layout.addStretch()
        self.tab_widget.addTab(edge_tab, "Kenar Bulma")

    def create_morph_tab(self):
        morph_tab = QWidget()
        layout = QVBoxLayout(morph_tab)
        btn_erode = QPushButton("Erode")
        btn_erode.clicked.connect(self.apply_erode)
        layout.addWidget(btn_erode)
        btn_dilate = QPushButton("Dilate")
        btn_dilate.clicked.connect(self.apply_dilate)
        layout.addWidget(btn_dilate)
        layout.addStretch()
        self.tab_widget.addTab(morph_tab, "Morfolojik İşlemler")

    def create_segment_tab(self):
        segment_tab = QWidget()
        layout = QVBoxLayout(segment_tab)
        btn_kmeans = QPushButton("K-means Segmentasyon")
        btn_kmeans.clicked.connect(self.apply_kmeans)
        layout.addWidget(btn_kmeans)
        layout.addStretch()
        self.tab_widget.addTab(segment_tab, "Segmentasyon")

    def create_advanced_filter_tab(self):
        adv_tab = QWidget()
        layout = QVBoxLayout(adv_tab)
        btn_gaussian_lpf = QPushButton("Gaussian LPF")
        btn_gaussian_lpf.clicked.connect(self.apply_gaussian_lpf)
        layout.addWidget(btn_gaussian_lpf)
        btn_gaussian_hpf = QPushButton("Gaussian HPF")
        btn_gaussian_hpf.clicked.connect(self.apply_gaussian_hpf)
        layout.addWidget(btn_gaussian_hpf)
        layout.addStretch()
        self.tab_widget.addTab(adv_tab, "Gelişmiş Filtreler")

    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.update_display()

    def load_image(self):
        try:
            # Gri tonda okuma seçeneği için dialog
            class LoadDialog(QDialog):
                def __init__(self):
                    super().__init__()
                    self.setWindowTitle("Görüntü Seçenekleri")
                    self.checkbox = QCheckBox("Gri tonda oku (grayscale)")
                    layout = QVBoxLayout()
                    layout.addWidget(QLabel("Görüntü gri tonda mı yüklensin?"))
                    layout.addWidget(self.checkbox)
                    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
                    buttons.accepted.connect(self.accept)
                    buttons.rejected.connect(self.reject)
                    layout.addWidget(buttons)
                    self.setLayout(layout)
            # Önce dosya seç
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Görüntü Seç",
                "",
                "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;All Files (*.*)"
            )
            if file_name and os.path.exists(file_name):
                # Gri tonda okuma seçeneği sor
                dlg = LoadDialog()
                grayscale = False
                if dlg.exec() == QDialog.DialogCode.Accepted:
                    grayscale = dlg.checkbox.isChecked()
                else:
                    return
                pil_image = Image.open(file_name)
                if grayscale:
                    pil_image = pil_image.convert('L')
                elif pil_image.mode in ['RGBA', 'LA']:
                    background = Image.new('RGB', pil_image.size, (255, 255, 255))
                    background.paste(pil_image, mask=pil_image.split()[-1])
                    pil_image = background
                elif pil_image.mode not in ['RGB', 'L']:
                    pil_image = pil_image.convert('RGB')
                self.original_image = np.array(pil_image)
                if len(self.original_image.shape) == 2:
                    self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)
                self.processed_image = self.original_image.copy()
                self.update_display()
                QMessageBox.information(self, "Başarılı", "Görüntü başarıyla yüklendi!")
        except Exception as e:
            QMessageBox.critical(self, "Hata", 
                f"Görüntü yüklenirken bir hata oluştu:\n{str(e)}\n\n"
                "Lütfen geçerli bir görüntü dosyası seçtiğinizden emin olun.\n"
                "Desteklenen formatlar: PNG, JPG, JPEG, BMP, GIF, TIFF")

    def save_image(self):
        try:
            if self.processed_image is not None:
                file_name, _ = QFileDialog.getSaveFileName(self, "Görüntüyü Kaydet",
                                                         "", "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)")
                if file_name:
                    # RGB'den BGR'ye dönüştür
                    save_image = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(file_name, save_image)
                    QMessageBox.information(self, "Başarılı", "Görüntü başarıyla kaydedildi!")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Görüntü kaydedilirken bir hata oluştu: {str(e)}")

    def convert_to_grayscale(self):
        try:
            if self.processed_image is not None:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                self.processed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Gri tonlamaya dönüştürürken bir hata oluştu: {str(e)}")

    def convert_to_negative(self):
        try:
            if self.processed_image is not None:
                self.processed_image = 255 - self.processed_image
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Negatif görüntü oluştururken bir hata oluştu: {str(e)}")

    def adjust_brightness(self):
        try:
            if self.original_image is not None:
                value = self.brightness_slider.value()
                self.processed_image = cv2.convertScaleAbs(self.original_image, alpha=1, beta=value)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Parlaklık ayarlanırken bir hata oluştu: {str(e)}")

    def adjust_contrast(self):
        try:
            if self.original_image is not None:
                value = self.contrast_slider.value()
                alpha = 1.0 + (value / 100.0)
                self.processed_image = cv2.convertScaleAbs(self.original_image, alpha=alpha, beta=0)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Kontrast ayarlanırken bir hata oluştu: {str(e)}")

    def apply_threshold(self):
        try:
            if self.original_image is not None:
                threshold_value = self.threshold_slider.value()
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
                self.processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Eşikleme işlemi sırasında bir hata oluştu: {str(e)}")

    def flip_image(self, direction):
        try:
            if self.processed_image is not None:
                self.processed_image = cv2.flip(self.processed_image, direction)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Görüntü çevrilirken bir hata oluştu: {str(e)}")

    def rotate_image(self):
        try:
            if self.processed_image is not None:
                self.processed_image = cv2.rotate(self.processed_image, 
                                                cv2.ROTATE_90_CLOCKWISE)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Görüntü döndürülürken bir hata oluştu: {str(e)}")

    def update_display(self):
        try:
            if self.original_image is not None:
                # Orijinal görüntüyü göster
                height, width = self.original_image.shape[:2]
                bytes_per_line = 3 * width
                # Görüntü boyutlarını kontrol et
                if height > 0 and width > 0 and self.original_image.data and bytes_per_line > 0:
                    q_img = QImage(self.original_image.tobytes(), width, height, 
                                bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    scaled_pixmap = pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio)
                    self.original_label.setPixmap(scaled_pixmap)

            if self.processed_image is not None:
                # İşlenmiş görüntüyü göster
                height, width = self.processed_image.shape[:2]
                bytes_per_line = 3 * width
                # Görüntü boyutlarını kontrol et
                if height > 0 and width > 0 and self.processed_image.data and bytes_per_line > 0:
                    q_img = QImage(self.processed_image.tobytes(), width, height, 
                                bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    scaled_pixmap = pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio)
                    self.processed_label.setPixmap(scaled_pixmap)
                    
        except Exception as e:
            QMessageBox.critical(self, "Hata", 
                f"Görüntü gösterilirken bir hata oluştu:\n{str(e)}\n\n"
                "Lütfen geçerli bir görüntü dosyası seçtiğinizden emin olun.")

    def show_histogram(self):
        try:
            if self.processed_image is not None:
                # Yeni bir pencere oluştur
                self.hist_window = QMainWindow()  # self ile sakla
                self.hist_window.setWindowTitle("Histogram")
                self.hist_window.setGeometry(200, 200, 800, 600)
                
                # Merkezi widget ve layout oluştur
                central_widget = QWidget()
                self.hist_window.setCentralWidget(central_widget)
                layout = QVBoxLayout(central_widget)
                
                # matplotlib figure oluştur
                fig = Figure(figsize=(8, 6), facecolor='#353535')
                canvas = FigureCanvas(fig)
                layout.addWidget(canvas)
                
                # Alt grafik için axes oluştur
                ax = fig.add_subplot(111)
                ax.set_facecolor('#353535')
                
                # Izgara çizgilerini ayarla
                ax.grid(True, color='#666666', linestyle='--', alpha=0.3)
                
                # RGB kanalları için histogram hesapla ve çiz
                colors = ('r', 'g', 'b')
                labels = ('Kırmızı', 'Yeşil', 'Mavi')
                
                for i, (color, label) in enumerate(zip(colors, labels)):
                    hist = cv2.calcHist([self.processed_image], [i], None, [256], [0, 256])
                    ax.plot(hist, color=color, label=label, linewidth=2)
                
                # Grafik özelliklerini ayarla
                ax.set_title('RGB Histogram', color='white', pad=20, fontsize=12)
                ax.set_xlabel('Piksel Değeri', color='white', fontsize=10)
                ax.set_ylabel('Piksel Sayısı', color='white', fontsize=10)
                
                # Eksen renklerini ayarla
                ax.tick_params(axis='both', colors='white')
                for spine in ax.spines.values():
                    spine.set_color('white')
                
                # Gösterge kutusunu ayarla
                legend = ax.legend(facecolor='#353535', edgecolor='white')
                plt.setp(legend.get_texts(), color='white')
                
                # Grafik kenarlarında boşluk bırak
                fig.tight_layout()
                
                # Pencereyi göster
                self.hist_window.show()
                
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Histogram gösterimi sırasında bir hata oluştu: {str(e)}")
            print(f"Detaylı hata: {str(e)}")  # Konsola detaylı hata mesajı yazdır

    def equalize_histogram(self):
        try:
            if self.processed_image is not None:
                # RGB kanallarını ayır
                b, g, r = cv2.split(self.processed_image)
                
                # Her kanal için histogram eşitleme uygula
                b_eq = cv2.equalizeHist(b)
                g_eq = cv2.equalizeHist(g)
                r_eq = cv2.equalizeHist(r)
                
                # Kanalları birleştir
                self.processed_image = cv2.merge([b_eq, g_eq, r_eq])
                
                # Görüntüyü güncelle
                self.update_display()
                
                # Başarı mesajı göster
                QMessageBox.information(self, "Başarılı", "Histogram eşitleme işlemi tamamlandı!")
                
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Histogram eşitleme sırasında bir hata oluştu: {str(e)}")

    def split_channels(self):
        try:
            if self.processed_image is not None:
                # Kanalları ayır (RGB sırası!)
                r, g, b = cv2.split(self.processed_image)
                channels = {'Kırmızı Kanal': r, 'Yeşil Kanal': g, 'Mavi Kanal': b}
                if not hasattr(self, 'channel_windows'):
                    self.channel_windows = []
                for title, channel in channels.items():
                    window = QMainWindow()
                    window.setWindowTitle(title)
                    window.setGeometry(200, 200, 400, 400)
                    label = QLabel()
                    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    rgb_channel = cv2.cvtColor(channel, cv2.COLOR_GRAY2RGB)
                    height, width = rgb_channel.shape[:2]
                    bytes_per_line = 3 * width
                    q_img = QImage(rgb_channel.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
                    label.setPixmap(QPixmap.fromImage(q_img).scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio))
                    window.setCentralWidget(label)
                    window.show()
                    self.channel_windows.append(window)
                    # Pencere kapatılınca referansı listeden sil
                    def remove_ref(w=window):
                        if w in self.channel_windows:
                            self.channel_windows.remove(w)
                    window.destroyed.connect(remove_ref)
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Kanalları ayırma işlemi sırasında bir hata oluştu: {str(e)}")

    def translate_image(self):
        try:
            if self.processed_image is not None:
                # Taşıma matrisi oluştur
                tx, ty = self.tx_spin.value(), self.ty_spin.value()
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                
                # Görüntüyü taşı
                height, width = self.processed_image.shape[:2]
                self.processed_image = cv2.warpAffine(self.processed_image, M, (width, height))
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Görüntü taşıma sırasında bir hata oluştu: {str(e)}")

    def scale_image(self):
        try:
            if self.processed_image is not None:
                scale = self.scale_spin.value()
                height, width = self.processed_image.shape[:2]
                new_width = max(1, int(width * scale))
                new_height = max(1, int(height * scale))
                new_size = (new_width, new_height)
                self.processed_image = cv2.resize(self.processed_image, new_size, 
                                               interpolation=cv2.INTER_LINEAR)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Görüntü ölçekleme sırasında bir hata oluştu: {str(e)}")

    def shear_image(self):
        try:
            if self.processed_image is not None:
                shear_factor = self.shear_spin.value()
                height, width = self.processed_image.shape[:2]
                
                # Eğme matrisi oluştur
                M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
                
                # Yeni görüntü boyutunu hesapla
                new_width = int(width + abs(shear_factor * height))
                
                # Görüntüyü eğ
                self.processed_image = cv2.warpAffine(self.processed_image, M, (new_width, height))
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Görüntü eğme sırasında bir hata oluştu: {str(e)}")

    def start_perspective_correction(self):
        try:
            if self.processed_image is not None:
                # Önce bilgi mesajı göster
                QMessageBox.information(self, "Bilgi", "Lütfen düzeltmek istediğiniz dörtgenin 4 köşesini saat yönünde tıklayın.")
                self.is_selecting_points = True
                self.perspective_points = []
                self.perspective_window = QMainWindow()
                self.perspective_window.setWindowTitle("Perspektif Düzeltme")
                self.perspective_window.setGeometry(200, 200, 800, 600)
                self.perspective_label = QLabel()
                self.perspective_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                height, width = self.processed_image.shape[:2]
                bytes_per_line = 3 * width
                q_img = QImage(self.processed_image.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
                self.perspective_label.setPixmap(QPixmap.fromImage(q_img))
                self.perspective_label.mousePressEvent = self.perspective_point_click
                self.perspective_window.setCentralWidget(self.perspective_label)
                self.perspective_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Perspektif düzeltme başlatılırken bir hata oluştu: {str(e)}")

    def perspective_point_click(self, event):
        try:
            if len(self.perspective_points) < 4:
                # Noktayı kaydet
                point = (event.pos().x(), event.pos().y())
                self.perspective_points.append(point)
                
                # Noktayı görüntüle
                pixmap = self.perspective_label.pixmap()
                painter = QPainter(pixmap)
                painter.setPen(QPen(Qt.GlobalColor.red, 5))
                painter.drawPoint(event.pos())
                painter.end()
                self.perspective_label.setPixmap(pixmap)
                
                # 4 nokta seçildiyse perspektif düzeltmeyi uygula
                if len(self.perspective_points) == 4:
                    self.apply_perspective_correction()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Nokta seçimi sırasında bir hata oluştu: {str(e)}")

    def apply_perspective_correction(self):
        try:
            src_points = np.float32(self.perspective_points)
            width = max(
                np.linalg.norm(src_points[0] - src_points[1]),
                np.linalg.norm(src_points[2] - src_points[3])
            )
            height = max(
                np.linalg.norm(src_points[1] - src_points[2]),
                np.linalg.norm(src_points[3] - src_points[0])
            )
            dst_points = np.float32([
                [0, 0],
                [width-1, 0],
                [width-1, height-1],
                [0, height-1]
            ])
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            self.processed_image = cv2.warpPerspective(self.processed_image, M, (int(width), int(height)))
            self.update_display()
            self.perspective_window.close()
            self.is_selecting_points = False
            self.perspective_points = []
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Perspektif düzeltme sırasında bir hata oluştu: {str(e)}")

    def apply_average_filter(self):
        try:
            if self.processed_image is not None:
                kernel_size = int(self.kernel_size.currentText().split('x')[0])
                kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
                self.processed_image = cv2.filter2D(self.processed_image, -1, kernel)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Ortalama filtre uygulanırken bir hata oluştu: {str(e)}")

    def apply_median_filter(self):
        try:
            if self.processed_image is not None:
                kernel_size = int(self.kernel_size.currentText().split('x')[0])
                self.processed_image = cv2.medianBlur(self.processed_image, kernel_size)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Medyan filtre uygulanırken bir hata oluştu: {str(e)}")

    def apply_gaussian_filter(self):
        try:
            if self.processed_image is not None:
                kernel_size = int(self.kernel_size.currentText().split('x')[0])
                self.processed_image = cv2.GaussianBlur(self.processed_image, 
                                                      (kernel_size, kernel_size), 0)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Gauss filtresi uygulanırken bir hata oluştu: {str(e)}")

    def apply_frequency_filter(self, filter_type):
        try:
            if self.processed_image is not None:
                # Görüntüyü gri tonlamaya çevir
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                
                # Fourier dönüşümü
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                
                # Görüntü boyutları
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                
                # Filtre maskesi oluştur
                mask = np.ones((rows, cols), np.uint8)
                
                if filter_type == "lpf":
                    # Alçak geçiren filtre
                    r = 30
                    mask[crow-r:crow+r, ccol-r:ccol+r] = 0
                    mask = 1 - mask
                elif filter_type == "hpf":
                    # Yüksek geçiren filtre
                    r = 30
                    mask[crow-r:crow+r, ccol-r:ccol+r] = 0
                elif filter_type == "band_pass":
                    # Band geçiren filtre
                    r_out, r_in = 50, 20
                    mask[crow-r_out:crow+r_out, ccol-r_out:ccol+r_out] = 0
                    mask[crow-r_in:crow+r_in, ccol-r_in:ccol+r_in] = 1
                elif filter_type == "band_stop":
                    # Band durduran filtre
                    r_out, r_in = 50, 20
                    mask[crow-r_out:crow+r_out, ccol-r_out:ccol+r_out] = 1
                    mask[crow-r_in:crow+r_in, ccol-r_in:ccol+r_in] = 0
                elif filter_type == "gaussian":
                    # Gaussian filtre
                    mask = np.zeros((rows, cols))
                    sigma = 30
                    for i in range(rows):
                        for j in range(cols):
                            mask[i,j] = np.exp(-((i-crow)**2 + (j-ccol)**2)/(2*sigma**2))
                
                # Filtreyi uygula
                fshift = fshift * mask
                
                # Ters Fourier dönüşümü
                f_ishift = np.fft.ifftshift(fshift)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                
                # Görüntüyü normalize et
                img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
                
                # RGB'ye dönüştür
                self.processed_image = cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Frekans filtresi uygulanırken bir hata oluştu: {str(e)}")

    def apply_butterworth(self):
        try:
            if self.processed_image is not None:
                # Görüntüyü gri tonlamaya çevir
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                
                # Fourier dönüşümü
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                
                # Görüntü boyutları
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                
                # Butterworth filtre parametreleri
                D0 = 30  # Kesme frekansı
                n = 2    # Filtre derecesi
                
                # Butterworth filtre maskesi oluştur
                mask = np.zeros((rows, cols))
                for i in range(rows):
                    for j in range(cols):
                        D = np.sqrt((i-crow)**2 + (j-ccol)**2)
                        mask[i,j] = 1 / (1 + (D/D0)**(2*n))
                
                # Filtreyi uygula
                fshift = fshift * mask
                
                # Ters Fourier dönüşümü
                f_ishift = np.fft.ifftshift(fshift)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                
                # Görüntüyü normalize et
                img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
                
                # RGB'ye dönüştür
                self.processed_image = cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Butterworth filtresi uygulanırken bir hata oluştu: {str(e)}")

    def apply_homomorphic(self):
        try:
            if self.processed_image is not None:
                # Görüntüyü gri tonlamaya çevir
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
                
                # Sıfır değerlerini küçük bir sayı ile değiştir (log(0) tanımsız olduğu için)
                gray = np.maximum(gray, 0.001)
                
                # Log dönüşümü
                img_log = np.log(gray)
                
                # Fourier dönüşümü
                f = np.fft.fft2(img_log)
                fshift = np.fft.fftshift(f)
                
                # Görüntü boyutları
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                
                # Homomorfik filtre parametreleri
                rh = 2.5    # Yüksek frekans kazancı
                rl = 0.5    # Düşük frekans kazancı
                d0 = 10     # Kesme frekansı
                c = 1       # Keskinlik kontrolü
                
                # Filtre maskesi oluştur
                y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
                d = np.sqrt(x*x + y*y)
                mask = (rh - rl) * (1 - np.exp(-c * (d*d)/(d0*d0))) + rl
                
                # Filtreyi uygula
                fshift_filtered = fshift * mask
                
                # Ters Fourier dönüşümü
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                
                # Üstel dönüşüm
                img_exp = np.exp(img_back)
                
                # Görüntüyü normalize et
                img_norm = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX)
                
                # RGB'ye dönüştür
                self.processed_image = cv2.cvtColor(img_norm.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                self.update_display()
                
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Homomorfik filtre uygulanırken bir hata oluştu: {str(e)}")

    def apply_conservative_filter(self):
        print("Konservatif filtre tıklandı")
        try:
            if self.processed_image is not None:
                kernel_size = int(self.kernel_size.currentText().split('x')[0])
                result = self.processed_image.copy()
                for c in range(3):  # Her kanal için uygula
                    channel = self.processed_image[:, :, c]
                    min_img = cv2.erode(channel, np.ones((kernel_size, kernel_size), np.uint8))
                    max_img = cv2.dilate(channel, np.ones((kernel_size, kernel_size), np.uint8))
                    # Konservatif filtre: min ve max arasında olmayan pikselleri sınırla
                    result[:, :, c] = np.where(channel < min_img, min_img,
                                       np.where(channel > max_img, max_img, channel))
                self.processed_image = result
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Konservatif filtre uygulanırken bir hata oluştu: {str(e)}")

    def apply_crimmins(self):
        print("Crimmins filtre tıklandı")
        try:
            if self.processed_image is not None:
                img = self.processed_image.copy()
                def crimmins_iteration(image, direction):
                    result = np.copy(image)
                    if direction == 'dark':
                        for _ in range(4):
                            tmp1 = np.roll(image, 1, axis=0)
                            tmp2 = np.roll(image, -1, axis=0)
                            tmp3 = np.roll(image, 1, axis=1)
                            tmp4 = np.roll(image, -1, axis=1)
                            result = np.where((image < tmp1) & (image < tmp2) & \
                                            (image < tmp3) & (image < tmp4),
                                            image + 1, result)
                    else:  # 'light'
                        for _ in range(4):
                            tmp1 = np.roll(image, 1, axis=0)
                            tmp2 = np.roll(image, -1, axis=0)
                            tmp3 = np.roll(image, 1, axis=1)
                            tmp4 = np.roll(image, -1, axis=1)
                            result = np.where((image > tmp1) & (image > tmp2) & \
                                            (image > tmp3) & (image > tmp4),
                                            image - 1, result)
                    return result
                for i in range(3):
                    channel = img[:,:,i]
                    channel = crimmins_iteration(channel, 'dark')
                    channel = crimmins_iteration(channel, 'light')
                    img[:,:,i] = channel
                self.processed_image = img
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Crimmins Speckle filtresi uygulanırken bir hata oluştu: {str(e)}")

    def start_crop(self):
        if self.processed_image is None:
            QMessageBox.warning(self, "Uyarı", "Önce bir görüntü yükleyin!")
            return
        QMessageBox.information(self, "Kırpma Bilgisi", "Köşe tutamaçlarını sürükleyerek istediğiniz alanı seçin. Seçili alan dışı yarı saydam gösterilecektir. 'Kırp' butonuna basınca sadece seçili alan kalacaktır.")
        img = self.processed_image.copy()
        height, width = img.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(img.tobytes(), width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(500, 500, Qt.AspectRatioMode.KeepAspectRatio)
        dialog = QDialog(self)
        dialog.setWindowTitle("Gelişmiş Kırpma")
        layout = QVBoxLayout(dialog)
        crop_widget = CropWidget(pixmap)
        layout.addWidget(crop_widget)
        btn_crop = QPushButton("Kırp")
        layout.addWidget(btn_crop)
        def do_crop():
            r = crop_widget.get_crop_rect()
            # Orijinal görseldeki koordinatlara dönüştür
            scale_x = self.processed_image.shape[1] / crop_widget.width()
            scale_y = self.processed_image.shape[0] / crop_widget.height()
            x1 = int(r.left() * scale_x)
            y1 = int(r.top() * scale_y)
            x2 = int(r.right() * scale_x)
            y2 = int(r.bottom() * scale_y)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.processed_image.shape[1]-1, x2), min(self.processed_image.shape[0]-1, y2)
            if x2 > x1 and y2 > y1:
                self.processed_image = self.processed_image[y1:y2, x1:x2]
                self.update_display()
            else:
                QMessageBox.warning(self, "Uyarı", "Geçerli bir alan seçilmedi!")
            dialog.accept()
        btn_crop.clicked.connect(do_crop)
        dialog.exec()

    def apply_sobel(self):
        try:
            if self.processed_image is not None:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel = cv2.magnitude(sobelx, sobely)
                sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
                self.processed_image = cv2.cvtColor(sobel.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Sobel uygulanırken hata: {str(e)}")

    def apply_prewitt(self):
        try:
            if self.processed_image is not None:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=np.float32)
                kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.float32)
                prewittx = cv2.filter2D(gray, -1, kernelx)
                prewitty = cv2.filter2D(gray, -1, kernely)
                prewitt = cv2.magnitude(prewittx.astype(np.float32), prewitty.astype(np.float32))
                prewitt = cv2.normalize(prewitt, None, 0, 255, cv2.NORM_MINMAX)
                self.processed_image = cv2.cvtColor(prewitt.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Prewitt uygulanırken hata: {str(e)}")

    def apply_roberts(self):
        try:
            if self.processed_image is not None:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
                kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)
                robertsx = cv2.filter2D(gray, -1, kernelx)
                robertsy = cv2.filter2D(gray, -1, kernely)
                roberts = cv2.magnitude(robertsx.astype(np.float32), robertsy.astype(np.float32))
                roberts = cv2.normalize(roberts, None, 0, 255, cv2.NORM_MINMAX)
                self.processed_image = cv2.cvtColor(roberts.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Roberts uygulanırken hata: {str(e)}")

    def apply_compass(self):
        try:
            if self.processed_image is not None:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                # 8 yönlü Kirsch kernel'leri
                kernels = [
                    np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]]),
                    np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]]),
                    np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]]),
                    np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]]),
                    np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]]),
                    np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]]),
                    np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]]),
                    np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])
                ]
                max_response = np.zeros_like(gray, dtype=np.float32)
                for k in kernels:
                    response = cv2.filter2D(gray, -1, k)
                    max_response = np.maximum(max_response, response.astype(np.float32))
                max_response = cv2.normalize(max_response, None, 0, 255, cv2.NORM_MINMAX)
                self.processed_image = cv2.cvtColor(max_response.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Compass uygulanırken hata: {str(e)}")

    def apply_canny(self):
        try:
            if self.processed_image is not None:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                self.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Canny uygulanırken hata: {str(e)}")

    def apply_laplace(self):
        try:
            if self.processed_image is not None:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                laplace = cv2.Laplacian(gray, cv2.CV_64F)
                laplace = np.abs(laplace)
                laplace = cv2.normalize(laplace, None, 0, 255, cv2.NORM_MINMAX)
                self.processed_image = cv2.cvtColor(laplace.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Laplace uygulanırken hata: {str(e)}")

    def apply_gabor(self):
        try:
            if self.processed_image is not None:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                gabor = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                self.processed_image = cv2.cvtColor(gabor, cv2.COLOR_GRAY2RGB)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Gabor uygulanırken hata: {str(e)}")

    def apply_hough(self):
        try:
            if self.processed_image is not None:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                lines = cv2.HoughLines(edges, 1, np.pi/180, 120)
                hough_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                if lines is not None:
                    for i, line in enumerate(lines):
                        if i > 100: break
                        rho, theta = line[0]
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        cv2.line(hough_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                self.processed_image = hough_img
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Hough uygulanırken hata: {str(e)}")

    def apply_erode(self):
        try:
            if self.processed_image is not None:
                kernel = np.ones((3, 3), np.uint8)
                eroded = cv2.erode(self.processed_image, kernel, iterations=1)
                self.processed_image = eroded
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Erode uygulanırken hata: {str(e)}")

    def apply_dilate(self):
        try:
            if self.processed_image is not None:
                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(self.processed_image, kernel, iterations=1)
                self.processed_image = dilated
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Dilate uygulanırken hata: {str(e)}")

    def apply_kmeans(self):
        try:
            if self.processed_image is not None:
                img = self.processed_image.copy()
                Z = img.reshape((-1, 3))
                Z = np.float32(Z)
                K = 4  # Sabit, istersen kullanıcıdan alabilirsin
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                center = np.uint8(center)
                res = center[label.flatten()]
                result_image = res.reshape((img.shape))
                self.processed_image = result_image
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"K-means uygulanırken hata: {str(e)}")

    def apply_gaussian_lpf(self):
        try:
            if self.processed_image is not None:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                rows, cols = gray.shape
                crow, ccol = rows // 2, cols // 2
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                sigma = 30
                x = np.arange(-ccol, ccol)
                y = np.arange(-crow, crow)
                X, Y = np.meshgrid(x, y)
                gaussian_lpf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
                fshift = fshift * gaussian_lpf
                f_ishift = np.fft.ifftshift(fshift)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
                self.processed_image = cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Gaussian LPF uygulanırken hata: {str(e)}")

    def apply_gaussian_hpf(self):
        try:
            if self.processed_image is not None:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2GRAY)
                rows, cols = gray.shape
                crow, ccol = rows // 2, cols // 2
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                # Gaussian HPF maskesi
                sigma = 30
                x = np.arange(-ccol, ccol)
                y = np.arange(-crow, crow)
                X, Y = np.meshgrid(x, y)
                gaussian_lpf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
                gaussian_hpf = 1 - gaussian_lpf
                fshift = fshift * gaussian_hpf
                f_ishift = np.fft.ifftshift(fshift)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
                self.processed_image = cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Gaussian HPF uygulanırken hata: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec()) 