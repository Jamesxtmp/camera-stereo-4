import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QComboBox, QPushButton)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Control de Cámara Pro - Windows")
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.slider_labels = {}
        
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- Panel de Controles ---
        controls_layout = QVBoxLayout()
        
        controls_layout.addWidget(QLabel("Seleccionar Cámara:"))
        self.combo_camaras = QComboBox()
        self.detectar_camaras()
        controls_layout.addWidget(self.combo_camaras)

        self.btn_conectar = QPushButton("Conectar Cámara")
        self.btn_conectar.clicked.connect(self.toggle_camera)
        controls_layout.addWidget(self.btn_conectar)

        self.btn_restablecer = QPushButton("Restablecer")
        self.btn_restablecer.clicked.connect(self.restablecer_controles)
        controls_layout.addWidget(self.btn_restablecer)

        # Diccionario de controles (Nombre: [ID_OpenCV, Min, Max, Default])
        self.params = {
            "Brillo": [cv2.CAP_PROP_BRIGHTNESS, 0, 255, 128],
            "Contraste": [cv2.CAP_PROP_CONTRAST, 0, 255, 32],
            "Saturación": [cv2.CAP_PROP_SATURATION, 0, 255, 64],
            "Exposición": [cv2.CAP_PROP_EXPOSURE, -13, -1, -5],
            "Balance Blancos": [cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 2000, 8000, 4000]
        }

        self.sliders = {}
        for name, info in self.params.items():
            row = QHBoxLayout()
            label = QLabel(f"{name}:")
            value_label = QLabel(str(info[3]))
            value_label.setMinimumWidth(50)
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(info[1])
            slider.setMaximum(info[2])
            slider.setValue(info[3])
            slider.setEnabled(False)
            slider.valueChanged.connect(lambda val, p=name: self.cambiar_parametro(p, val))
            slider.valueChanged.connect(lambda val, p=name: self.actualizar_valor_slider(p, val))
            
            row.addWidget(label)
            row.addWidget(value_label)
            controls_layout.addLayout(row)
            controls_layout.addWidget(slider)
            self.sliders[name] = slider
            self.slider_labels[name] = value_label

        controls_layout.addStretch()
        layout.addLayout(controls_layout, 1)

        # --- Visualizador de Video ---
        self.video_label = QLabel("Cámara Desconectada")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.video_label, 4)

    def detectar_camaras(self):
        # Escaneo rápido de índices de cámaras
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.combo_camaras.addItem(f"Cámara {i}", i)
                cap.release()

    def toggle_camera(self):
        if self.cap is None:
            cam_idx = self.combo_camaras.currentData()
            # Usamos CAP_DSHOW para mejor compatibilidad con parámetros en Windows
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            
            if self.cap.isOpened():
                self.timer.start(30)
                self.btn_conectar.setText("Desconectar")
                for s in self.sliders.values(): s.setEnabled(True)
        else:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.video_label.clear()
            self.video_label.setText("Cámara Desconectada")
            self.btn_conectar.setText("Conectar Cámara")
            for s in self.sliders.values(): s.setEnabled(False)

    def cambiar_parametro(self, nombre, valor):
        if self.cap and self.cap.isOpened():
            prop_id = self.params[nombre][0]
            # Desactivar auto-ajustes si es necesario (depende del driver)
            if nombre == "Exposición":
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 0.25 suele ser manual en Windows
            
            self.cap.set(prop_id, valor)

    def actualizar_valor_slider(self, nombre, valor):
        self.slider_labels[nombre].setText(str(valor))

    def restablecer_controles(self):
        for nombre, info in self.params.items():
            self.sliders[nombre].setValue(info[3])

        if self.cap and self.cap.isOpened():
            for nombre, info in self.params.items():
                prop_id = info[0]
                valor = info[3]
                if nombre == "Exposición":
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                self.cap.set(prop_id, valor)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(img).scaled(
                self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.resize(1000, 600)
    window.show()
    sys.exit(app.exec_())