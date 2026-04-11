import sys
from pathlib import Path
import cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QSlider, QComboBox, QPushButton)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera_utils import (
    DROIDCAM_FIXED_INDEX,
    CAMERA_BACKENDS,
    abrir_camara_por_indice,
    abrir_camara_con_fallback,
)

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Control de Cámara DroidCam - Proyecto Estable")
        self.cap = None
        self.current_cam_data = None
        self.frame_fail_count = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Tamaño inicial de la ventana
        self.resize(1100, 700)
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- Panel de Controles ---
        controls_layout = QVBoxLayout()
        
        controls_layout.addWidget(QLabel("Cámara Detectada:"))
        self.combo_camaras = QComboBox()
        self.configurar_camara_fija()
        controls_layout.addWidget(self.combo_camaras)

        self.btn_conectar = QPushButton("Iniciar Stream")
        self.btn_conectar.setStyleSheet("""
            QPushButton {
                background-color: #27ae60; 
                color: white; 
                font-weight: bold; 
                height: 35px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #2ecc71; }
        """)
        self.btn_conectar.clicked.connect(self.toggle_camera)
        controls_layout.addWidget(self.btn_conectar)

        # Parámetros de imagen
        self.params = {
            "Brillo": [cv2.CAP_PROP_BRIGHTNESS, 0, 255, 128],
            "Contraste": [cv2.CAP_PROP_CONTRAST, 0, 255, 32]
        }

        self.sliders = {}
        for name, info in self.params.items():
            controls_layout.addWidget(QLabel(f"{name}:"))
            slider = QSlider(Qt.Horizontal)
            slider.setRange(info[1], info[2])
            slider.setValue(info[3])
            slider.setEnabled(False)
            slider.valueChanged.connect(lambda val, p=name: self.cambiar_parametro(p, val))
            controls_layout.addWidget(slider)
            self.sliders[name] = slider

        controls_layout.addStretch()
        layout.addLayout(controls_layout, 1)

        # --- Visualizador de Video (Fijo) ---
        self.video_label = QLabel("DroidCam lista para conectar")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #111; 
            color: #777; 
            border: 2px solid #333;
            font-family: Arial;
            font-size: 14px;
        """)
        
        # Crucial: Esto evita que la ventana crezca cuando llega el video
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label, 4)

    def configurar_camara_fija(self):
        self.combo_camaras.clear()

        for backend_name, backend_id in CAMERA_BACKENDS:
            texto = f"DroidCam Video | idx={DROIDCAM_FIXED_INDEX} | {backend_name}"
            self.combo_camaras.addItem(
                texto,
                {"index": DROIDCAM_FIXED_INDEX, "backend": backend_id, "nombre": "DroidCam Video"}
            )

    def toggle_camera(self):
        if self.cap is None:
            cam_data = self.combo_camaras.currentData()
            if cam_data is None:
                self.video_label.setText("No hay camaras disponibles")
                return

            idx = cam_data["index"]
            backend = cam_data["backend"]
            self.cap = abrir_camara_por_indice(
                indice=idx,
                backend=backend,
                validate_frames=True,
                frame_attempts=8,
            )
            self.current_cam_data = cam_data
            self.frame_fail_count = 0
            
            # Opcional: Limitar resolución para mejorar rendimiento
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            if self.cap is not None and self.cap.isOpened():
                self.timer.start(30)
                self.btn_conectar.setText("Detener Stream")
                self.btn_conectar.setStyleSheet("background-color: #c0392b; color: white; height: 35px; border-radius: 5px;")
                for s in self.sliders.values(): s.setEnabled(True)
            else:
                self.video_label.setText(f"No se pudo abrir idx={idx} con backend seleccionado")
        else:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.current_cam_data = None
            self.frame_fail_count = 0
            self.video_label.setText("Cámara Desconectada")
            self.btn_conectar.setText("Iniciar Stream")
            self.btn_conectar.setStyleSheet("background-color: #27ae60; color: white; height: 35px; border-radius: 5px;")
            for s in self.sliders.values(): s.setEnabled(False)

    def _try_reopen_with_fallback(self):
        if not self.current_cam_data:
            return False

        idx = self.current_cam_data["index"]
        backend_actual = self.current_cam_data["backend"]
        candidatos_backend = [
            (name, b_id) for name, b_id in CAMERA_BACKENDS if b_id != backend_actual
        ]

        try:
            nueva, backend_name, backend_id = abrir_camara_con_fallback(
                indice=idx,
                backend_order=candidatos_backend,
                validate_frames=True,
                frame_attempts=8,
            )
        except RuntimeError:
            return False

        if self.cap:
            self.cap.release()
        self.cap = nueva
        self.current_cam_data = {
            "index": idx,
            "backend": backend_id,
            "nombre": self.current_cam_data["nombre"],
        }
        self.frame_fail_count = 0
        self.video_label.setText(f"Se aplico backend alternativo: {backend_name}")
        return True

    def cambiar_parametro(self, nombre, valor):
        if self.cap:
            self.cap.set(self.params[nombre][0], valor)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_fail_count = 0
                # Conversión de color BGR a RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                
                img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Obtener el tamaño actual del contenedor para escalar la imagen
                # sin forzar el crecimiento de la ventana
                lbl_w = self.video_label.width()
                lbl_h = self.video_label.height()
                
                pixmap = QPixmap.fromImage(img).scaled(
                    lbl_w, lbl_h, 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                
                self.video_label.setPixmap(pixmap)
            else:
                self.frame_fail_count += 1
                if self.frame_fail_count >= 12:
                    if self._try_reopen_with_fallback():
                        pass
                    else:
                        self.timer.stop()
                        if self.cap:
                            self.cap.release()
                        self.cap = None
                        self.current_cam_data = None
                        self.btn_conectar.setText("Iniciar Stream")
                        self.btn_conectar.setStyleSheet("background-color: #27ae60; color: white; height: 35px; border-radius: 5px;")
                        for s in self.sliders.values():
                            s.setEnabled(False)
                        self.video_label.setText("Sin frames en idx fijo 1402")

    def closeEvent(self, event):
        """Asegura liberar la cámara si cierras la ventana"""
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())
