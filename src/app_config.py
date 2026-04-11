import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_FOLDER = "dataset_stereo"
LEFT_FOLDER = os.path.join(DATASET_FOLDER, "left")

MODEL_PATH = os.path.join(BASE_DIR, "models", "Limpia.onnx")
CALIBRATION_FILE = os.path.join(BASE_DIR, "configuration", "calibracion_estereo.npz")
FLOOR_HOMOGRAPHY_FILE = os.path.join(BASE_DIR, "configuration", "homografia_piso.json")

CAMERA_NAME = "3D USB Camera"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 360

YOLO_CONF = 0.35
YOLO_IMGSZ = 320
YOLO_EVERY_N_FRAMES = 8
BLUE_EVERY_N_FRAMES = 10

# Resolucion usada solo para inferencia (se reescala desde el frame de camara).
INFER_FRAME_WIDTH = 416
INFER_FRAME_HEIGHT = 234

# Limita cuantas detecciones se dibujan para reducir costo de render.
MAX_DRAW_DETECTIONS = 3

# Desactivar por defecto para maximizar FPS en CPU.
ENABLE_BLUE_FLOOR = False

# Mantiene cajas por unos frames cuando se pierde deteccion temporalmente.
DETECTION_HOLD_FRAMES = 20

# Segmentacion inicial para delimitante azul en piso.
BLUE_HSV_LOWER = (85, 50, 40)
BLUE_HSV_UPPER = (140, 255, 255)
BLUE_ROI_Y_MIN_RATIO = 0.45
BLUE_MIN_AREA_PX = 900
