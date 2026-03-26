import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_FOLDER = "dataset_stereo"
LEFT_FOLDER = os.path.join(DATASET_FOLDER, "left")

MODEL_PATH = os.path.join(BASE_DIR, "models", "exp.onnx")
CALIBRATION_FILE = os.path.join(BASE_DIR, "configuration", "calibracion_estereo.npz")

CAMERA_NAME = "3D USB Camera"
CAMERA_WIDTH = 2560
CAMERA_HEIGHT = 720

YOLO_CONF = 0.35
YOLO_IMGSZ = 960

# Calcula distancia estereo una vez cada N frames para reducir carga.
STEREO_EVERY_N_FRAMES = 5

# Limita cuantas detecciones se usan para estereo por frame de calculo.
MAX_STEREO_TARGETS = 2

# Mantiene cajas por unos frames cuando se pierde deteccion temporalmente.
DETECTION_HOLD_FRAMES = 8

# Conversion de unidad de calibracion a centimetros.
# En esta calibracion T esta en milimetros (baseline ~53.54), por eso la escala correcta es 0.1.
# Referencias utiles:
# - si esta en mm -> 0.1
# - si esta en cm -> 1.0
# - si esta en m  -> 100.0
DISTANCE_TO_CM_SCALE = 0.1
DISTANCE_UNIT_LABEL = "cm"
