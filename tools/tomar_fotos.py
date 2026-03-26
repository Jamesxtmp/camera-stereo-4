import cv2
import os
import sys
import time
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from app_config import CALIBRATION_FILE, CAMERA_HEIGHT, CAMERA_NAME, CAMERA_WIDTH, LEFT_FOLDER
except Exception:
    LEFT_FOLDER = os.path.join("dataset_stereo", "left")
    CALIBRATION_FILE = os.path.join(BASE_DIR, "configuration", "calibracion_estereo.npz")
    CAMERA_NAME = "3D USB Camera"
    CAMERA_WIDTH = 2560
    CAMERA_HEIGHT = 720

left_folder = LEFT_FOLDER if os.path.isabs(LEFT_FOLDER) else os.path.join(BASE_DIR, LEFT_FOLDER)
os.makedirs(left_folder, exist_ok=True)

# --- CARGA DE DATOS DE CALIBRACIÓN ---
calib_data = None
calib_params = None
if os.path.exists(CALIBRATION_FILE):
    with np.load(CALIBRATION_FILE) as data:
        keys = set(data.files)

        # Caso único: el archivo trae matrices de calibración para generar mapas en runtime.
        if {'mtx_l', 'dist_l', 'mtx_r', 'dist_r', 'R1', 'R2', 'P1', 'P2'}.issubset(keys):
            calib_params = {
                "mtx_l": data['mtx_l'],
                "dist_l": data['dist_l'],
                "R1": data['R1'],
                "P1": data['P1'],
            }
        else:
            print(f"ADVERTENCIA: El archivo no tiene las matrices esperadas, Claves encontradas: {sorted(keys)}")
else:
    print(f"ADVERTENCIA: No se encontró '{CALIBRATION_FILE}'. Se guardarán las imágenes originales sin rectificar.")

def buscar_indice_camara(nombre_objetivo):
    for camera_info in enumerate_cameras():
        if nombre_objetivo in camera_info.name:
            return camera_info.index
    return None

indice = buscar_indice_camara(CAMERA_NAME)
if indice is None:
    print(f"ADVERTENCIA: No se encontro camara con nombre '{CAMERA_NAME}'. Usando indice 0.")
    indice = 0

cap = cv2.VideoCapture(indice)

# Resolución nativa para capturar ambos lentes sin pérdida
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

print("Captura preparada. Teclas: [S] Guardar | [Q] Salir")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Dividimos el frame en izquierdo y derecho
    left_frame_raw = frame[:, :1280]

    # Si tenemos matrices de calibración (pero no mapas), generamos los mapas una sola vez.
    if calib_data is None and calib_params is not None:
        h, w = left_frame_raw.shape[:2]
        image_size = (w, h)
        calib_data = {
            "map_l_x": cv2.initUndistortRectifyMap(
                calib_params['mtx_l'], calib_params['dist_l'], calib_params['R1'], calib_params['P1'], image_size, cv2.CV_32FC1
            )[0],
            "map_l_y": cv2.initUndistortRectifyMap(
                calib_params['mtx_l'], calib_params['dist_l'], calib_params['R1'], calib_params['P1'], image_size, cv2.CV_32FC1
            )[1],
        }
        calib_params = None
        print(f"Mapas de rectificación generados para resolución {w}x{h}.")

    # Si tenemos datos de calibración, rectificamos la imagen izquierda
    if calib_data:
        left_frame_rect = cv2.remap(left_frame_raw, calib_data['map_l_x'], calib_data['map_l_y'], cv2.INTER_LANCZOS4)
        preview = left_frame_rect
    else:
        # Si no, usamos la imagen izquierda original para guardar y previsualizar
        left_frame_rect = left_frame_raw
        preview = left_frame_raw

    cv2.imshow('Captura para Entrenamiento 3D', preview)

    key = cv2.waitKey(1)
    
    if cv2.getWindowProperty('Captura para Entrenamiento 3D', cv2.WND_PROP_VISIBLE) < 1:
        break
    
    if key == ord('s'):
        # Timestamp en milisegundos para evitar colisiones por capturas en el mismo segundo.
        ts = int(time.time() * 1000)
        left_img_path = os.path.join(left_folder, f"left_{ts}.jpg")
        
        # Guardamos solo el frame izquierdo para dataset de detección
        cv2.imwrite(left_img_path, left_frame_rect)
        print(f"Imagen izquierda guardada: left_{ts}.jpg")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()