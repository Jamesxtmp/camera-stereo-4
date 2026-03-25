try:
    import cv2
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "No se encontro el modulo cv2. Instala dependencias con: pip install -r requirements.txt"
    ) from exc
import os
import time
import numpy as np
import torch
from ultralytics import YOLO
from cv2_enumerate_cameras import enumerate_cameras

# --- CONFIGURACIÓN ---
folder = 'dataset_stereo'
left_folder = os.path.join(folder, 'left')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO_ONNX = os.path.join(BASE_DIR, 'models', 'exp.onnx')
CALIBRATION_FILE = os.path.join(BASE_DIR, 'configuration', 'calibracion_estereo.npz')

if not os.path.exists(left_folder): os.makedirs(left_folder)

# --- INICIALIZACIÓN DE MODELO Y DISPOSITIVO ---
# Autodetectar GPU (RTX 5070)
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"Trabajando con dispositivo: {device}")

try:
    # Cargamos el modelo ONNX. Ultralytics gestiona la inferencia en GPU automáticamente si el device es '0'
    model = YOLO(RUTA_MODELO_ONNX, task='detect')
    print("Modelo ONNX cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# --- CARGA DE DATOS DE CALIBRACIÓN ---
calib_data = None
calib_params = None
if os.path.exists(CALIBRATION_FILE):
    with np.load(CALIBRATION_FILE) as data:
        keys = set(data.files)
        if {'mtx_l', 'dist_l', 'R1', 'P1'}.issubset(keys):
            calib_params = {
                "mtx_l": data['mtx_l'],
                "dist_l": data['dist_l'],
                "R1": data['R1'],
                "P1": data['P1'],
            }
        else:
            print(f"ADVERTENCIA: Claves incompletas en {CALIBRATION_FILE}")
else:
    print(f"ADVERTENCIA: No se encontró archivo de calibración.")

def buscar_indice_camara(nombre_objetivo):
    for camera_info in enumerate_cameras():
        if nombre_objetivo in camera_info.name:
            return camera_info.index
    return None

indice = buscar_indice_camara("3D USB Camera")
cap = cv2.VideoCapture(indice)

# Resolución para el sensor doble (1280x2 = 2560)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Sistema listo. Presiona [S] para capturar imagen limpia | [Q] Salir")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Separar frame izquierdo
    left_frame_raw = frame[:, :1280]

    # 2. Generar mapas de rectificación si es necesario
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
        print(f"Rectificación activa para {w}x{h}.")

    # 3. Aplicar rectificación
    if calib_data:
        left_frame_rect = cv2.remap(left_frame_raw, calib_data['map_l_x'], calib_data['map_l_y'], cv2.INTER_LANCZOS4)
    else:
        left_frame_rect = left_frame_raw.copy()

    # 4. INFERENCIA CON YOLO (ONNX)
    # Ejecutamos la detección sobre el frame rectificado
    results = model.predict(left_frame_rect, conf=0.5, device=device, verbose=False)
    
    # Dibujar los recuadros usando el método plot() de Ultralytics
    # Esto genera una copia del frame con las cajas y etiquetas dibujadas
    preview_annotated = results[0].plot()

    # Mostrar el resultado con cajas
    cv2.imshow('Deteccion de Latas - Stereo Left', preview_annotated)

    key = cv2.waitKey(1)
    
    # Salir si se cierra la ventana manualmente
    if cv2.getWindowProperty('Deteccion de Latas - Stereo Left', cv2.WND_PROP_VISIBLE) < 1:
        break
    
    if key == ord('s'):
        ts = int(time.time())
        left_img_path = os.path.join(left_folder, f"left_{ts}.jpg")
        # Guardamos la imagen RECTIFICADA pero LIMPIA (sin los cuadros de YOLO) para tu dataset
        cv2.imwrite(left_img_path, left_frame_rect)
        print(f"Imagen limpia guardada para entrenamiento: left_{ts}.jpg")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()