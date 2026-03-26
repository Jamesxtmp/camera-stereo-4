import cv2
import os
import time
import torch
from app_config import (
    CALIBRATION_FILE,
    CAMERA_HEIGHT,
    CAMERA_NAME,
    CAMERA_WIDTH,
    DETECTION_HOLD_FRAMES,
    DISTANCE_TO_CM_SCALE,
    DISTANCE_UNIT_LABEL,
    LEFT_FOLDER,
    MAX_STEREO_TARGETS,
    MODEL_PATH,
    STEREO_EVERY_N_FRAMES,
    YOLO_CONF,
    YOLO_IMGSZ,
)
from camera_utils import abrir_camara_estereo
from detection_utils import cargar_modelo, obtener_detecciones_latas
from stereo_utils import (
    cargar_calibracion,
    crear_mapas_rectificacion,
    distancia_desde_disparidad,
    estimar_disparidad_bbox,
    rectificar_frames,
)
from visualization_utils import dibujar_detecciones, obtener_indice_mas_cercana


def _centro_bbox(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _asignar_distancias_desde_cache(detections, cache, max_center_dist=140.0):
    if not detections or not cache:
        return

    for d in detections:
        cx, cy = _centro_bbox(d["box"])
        best_item = None
        best_dist = float("inf")
        for c in cache:
            if c.get("distance") is None:
                continue
            ccx, ccy = _centro_bbox(c["box"])
            center_dist = ((cx - ccx) ** 2 + (cy - ccy) ** 2) ** 0.5
            if center_dist < best_dist:
                best_dist = center_dist
                best_item = c

        if best_item is not None and best_dist <= max_center_dist:
            d["distance"] = best_item["distance"]

# --- CONFIGURACIÓN ---
if not os.path.exists(LEFT_FOLDER):
    os.makedirs(LEFT_FOLDER)

# --- INICIALIZACIÓN DE MODELO Y DISPOSITIVO ---
# Autodetectar GPU (RTX 5070)
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"Trabajando con dispositivo: {device}")

try:
    model = cargar_modelo(MODEL_PATH)
    print("Modelo ia cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# --- CARGA DE DATOS DE CALIBRACIÓN ---
rect_maps = None
calibration = cargar_calibracion(CALIBRATION_FILE)
calib_params = None
stereo_ready = False
fx_px = None
baseline_units = None
if calibration is not None:
    calib_params = calibration["params"]
    stereo_ready = calibration["stereo_ready"]
    fx_px = calibration["fx_px"]
    baseline_units = calibration["baseline_units"]
    if stereo_ready:
        baseline_cm = baseline_units * DISTANCE_TO_CM_SCALE
        print(
            f"Stereo activo. fx={fx_px:.2f}px, baseline={baseline_units:.4f} unidades "
            f"(~{baseline_cm:.2f} {DISTANCE_UNIT_LABEL})."
        )
    else:
        print("ADVERTENCIA: No se pudo obtener fx/baseline válidos para distancia estéreo.")
else:
    print(f"ADVERTENCIA: No se encontró calibración válida en {CALIBRATION_FILE}.")

cap = abrir_camara_estereo(CAMERA_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)

print("Sistema listo. Presiona [S] para capturar imagen limpia | [Q] Salir")

frame_idx = 0
frames_without_detection = DETECTION_HOLD_FRAMES + 1
last_detections = []

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1

    # 1. Separar frames izquierdo y derecho
    left_frame_raw = frame[:, :1280]
    right_frame_raw = frame[:, 1280:]

    # 2. Generar mapas de rectificación si es necesario
    if rect_maps is None and calib_params is not None:
        h, w = left_frame_raw.shape[:2]
        image_size = (w, h)
        rect_maps = crear_mapas_rectificacion(calib_params, image_size)
        print(f"Rectificación activa para {w}x{h}.")

    # 3. Aplicar rectificación
    if rect_maps:
        left_frame_rect, right_frame_rect = rectificar_frames(left_frame_raw, right_frame_raw, rect_maps)
    else:
        left_frame_rect = left_frame_raw.copy()
        right_frame_rect = right_frame_raw.copy()

    # 4. INFERENCIA CON YOLO (ONNX)
    detections = obtener_detecciones_latas(
        model,
        left_frame_rect,
        device,
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
    )
    has_detection_now = len(detections) > 0

    if has_detection_now:
        frames_without_detection = 0
        for d in detections:
            d["distance"] = None

        # Reusa distancia previa en frames donde no toca recalcular estereo.
        _asignar_distancias_desde_cache(detections, last_detections)

        should_update_stereo = stereo_ready and (frame_idx % STEREO_EVERY_N_FRAMES == 0)
        if should_update_stereo:
            left_gray = cv2.cvtColor(left_frame_rect, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame_rect, cv2.COLOR_BGR2GRAY)

            # Prioriza pocas latas con mayor confianza para recortar costo.
            top_indices = sorted(
                range(len(detections)),
                key=lambda idx: detections[idx]["conf"],
                reverse=True,
            )[:MAX_STEREO_TARGETS]

            for idx in top_indices:
                d = detections[idx]
                disparity = estimar_disparidad_bbox(left_gray, right_gray, d["box"])
                distance_units = distancia_desde_disparidad(disparity, fx_px, baseline_units)
                if distance_units is not None:
                    d["distance"] = distance_units * DISTANCE_TO_CM_SCALE

        # Actualiza cache para robustez ante perdidas de confianza.
        last_detections = [
            {"box": d["box"], "conf": d["conf"], "distance": d.get("distance")}
            for d in detections
        ]
    else:
        frames_without_detection += 1
        if frames_without_detection <= DETECTION_HOLD_FRAMES and last_detections:
            detections = [
                {"box": d["box"], "conf": d["conf"], "distance": d.get("distance")}
                for d in last_detections
            ]
        else:
            detections = []

    nearest_idx = obtener_indice_mas_cercana(detections)
    preview_annotated = dibujar_detecciones(
        left_frame_rect,
        detections,
        nearest_idx=nearest_idx,
        unit_label=DISTANCE_UNIT_LABEL,
    )

    # Mostrar el resultado con cajas
    cv2.imshow('Deteccion de Latas - Stereo Left', preview_annotated)

    key = cv2.waitKey(1)
    
    # Salir si se cierra la ventana manualmente
    if cv2.getWindowProperty('Deteccion de Latas - Stereo Left', cv2.WND_PROP_VISIBLE) < 1:
        break
    
    if key == ord('s'):
        ts = int(time.time())
        left_img_path = os.path.join(LEFT_FOLDER, f"left_{ts}.jpg")
        # Guardamos la imagen RECTIFICADA pero LIMPIA (sin los cuadros de YOLO) para tu dataset
        cv2.imwrite(left_img_path, left_frame_rect)
        print(f"Imagen limpia guardada para entrenamiento: left_{ts}.jpg")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()