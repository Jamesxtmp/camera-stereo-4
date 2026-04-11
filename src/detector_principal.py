import cv2
import os
import time
from concurrent.futures import ThreadPoolExecutor
import torch
from app_config import (
    BLUE_HSV_LOWER,
    BLUE_HSV_UPPER,
    BLUE_MIN_AREA_PX,
    BLUE_ROI_Y_MIN_RATIO,
    CAMERA_HEIGHT,
    DROIDCAM_FIXED_INDEX,
    CAMERA_WIDTH,
    DETECTION_HOLD_FRAMES,
    ENABLE_BLUE_FLOOR,
    FLOOR_HOMOGRAPHY_FILE,
    INFER_FRAME_HEIGHT,
    INFER_FRAME_WIDTH,
    LEFT_FOLDER,
    MAX_DRAW_DETECTIONS,
    BLUE_EVERY_N_FRAMES,
    MODEL_PATH,
    YOLO_CONF,
    YOLO_EVERY_N_FRAMES,
    YOLO_IMGSZ,
)
from camera_utils import abrir_camara_con_fallback
from blue_floor_utils import cargar_homografia_piso, estimar_borde_azul_mas_cercano_cm
from detection_utils import cargar_modelo, obtener_detecciones_latas
from visualization_utils import dibujar_detecciones


def _escalar_detecciones_a_frame_original(detections, sx, sy, w, h):
    if not detections:
        return detections

    scaled = []
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        nx1 = max(0, min(int(x1 * sx), w - 1))
        ny1 = max(0, min(int(y1 * sy), h - 1))
        nx2 = max(nx1 + 1, min(int(x2 * sx), w))
        ny2 = max(ny1 + 1, min(int(y2 * sy), h))
        scaled.append({
            "box": (nx1, ny1, nx2, ny2),
            "conf": d["conf"],
        })
    return scaled


def _inferir_y_escalar(model, infer_frame, device, conf, imgsz, sx, sy, frame_w, frame_h):
    t0 = time.perf_counter()
    detections_small = obtener_detecciones_latas(
        model,
        infer_frame,
        device,
        conf=conf,
        imgsz=imgsz,
    )
    detections = _escalar_detecciones_a_frame_original(detections_small, sx, sy, frame_w, frame_h)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    return detections, infer_ms


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

floor_calib = None
if ENABLE_BLUE_FLOOR:
    floor_calib = cargar_homografia_piso(FLOOR_HOMOGRAPHY_FILE)
    if floor_calib is None:
        print(f"ADVERTENCIA: No se encontro homografia de piso en {FLOOR_HOMOGRAPHY_FILE}")
    else:
        print("Homografia de piso cargada correctamente.")

try:
    cap, backend_name, _ = abrir_camara_con_fallback(
        indice=DROIDCAM_FIXED_INDEX,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        validate_frames=True,
        frame_attempts=8,
    )
    print(f"Camara unica activa en idx={DROIDCAM_FIXED_INDEX} usando backend {backend_name}.")
except RuntimeError as e:
    print(f"Error al abrir camara unica: {e}")
    exit()

# Reducir cola interna para evitar "arrastre" de frames viejos.
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Sistema listo. Presiona [S] para capturar imagen limpia | [Q] Salir")
print(
    f"Modo extremo: cam={CAMERA_WIDTH}x{CAMERA_HEIGHT}, infer={INFER_FRAME_WIDTH}x{INFER_FRAME_HEIGHT}, "
    f"imgsz={YOLO_IMGSZ}, yolo_cada={YOLO_EVERY_N_FRAMES} frames"
)

frame_idx = 0
frames_without_detection = DETECTION_HOLD_FRAMES + 1
last_detections = []
last_blue_info = None
last_infer_ms = 0.0

executor = ThreadPoolExecutor(max_workers=1)
inference_future = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_h, frame_w = frame.shape[:2]

        # Consume resultado de inferencia si ya termino (sin bloquear el loop de video).
        if inference_future is not None and inference_future.done():
            try:
                detections, last_infer_ms = inference_future.result()
            except Exception as e:
                print(f"Error de inferencia: {e}")
                detections = []
                last_infer_ms = 0.0

            inference_future = None
            if detections:
                frames_without_detection = 0
                for d in detections:
                    d["distance"] = None
                last_detections = [
                    {"box": d["box"], "conf": d["conf"], "distance": d.get("distance")}
                    for d in detections
                ]
            else:
                frames_without_detection += 1
                if frames_without_detection > DETECTION_HOLD_FRAMES:
                    last_detections = []

        # Lanza inferencia solo si no hay una en progreso.
        should_run_yolo = (frame_idx % YOLO_EVERY_N_FRAMES == 0) or (not last_detections)
        if inference_future is None and should_run_yolo:
            infer_frame = cv2.resize(frame, (INFER_FRAME_WIDTH, INFER_FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
            sx = frame_w / float(INFER_FRAME_WIDTH)
            sy = frame_h / float(INFER_FRAME_HEIGHT)
            inference_future = executor.submit(
                _inferir_y_escalar,
                model,
                infer_frame,
                device,
                YOLO_CONF,
                YOLO_IMGSZ,
                sx,
                sy,
                frame_w,
                frame_h,
            )

        if frames_without_detection <= DETECTION_HOLD_FRAMES and last_detections:
            detections = [
                {"box": d["box"], "conf": d["conf"], "distance": d.get("distance")}
                for d in last_detections
            ]
        else:
            detections = []

        draw_detections = detections
        if len(draw_detections) > MAX_DRAW_DETECTIONS:
            draw_detections = sorted(
                draw_detections,
                key=lambda d: d["conf"],
                reverse=True,
            )[:MAX_DRAW_DETECTIONS]

        if draw_detections:
            preview_annotated = dibujar_detecciones(
                frame,
                draw_detections,
                nearest_idx=None,
                unit_label="cm",
            )
        else:
            # Sin detecciones evitamos copia extra para ahorrar CPU.
            preview_annotated = frame

        blue_info = None
        if floor_calib is not None:
            should_run_blue = (frame_idx % BLUE_EVERY_N_FRAMES == 0) or (last_blue_info is None)
            if should_run_blue:
                last_blue_info = estimar_borde_azul_mas_cercano_cm(
                    frame,
                    floor_calib,
                    BLUE_HSV_LOWER,
                    BLUE_HSV_UPPER,
                    roi_y_min_ratio=BLUE_ROI_Y_MIN_RATIO,
                    min_area_px=BLUE_MIN_AREA_PX,
                )
            blue_info = last_blue_info

        if blue_info is not None:
            blue_mask = blue_info["mask"]
            if blue_mask is not None:
                tint = preview_annotated.copy()
                tint[blue_mask > 0] = (255, 80, 40)
                preview_annotated = cv2.addWeighted(preview_annotated, 0.78, tint, 0.22, 0.0)

            blue_dist_cm = blue_info["distance_cm"]
            nearest_blue_px = blue_info["nearest_point_px"]
            if nearest_blue_px is not None:
                cv2.circle(preview_annotated, nearest_blue_px, 6, (255, 0, 0), -1)

            text = "Borde azul: sin deteccion"
            color = (0, 0, 255)
            if blue_dist_cm is not None:
                text = f"Borde azul: {blue_dist_cm:.1f} cm"
                color = (0, 255, 255)

            cv2.putText(
                preview_annotated,
                text,
                (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )

        yolo_state = "YOLO: procesando" if inference_future is not None else "YOLO: libre"
        cv2.putText(
            preview_annotated,
            f"{yolo_state} | infer {last_infer_ms:.1f} ms",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (50, 255, 50),
            2,
        )

        # Mostrar el resultado con cajas
        cv2.imshow('Deteccion de Latas - Camara Unica', preview_annotated)

        key = cv2.waitKey(1)
        
        # Salir si se cierra la ventana manualmente
        if cv2.getWindowProperty('Deteccion de Latas - Camara Unica', cv2.WND_PROP_VISIBLE) < 1:
            break
        
        if key == ord('s'):
            ts = int(time.time())
            left_img_path = os.path.join(LEFT_FOLDER, f"left_{ts}.jpg")
            # Guardamos la imagen limpia (sin los cuadros de YOLO) para dataset.
            cv2.imwrite(left_img_path, frame)
            print(f"Imagen limpia guardada para entrenamiento: left_{ts}.jpg")

        elif key == ord('q'):
            break
finally:
    if inference_future is not None:
        try:
            inference_future.cancel()
        except Exception:
            pass
    executor.shutdown(wait=False)

cap.release()
cv2.destroyAllWindows()