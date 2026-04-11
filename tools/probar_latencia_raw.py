import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera_utils import DROIDCAM_FIXED_INDEX, abrir_camara_con_fallback
from src.app_config import CAMERA_WIDTH, CAMERA_HEIGHT


WINDOW_NAME = "Prueba Latencia RAW"


def main():
    try:
        cap, backend_name, _ = abrir_camara_con_fallback(
            indice=DROIDCAM_FIXED_INDEX,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            validate_frames=True,
            frame_attempts=8,
        )
    except RuntimeError as exc:
        print(f"No se pudo abrir la camara: {exc}")
        return 1

    # Intento de reducir cola interna de frames para menor latencia percibida.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("=== Prueba de latencia RAW ===")
    print(f"Indice fijo: {DROIDCAM_FIXED_INDEX}")
    print(f"Backend: {backend_name}")
    print("Sin procesamiento intermedio entre frames.")
    print("Controles: [Q] salir")

    last_t = time.perf_counter()
    frame_count = 0
    avg_dt_ms = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            t1 = time.perf_counter()

            if not ok:
                print("Frame no disponible. Saliendo...")
                break

            frame_count += 1
            dt_ms = (t1 - last_t) * 1000.0
            last_t = t1

            # Promedio movil simple para no tener lecturas tan inestables.
            alpha = 0.05
            avg_dt_ms = dt_ms if frame_count == 1 else (1 - alpha) * avg_dt_ms + alpha * dt_ms

            inst_fps = 1000.0 / dt_ms if dt_ms > 0 else 0.0
            avg_fps = 1000.0 / avg_dt_ms if avg_dt_ms > 0 else 0.0
            read_ms = (t1 - t0) * 1000.0

            cv2.putText(frame, f"FPS inst: {inst_fps:6.2f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS prom: {avg_fps:6.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"read() ms: {read_ms:6.2f}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
