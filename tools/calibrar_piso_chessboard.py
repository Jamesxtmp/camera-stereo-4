import argparse
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras

REQUIRED_STEREO_KEYS = {"mtx_l", "dist_l", "R1", "P1"}


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from app_config import CALIBRATION_FILE, CAMERA_HEIGHT, CAMERA_NAME, CAMERA_WIDTH

    DEFAULT_CALIB_STEREO = CALIBRATION_FILE
    DEFAULT_CAMERA_NAME = CAMERA_NAME
    DEFAULT_FRAME_WIDTH = CAMERA_WIDTH
    DEFAULT_FRAME_HEIGHT = CAMERA_HEIGHT
except Exception:
    DEFAULT_CALIB_STEREO = "configuration/calibracion_estereo.npz"
    DEFAULT_CAMERA_NAME = "3D USB Camera"
    DEFAULT_FRAME_WIDTH = 2560
    DEFAULT_FRAME_HEIGHT = 720


def resolver_indice_camara(camera_name, fallback_index):
    for camera_info in enumerate_cameras():
        if camera_name in camera_info.name:
            return camera_info.index, camera_info.name
    return fallback_index, None


def cargar_parametros_rectificacion(calib_stereo_path):
    data = np.load(calib_stereo_path)
    with data:
        keys = set(data.files)
        if not REQUIRED_STEREO_KEYS.issubset(keys):
            raise RuntimeError(
                "Archivo de calibracion estereo incompleto. "
                f"Se requieren claves: {sorted(REQUIRED_STEREO_KEYS)}"
            )
        mtx_l = data["mtx_l"]
        dist_l = data["dist_l"]
        r1 = data["R1"]
        p1 = data["P1"]
    return mtx_l, dist_l, r1, p1


def construir_obj_points_cm(corners_x, corners_y, square_size_mm):
    square_size_cm = float(square_size_mm) / 10.0
    obj = np.zeros((corners_y * corners_x, 2), dtype=np.float32)

    idx = 0
    for y in range(corners_y):
        for x in range(corners_x):
            obj[idx] = [x * square_size_cm, y * square_size_cm]
            idx += 1

    return obj


def guardar_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Calibra homografia de piso con tablero de ajedrez y guarda en configuration/."
    )
    parser.add_argument("--calib-stereo", default=DEFAULT_CALIB_STEREO)
    parser.add_argument("--output", default="configuration/homografia_piso.json")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--camera-name",
        default=DEFAULT_CAMERA_NAME,
        help="Nombre de camara a buscar. Si no aparece, usa --camera-index",
    )
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--frame-height", type=int, default=DEFAULT_FRAME_HEIGHT)
    parser.add_argument("--left-width", type=int, default=1280)
    parser.add_argument("--left-height", type=int, default=720)
    parser.add_argument(
        "--corners",
        default="7x10",
        help="Esquinas internas del tablero (ejemplo: 7x10)",
    )
    parser.add_argument("--square-mm", type=float, default=22.0)
    parser.add_argument(
        "--robot-ref",
        default="center-bottom",
        help="Punto referencia del robot en imagen: center-bottom o x,y",
    )
    args = parser.parse_args()

    corners_x, corners_y = [int(v) for v in args.corners.lower().split("x")]
    if corners_x < 2 or corners_y < 2:
        raise RuntimeError("El patron de esquinas debe ser >= 2x2")

    mtx_l, dist_l, r1, p1 = cargar_parametros_rectificacion(args.calib_stereo)
    map_x, map_y = cv2.initUndistortRectifyMap(
        mtx_l,
        dist_l,
        r1,
        p1,
        (args.left_width, args.left_height),
        cv2.CV_32FC1,
    )

    if args.robot_ref == "center-bottom":
        robot_ref_px = [args.left_width // 2, args.left_height - 1]
    else:
        rx, ry = [int(v) for v in args.robot_ref.split(",")]
        robot_ref_px = [rx, ry]

    obj_points_cm = construir_obj_points_cm(corners_x, corners_y, args.square_mm)

    selected_index, selected_name = resolver_indice_camara(args.camera_name, args.camera_index)
    if selected_name is not None:
        print(f"Camara seleccionada por nombre: {selected_name} (indice {selected_index})")
    else:
        print(
            f"No se encontro camara con nombre '{args.camera_name}'. "
            f"Usando indice de respaldo {selected_index}."
        )

    cap = cv2.VideoCapture(selected_index)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir camara indice {selected_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    win = "Calibracion Piso Chessboard (C capturar, G guardar, Q salir)"
    cv2.namedWindow(win)

    frozen_left = None
    frozen_gray = None
    frozen_corners = None

    print("Instrucciones:")
    print("1) Coloca el tablero en el piso y visible completo en la camara izquierda.")
    print("2) Presiona C para congelar frame.")
    print("3) Si detecta esquinas, presiona G para guardar homografia.")
    print("4) Si falla, mejora luz/enfoque y vuelve a presionar C.")

    while True:
        if frozen_left is None:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer frame de camara.")
                break

            left_raw = frame[:, : args.left_width]
            left_rect = cv2.remap(left_raw, map_x, map_y, cv2.INTER_LINEAR)
            display = left_rect.copy()

            cv2.circle(display, tuple(robot_ref_px), 6, (0, 0, 255), -1)
            cv2.putText(
                display,
                "RobotRef",
                (robot_ref_px[0] + 8, max(16, robot_ref_px[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                display,
                f"Patron: {corners_x}x{corners_y} | Cuadro: {args.square_mm} mm",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
        else:
            display = frozen_left.copy()
            if frozen_corners is not None:
                cv2.drawChessboardCorners(
                    display,
                    (corners_x, corners_y),
                    frozen_corners,
                    True,
                )
                cv2.putText(
                    display,
                    "Tablero detectado - presiona G para guardar",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    display,
                    "No detectado - presiona C para reintentar",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                )

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer frame para captura.")
                continue

            left_raw = frame[:, : args.left_width]
            frozen_left = cv2.remap(left_raw, map_x, map_y, cv2.INTER_LINEAR)
            frozen_gray = cv2.cvtColor(frozen_left, cv2.COLOR_BGR2GRAY)

            found, corners = cv2.findChessboardCorners(
                frozen_gray,
                (corners_x, corners_y),
                flags,
            )
            if found:
                refined = cv2.cornerSubPix(
                    frozen_gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria,
                )
                frozen_corners = refined
                print(f"Tablero detectado con {len(refined)} esquinas.")
            else:
                frozen_corners = None
                print("No se detecto el tablero en este frame.")

        elif key == ord("g"):
            if frozen_left is None or frozen_corners is None:
                print("Primero captura un frame valido con C.")
                continue

            img_pts = frozen_corners.reshape(-1, 2).astype(np.float32)
            world_pts = obj_points_cm.astype(np.float32)

            h_img_to_world, _ = cv2.findHomography(img_pts, world_pts, method=0)
            if h_img_to_world is None:
                print("No se pudo calcular homografia.")
                continue

            h_world_to_img = np.linalg.inv(h_img_to_world)

            pred_world = cv2.perspectiveTransform(
                img_pts.reshape(-1, 1, 2), h_img_to_world
            ).reshape(-1, 2)
            rmse_cm = float(np.sqrt(np.mean(np.sum((pred_world - world_pts) ** 2, axis=1))))

            payload = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source": {
                    "calib_stereo": args.calib_stereo,
                    "camera_index": selected_index,
                    "camera_name_requested": args.camera_name,
                    "camera_name_selected": selected_name,
                    "left_image_size": [args.left_width, args.left_height],
                },
                "chessboard": {
                    "inner_corners": [corners_x, corners_y],
                    "square_mm": args.square_mm,
                    "square_cm": args.square_mm / 10.0,
                },
                "reference": {
                    "world_unit": "cm",
                    "robot_ref_px": robot_ref_px,
                },
                "homography": {
                    "img_to_world": h_img_to_world.tolist(),
                    "world_to_img": h_world_to_img.tolist(),
                    "reprojection_rmse_cm": rmse_cm,
                },
                "samples": {
                    "img_points_px": img_pts.tolist(),
                    "world_points_cm": world_pts.tolist(),
                },
            }

            guardar_json(args.output, payload)
            print(f"Homografia guardada en: {args.output}")
            print(f"RMSE reproyeccion: {rmse_cm:.2f} cm")
            print("Calibracion de piso completada.")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
