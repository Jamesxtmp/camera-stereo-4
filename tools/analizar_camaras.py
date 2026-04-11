import argparse
import sys
from pathlib import Path

import cv2
from cv2_enumerate_cameras import enumerate_cameras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera_utils import CAMERA_BACKENDS, abrir_camara_por_indice


def listar_enumeradas():
    camaras = []
    for cam in enumerate_cameras():
        camaras.append(
            {
                "index": int(cam.index),
                "name": str(cam.name),
                "backend": str(getattr(cam, "backend", "N/A")),
            }
        )
    return camaras


def sondear_indices(max_index, frame_attempts):
    detectadas = []
    vistos = set()

    for idx in range(max_index + 1):
        for backend_name, backend_id in CAMERA_BACKENDS:
            cap = abrir_camara_por_indice(
                indice=idx,
                backend=backend_id,
                validate_frames=True,
                frame_attempts=frame_attempts,
            )
            if cap is None:
                continue

            signature = (idx, backend_name)
            if signature not in vistos:
                detectadas.append(
                    {
                        "index": idx,
                        "name": f"Camara detectada por sondeo ({backend_name})",
                        "backend": backend_name,
                    }
                )
                vistos.add(signature)

            cap.release()

    return detectadas


def combinar_resultados(enumeradas, sondeo):
    # Priorizamos nombre real de enumerate_cameras y anexamos hallazgos de sondeo.
    por_indice = {}
    for cam in enumeradas:
        por_indice[cam["index"]] = cam

    for cam in sondeo:
        if cam["index"] not in por_indice:
            por_indice[cam["index"]] = cam

    return [por_indice[idx] for idx in sorted(por_indice)]


def main():
    parser = argparse.ArgumentParser(
        description="Analiza camaras disponibles y muestra indice + nombre"
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=20,
        help="Indice maximo para sondeo adicional (default: 20)",
    )
    parser.add_argument(
        "--frame-attempts",
        type=int,
        default=6,
        help="Cantidad de lecturas para validar frames (default: 6)",
    )
    parser.add_argument(
        "--sin-sondeo",
        action="store_true",
        help="Solo usa enumerate_cameras sin abrir dispositivos",
    )
    args = parser.parse_args()

    enumeradas = listar_enumeradas()
    sondeo = [] if args.sin_sondeo else sondear_indices(args.max_index, args.frame_attempts)
    resultado = combinar_resultados(enumeradas, sondeo)

    if not resultado:
        print("No se detectaron camaras.")
        return

    print("Camaras detectadas:\n")
    for cam in resultado:
        print(f"- idx={cam['index']} | nombre={cam['name']} | backend={cam['backend']}")


if __name__ == "__main__":
    main()