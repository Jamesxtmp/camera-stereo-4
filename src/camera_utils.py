import cv2
from cv2_enumerate_cameras import enumerate_cameras


def buscar_indice_camara(nombre_objetivo):
    for camera_info in enumerate_cameras():
        if nombre_objetivo in camera_info.name:
            return camera_info.index
    return None


def abrir_camara_estereo(nombre_objetivo, width, height):
    indice = buscar_indice_camara(nombre_objetivo)
    if indice is None:
        raise RuntimeError(f"No se encontro la camara: {nombre_objetivo}")

    cap = cv2.VideoCapture(indice)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara en indice {indice}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap
