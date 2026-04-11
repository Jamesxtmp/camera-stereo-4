import cv2
from cv2_enumerate_cameras import enumerate_cameras


CAMERA_BACKENDS = [
    ("Auto", cv2.CAP_ANY),
    ("DirectShow", cv2.CAP_DSHOW),
    ("Media Foundation", cv2.CAP_MSMF),
]


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


def camera_delivers_frames(cap, attempts=8):
    for _ in range(attempts):
        ret, frame = cap.read()
        if ret and frame is not None:
            return True
    return False


def abrir_camara_por_indice(
    indice,
    backend,
    width=None,
    height=None,
    validate_frames=True,
    frame_attempts=8,
):
    cap = cv2.VideoCapture(indice, backend)
    if not cap.isOpened():
        cap.release()
        return None

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if validate_frames and not camera_delivers_frames(cap, attempts=frame_attempts):
        cap.release()
        return None

    return cap


def abrir_camara_con_fallback(
    indice,
    backend_order=None,
    width=None,
    height=None,
    validate_frames=True,
    frame_attempts=8,
):
    orden = backend_order or CAMERA_BACKENDS

    for backend_name, backend_id in orden:
        cap = abrir_camara_por_indice(
            indice=indice,
            backend=backend_id,
            width=width,
            height=height,
            validate_frames=validate_frames,
            frame_attempts=frame_attempts,
        )
        if cap is not None:
            return cap, backend_name, backend_id

    raise RuntimeError(f"No se pudo abrir la camara en indice {indice} con ningun backend")
