import cv2
import numpy as np


REQUIRED_CALIB_KEYS = {"mtx_l", "dist_l", "mtx_r", "dist_r", "R1", "R2", "P1", "P2"}


def cargar_calibracion(calibration_file):
    if not calibration_file:
        return None

    try:
        data = np.load(calibration_file)
    except FileNotFoundError:
        return None

    with data:
        keys = set(data.files)
        if not REQUIRED_CALIB_KEYS.issubset(keys):
            return None

        calib_params = {
            "mtx_l": data["mtx_l"],
            "dist_l": data["dist_l"],
            "mtx_r": data["mtx_r"],
            "dist_r": data["dist_r"],
            "R1": data["R1"],
            "R2": data["R2"],
            "P1": data["P1"],
            "P2": data["P2"],
        }

    fx_px = float(calib_params["P1"][0, 0])
    baseline_units = abs(float(calib_params["P2"][0, 3]) / float(calib_params["P2"][0, 0]))
    stereo_ready = fx_px > 0 and baseline_units > 0

    return {
        "params": calib_params,
        "fx_px": fx_px,
        "baseline_units": baseline_units,
        "stereo_ready": stereo_ready,
    }


def crear_mapas_rectificacion(calib_params, image_size):
    map_l_x, map_l_y = cv2.initUndistortRectifyMap(
        calib_params["mtx_l"],
        calib_params["dist_l"],
        calib_params["R1"],
        calib_params["P1"],
        image_size,
        cv2.CV_32FC1,
    )
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(
        calib_params["mtx_r"],
        calib_params["dist_r"],
        calib_params["R2"],
        calib_params["P2"],
        image_size,
        cv2.CV_32FC1,
    )

    return {
        "map_l_x": map_l_x,
        "map_l_y": map_l_y,
        "map_r_x": map_r_x,
        "map_r_y": map_r_y,
    }


def rectificar_frames(left_frame_raw, right_frame_raw, rect_maps):
    left_frame_rect = cv2.remap(
        left_frame_raw,
        rect_maps["map_l_x"],
        rect_maps["map_l_y"],
        cv2.INTER_LINEAR,
    )
    right_frame_rect = cv2.remap(
        right_frame_raw,
        rect_maps["map_r_x"],
        rect_maps["map_r_y"],
        cv2.INTER_LINEAR,
    )
    return left_frame_rect, right_frame_rect


def estimar_disparidad_bbox(left_gray, right_gray, bbox, max_disp=320):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = left_gray.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    roi_w, roi_h = x2 - x1, y2 - y1
    if roi_w < 12 or roi_h < 12:
        return None

    template = left_gray[y1:y2, x1:x2]
    pad_y = max(8, roi_h // 4)
    sy1 = max(0, y1 - pad_y)
    sy2 = min(h, y2 + pad_y)

    sx1 = max(0, x1 - max_disp)
    sx2 = min(w - roi_w, x1 + 24)
    if sx2 <= sx1 or sy2 - sy1 < roi_h:
        return None

    search = right_gray[sy1:sy2, sx1 : sx2 + roi_w]
    if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
        return None

    corr = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(corr)

    if score < 0.35:
        return None

    match_x_left = sx1 + loc[0]
    disparity = float(x1 - match_x_left)
    if disparity <= 1.0:
        return None

    return disparity


def distancia_desde_disparidad(disparity, fx_px, baseline_units):
    if disparity is None or disparity <= 0:
        return None
    return (fx_px * baseline_units) / disparity
