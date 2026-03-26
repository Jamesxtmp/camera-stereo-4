import json
import os

import cv2
import numpy as np


def cargar_homografia_piso(path):
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h_img_to_world = np.array(data["homography"]["img_to_world"], dtype=np.float32)
    robot_ref_px = tuple(data["reference"]["robot_ref_px"])

    ref_img = np.array([[robot_ref_px]], dtype=np.float32)
    ref_world = cv2.perspectiveTransform(ref_img, h_img_to_world)[0, 0]

    return {
        "h_img_to_world": h_img_to_world,
        "robot_ref_px": robot_ref_px,
        "robot_ref_world": (float(ref_world[0]), float(ref_world[1])),
    }


def estimar_borde_azul_mas_cercano_cm(
    frame_bgr,
    floor_calib,
    hsv_lower,
    hsv_upper,
    roi_y_min_ratio=0.45,
    min_area_px=900,
):
    h, w = frame_bgr.shape[:2]
    y0 = max(0, min(int(h * roi_y_min_ratio), h - 1))

    roi = frame_bgr[y0:, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)
    roi_mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y0:, :] = roi_mask

    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    robot_world = np.array(floor_calib["robot_ref_world"], dtype=np.float32)
    h_img_to_world = floor_calib["h_img_to_world"]

    best_distance = None
    best_point_img = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue

        world_pts = cv2.perspectiveTransform(cnt.astype(np.float32), h_img_to_world).reshape(-1, 2)
        if world_pts.size == 0:
            continue

        deltas = world_pts - robot_world
        dists = np.sqrt(np.sum(deltas * deltas, axis=1))
        idx = int(np.argmin(dists))
        dist_cm = float(dists[idx])

        if best_distance is None or dist_cm < best_distance:
            best_distance = dist_cm
            best_point_img = tuple(int(v) for v in cnt.reshape(-1, 2)[idx])

    return {
        "distance_cm": best_distance,
        "nearest_point_px": best_point_img,
        "mask": full_mask,
    }
