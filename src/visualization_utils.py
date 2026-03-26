import cv2

def obtener_indice_mas_cercana(detections):
    valid_distances = [
        (i, d["distance"])
        for i, d in enumerate(detections)
        if d.get("distance") is not None
    ]
    if not valid_distances:
        return None
    return min(valid_distances, key=lambda t: t[1])[0]


def dibujar_detecciones(frame, detections, nearest_idx=None, unit_label="u"):
    preview = frame.copy()

    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["box"]
        is_nearest = i == nearest_idx
        color = (0, 0, 255) if is_nearest else (0, 200, 255)

        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)

        label = f"Lata {d['conf']:.2f}"
        if d.get("distance") is not None:
            label += f" | {d['distance']:.2f} {unit_label}"
        else:
            label += " | sin distancia"

        cv2.putText(
            preview,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    if nearest_idx is not None:
        nearest_d = detections[nearest_idx]["distance"]
        cv2.putText(
            preview,
            f"Lata mas cercana: {nearest_d:.2f} {unit_label}",
            (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    return preview