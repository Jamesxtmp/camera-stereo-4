from ultralytics import YOLO


def cargar_modelo(ruta_modelo_onnx):
    return YOLO(ruta_modelo_onnx, task="detect")


def obtener_detecciones_latas(model, frame_left_rect, device, conf=0.5, imgsz=None):
    predict_kwargs = {
        "conf": conf,
        "device": device,
        "verbose": False,
    }
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz

    results = model.predict(frame_left_rect, **predict_kwargs)
    boxes = results[0].boxes

    detecciones = []
    if boxes is None or len(boxes) == 0:
        return detecciones

    xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy()

    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = box.tolist()
        detecciones.append({
            "box": (x1, y1, x2, y2),
            "conf": float(confs[i]),
        })

    return detecciones
