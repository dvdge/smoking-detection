"""YOLO11 ONNX 推理：等比例缩放、坐标还原和重复框抑制。"""
import ast
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")


class OnnxDetector:
    """CPU inference for YOLOv8/11 detection ONNX with [1,4+classes,N] output."""
    def __init__(self, model_path, positive_names, confidence, iou):
        if not 0 < confidence <= 1 or not 0 < iou <= 1:
            raise ValueError("confidence and iou must be in (0, 1]")
        if not Path(model_path).is_file():
            raise ValueError("缺少模型，请先运行 python download_model.py，或用 --model 指定 ONNX 文件。")
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = {int(k): str(v) for k, v in ast.literal_eval(meta["names"]).items()}
        input_node = self.session.get_inputs()[0]
        if input_node.type != "tensor(float)" or len(input_node.shape) != 4:
            raise ValueError("Only float32 NCHW ONNX models are supported")
        _, channels, height, width = input_node.shape
        if channels != 3 or not isinstance(height, int) or not isinstance(width, int):
            raise ValueError("This example requires a fixed input shape with 3 channels")
        self.height, self.width = height, width
        self.input_name = input_node.name
        desired = {name.strip().casefold() for name in positive_names}
        self.positive_ids = {key for key, name in self.names.items() if name.casefold() in desired}
        if not self.positive_ids:
            raise ValueError(f"No matching smoking class. Actual model classes: {self.names}")
        self.confidence, self.iou = confidence, iou

    def predict(self, frame):
        import cv2
        import numpy as np
        original_h, original_w = frame.shape[:2]
        scale = min(self.width / original_w, self.height / original_h)
        resized_w, resized_h = round(original_w * scale), round(original_h * scale)
        left = (self.width - resized_w) // 2
        top = (self.height - resized_h) // 2
        canvas = np.full((self.height, self.width, 3), 114, dtype=np.uint8)
        canvas[top:top + resized_h, left:left + resized_w] = cv2.resize(frame, (resized_w, resized_h))
        tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        output = self.session.run(None, {self.input_name: tensor})[0]
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 4 + len(self.names):
            raise ValueError(f"Unsupported ONNX output shape: {output.shape}")
        rows = output[0].T
        class_ids = rows[:, 4:].argmax(axis=1)
        scores = rows[:, 4:].max(axis=1)
        selected = (scores >= self.confidence) & np.isin(class_ids, list(self.positive_ids))
        rows, scores, class_ids = rows[selected], scores[selected], class_ids[selected]
        if not len(rows):
            return []
        boxes, detections = [], []
        for row, score, class_id in zip(rows, scores, class_ids):
            cx, cy, width, height = map(float, row[:4])
            x1 = max(0.0, (cx - width / 2 - left) / scale)
            y1 = max(0.0, (cy - height / 2 - top) / scale)
            x2 = min(float(original_w), (cx + width / 2 - left) / scale)
            y2 = min(float(original_h), (cy + height / 2 - top) / scale)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            detections.append({"class_id": int(class_id), "label": self.names[int(class_id)],
                               "confidence": float(score), "bbox_xyxy": [x1, y1, x2, y2]})
        if not detections:
            return []
        # Only positive smoking labels are passed to NMS; negative labels cannot trigger an alarm.
        kept = cv2.dnn.NMSBoxes(boxes, [d["confidence"] for d in detections], self.confidence, self.iou)
        return [detections[int(index)] for index in np.asarray(kept).reshape(-1)]
