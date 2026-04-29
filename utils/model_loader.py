import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import cv2
import numpy as np

# 关键：PyTorch 安全白名单（新版本依然需要）
torch.serialization.add_safe_globals([DetectionModel])

class Detector:
    def __init__(self, scratch_path, missing_path,
                 scratch_conf=0.25, missing_conf=0.25):
        self.scratch_model = YOLO(scratch_path)
        self.missing_model = YOLO(missing_path)
        self.scratch_conf = scratch_conf
        self.missing_conf = missing_conf

    def set_scratch_conf(self, conf):
        self.scratch_conf = conf

    def set_missing_conf(self, conf):
        self.missing_conf = conf

    def detect_both(self, image):
        # 划痕检测（使用独立阈值）
        scratch_results = self.scratch_model(image, conf=self.scratch_conf)[0]
        scratch_boxes = scratch_results.boxes
        scratch_count = len(scratch_boxes) if scratch_boxes is not None else 0

        # 漏装检测（使用独立阈值）
        missing_results = self.missing_model(image, conf=self.missing_conf)[0]
        missing_boxes = missing_results.boxes
        missing_count = len(missing_boxes) if missing_boxes is not None else 0

        # 绘制结果：先画划痕（默认颜色），再画漏装（红色）
        combined_img = scratch_results.plot()
        if missing_boxes is not None:
            boxes = missing_boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes:
                cv2.rectangle(combined_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                cv2.putText(combined_img, "missing_screw", (box[0], box[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        return combined_img, {"scratch_count": scratch_count, "missing_count": missing_count}
