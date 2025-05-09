from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)  # Load pre-trained YOLOv8 nano model

    def detect(self, frame):
        results = self.model(frame)
        detections = []
        for result in results:
            for box in result.boxes:
                x, y, w, h = box.xywh[0].cpu().numpy()  # Get bounding box
                conf = box.conf.cpu().numpy()  # Confidence score
                class_id = int(box.cls.cpu().numpy())  # Class ID
                detections.append({
                    'bbox': [int(x - w/2), int(y - h/2), int(w), int(h)],
                    'conf': float(conf),
                    'class_id': class_id
                })
        return detections