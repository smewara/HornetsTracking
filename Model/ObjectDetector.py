from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path = None):
        if model_path is not None:
            self.model = YOLO(model_path)

    def train_model(self, epochs):
        model = YOLO('yolov8m.pt')
        results = model.train(data='Model/data.yaml', epochs=epochs)

    def load_model(self, model_path):
        self.model = YOLO(model_path)
    
    def get_model(self):
        return self.model
    
    def detect_hornets(self, frame):
        detections = self.model(frame)
        results = []
        for detection in detections:
            if detection['confidence'] > self.conf_threshold:
                label = detection['label']
                score = detection['confidence']
                bbox = detection['bbox']
                results.append({
                    'label': label,
                    'confidence': score,
                    'bbox': bbox
                })
        
        return results