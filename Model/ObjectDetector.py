import cv2
import pandas as pd
from ultralytics import YOLO
import supervision as sv

class ObjectDetector:
    def __init__(self, model_path = None):
        if model_path is not None:
            self.model = YOLO(model_path)

    def train_model(self, epochs):
        model = YOLO('yolov8n.pt')
        # Drop out of 20% + Early stopping if validation metrics don't improve for consecutively for 10 epochs
        model.train(data='Model/data.yaml', epochs=epochs, dropout=0.4, plots=True, patience=10)

    def load_model(self, model_path):
        self.model = YOLO(model_path)
    
    def get_model(self):
        return self.model
    
    def detect_hornets(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f'\nFPS:{fps}\n')
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            detections, processed_frame = self.process_frame(frame=frame)
            cv2.imshow('YOLOv8_HornetDetector', processed_frame)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        results = self.model(frame, conf=0.6)

        if results is None or len(results[0]) == 0:
            return pd.DataFrame(), frame
        
        detections = sv.Detections.from_ultralytics(results[0])
        if results[0].boxes.id is not None:
            detections.tracker_id = results[0].boxes.id.cpu().numpy().astype(int)  
   
        return self._annotate_detections(frame=frame, detections=detections)
    
    def _annotate_detections(self, frame, detections):
        box_annotator = sv.BoundingBoxAnnotator(thickness=1)
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)
        
        if detections is None:
            return pd.DataFrame(), frame
        
        detections = detections[detections.confidence > 0.6]

        labels = [f'Conf:{confidence:.2f}' 
                  for xyxy, mask, confidence, class_id, tracker_id, data 
                  in detections]
        
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        return detections, frame
