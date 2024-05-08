import pandas as pd
import supervision as sv
from ultralytics import YOLO

class ByteTracker:
    def __init__(self, model_path):
        self.model = YOLO(model=model_path)
    
    def track(self, frame, confidence_threshold):
        results = self.model.track(frame, persist=True, tracker='bytetrack.yaml')
        detections, frame = self._annotate_results(frame=frame, results=results)
        return detections, frame
    
    def _annotate_results(self, frame, results):
        if results is None or len(results[0]) == 0:
            return pd.DataFrame(), frame
        
        detections = sv.Detections.from_ultralytics(results[0])
        if results[0].boxes.id is not None:
            detections.tracker_id = results[0].boxes.id.cpu().numpy().astype(int)
        
        return self._annotate_detections(frame=frame, detections=detections)
    
    def _annotate_detections(self, frame, detections):
        box_annotator = sv.BoundingBoxAnnotator(thickness=1)
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)
        
        if detections is None or detections.tracker_id is None:
            return pd.DataFrame(), frame
        
        labels = [f'#{tracker_id} Conf:{confidence:.2f}' 
                  for xyxy, mask, confidence, class_id, tracker_id, data 
                  in detections]
        
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        return detections, frame
