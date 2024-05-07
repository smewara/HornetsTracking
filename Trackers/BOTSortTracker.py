import pandas as pd
import supervision as sv
from ultralytics import YOLO

class BotSortTracker:
    def __init__(self, model_path):
        self.model = YOLO(model=model_path)
    
    def update(self, detections, frame, confidence_threshold):
        results = self.model.track(frame, persist=True)
        filtered_detections, frame = self._annotate_results(frame=frame, results=results, confidence_threshold=confidence_threshold)
        return filtered_detections, frame
    
    def _annotate_results(self, frame, results, confidence_threshold):
        if results is None or len(results[0]) == 0:
            return pd.DataFrame(), frame
        
        detections = sv.Detections.from_ultralytics(results[0])
        if results[0].boxes.id is not None:
            detections.tracker_id = results[0].boxes.id.cpu().numpy().astype(int)
        
        return self._annotate_detections(frame=frame, detections=detections, confidence_threshold=confidence_threshold)
    
    def _annotate_detections(self, frame, detections, confidence_threshold):
        box_annotator = sv.BoundingBoxAnnotator(thickness=1)
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)
        
        if detections is None or detections.tracker_id is None:
            return pd.DataFrame(), frame
        
        filtered_detections = detections[(detections.confidence != None) &
                                         (detections.confidence >= confidence_threshold)]
        
        labels = [f'#{tracker_id} Conf:{confidence:.2f}' 
                  for xyxy, mask, confidence, class_id, tracker_id, data 
                  in filtered_detections]
        
        frame = box_annotator.annotate(scene=frame, detections=filtered_detections)
        frame = label_annotator.annotate(scene=frame, detections=filtered_detections, labels=labels)
        return filtered_detections, frame
