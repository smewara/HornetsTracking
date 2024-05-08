import supervision as sv
from ultralytics import YOLO

class ByteTracker:
    def __init__(self, model_path):
        self.model = YOLO(model=model_path)
        self.tracker = sv.ByteTrack()
    
    def track(self, frame, confidence_threshold):
        result = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        # filter detections < confidence threshold from being tracked
        #detections = detections[detections.confidence >= confidence_threshold]
        detections = self.tracker.update_with_detections(detections=detections)
        detections, frame = self._annotate_detections(frame=frame, detections=detections)
        return detections, frame
    
    def _annotate_detections(self, frame, detections):
        box_annotator = sv.BoundingBoxAnnotator(thickness=1)
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)
        
        if detections is None:
            return
        
        detections = detections[detections.tracker_id != None]
        
        labels = [f'#{tracker_id} Conf:{confidence:.2f}' 
                  for xyxy, mask, confidence, class_id, tracker_id, data 
                  in detections]
        
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        return detections, frame
