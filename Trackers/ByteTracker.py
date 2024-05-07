import supervision as sv

class ByteTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()
    
    def update(self, detections, frame, confidence_threshold):
        detections = self.tracker.update_with_detections(detections=detections)
        filtered_detections, frame = self._annotate_detections(frame=frame, detections=detections, confidence_threshold=confidence_threshold)
        return filtered_detections, frame
    
    def _annotate_detections(self, frame, detections, confidence_threshold):
        box_annotator = sv.BoundingBoxAnnotator(thickness=1)
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)
        
        if detections is None:
            return
        
        detections = detections[detections.tracker_id != None]
        
        filtered_detections = detections[(detections.confidence != None) &
                                         (detections.confidence >= confidence_threshold)]
        
        labels = [f'#{tracker_id} Conf:{confidence:.2f}' 
                  for xyxy, mask, confidence, class_id, tracker_id, data 
                  in filtered_detections]
        
        frame = box_annotator.annotate(scene=frame, detections=filtered_detections)
        frame = label_annotator.annotate(scene=frame, detections=filtered_detections, labels=labels)

        return filtered_detections, frame
