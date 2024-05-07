from ultralytics.trackers.bot_sort import BOTrack
import supervision as sv
import ultralytics.trackers as t

class BotSortTracker:
    def __init__(self):
        t.BOTSORT()
        self.tracker = BOTrack()
    
    def update(self, detections, frame):
        self.tracker.update(new_track=detections)
        frame = self._annotate_detections(frame=frame, detections=detections)
        return detections, frame
    
    def _annotate_detections(self, frame, detections):
        box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)
        
        if detections.tracker_id is not None:
            labels = [f'#{tracker_id}'
                    for tracker_id
                    in detections.tracker_id]
            
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        return frame
