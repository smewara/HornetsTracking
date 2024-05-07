import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd

class DeepSortTracker:
    def __init__(self):
        self.tracker = DeepSort()

    def update(self, detections, frame, confidence_threshold):
        results = []
        xyxy = detections.xyxy
        confs = detections.confidence
        class_ids = detections.class_id

        for detection_idx in range(len(xyxy)):    
            # Extract attributes for the current detection
            bbox = xyxy[detection_idx]
            confidence = confs[detection_idx]
            cls_id = class_ids[detection_idx]
            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]  # Extract bounding box coordinates
            
            results.append([[int(x_min), int(y_min), int(x_max)-int(x_min), int(y_max)-int(y_min)], confidence, int(cls_id)])
        tracks = self.tracker.update_tracks(raw_detections=results, frame=frame)
        filtered_detections, frame = self._annotate_detections(tracks=tracks, frame=frame, confidence_threshold=confidence_threshold)
        return filtered_detections, frame

    def _annotate_detections(self, tracks, frame, confidence_threshold):
        filtered_detections = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            class_id = track.det_class
            conf = track.det_conf
            bboxes = track.to_ltrb()
            x_min, y_min, x_max, y_max = int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3])

            if track_id is None or conf is None or conf < confidence_threshold:
                continue

            filtered_detections.append({
                'xyxy': bboxes,
                'tracker_id': track_id,
                'confidence': conf,
                'class_id': class_id
            })

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,0,0), 1)
            cv2.putText(frame, f'#{track_id} Conf: {conf:.2f}', (x_min + 2, y_min - 2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)

        return pd.DataFrame(filtered_detections), frame