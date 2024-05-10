import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import pandas as pd
import supervision as sv
from ultralytics import YOLO

from Trackers.ObjectTracker import ObjectTracker

class DeepSortTracker(ObjectTracker):
    def __init__(self, model_path):
        self.model = YOLO(model=model_path)
        self.tracker = DeepSort(max_age=900, max_iou_distance=0.6)

    def track(self, frame):
        detections = sv.Detections.from_ultralytics(self.model(frame)[0])
        xyxy = detections.xyxy
        confs = detections.confidence
        class_ids = detections.class_id
        results = []

        for detection_idx in range(len(xyxy)):    
            bbox = xyxy[detection_idx]
            confidence = confs[detection_idx]
            cls_id = class_ids[detection_idx]
            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
            
            if confidence <= 0.1:
                continue

            results.append([[int(x_min), int(y_min), int(x_max)-int(x_min), int(y_max)-int(y_min)], confidence, int(cls_id)])
        tracks = self.tracker.update_tracks(raw_detections=results, frame=frame)
        tracked_detections, frame = self._annotate_detections(tracks=tracks, frame=frame)
        return tracked_detections, frame

    def _annotate_detections(self, tracks, frame):
        tracked_detections = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            class_id = track.det_class
            conf = track.det_conf
            bboxes = track.to_ltrb()
            x_min, y_min, x_max, y_max = int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3])

            if track_id is None or conf is None:
                continue

            tracked_detections.append({
                'xyxy': bboxes,
                'tracker_id': track_id,
                'confidence': conf,
                'class_id': class_id
            })

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
            cv2.putText(
                img=frame,
                text=f'#{track_id} Conf: {conf:.2f}',
                org=(x_min, y_min - 2),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255,255,255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        return pd.DataFrame(tracked_detections), frame