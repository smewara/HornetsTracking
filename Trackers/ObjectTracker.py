import numpy as np
import pandas as pd
import supervision as sv
import cv2

class ObjectTracker:

    def __init__(self, model, tracker, bbox=None):
        self.model = model
        self.tracking_results = []
        self.tracker = tracker
        self.bbox = bbox
        self.confidence_threshold = 0.7

    def process_frame(self, frame, frame_id):
        result = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections, annotated_frame = self.tracker.update(detections=detections, 
                                                          frame=frame, 
                                                          confidence_threshold=self.confidence_threshold)
        self.log_detections(detections=detections, frame_id=frame_id)
        return annotated_frame       
    
    def log_detections(self, detections, frame_id):
        if detections is None or len(detections) == 0:
            return

        xyxy = detections.xyxy
        tracker_id = detections.tracker_id
        confidences = detections.confidence
        class_ids = detections.class_id

        for detection_idx in range(len(xyxy)):    
            bbox = xyxy[detection_idx]
            trk_id = tracker_id[detection_idx]
            conf = confidences[detection_idx]
            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]  # Extract bounding box coordinates
            cls_id = class_ids[detection_idx]  # Extract class name (if available)
            width = x_max - x_min
            height = y_max - y_min
            
            # Create a tracking result entry for the current object
            tracking_result = {
                'FrameId': frame_id,
                'Id': trk_id,
                'X': f'{x_min:.6f}',
                'Y': f'{y_min:.6f}',
                'Width': f'{width:.6f}',
                'Height': f'{height:.6f}',
                'Confidence': f'{conf:.6f}',
                'Class': cls_id
            }
            
            # Append the tracking result to the list
            self.tracking_results.append(tracking_result)

    def save_tracking_results(self, output_csv_file):
        # Convert tracking results to a DataFrame
        tracking_results_df = pd.DataFrame(self.tracking_results)
        
        # Save tracking results to a CSV file
        tracking_results_df.to_csv(output_csv_file, index=False)
        
        print(f"Tracking results saved to: {output_csv_file}")

    def track_hornets(self, video_path):
        #frame_generator = sv.get_video_frames_generator(source_path=video_path)
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            processed_frame = self.process_frame(frame=frame, frame_id=frame_id)
            cv2.imshow(f'HornetTracking', processed_frame)
            cv2.putText(img=frame, text=f'{frame_id}', org=(50,50), fontFace= cv2.FONT_HERSHEY_COMPLEX, fontScale= 3, color= (255,255,255), thickness= 3)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        cap.release()
        cv2.destroyAllWindows()