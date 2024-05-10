from abc import ABC, abstractmethod
import cv2
import pandas as pd
import supervision as sv

class ObjectTracker(ABC):
    @abstractmethod
    def track(self, frame):
        pass
    
    @abstractmethod 
    def _annotate_detections(self, frame, detections):
        pass

