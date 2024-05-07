from Trackers.BOTSortTracker import BOTSortTracker
from Trackers.ByteTracker import ByteTracker
from Trackers.DeepSortTracker import DeepSortTracker
from Model.ObjectDetector import ObjectDetector
import os
from Trackers.ObjectTracker import ObjectTracker
from Inference.MOTMetrics import MOTMetrics
from Trackers.TrackerEnum import TrackerEnum

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train_model():
    model = ObjectDetector()
    model.train_model(epochs=50)

def model_tracking(model_path, tracker, video_path, results_file):
    model = ObjectDetector(model_path=model_path).get_model()
    tracker = ObjectTracker(model=model, tracker=tracker)
    tracker.track_hornets(video_path=video_path)
    tracker.save_tracking_results(results_file)

def compare_results(ground_truth_file, model_tracking_file, results_file):
    motMetrics = MOTMetrics(ground_truth_file, model_tracking_file)
    motMetrics.save_tracking_results(results_file=results_file)

def get_tracker(tracker_name, video_name, model_path):
    if tracker_name == TrackerEnum.BOTSORT:
        botsort_tracker = BOTSortTracker(model_path=model_path)
        botsort_tracker_file = f'Inference/{video_name}_botsort_tracker.csv'
        return botsort_tracker, botsort_tracker_file
    elif tracker_name == TrackerEnum.BYTETRACK:
        byte_tracker = ByteTracker(model_path=model_path)
        byte_tracker_file = f'Inference/{video_name}_byte_tracker.csv'
        return byte_tracker, byte_tracker_file
    elif tracker_name == TrackerEnum.DEEPSORT:
        deep_sort_tracker = DeepSortTracker()
        deep_sort_tracker_file = f'Inference/{video_name}_deepsort_tracker.csv'
        return deep_sort_tracker, deep_sort_tracker_file
    
def main():
    video_name = 'MAH00002'
    video_path = f'Datasource/Hornet_videos/Hornet_Colony_{video_name}.mov'
    model_path = r'Model\train4\weights\best.pt'
    ground_truth_file = r'Inference\MAH00002_gt.txt'
    tracker_name = TrackerEnum.BOTSORT
    metrics_results_file = f'Inference/{tracker_name}_motmetrics_results.csv'

    tracker, tracker_file = get_tracker(tracker_name=tracker_name, 
                                        video_name=video_name,
                                        model_path=model_path)

    #Utils.copy_images_labels(dataset_dir=dataset_dir, video_name=video_name, images_dir=images_dir, labels_dir=labels_dir)

    #train_model()

    model_tracking(model_path=model_path, tracker=tracker, 
                   video_path=video_path, 
                   results_file=tracker_file)

    compare_results(ground_truth_file=ground_truth_file, 
                    model_tracking_file=tracker_file,
                    results_file=metrics_results_file)

if __name__ == "__main__":
    main()