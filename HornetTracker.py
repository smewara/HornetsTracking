from Trackers.BOTSortTracker import BotSortTracker
from Trackers.ByteTracker import ByteTracker
from Trackers.DeepSortTracker import DeepSortTracker
from Model.ObjectDetector import ObjectDetector
from Trackers.ObjectTrackManager import ObjectTrackManager
from Inference.MOTMetrics import MOTMetrics
from Trackers.TrackerEnum import TrackerEnum
from Utils import Utils
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def create_train_data():
    train_dataset_dir = r'Datasets\train_DS'
    train_images_dir = r'Datasets\train\images'
    train_labels_dir = r'Datasets\train\labels'
    val_dataset_dir = r'Datasets\val_DS'
    val_images_dir = r'Datasets\val\images'
    val_labels_dir = r'Datasets\val\labels'
    print('Creating training images and labels')
    Utils.copy_images_labels(dataset_dir=train_dataset_dir, images_dir=train_images_dir, labels_dir=train_labels_dir)
    print('Creating validation images and labels')
    Utils.copy_images_labels(dataset_dir=val_dataset_dir, images_dir=val_images_dir, labels_dir=val_labels_dir)
    print('Successfully created images and labels')

def train_model():
    model = ObjectDetector()
    model.train_model(epochs=10)

def model_tracking(track_manager, tracker_name, video_path, results_file):
    track_manager = ObjectTrackManager(tracker=track_manager)
    track_manager.track_hornets(video_path=video_path, tracker_name=tracker_name)
    track_manager.save_tracking_results(results_file)

def model_detect(model_path, video_path):
    objectDetector = ObjectDetector(model_path=model_path)
    objectDetector.detect_hornets(video_path)

def calculate_mot_metrics(ground_truth_file, model_tracking_file, results_file):
    motMetrics = MOTMetrics(ground_truth_file, model_tracking_file)
    motMetrics.save_tracking_results(results_file=results_file)

def get_tracker(tracker_name, video_name, model_path):
    if tracker_name == TrackerEnum.BOTSORT:
        botsort_tracker = BotSortTracker(model_path=model_path)
        botsort_tracker_file = f'Inference/{video_name}_botsort_tracker.csv'
        return botsort_tracker, botsort_tracker_file
    elif tracker_name == TrackerEnum.BYTETRACK:
        byte_tracker = ByteTracker(model_path=model_path)
        byte_tracker_file = f'Inference/{video_name}_byte_tracker.csv'
        return byte_tracker, byte_tracker_file
    elif tracker_name == TrackerEnum.DEEPSORT:
        deep_sort_tracker = DeepSortTracker(model_path=model_path)
        deep_sort_tracker_file = f'Inference/{video_name}_deepsort_tracker.csv'
        return deep_sort_tracker, deep_sort_tracker_file
    
def track_and_log_metrics(model_path, tracker_name, video_name, video_path, ground_truth_file):
    tracker, tracker_file = get_tracker(tracker_name=tracker_name, video_name=video_name, model_path=model_path)

    metrics_results_file = f'Inference/{tracker_name}_motmetrics_results.csv'

    model_tracking(track_manager=tracker, 
                   tracker_name=tracker_name,
                   video_path=video_path, 
                   results_file=tracker_file)

    calculate_mot_metrics(ground_truth_file=ground_truth_file, 
                    model_tracking_file=tracker_file,
                    results_file=metrics_results_file)
    
def main():
    video_name = 'MAH00002'
    video_path = f'Datasource/Hornet_videos/Hornet_Colony_{video_name}.mov'
    model_path = r'Model\train\weights\best.pt'
    ground_truth_file = r'Inference\MAH00002_gt.txt'
    tracker_name = TrackerEnum.BOTSORT
    
    #create_train_data()
    #train_model()
    #model_detect(model_path=model_path, video_path=video_path)
    track_and_log_metrics(model_path=model_path, tracker_name=tracker_name, video_name=video_name, video_path=video_path, ground_truth_file=ground_truth_file)

if __name__ == "__main__":
    main()