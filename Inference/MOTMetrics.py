import motmetrics as mm
import numpy as np
import pandas as pd

class MOTMetrics:
    def __init__(self, ground_truth_file, tracker_results_file):
        self.gt = pd.read_csv(ground_truth_file).sort_values(by=['FrameId', 'Id'], ascending=True)
        self.dt = pd.read_csv(tracker_results_file).sort_values(by=['FrameId', 'Id'], ascending=True)
    
    def save_tracking_results(self, results_file):  
        key = ["FrameId", "Id"]      
        self.gt.set_index(key, inplace=True, drop=False, append=False)
        self.dt.set_index(key, inplace=True, drop=False, append=False)
        cols = ['X', 'Y', 'Width', 'Height']

        mh = mm.metrics.create()
        acc = mm.utils.compare_to_groundtruth(self.gt, self.dt, 'iou', distth=0.5)

        mh = mm.metrics.create()
        metrics = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics)
        print('\n')
        for key in mm.metrics.motchallenge_metrics:
            print(f"{key}:{metrics.loc[0][key]}\n") 

        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(results_file, index=False)

        print(f"Tracking results saved to: {results_file}")

        

