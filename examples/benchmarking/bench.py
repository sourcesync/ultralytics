# Imports
import ultralytics
from ultralytics import YOLO
import time
import os
import statistics
import traceback

#
# Config
#
MODEL_PATH = "/Users/cuongwilliams/Projects/GSI/model_zoo/examples/yolo_val_bench/skokate/experiment_yolo_v5/runs/detect/train/weights/best.pt"
YAML_PATH  = "/Users/cuongwilliams/Projects/GSI/model_zoo/examples/yolo_val_bench/skokate/test_fake_rp.yaml"
BATCH_SIZE = 1
VERBOSE    = True

# Turn on/off ultralytics verbosity (such as progress bar)
if VERBOSE: os.environ["YOLO_VERBOSE"]="True"
else: os.environ["YOLO_VERBOSE"]="False"

# Initialize YOLO model from weights
m= YOLO( model=MODEL_PATH)

def on_val_end(predictor):
    '''This function will get called after all images are processed'''
    try:
        global computed_metrics, speed
        print("predictor=", predictor.profilers['inference'].timings)
        computed_metrics = predictor.metrics.results_dict
        speed = predictor.speed
    except:
        traceback.print_exc()

m.add_callback("on_val_end", on_val_end)

stats = m.val( data=YAML_PATH, validator=ultralytics.models.yolo.detect.val.DetectionValidator, batch=BATCH_SIZE)
print("Final speed=", stats.speed)
print("Final metrics=", stats.results_dict)
