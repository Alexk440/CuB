import numpy as np

from typing import List, Tuple

from pipeline.pipeline import Pipeline
from pipeline.steps.add import AddStep
from pipeline.steps.camera import CameraStep
from pipeline.steps.channel_extraction import ChannelExtractionStep
from pipeline.steps.fast_convolution import FastConvStep
from pipeline.steps.difference import DiffStep
from pipeline.steps.fast_linear_scaling import FastLinearScalingStep
from pipeline.steps.load_file import LoadFileStep
from pipeline.steps.set_channel import SetChannelToConstStep
from pipeline.steps.slow_convolution import SlowConvStep
from pipeline.steps.slow_linear_scaling import SlowLinearScalingStep
from pipeline.steps.subtract import SubtractStep
from pipeline.steps.morph_ops import MorphOpStep
from pipeline.steps.median_filter import MedianFilterStep
from pipeline.steps.yolo_object_detection import YoloObjectDetectionStep


def pipeline_simple():
    steps = [
        ('Load File', LoadFileStep(), {'file': 'img/landscape.png'}),
        ('Select Channels', ChannelExtractionStep(), {'channels': '0003'})
    ]

    return Pipeline('Simple Pipeline', steps)


def pipeline_camera():
    steps = [
        ('Get Camera Frame', CameraStep(), {}),
        ('Select Channels', ChannelExtractionStep(), {'channels': '0003'})
    ]

    return Pipeline('Camera Pipeline', steps)


def pipeline_yolo():
    steps = [
        ('Get Camera Frame', CameraStep(), {}),
        ('Predict', YoloObjectDetectionStep(), {
            'weights': 'models/yolov10n.pt',
            # Path to the YOLOv8 model weights (can be a pre-trained one like 'yolov8n.pt')
            'imgsz': 320,  # Image size for inference
            'conf': 0.5  # Confidence threshold
        })
    ]

    return Pipeline('Yolo Pipeline', steps)

def pipeline_slow_linear_scaling():
    steps = [
        ('Load File', LoadFileStep(), {'file': 'img/landscape.png'}),
        ('Slow Linear Scaling', SlowLinearScalingStep(), {})
    ]

    return Pipeline('Slow Linear Scaling', steps)


def pipeline_fast_linear_scaling():
    steps = [
        ('Load File', LoadFileStep(), {'file': 'img/landscape.png'}),
        ('Fast Linear Scaling', FastLinearScalingStep(), {})
    ]

    return Pipeline('Fast Linear Scaling', steps)


def pipeline_compare_fast_slow_scaling():
    steps = [
        ('Load File', LoadFileStep(), {'file': 'img/landscape.png'}),
        ('Subtract', DiffStep((FastLinearScalingStep(), SlowLinearScalingStep())), {'minuend': {}, 'subtrahend': {}}),
        ('Set Alpha to 255', SetChannelToConstStep(), {'value': 255, 'channel_index': 3})
    ]

    return Pipeline('Compare Fast/Slow Linear Scaling', steps)


def pipeline_compare_fast_slow_convolution():
    kernel = np.array([[-1 / 16, 1 / 16, -1 / 16, -1 / 16],
                       [-1 / 16, -1 / 16, -1 / 16, -1 / 16],
                       [10 / 16, 1 / 16, -1 / 16, -1 / 16],
                       [-1 / 16, -1 / 16, -1 / 16, 10 / 16]])


    steps = [
        ('Load File', LoadFileStep(), {'file': 'img/landscape_small.png'}),
        ('Subtract', DiffStep((FastConvStep(kernel), SlowConvStep(kernel))), {'minuend': {}, 'subtrahend': {}}),
        ('Set Alpha to 255', SetChannelToConstStep(), {'value': 255, 'channel_index': 3})
    ]

    return Pipeline('Compare Fast/Slow Convolution', steps)


def pipeline_slow_convolution():
    steps = [
        ('Load File', LoadFileStep(), {'file': 'img/landscape.png'})
        # TODO: Add next steps.
    ]

    return Pipeline('Slow Convolution', steps)


def pipeline_fast_convolution():
    steps = [
        ('Load File', LoadFileStep(), {'file': 'img/landscape.png'})
        # TODO: Add next steps.

    ]

    return Pipeline('Fast Convolution', steps)

def pipeline_unsharp_masking():
    steps = [
        ('Load Original', LoadFileStep(), {'file': 'img/landscape_small.png'})
        # TODO: Add next steps.
    ]

    return Pipeline('Unsharp Masking', steps)

def pipeline_morph():
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]])

    steps = [
        ('Load File', LoadFileStep(), {'file': 'img/landscape_small.png'}),
        ('Morph', MorphOpStep(kernel), {})
    ]

    return Pipeline('Morph Filter', steps)

def pipeline_median():
    steps = [
        ('Load File', LoadFileStep(), {'file': 'img/landscape.png'}),
        ('Median', MedianFilterStep(), {'n': 3, 'm': 3})
    ]

    return Pipeline('Median Filter', steps)


defined_pipelines: List[Tuple[str, Pipeline]] = [
    pipeline_camera(),
    pipeline_yolo(),
    pipeline_simple(),
    pipeline_slow_convolution(),
    pipeline_fast_convolution(),
    pipeline_slow_linear_scaling(),
    pipeline_fast_linear_scaling(),
    pipeline_compare_fast_slow_scaling(),
    pipeline_compare_fast_slow_convolution(),
    pipeline_unsharp_masking(),
    pipeline_morph(),
    pipeline_median()
]
