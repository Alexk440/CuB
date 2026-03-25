import numpy as np

import cv2
from ultralytics import YOLO
from pipeline.steps.step import Step, StepResult


class YoloObjectDetectionStep(Step):
    def __init__(self):
        self.model = None

    def load_yolo_model(self, config):
        # Load the YOLO model from the specified config
        self.model = YOLO(config['weights'])

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:
        # Load the YOLO model if it hasn't been loaded yet
        if self.model is None:
            self.load_yolo_model(config)

        # Run inference on the input image
        results = self.model.predict(source=input_img, imgsz=config.get('imgsz', 320), conf=config.get('conf', 0.5),
                                     verbose=False)

        # Get the predicted results in the format we need (e.g., with bounding boxes and labels)
        result_img = results[0].plot()  # Draws bounding boxes and labels directly on the image

        return StepResult(result_img)

    def config_schema(self):
        return {
            'type': 'object',
            'properties': {
                'weights': {'type': 'string'},  # Path to YOLO model weights (can be 'yolov8n.pt' for example)
                'imgsz': {'type': 'integer', 'default': 320},  # Image size for inference
                'conf': {'type': 'number', 'default': 0.5},  # Confidence threshold for detections
            },
            'required': ['weights']
        }
