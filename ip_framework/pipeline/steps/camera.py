import numpy as np

import cv2
from pipeline.steps.step import Step, StepResult
class CameraStep(Step):
    def apply(self, input_img: np.ndarray = None, config: dict = None) -> StepResult:
        # Open the camera device, default device index is 0
        cap = cv2.VideoCapture(config.get('camera_index', 0))

        # Check if the camera opened successfully
        if not cap.isOpened():
            raise Exception(f"Unable to open camera with index {config.get('camera_index', 0)}")

        # Capture one frame
        ret, frame = cap.read()

        # Convert input image to a format YOLO can process
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Release the camera
        cap.release()

        if not ret:
            raise Exception("Failed to capture frame from camera")

        # Return the captured frame as a StepResult
        return StepResult(frame_rgb.astype(np.float32))

    def config_schema(self):
        return {
            'type': 'object',
            'properties': {
                'camera_index': {'type': 'integer', 'default': 0, 'minimum': 0, 'maximum': 16},
            },
            'required': []
        }