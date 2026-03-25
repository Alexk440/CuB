import torch
import numpy as np
#TODO Add necessary imports.

from pipeline.steps.step import Step, StepResult


class SemanticSegmentationStep(Step):
    def __init__(self):
        self.models = {}

    def load_model(self, model_name):

        # TODO Load model model_name, add it to self.models and return it.
        return None

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:

        model_name = config['model'].lower()

        # TODO Load the model if it hasn't been loaded yet.
        model = None

        # TODO Apply the necessary transformations to input_img and create the input_tensor.
        input_tensor = None

        # Perform the segmentation.
        with torch.no_grad():
            output = model(input_tensor)['out'][0]

        # Convert output to a segmentation mask.
        seg_mask = output.argmax(0).byte().cpu().numpy()

        # Map the segmentation mask to colors.
        seg_mask_color = self.visualize_segmentation_mask(seg_mask)

        # Return the color segmentation mask as the output.
        return StepResult(seg_mask_color)

    def visualize_segmentation_mask(self, seg_mask: np.ndarray):

        # TODO Create color_mask based on seg_mask. Use COCO/VOC-like classes with basic colors.

        color_mask = None
        return color_mask

    def config_schema(self):
        return {
            'type': 'object',
            'properties': {
                'model': {'type': 'string', 'default': 'fcn'}
            },
            'required': ['model']
        }
