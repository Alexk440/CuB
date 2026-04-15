import torch
import numpy as np
from torchvision import transforms
from torchvision.models.segmentation import (
    FCN_ResNet101_Weights,
    DeepLabV3_ResNet101_Weights,
    deeplabv3_resnet101,
    fcn_resnet101,
)

from pipeline.steps.step import Step, StepResult


class SemanticSegmentationStep(Step):
    def __init__(self):
        self.models = {}

    def load_model(self, model_name):
        if model_name == 'fcn':
            model = fcn_resnet101(weights=FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
        elif model_name == 'deeplab':
            model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
        else:
            raise ValueError(f"Unknown semantic segmentation model '{model_name}'. Expected 'fcn' or 'deeplab'.")

        model.eval()
        self.models[model_name] = model
        return model

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:

        model_name = config.get('model', 'fcn').lower()

        model = self.models.get(model_name)
        if model is None:
            model = self.load_model(model_name)

        rgb_img = np.array(input_img, dtype=np.uint8, copy=True)

        input_tensor = transforms.ToTensor()(rgb_img)

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
                'model': {
                    'type': 'string',
                    'default': 'fcn',
                    'enum': ['fcn', 'deeplab']
                }
            },
            'required': ['model']
        }
