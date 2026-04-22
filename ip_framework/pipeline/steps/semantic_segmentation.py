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
        if self.models.get(model_name) is not None:
            return self.models[model_name]
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

        model_name = config.get('model')

        model = self.load_model(model_name)

        if input_img.shape[-1] == 4:
            input_img = input_img[..., :3]

        rgb_img = np.array(input_img, dtype=np.uint8, copy=True)

        input_tensor = transforms.ToTensor()(rgb_img).unsqueeze(0)

        # Map the segmentation mask to colors.
        seg_mask_color = self.visualize_segmentation_mask(model(input_tensor))

        # Return the color segmentation mask as the output.
        return StepResult(seg_mask_color)

    def visualize_segmentation_mask(self, seg_mask: np.ndarray):
        colors = [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]

        predictions = seg_mask['out'][0]
        seg_mask_indices = torch.argmax(predictions, dim=0)

        h, w = seg_mask_indices.shape
        color_mask = np.zeros((h, w, 3))

        for class_idx, color in enumerate(colors):
            color_mask[seg_mask_indices == class_idx] = color

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
