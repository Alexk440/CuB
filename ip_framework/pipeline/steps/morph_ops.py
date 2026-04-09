import numpy as np
from sympy.strategies.core import switch

from pipeline.steps.step import Step, StepResult, StepWrapper
from skimage.morphology import opening, closing, erosion, dilation


class MorphOpStep(Step):

    def __init__(self, filter_kernel: np.ndarray):
        self.filter_kernel = filter_kernel

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:

        op = config['operation'].lower()

        # iterate over all channel dimensions.
        num_channels = input_img.shape[2]
        output_imgs = []

        for i in range(num_channels):
            channel_img = input_img[:, :, i]
            if op == 'erosion':
                output_imgs.append(erosion(channel_img))
            elif op == 'dilation':
                output_imgs.append(dilation(channel_img))
            elif op == 'closing':
                output_imgs.append(closing(channel_img))
            elif op == 'opening':
                output_imgs.append(opening(channel_img))

        combined_output_img = np.stack(output_imgs, axis=2)

        return StepResult(combined_output_img)

    def config_schema(self):
        return {
            'type': 'object',
            'properties': {
                'operation': {'type': 'string', 'default': 'dilation'}
            },
            'required': ['operation']
        }
