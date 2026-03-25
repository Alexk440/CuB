import numpy as np
import scipy as scipy

from scipy import ndimage
from pipeline.steps.step import Step, StepResult, StepWrapper


class FastConvStep(Step):

    def __init__(self, filter_kernel: np.ndarray):
        self.filter_kernel = filter_kernel

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:

        # iterate over all channel dimensions.
        num_channels = input_img.shape[2]
        output_imgs = []

        for i in range(num_channels):

            # TODO: replace this line with the convolution operation.
            output_img = np.empty_like(input_img[:, :, i])

            output_imgs.append(output_img)

        combined_output_img = np.stack(output_imgs, axis=2)

        return StepResult(combined_output_img)

    def config_schema(self):
        return {}
