import numpy as np

from pipeline.steps.step import Step, StepResult, StepWrapper


class SlowConvStep(Step):

    def __init__(self, filter_kernel: np.ndarray):
        self.filter_kernel = filter_kernel

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:

        # iterate over all channel dimensions.
        num_channels = input_img.shape[2]
        output_imgs = []

        # See also https://towardsdatascience.com/convolution-vs-correlation-af868b6b4fb5

        for i in range(num_channels):

            output_img = np.empty_like(input_img[:, :, i])

            # TODO implement convolution for channel i and store result in output_img
            #      input is input_img[:, :, i] for channel i.

            output_imgs.append(output_img)

        combined_output_img = np.stack(output_imgs, axis=2)

        return StepResult(combined_output_img)

    def config_schema(self):
        return {}
