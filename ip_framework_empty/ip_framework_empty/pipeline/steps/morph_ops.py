import numpy as np
from pipeline.steps.step import Step, StepResult, StepWrapper


class MorphOpStep(Step):

    def __init__(self, filter_kernel: np.ndarray):
        self.filter_kernel = filter_kernel

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:

        op = config['operation'].lower()

        # iterate over all channel dimensions.
        num_channels = input_img.shape[2]
        output_imgs = []

        # TODO

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
