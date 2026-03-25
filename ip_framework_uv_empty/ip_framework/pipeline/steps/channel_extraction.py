import numpy as np
import scipy as scipy

from scipy import ndimage
from pipeline.steps.step import Step, StepResult, StepWrapper


class ChannelExtractionStep(Step):

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:

        channels = [int(x) for x in config['channels']]

        num_channels = input_img.shape[2]
        output_channels = []
        for i in channels:
            if i >= num_channels:
                continue
            output_channels.append(input_img[:, :, i])

        combined_channels = np.stack(output_channels, axis=2)

        return StepResult(combined_channels)

    def config_schema(self):
        return {
            'type': 'object',
            'properties': {
                'channels': {'type': 'string', 'minLength': 1, 'maxLength': 4, 'default': '0123', 'pattern': '^[0-3]+$'}
            },
            'required':['channels']
        }
