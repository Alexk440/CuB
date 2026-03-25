import numpy as np
import scipy as scipy

from scipy import ndimage
from pipeline.steps.step import Step, StepResult, StepWrapper


class MedianFilterStep(Step):

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:

        # iterate over all channel dimensions.
        num_channels = input_img.shape[2]
        output_imgs = []

        n = config['n']
        m = config['m']

        # TODO

        combined_output_img = np.stack(output_imgs, axis=2)

        return StepResult(combined_output_img)

    def config_schema(self):
        return {'type': 'object',
                'properties': {
                    'n': {'type': 'integer', 'minimum': 3, 'maximum': 7, 'default': 3 },
                    'm': {'type': 'integer', 'minimum': 3, 'maximum': 7, 'default': 3 }
                }
                }
