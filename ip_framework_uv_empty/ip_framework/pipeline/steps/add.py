from typing import List, Tuple

import numpy as np
import scipy as scipy

from scipy import ndimage
from pipeline.steps.step import Step, StepResult, StepWrapper


class AddStep(Step):

    def __init__(self, step: Step):
        self.step = step

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:
        step_result = StepWrapper(self.step).apply(input_img, config['step'])
        if step_result.error_occured():
            return StepResult(None, step_result.error)

        sum_img = step_result.output_img + input_img

        return StepResult(sum_img)

    def config_schema(self):
        return \
            {
                'type': 'object',
                'properties': {
                    'step': self.step.config_schema()
                }
            }
