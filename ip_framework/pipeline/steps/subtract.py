from typing import List, Tuple

import numpy as np
import scipy as scipy

from scipy import ndimage
from pipeline.steps.step import Step, StepResult, StepWrapper


class SubtractStep(Step):

    def __init__(self, step: Step):
        self.step = step

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:
        step_result = StepWrapper(self.step).apply(input_img, config['step'])
        if step_result.error_occured():
            return StepResult(None, step_result.error)

        if config['input from step']:
            return StepResult((step_result.output_img - input_img))
        else:
            return StepResult((input_img - step_result.output_img))

    def config_schema(self):
        return \
            {
                'type': 'object',
                'properties': {
                    'step': self.step.config_schema(),
                    'input from step': {'type': 'boolean', 'default': False}
                }
            }
