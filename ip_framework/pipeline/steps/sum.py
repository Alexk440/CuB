from typing import List, Tuple

import numpy as np
import scipy as scipy

from scipy import ndimage
from pipeline.steps.step import Step, StepResult, StepWrapper


class SumStep(Step):

    def __init__(self, steps: Tuple[Step, Step]):
        self.steps = list(steps)

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:

        step_result_0 = StepWrapper(self.steps[0]).apply(input_img, config['summand 1'])
        if step_result_0.error_occured():
            return StepResult(None, step_result_0.error)

        step_result_1 = StepWrapper(self.steps[1]).apply(input_img, config['summand 2'])
        if step_result_1.error_occured():
            return StepResult(None, step_result_1.error)

        sum_img = step_result_0.output_img + step_result_1.output_img

        return StepResult(sum_img)

    def config_schema(self):
        return \
            {
                'type': 'object',
                'properties': {
                    'summand 1': self.steps[0].config_schema(),
                    'summand 2': self.steps[1].config_schema()
                }
            }
