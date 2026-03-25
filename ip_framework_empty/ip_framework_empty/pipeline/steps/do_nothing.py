import numpy as np

from pipeline.steps.step import Step, StepResult


class DoNothingStep(Step):

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:

        return StepResult(input_img)

    def config_schema(self):
        return {}