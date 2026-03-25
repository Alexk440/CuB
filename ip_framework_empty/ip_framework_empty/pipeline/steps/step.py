from abc import abstractmethod
from typing import Optional, List

import numpy as np

from schema.extended_schema_validator import ExtendedValidator


class StepResult:

    def __init__(self, output_img: Optional[np.ndarray] = None, error: Optional[Exception] = None):
        self.output_img = output_img
        self.error = error

    def error_occured(self) -> bool:
        return self.error is not None


class Step:

    def __init__(self):
        self.steps: List[Step] = []

    @abstractmethod
    def apply(self, input_img: np.ndarray, config: Optional[dict] = None) -> StepResult:
        pass

    @abstractmethod
    def config_schema(self):
        pass


class StepWrapper(Step):

    def __init__(self, step: Step):

        self.inner_step = step

    def apply(self, input_img: np.ndarray, config: Optional[dict] = None) -> StepResult:

        if input_img is None:
            # Simply create a black RGB image if there is no input image.
            input_img = np.zeros([128,128,3],dtype=np.uint8)

        if not config:
            config = {}

        try:

            ExtendedValidator(schema=self.inner_step.config_schema()).validate(config)

            return self.inner_step.apply(input_img, config)

        except Exception as ex:

            return StepResult(None, ex)

    def config_schema(self):
        return self.inner_step.config_schema()
