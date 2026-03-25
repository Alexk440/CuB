import numpy as np

from skimage import io
from pipeline.steps.step import Step, StepResult


class LoadFileStep(Step):

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:

        return StepResult(io.imread(config['file']).astype(np.float32))

    def config_schema(self):
        return {
            'type': 'object',
            'properties': {
                'file': {'type': 'string'}
            },
            'required': ['file']
        }
