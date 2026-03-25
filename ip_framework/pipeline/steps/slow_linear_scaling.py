import numpy as np

from pipeline.steps.step import Step, StepResult


class SlowLinearScalingStep(Step):

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:
        c1 = config['c1']
        c2 = config['c2']

        output_img = input_img.copy()

        # TODO

        return StepResult(output_img)

    def config_schema(self):
        return {
            'type': 'object',
            'properties': {

                'c1': {'type': 'number', 'minimum': -255.0, 'maximum': 255.0, 'default': 0.0},
                'c2': {'type': 'number', 'minimum': -10.0, 'maximum': 10.0, 'default': 1.0},

            }
        }
