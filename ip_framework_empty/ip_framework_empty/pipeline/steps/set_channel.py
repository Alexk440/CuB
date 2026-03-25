import numpy as np

from pipeline.steps.step import Step, StepResult


class SetChannelToConstStep(Step):

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:

        channel_index = config['channel_index']
        value = config['value']

        if channel_index < input_img.shape[2]:
            output_img = input_img.copy()
            output_img[:, :, channel_index] = value
            return StepResult(output_img)
        else:
            return StepResult(input_img)

    def config_schema(self):
        return {'type': 'object',
                'properties': {
                    'value': {'type': 'number', 'minimum': 0.0, 'maximum': 255.0, 'default': 0.0, },
                    'channel_index': {'type': 'integer', 'minimum': 0, 'maximum': 255, 'default': 0, }
                }
                }
