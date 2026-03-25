
import numpy as np
import cv2 as cv

from pipeline.steps.step import Step, StepResult

class Canny(Step):

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:
        x = input_img.shape[0]
        y = input_img.shape[1]
        inp = np.delete(input_img, 3, 2).astype('uint8')
        canny = cv.Canny(inp, config['l'], config['h']).reshape(x, y, 1).astype('float')
        stacked = np.stack([canny, canny, canny], axis=2)
        return StepResult(stacked)

    def config_schema(self):
        return {
            'type': 'object',
            'properties': {

                'h': {'type': 'number', 'minimum': -255.0, 'maximum': 255.0, 'default': 100.0},
                'l': {'type': 'number', 'minimum': -255.0, 'maximum': 255.0, 'default': 200.0},

            }
        }