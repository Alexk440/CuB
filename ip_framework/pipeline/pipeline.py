import logging
import os
import time

import numpy as np
from skimage.io import imsave
from typing import Optional, List, Tuple
from pipeline.steps.step import Step, StepResult, StepWrapper

logger = logging.getLogger(__name__)

class PipelineResult:

    def __init__(self, error: Optional[Exception] = None):
        self.step_results = []
        self.error = error

    def error_occured(self) -> bool:
        return self.error is not None

    def successful_execution(self) -> bool:
        return self.error is None

    def add_step_result(self, step_result: StepResult):
        self.step_results.append(step_result)


class Pipeline:

    def __init__(self, name: str, steps: List[Tuple[str, Step, dict]]):
        self.steps = steps
        self.name = name

    def execute(self) -> PipelineResult:
        if not self.steps:
            return PipelineResult(ValueError('There are no steps to process.'))

        pipeline_result = PipelineResult()
        current_input_img = None

        for (step_name, step, config) in self.steps:

            logging.info(f'Executing step "{step_name}" of type "{type(step).__name__}" with config {config}...')

            start_time = time.time()

            step_result = StepWrapper(step).apply(current_input_img, config)

            duration = time.time() - start_time

            pipeline_result.add_step_result(step_result)
            if step_result.error_occured():
                logging.info(f'An error occurred while executing step "{step_name}": {step_result.error}.')

                pipeline_result.error = RuntimeError(f'Step "{step_name}" failed. Reason: {step_result.error}')
                break
            else:
                logging.info(f'Successfully executed step "{step_name}" in {duration * 1000:.1f} ms.')

                current_input_img = step_result.output_img

        return pipeline_result


def save_pipeline_result(pipeline: Pipeline, pipeline_result: PipelineResult, path: str):
    if pipeline_result.error_occured():
        return

    try:
        folder = str(round(time.time() * 1000))
        folder_path = os.path.join(path, folder)
        os.mkdir(folder_path)

        for i, step_result in enumerate(pipeline_result.step_results):
            # https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
            imsave(os.path.join(folder_path, pipeline.steps[i][0] + '.png'), step_result.output_img.astype(np.uint8))
            logging.info(f'Successfully saved pipeline results to directory "{folder_path}".')

    except Exception as e:
        logging.error(f'Could not save pipeline results to directory "{folder_path}". Reason: {str(e)}')
