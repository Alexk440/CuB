import numpy as np

from pipeline.steps.step import Step, StepResult


class SlowConvStep(Step):

    def __init__(self, filter_kernel: np.ndarray):
        self.filter_kernel = filter_kernel

    def apply(self, input_img: np.ndarray, config: dict = None) -> StepResult:

        # iterate over all channel dimensions.
        num_channels = input_img.shape[2]
        output_imgs = []

        x_kernel, y_kernel = self.filter_kernel.shape
        x_center = x_kernel // 2
        y_center = y_kernel // 2

        for i in range(num_channels):

            output_img = np.empty_like(input_img[:, :, i])

            def get_pixel(x, y):
                # Zero-Padding
                if x < 0 or x >= input_img.shape[0] or y < 0 or y >= input_img.shape[1]:
                    return 0
                else:
                    return input_img[int(x), int(y), i]

            for x in range(input_img.shape[0]):
                for y in range(input_img.shape[1]):
                    new_pixel = 0
                    for u in range(x_kernel):
                        for v in range(y_kernel):
                            x_offset = u - x_center
                            y_offset = v - y_center
                            new_pixel += self.filter_kernel[u, v] * get_pixel(x - x_offset, y - y_offset)

                    output_img[x, y] = new_pixel

            output_imgs.append(output_img)

        combined_output_img = np.stack(output_imgs, axis=2)

        return StepResult(combined_output_img)

    def config_schema(self):
        return {}
