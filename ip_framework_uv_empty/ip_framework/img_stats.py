import numpy as np
import skimage.measure


def compute_img_stats(input_img: np.ndarray):
    return {
        'dims': compute_dims(input_img),
        'mean': compute_mean(input_img),
        'histo': compute_histogram(input_img),
        'entropy': compute_entropy(input_img)
    }


def compute_dims(input_img: np.ndarray) -> tuple[float, float]:
    dims = 0.0, 0.0  # To be replaced.

    # TODO

    return dims


def compute_mean(input_img: np.ndarray) -> list[float]:
    means = [0.0] * input_img.shape[2]  # To be replaced.

    # TODO

    return means


def compute_histogram(input_img) -> list[tuple]:
    num_bins = 256
    num_channels = input_img.shape[2]
    histograms = [(np.zeros(num_bins), np.zeros(num_bins))] * num_channels  # To be replaced.

    # TODO

    return histograms


def compute_entropy(input_img) -> list[float]:
    entropies = [0.0] * input_img.shape[2]  # To be replaced.

    # TODO

    return entropies