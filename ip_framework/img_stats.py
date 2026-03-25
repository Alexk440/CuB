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
    return float(input_img.shape[0]), float(input_img.shape[1])


def compute_mean(input_img: np.ndarray) -> list[float]:
    return np.mean(input_img, axis=(0, 1)).tolist()


def _map_channels(num_channels, func):

    return [func(i) for i in range(num_channels)]


def compute_histogram(input_img) -> list[tuple]:
    num_bins = 256
    num_channels = input_img.shape[2]
    def _histogram_per_channel(channel):
        return np.histogram(input_img[:, :, channel], bins=num_bins, range=(0,255))
    return _map_channels(num_channels, _histogram_per_channel)


def compute_entropy(input_img) -> list[float]:
    num_channels = input_img.shape[2]
    def _entropy_per_channel(channel):
        return skimage.measure.shannon_entropy(input_img[:, :, channel])
    return _map_channels(num_channels, _entropy_per_channel)
