"""Проверка, что нативная реализация даёт тот же результат, что и OpenCV."""

import numpy as np
import pytest

from median_filter import median_native, median_opencv


@pytest.mark.parametrize("ksize", [3, 5, 7, 9, 11, 15])
@pytest.mark.parametrize("shape", [(17, 23), (40, 31, 3)])
def test_native_equals_opencv(ksize, shape):
    image = np.random.default_rng(ksize).integers(0, 256, size=shape, dtype=np.uint8)
    assert np.array_equal(median_native(image, ksize), median_opencv(image, ksize))


def test_salt_and_pepper_removed():
    image = np.full((20, 20), 128, dtype=np.uint8)
    image[5, 5] = 255
    image[10, 12] = 0
    assert np.all(median_native(image, 3) == 128)


@pytest.mark.parametrize("ksize", [1, 2, 4, 0, -3])
def test_wrong_ksize(ksize):
    image = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        median_native(image, ksize)
    with pytest.raises(ValueError):
        median_opencv(image, ksize)
