"""Медианный фильтр с переменным размером ядра: реализация через OpenCV и на чистом Python."""

import cv2
import numpy as np


def check_ksize(ksize):
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError("Размер ядра должен быть нечётным числом не меньше 3")


def median_opencv(image, ksize):
    """Медианный фильтр встроенной функцией OpenCV."""
    check_ksize(ksize)
    return cv2.medianBlur(image, ksize)


def median_native(image, ksize):
    """Медианный фильтр на чистом Python (без NumPy и OpenCV в самом алгоритме).

    Края дополняются повторением крайних пикселей, как в OpenCV (BORDER_REPLICATE).
    Цветное изображение обрабатывается по каналам независимо.
    """
    check_ksize(ksize)
    if image.ndim == 3:
        channels = [median_native(image[:, :, c], ksize) for c in range(image.shape[2])]
        return np.dstack(channels)

    r = ksize // 2
    mid = ksize * ksize // 2
    rows = image.tolist()
    height = len(rows)

    # дополняем каждую строку слева и справа на r пикселей
    padded = [[row[0]] * r + row + [row[-1]] * r for row in rows]
    width = len(rows[0])

    result = []
    for y in range(height):
        # строки окна; за верхней и нижней границей берём крайние строки
        window_rows = [padded[min(max(y + dy, 0), height - 1)] for dy in range(-r, r + 1)]
        out_row = []
        for x in range(width):
            window = []
            for row in window_rows:
                window.extend(row[x:x + ksize])
            window.sort()
            out_row.append(window[mid])
        result.append(out_row)

    return np.array(result, dtype=image.dtype)
