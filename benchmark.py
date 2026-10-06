"""Сравнение быстродействия реализаций медианного фильтра."""

import csv
import os
import platform
import time

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from median_filter import median_native, median_opencv  # noqa: E402

SIZES = [128, 256, 512, 1024]   # опыт 1: меняем размер изображения при ядре 5x5
SIZE_KSIZE = 5
KSIZES = [3, 5, 7, 9, 11, 15, 17, 21]   # опыт 2: меняем размер ядра на изображении 512x512
KSIZE_SIZE = 512
OPENCV_REPEATS = 20
NATIVE_REPEATS = 3
SERIES = [("OpenCV", 3, "#2a78d6", "o"), ("OpenCV, 1 поток", 4, "#1baf7a", "s"), ("Native", 5, "#eb6834", "^")]


def measure(func, image, ksize, repeats):
    """Лучшее время из нескольких запусков, в секундах."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        func(image, ksize)
        best = min(best, time.perf_counter() - start)
    return best


def run(image, ksize):
    threads = cv2.getNumThreads()
    t_cv = measure(median_opencv, image, ksize, OPENCV_REPEATS)
    cv2.setNumThreads(1)
    t_cv1 = measure(median_opencv, image, ksize, OPENCV_REPEATS)
    cv2.setNumThreads(threads)
    t_native = measure(median_native, image, ksize, NATIVE_REPEATS)
    print(f"{image.shape[1]}x{image.shape[0]}, ядро {ksize}: OpenCV {t_cv * 1000:.3f} мс, "
          f"OpenCV 1 поток {t_cv1 * 1000:.3f} мс, Native {t_native:.3f} с, разница x{t_native / t_cv:.0f}")
    return t_cv, t_cv1, t_native


def plot(x, rows, xlabel, title, path):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for label, column, color, marker in SERIES:
        values = [row[column] for row in rows]
        ax.plot(x, values, marker=marker, markersize=6, linewidth=2, color=color, label=label)
        ax.annotate(label, (x[-1], values[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10, color="#333333")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Время, с (логарифмическая шкала)")
    ax.set_title(title)
    ax.grid(color="#e5e5e5", linewidth=0.8, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(x[0] - (x[-1] - x[0]) * 0.04, x[-1] + (x[-1] - x[0]) * 0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    os.makedirs("results", exist_ok=True)
    print(f"{platform.platform()}, Python {platform.python_version()}, "
          f"NumPy {np.__version__}, OpenCV {cv2.__version__}, потоков OpenCV: {cv2.getNumThreads()}")
    print("KleidiCV HAL в сборке OpenCV:", "kleidicv_hal" in cv2.getBuildInformation())

    base = cv2.imread("images/camera_noisy.png", cv2.IMREAD_GRAYSCALE)
    median_opencv(base, 3)  # прогрев OpenCV

    size_rows = []
    for size in SIZES:
        image = cv2.resize(base, (size, size), interpolation=cv2.INTER_NEAREST)
        size_rows.append(["size", size, SIZE_KSIZE, *run(image, SIZE_KSIZE)])

    kernel_rows = []
    image = cv2.resize(base, (KSIZE_SIZE, KSIZE_SIZE), interpolation=cv2.INTER_NEAREST)
    for k in KSIZES:
        kernel_rows.append(["kernel", KSIZE_SIZE, k, *run(image, k)])

    with open("results/benchmark.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "size", "ksize", "opencv_s", "opencv_1thread_s", "native_s", "ratio"])
        for row in size_rows + kernel_rows:
            writer.writerow([*row[:3], f"{row[3]:.6f}", f"{row[4]:.6f}", f"{row[5]:.4f}", round(row[5] / row[3])])

    plot(SIZES, size_rows, "Сторона изображения, пикс.",
         f"Время работы от размера изображения (ядро {SIZE_KSIZE}x{SIZE_KSIZE})", "results/benchmark_size.png")
    plot(KSIZES, kernel_rows, "Размер ядра",
         f"Время работы от размера ядра (изображение {KSIZE_SIZE}x{KSIZE_SIZE})", "results/benchmark_kernel.png")


if __name__ == "__main__":
    main()
