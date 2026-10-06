"""Демонстрация работы фильтра: результаты для разных размеров ядра, сравнение реализаций, PSNR."""

import csv
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from median_filter import median_native, median_opencv  # noqa: E402

KSIZES = [3, 5, 7, 9]
IMAGES = ["camera", "astronaut"]
COLORS = {"camera": "#2a78d6", "astronaut": "#eb6834"}


def show(ax, image, title):
    if image.ndim == 3:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def save_grid(panels, path, cols):
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.2 * rows))
    for ax, (image, title) in zip(np.ravel(axes), panels):
        show(ax, image, title)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs("results", exist_ok=True)
    psnr_rows = []

    for name in IMAGES:
        clean = cv2.imread(f"images/{name}.png", cv2.IMREAD_UNCHANGED)
        noisy = cv2.imread(f"images/{name}_noisy.png", cv2.IMREAD_UNCHANGED)
        noisy_psnr = cv2.PSNR(clean, noisy)
        panels = [(clean, "Исходное"), (noisy, f"Шум 10%, PSNR {noisy_psnr:.2f} дБ")]
        psnr_rows.append([name, 0, round(noisy_psnr, 2), ""])

        for k in KSIZES:
            res_cv = median_opencv(noisy, k)
            res_native = median_native(noisy, k)
            max_diff = int(np.max(cv2.absdiff(res_cv, res_native)))
            psnr = cv2.PSNR(clean, res_native)
            psnr_rows.append([name, k, round(psnr, 2), max_diff])
            cv2.imwrite(f"results/{name}_k{k}.png", res_native)
            panels.append((res_native, f"Ядро {k}x{k}, PSNR {psnr:.2f} дБ"))
            print(f"{name}, ядро {k}: PSNR {psnr:.2f} дБ, макс. разница OpenCV/native = {max_diff}")

        save_grid(panels, f"results/{name}_kernels.png", cols=3)

    # сравнение реализаций на одном изображении
    noisy = cv2.imread("images/camera_noisy.png", cv2.IMREAD_UNCHANGED)
    res_cv = median_opencv(noisy, 5)
    res_native = median_native(noisy, 5)
    diff = cv2.absdiff(res_cv, res_native)
    save_grid([(res_cv, "OpenCV, ядро 5x5"), (res_native, "Native, ядро 5x5"),
               (diff, f"|OpenCV - Native|, максимум {int(diff.max())}")],
              "results/compare_k5.png", cols=3)

    # увеличенный фрагмент: как большое ядро размывает мелкие детали
    clean = cv2.imread("images/camera.png", cv2.IMREAD_UNCHANGED)
    crop = (slice(60, 260), slice(180, 380))
    panels = [(clean[crop], "Исходное (фрагмент)"), (noisy[crop], "С шумом")]
    panels += [(cv2.imread(f"results/camera_k{k}.png", cv2.IMREAD_UNCHANGED)[crop], f"Ядро {k}x{k}")
               for k in (3, 9)]
    save_grid(panels, "results/camera_zoom.png", cols=4)

    with open("results/psnr.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "ksize", "psnr_db", "max_diff_opencv_native"])
        writer.writerows(psnr_rows)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name in IMAGES:
        values = [row[2] for row in psnr_rows if row[0] == name and row[1] > 0]
        ax.plot(KSIZES, values, marker="o", markersize=6, linewidth=2, color=COLORS[name], label=name)
        ax.annotate(name, (KSIZES[-1], values[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10, color="#333333")
    ax.set_xticks(KSIZES)
    ax.set_xlabel("Размер ядра")
    ax.set_ylabel("PSNR относительно исходного, дБ")
    ax.set_title("Качество восстановления после шума «соль и перец»")
    ax.grid(color="#e5e5e5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(2.5, 10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("results/psnr.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
