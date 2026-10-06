"""Подготовка тестовых изображений: исходники из scikit-image и их зашумлённые версии."""

import os

import cv2
import numpy as np
from skimage import data

NOISE_AMOUNT = 0.1  # доля пикселей, заменённых шумом "соль и перец"


def add_salt_and_pepper(image, amount, rng):
    noisy = image.copy()
    mask = rng.random(image.shape[:2])
    noisy[mask < amount / 2] = 0
    noisy[mask > 1 - amount / 2] = 255
    return noisy


def main():
    os.makedirs("images", exist_ok=True)
    rng = np.random.default_rng(42)

    camera = data.camera()
    astronaut = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)

    for name, image in (("camera", camera), ("astronaut", astronaut)):
        cv2.imwrite(f"images/{name}.png", image)
        cv2.imwrite(f"images/{name}_noisy.png", add_salt_and_pepper(image, NOISE_AMOUNT, rng))
        print(f"images/{name}.png и images/{name}_noisy.png сохранены")


if __name__ == "__main__":
    main()
