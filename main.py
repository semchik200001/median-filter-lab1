"""Применение медианного фильтра к изображению из командной строки."""

import argparse
import time

import cv2

from median_filter import median_native, median_opencv

METHODS = {"opencv": median_opencv, "native": median_native}


def main():
    parser = argparse.ArgumentParser(description="Медианный фильтр с переменным размером ядра")
    parser.add_argument("input", help="путь к исходному изображению")
    parser.add_argument("output", help="куда сохранить результат")
    parser.add_argument("-k", "--ksize", type=int, default=3, help="размер ядра (нечётный, >= 3)")
    parser.add_argument("-m", "--method", choices=METHODS, default="opencv", help="реализация")
    parser.add_argument("--gray", action="store_true", help="перевести изображение в оттенки серого")
    args = parser.parse_args()
    if args.ksize < 3 or args.ksize % 2 == 0:
        parser.error("размер ядра должен быть нечётным числом не меньше 3")

    flag = cv2.IMREAD_GRAYSCALE if args.gray else cv2.IMREAD_COLOR
    image = cv2.imread(args.input, flag)
    if image is None:
        parser.error(f"не удалось открыть {args.input}")

    start = time.perf_counter()
    result = METHODS[args.method](image, args.ksize)
    elapsed = time.perf_counter() - start

    cv2.imwrite(args.output, result)
    print(f"{args.method}, ядро {args.ksize}x{args.ksize}, {image.shape[1]}x{image.shape[0]}: {elapsed:.4f} с")


if __name__ == "__main__":
    main()
