from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

Matrix = list  # list[list[int]] hoac list[list[float]]


def height_of(matrix: Matrix) -> int:
    return len(matrix)


def width_of(matrix: Matrix) -> int:
    if len(matrix) == 0:
        return 0
    return len(matrix[0])


def clamp_byte(value: float) -> int:
    """Ep gia tri ve khoang [0, 255] va lam tron thanh int."""
    rounded = int(round(value))
    if rounded < 0:
        return 0
    if rounded > 255:
        return 255
    return rounded


def load_rgb_pixels(image_path: Path):
    """
    Mo anh va tra ve 3 ma tran R, G, B kieu list[list[int]].
    """
    pil_image = Image.open(image_path).convert("RGB")
    rgb_array = np.asarray(pil_image) 
    h, w, _ = rgb_array.shape

    red = [[int(rgb_array[y, x, 0]) for x in range(w)] for y in range(h)]
    green = [[int(rgb_array[y, x, 1]) for x in range(w)] for y in range(h)]
    blue = [[int(rgb_array[y, x, 2]) for x in range(w)] for y in range(h)]
    return red, green, blue


def to_grayscale(red: Matrix, green: Matrix, blue: Matrix) -> Matrix:
    """
    Cong thuc chuyen anh mau sang anh xam theo chuan ITU-R BT.601:
        Gray = 0.299*R + 0.587*G + 0.114*B
    """
    h = height_of(red)
    w = width_of(red)
    gray: Matrix = [[0 for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            r = red[y][x]
            g = green[y][x]
            b = blue[y][x]
            gray_value = 0.299 * r + 0.587 * g + 0.114 * b
            gray[y][x] = clamp_byte(gray_value)

    return gray


def save_matrix_as_image(matrix: Matrix, path: Path) -> None:
    """Ghi Matrix (gia tri 0-255) ra file anh xam"""
    h = height_of(matrix)
    w = width_of(matrix)
    out_image = Image.new("L", (w, h))

    for y in range(h):
        for x in range(w):
            out_image.putpixel((x, y), clamp_byte(matrix[y][x]))

    out_image.save(path)


def copy_matrix(matrix: Matrix) -> Matrix:
    return [row[:] for row in matrix]

def combine_rgb(red: Matrix, green: Matrix, blue: Matrix):
    """
    Ghep 3 Matrix R, G, B (tung kenh rieng) thanh 1 cau truc anh mau
    dang list[list[list[int]]] de hien thi bang
    matplotlib.imshow. 
    """
    h = height_of(red)
    w = width_of(red)
    return [
        [[red[y][x], green[y][x], blue[y][x]] for x in range(w)]
        for y in range(h)
    ]

def list_image_files(folder: Path):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in extensions
    ]
