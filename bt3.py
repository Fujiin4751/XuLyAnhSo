"""
bt3.py
------
Cau 3: Anh xam -> Local Binary Pattern (LBP) voi nhieu cau hinh (P, R).

  (P=8,  R=1), (P=8,  R=2)
  (P=16, R=2), (P=16, R=3)
  (P=24, R=3)

Voi mot pixel trung tam, lay P diem lan can nam tren duong tron ban kinh R
(toa do thuc, dung noi suy song tuyen tinh - bilinear - de lay gia tri xam
tai diem khong nam dung vao luoi pixel). So sanh tung diem lan can voi
pixel trung tam de tao 1 bit (1 neu lan can >= trung tam, nguoc lai 0),
ghep P bit thanh 1 chuoi nhi phan.

  - Neu P = 8  -> chuoi co dung 8 bit -> doi thang sang so thap phan (0..255).
  - Neu P = 16 -> tach chuoi 16 bit thanh 2 doan 8-bit, doi moi doan sang
    so thap phan, LAY GIA TRI LON HON trong 2 doan lam gia tri pixel.
  - Neu P = 24 -> tach chuoi 24 bit thanh 3 doan 8-bit, doi moi doan sang
    so thap phan, LAY GIA TRI LON NHAT trong 3 doan lam gia tri pixel.

Toan bo thuat toan (lay toa do lan can, noi suy bilinear, so sanh bit,
ghep chuoi nhi phan) la for-loop Python thuan tren Matrix = list[list[int]].
Khong dung skimage.feature.local_binary_pattern hay bat ky ham LBP co san.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

from imgcore import (
    Matrix,
    height_of,
    width_of,
    load_rgb_pixels,
    to_grayscale,
    save_matrix_as_image,
    list_image_files,
)

INPUT_DIR = Path("images/processed")
OUTPUT_DIR = Path("images/result_bt3")
DEFAULT_IMAGE_NAME = "Img_01.jpg"

# Danh sach cau hinh (so diem lan can P, ban kinh R) can xu ly theo de bai
LBP_CONFIGS = [
    (8, 1),
    (8, 2),
    (16, 2),
    (16, 3),
    (24, 3),
]


def clamp_coord(value: int, lower: int, upper: int) -> int:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def read_pixel_safe(gray: Matrix, x: int, y: int) -> int:
    """Doc gia tri pixel, neu toa do ngoai bien thi ghim (clamp) ve bien gan nhat."""
    h = height_of(gray)
    w = width_of(gray)
    safe_x = clamp_coord(x, 0, w - 1)
    safe_y = clamp_coord(y, 0, h - 1)
    return gray[safe_y][safe_x]


def bilinear_value(gray: Matrix, x: float, y: float) -> float:
    """
    Noi suy song tuyen tinh tai toa do thuc (x, y) khong nguyen.
    Lay 4 pixel nguyen gan nhat (x0,y0)-(x1,y0)-(x0,y1)-(x1,y1) roi
    noi suy theo ty le phan du a, b.
    """
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1

    a = x - x0  # ty le lech theo truc ngang
    b = y - y0  # ty le lech theo truc dung

    top_left = read_pixel_safe(gray, x0, y0)
    top_right = read_pixel_safe(gray, x1, y0)
    bottom_left = read_pixel_safe(gray, x0, y1)
    bottom_right = read_pixel_safe(gray, x1, y1)

    top = top_left * (1 - a) + top_right * a
    bottom = bottom_left * (1 - a) + bottom_right * a
    return top * (1 - b) + bottom * b


def neighbor_bits(gray: Matrix, cx: int, cy: int, num_points: int, radius: int) -> list[int]:
    """
    Tinh chuoi bit nhi phan cho mot pixel trung tam:
    voi moi diem p trong [0, num_points), lay toa do tren duong tron ban kinh
    radius quanh (cx, cy), so sanh gia tri noi suy tai do voi pixel trung tam.
    """
    center_value = gray[cy][cx]
    bits: list[int] = []

    for p in range(num_points):
        angle = 2.0 * math.pi * p / num_points
        sample_x = cx + radius * math.cos(angle)
        sample_y = cy - radius * math.sin(angle)  # truc y huong xuong nen lay dau am
        neighbor_value = bilinear_value(gray, sample_x, sample_y)

        if neighbor_value >= center_value:
            bits.append(1)
        else:
            bits.append(0)

    return bits


def bits_segment_to_decimal(bits: list[int], start_index: int) -> int:
    """Doi 8 bit lien tiep sang so thap phan, bit dau tien la trong so 2^0."""
    value = 0
    power_of_two = 1
    for offset in range(8):
        value += bits[start_index + offset] * power_of_two
        power_of_two *= 2
    return value


def pick_pixel_value_from_bits(bits: list[int]) -> int:
    """
    Tach chuoi bit thanh cac doan 8-bit, doi tung doan sang thap phan,
    tra ve gia tri LON NHAT trong cac doan (ap dung cho ca P=8,16,24:
    voi P=8 chi co 1 doan nen ket qua chinh la gia tri cua doan do).
    """
    num_segments = len(bits) // 8
    best_value = 0
    for seg in range(num_segments):
        seg_value = bits_segment_to_decimal(bits, seg * 8)
        if seg_value > best_value:
            best_value = seg_value
    return best_value


def lbp_transform(gray: Matrix, num_points: int, radius: int) -> Matrix:
    """Tinh anh LBP cho toan bo anh xam voi cau hinh (num_points, radius) cho truoc."""
    h = height_of(gray)
    w = width_of(gray)
    output: Matrix = [[0 for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            bits = neighbor_bits(gray, x, y, num_points, radius)
            output[y][x] = pick_pixel_value_from_bits(bits)

    return output


def process_into_dir(image_path: Path, save_dir: Path) -> None:
    """
    Xu ly 1 anh va luu KET QUA TRUC TIEP vao save_dir 
    """
    red, green, blue = load_rgb_pixels(image_path)
    gray = to_grayscale(red, green, blue)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_matrix_as_image(gray, save_dir / "00_gray.jpg")

    for num_points, radius in LBP_CONFIGS:
        lbp_result = lbp_transform(gray, num_points, radius)
        filename = f"LBP_P{num_points}_R{radius}.jpg"
        save_matrix_as_image(lbp_result, save_dir / filename)
        print(f"     LBP P={num_points} R={radius} done")


def run_on_one_image(image_path: Path, output_dir: Path) -> None:
    save_dir = output_dir / image_path.stem
    process_into_dir(image_path, save_dir)
    print(f"  -> {image_path.name} done")


def collect_targets(process_all: bool) -> list[Path]:
    if process_all:
        return list_image_files(INPUT_DIR)

    single = INPUT_DIR / DEFAULT_IMAGE_NAME
    if not single.exists():
        raise FileNotFoundError(f"Khong tim thay anh mac dinh: {single}")
    return [single]


def run_all(process_all: bool = False, output_dir: Path = OUTPUT_DIR, clean: bool = True) -> None:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = collect_targets(process_all)
    for image_path in targets:
        run_on_one_image(image_path, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Xu ly tat ca anh trong images/processed")
    args = parser.parse_args()

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc: {INPUT_DIR}")

    run_all(process_all=args.all)
    print("Done bt3.")


if __name__ == "__main__":
    main()
