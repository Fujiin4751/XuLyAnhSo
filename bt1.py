from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

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
OUTPUT_DIR = Path("images/result_bt1")
GRAY_LEVELS = 256
DEFAULT_IMAGE_NAME = "Img_01.jpg"


def build_histogram(gray: Matrix) -> list[int]:
    """
    H1: dem so luong pixel co tung muc xam 0..255.
    Tu tay quet tung pixel, khong dung np.bincount / np.histogram.
    """
    hist = [0 for _ in range(GRAY_LEVELS)]
    h = height_of(gray)
    w = width_of(gray)

    for y in range(h):
        for x in range(w):
            value = gray[y][x]
            hist[value] += 1

    return hist


def equalize_histogram(gray: Matrix, hist: list[int]) -> Matrix:
    """
    H2: can bang histogram bang phep bien doi:
        cdf(k)  = tong xac suat tu muc 0 den k
        s(k)    = round((L-1) * cdf(k)),  L = 256
    Sau do thay tung pixel gray bang gia tri s(gray[y][x]).
    """
    h = height_of(gray)
    w = width_of(gray)
    total_pixels = h * w

    # Bang tra cuu (lookup table) cho 256 muc xam, tinh tay bang for-loop
    lookup_table = [0 for _ in range(GRAY_LEVELS)]
    running_sum = 0
    for level in range(GRAY_LEVELS):
        running_sum += hist[level]
        cdf = running_sum / total_pixels if total_pixels > 0 else 0
        mapped_value = round((GRAY_LEVELS - 1) * cdf)
        if mapped_value < 0:
            mapped_value = 0
        if mapped_value > 255:
            mapped_value = 255
        lookup_table[level] = mapped_value

    equalized: Matrix = [[0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            old_value = gray[y][x]
            equalized[y][x] = lookup_table[old_value]

    return equalized


def shrink_range(equalized: Matrix, low: int = 30, high: int = 120) -> Matrix:
    """
    Hieu chinh thu hep dai gia tri cua anh H2 (dang 0..255) ve khoang [low, high]
    bang phep bien doi tuyen tinh:
        new = low + old * (high - low) / 255
    """
    h = height_of(equalized)
    w = width_of(equalized)
    span = high - low

    narrowed: Matrix = [[0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            old_value = equalized[y][x]
            new_value = low + (old_value * span) / 255.0
            if new_value < low:
                new_value = low
            if new_value > high:
                new_value = high
            narrowed[y][x] = round(new_value)

    return narrowed


def draw_histogram_chart(hist: list[int], path: Path, title: str) -> None:
    """Ve histogram """
    canvas_w, canvas_h = 520, 300
    pad_l, pad_r, pad_t, pad_b = 46, 14, 30, 32

    chart = Image.new("RGB", (canvas_w, canvas_h), "white")
    drawer = ImageDraw.Draw(chart)

    plot_w = canvas_w - pad_l - pad_r
    plot_h = canvas_h - pad_t - pad_b
    base_y = canvas_h - pad_b

    max_count = max(hist) if max(hist) > 0 else 1

    drawer.text((pad_l, 6), title, fill=(10, 10, 10))
    drawer.line((pad_l, pad_t, pad_l, base_y), fill=(0, 0, 0))
    drawer.line((pad_l, base_y, canvas_w - pad_r, base_y), fill=(0, 0, 0))
    drawer.text((pad_l - 10, base_y + 8), "0", fill=(10, 10, 10))
    drawer.text((canvas_w - pad_r - 26, base_y + 8), "255", fill=(10, 10, 10))

    bar_width = plot_w / float(GRAY_LEVELS)
    for level in range(GRAY_LEVELS):
        bar_h = int((hist[level] / max_count) * plot_h)
        x0 = pad_l + level * bar_width
        x1 = pad_l + (level + 1) * bar_width
        y0 = base_y - bar_h
        drawer.rectangle((x0, y0, max(x0, x1), base_y), fill=(60, 120, 160))

    chart.save(path)


def process_into_dir(image_path: Path, save_dir: Path) -> None:
    red, green, blue = load_rgb_pixels(image_path)
    gray = to_grayscale(red, green, blue)

    h1 = build_histogram(gray)
    equalized = equalize_histogram(gray, h1)
    h2 = build_histogram(equalized)
    narrowed = shrink_range(equalized, low=30, high=120)

    save_dir.mkdir(parents=True, exist_ok=True)

    save_matrix_as_image(gray, save_dir / "00_gray.jpg")
    draw_histogram_chart(h1, save_dir / "01_H1_histogram.jpg", "H1 - Histogram anh xam")
    save_matrix_as_image(equalized, save_dir / "02_H2_equalized.jpg")
    draw_histogram_chart(h2, save_dir / "03_H2_histogram.jpg", "H2 - Histogram sau can bang")
    save_matrix_as_image(narrowed, save_dir / "04_H2_narrow_30_120.jpg")


def run_on_one_image(image_path: Path, output_dir: Path) -> None:
    """Dung khi chay rieng bt1.py: tu tao thu muc con theo ten file anh."""
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
    print("Done bt1.")


if __name__ == "__main__":
    main()
