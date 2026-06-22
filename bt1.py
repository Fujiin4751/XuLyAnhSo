from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from imgcore import (
    Matrix,
    height_of,
    width_of,
    load_rgb_pixels,
    to_grayscale,
    save_matrix_as_image,
    combine_rgb,
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

    # Bang tra cuu cho 256 muc xam
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


def find_min_max(matrix: Matrix) -> tuple[int, int]:
    """Tim gia tri nho nhat (r_min) va lon nhat (r_max) thuc te xuat hien trong anh."""
    h = height_of(matrix)
    w = width_of(matrix)
 
    current_min = matrix[0][0]
    current_max = matrix[0][0]
                                     
    for y in range(h):
        for x in range(w):
            value = matrix[y][x]
            if value < current_min:
                current_min = value
            if value > current_max:
                current_max = value
 
    return current_min, current_max
 
 
def shrink_range(equalized: Matrix, low: int = 30, high: int = 120) -> Matrix:
    """
    Hieu chinh thu hep histogram:
 
        s = [(s_max - s_min) / (r_max - r_min)] * (r - r_min) + s_min
    """
    h = height_of(equalized)
    w = width_of(equalized)
 
    r_min, r_max = find_min_max(equalized)
    s_min, s_max = low, high
 
    # Truong hop dac biet: anh dong nhat (r_max == r_min), tranh chia cho 0
    if r_max == r_min:
        denom = 1
    else:
        denom = r_max - r_min
 
    ratio = (s_max - s_min) / denom
 
    narrowed: Matrix = [[0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            r = equalized[y][x]
            s = ratio * (r - r_min) + s_min
            narrowed[y][x] = round(s)
 
    return narrowed

def save_histogram_csv(hist: list[int], path: Path) -> None:
    """Ghi histogram ra file CSV voi 2 cot: gray_value, count."""
    lines = ["gray_value,count\n"]
    for gray_value, count in enumerate(hist):
        lines.append(f"{gray_value},{count}\n")
    path.write_text("".join(lines), encoding="utf-8")

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
    h3 = build_histogram(narrowed)

    save_dir.mkdir(parents=True, exist_ok=True)

    save_matrix_as_image(gray, save_dir / "00_gray.jpg")
    draw_histogram_chart(h1, save_dir / "01_H1_histogram.jpg", "H1 - Histogram anh xam")
    save_histogram_csv(h1, save_dir / "01_H1_histogram.csv")
    save_matrix_as_image(equalized, save_dir / "02_H2_equalized.jpg")
    draw_histogram_chart(h2, save_dir / "03_H2_histogram.jpg", "H2 - Histogram sau can bang")
    save_histogram_csv(h2, save_dir / "03_H2_histogram.csv")
    save_matrix_as_image(narrowed, save_dir / "04_H2_narrow_30_120.jpg")
    draw_histogram_chart(h3, save_dir / "05_H3_narrow_histogram.jpg", "H3 - Histogram sau thu hep [30,120]")
    save_histogram_csv(h3, save_dir / "05_H3_narrow_histogram.csv")

    color_rgb = combine_rgb(red, green, blue)
    draw_dashboard(color_rgb, gray, equalized, narrowed, h1, h2, h3, save_dir / "99_dashboard.png")

def draw_dashboard(
    color_rgb,
    gray: Matrix,
    equalized: Matrix,
    narrowed: Matrix,
    h1: list[int],
    h2: list[int],
    h3: list[int],
    save_path: Path,
) -> None:
    """
    Ve dashboard tong hop cho bai 1 (giong layout anh mau):
        Hang 1: Anh mau goc | Anh xam | Anh sau can bang Histogram | Anh thu hep [30,120]
        Hang 2: H1 - Histogram anh xam | H2 - Histogram can bang | H3 - Histogram thu hep
    Day chi la BUOC HIEN THI ket qua da tinh san bang for-loop tay; matplotlib
    khong tham gia vao bat ky phep tinh xu ly anh nao.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
 
    # ---- Hang 1: 4 anh ----
    axes[0, 0].imshow(color_rgb)
    axes[0, 0].set_title("Anh mau goc")
    axes[0, 0].axis("off")
 
    axes[0, 1].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Anh xam")
    axes[0, 1].axis("off")
 
    axes[0, 2].imshow(equalized, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("Anh sau can bang Histogram")
    axes[0, 2].axis("off")
 
    axes[0, 3].imshow(narrowed, cmap="gray", vmin=0, vmax=255)
    axes[0, 3].set_title("Anh thu hep [30, 120]")
    axes[0, 3].axis("off")
 
    # ---- Hang 2: 3 histogram, o con lai an di ----
    levels = list(range(GRAY_LEVELS))
 
    axes[1, 0].bar(levels, h1, width=1.0, color="tab:blue")
    axes[1, 0].set_title("H1 - Histogram anh xam")
    axes[1, 0].set_xlabel("Muc xam")
    axes[1, 0].set_ylabel("So pixel")
    axes[1, 0].set_xlim(0, 255)
 
    axes[1, 1].bar(levels, h2, width=1.0, color="tab:blue")
    axes[1, 1].set_title("H2 - Histogram can bang")
    axes[1, 1].set_xlabel("Muc xam")
    axes[1, 1].set_xlim(0, 255)
 
    axes[1, 2].bar(levels, h3, width=1.0, color="tab:blue")
    axes[1, 2].set_title("H3 - Histogram thu hep [30,120]")
    axes[1, 2].set_xlabel("Muc xam")
    axes[1, 2].axvline(30, color="tab:blue", linestyle="--", linewidth=1)
    axes[1, 2].axvline(120, color="tab:blue", linestyle="--", linewidth=1)
    axes[1, 2].set_xlim(0, 255)
 
    axes[1, 3].axis("off")  # o trong, khong dung den
 
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
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
