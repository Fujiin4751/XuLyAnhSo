from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from imgcore import (
    Matrix,
    height_of,
    width_of,
    clamp_byte,
    load_rgb_pixels,
    to_grayscale,
    save_matrix_as_image,
    list_image_files,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_DIR = Path("images/processed")
OUTPUT_DIR = Path("images/result_bt2")
DEFAULT_IMAGE_NAME = "Img_01.jpg"


def make_box_kernel(size: int) -> Matrix:
    """Kernel trung binh (box filter) kich thuoc size x size, tong cac phan tu = 1."""
    weight = 1.0 / (size * size)
    return [[weight for _ in range(size)] for _ in range(size)]


def pad_with_zero(matrix: Matrix, pad: int) -> Matrix:
    """Them vien gia tri 0 quanh matrix, moi ben pad pixel."""
    if pad <= 0:
        return [row[:] for row in matrix]

    h = height_of(matrix)
    w = width_of(matrix)
    new_h = h + 2 * pad
    new_w = w + 2 * pad

    padded: Matrix = [[0 for _ in range(new_w)] for _ in range(new_h)]
    for y in range(h):
        for x in range(w):
            padded[y + pad][x + pad] = matrix[y][x]

    return padded


def convolve(gray: Matrix, kernel: Matrix, padding: int, stride: int = 1) -> Matrix:
    """
    Phep tich chap: truot kernel tren anh da padding, tai moi
    vi tri nhan tung phan tu kernel voi pixel tuong ung roi cong lai.
    """
    k_h = len(kernel)
    k_w = len(kernel[0])
    in_h = height_of(gray)
    in_w = width_of(gray)

    padded = pad_with_zero(gray, padding)

    out_h = ((in_h + 2 * padding - k_h) // stride) + 1
    out_w = ((in_w + 2 * padding - k_w) // stride) + 1
    output: Matrix = [[0 for _ in range(out_w)] for _ in range(out_h)]

    for out_y in range(out_h):
        for out_x in range(out_w):
            top = out_y * stride
            left = out_x * stride

            total = 0.0
            for ky in range(k_h):
                for kx in range(k_w):
                    total += padded[top + ky][left + kx] * kernel[ky][kx]

            output[out_y][out_x] = clamp_byte(total)

    return output


def median_filter(source: Matrix, neighborhood: int) -> Matrix:
    """
    Loc trung vi: tai moi pixel, lay tat ca gia tri trong vung lan can
    neighborhood x neighborhood (co padding 0 o bien), sap xep va chon
    gia tri o giua. Kich thuoc dau ra giu nguyen kich thuoc dau vao.
    """
    pad = neighborhood // 2
    padded = pad_with_zero(source, pad)

    h = height_of(source)
    w = width_of(source)
    output: Matrix = [[0 for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            window_values = []
            for ky in range(neighborhood):
                for kx in range(neighborhood):
                    window_values.append(padded[y + ky][x + kx])

            window_values.sort()
            middle_index = len(window_values) // 2
            output[y][x] = window_values[middle_index]

    return output


def pad_to_match(a: Matrix, b: Matrix) -> tuple[Matrix, Matrix]:
    """
    Neu I4 va I5 khac kich thuoc, them padding 0 de hai anh
    co cung kich thuoc truoc khi so sanh tung pixel.
    """
    target_h = max(height_of(a), height_of(b))
    target_w = max(width_of(a), width_of(b))

    def pad_one(matrix: Matrix) -> Matrix:
        h = height_of(matrix)
        w = width_of(matrix)
        top = (target_h - h) // 2
        left = (target_w - w) // 2

        out: Matrix = [[0 for _ in range(target_w)] for _ in range(target_h)]
        for y in range(h):
            for x in range(w):
                out[y + top][x + left] = matrix[y][x]
        return out

    return pad_one(a), pad_one(b)


def build_i6(i4: Matrix, i5: Matrix) -> Matrix:
    """I6(x,y) = 0 neu I4 > I5, nguoc lai I6(x,y) = I5(x,y)."""
    same_i4, same_i5 = pad_to_match(i4, i5)
    h = height_of(same_i5)
    w = width_of(same_i5)

    output: Matrix = [[0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if same_i4[y][x] > same_i5[y][x]:
                output[y][x] = 0
            else:
                output[y][x] = same_i5[y][x]

    return output


def process_into_dir(image_path: Path, save_dir: Path) -> None:
    red, green, blue = load_rgb_pixels(image_path)
    gray = to_grayscale(red, green, blue)

    kernel_3 = make_box_kernel(3)
    kernel_5 = make_box_kernel(5)
    kernel_7 = make_box_kernel(7)

    i1 = convolve(gray, kernel_3, padding=1, stride=1)
    i2 = convolve(gray, kernel_5, padding=2, stride=1)
    i3 = convolve(gray, kernel_7, padding=3, stride=2)
    i4 = median_filter(i3, neighborhood=3)
    i5 = median_filter(i1, neighborhood=5)
    i6 = build_i6(i4, i5)

    save_dir.mkdir(parents=True, exist_ok=True)

    save_matrix_as_image(gray, save_dir / "00_gray.jpg")
    save_matrix_as_image(i1, save_dir / "01_I1_box3x3_p1_s1.jpg")
    save_matrix_as_image(i2, save_dir / "02_I2_box5x5_p2_s1.jpg")
    save_matrix_as_image(i3, save_dir / "03_I3_box7x7_p3_s2.jpg")
    save_matrix_as_image(i4, save_dir / "04_I4_median3x3_of_I3.jpg")
    save_matrix_as_image(i5, save_dir / "05_I5_median5x5_of_I1.jpg")
    save_matrix_as_image(i6, save_dir / "06_I6_threshold.jpg")

    draw_dashboard(gray, i1, i2, i3, i4, i5, i6, save_dir / "99_dashboard.png")

def draw_dashboard(
    gray: Matrix,
    i1: Matrix,
    i2: Matrix,
    i3: Matrix,
    i4: Matrix,
    i5: Matrix,
    i6: Matrix,
    save_path: Path,
) -> None:
    """
    Ve dashboard tong hop cho bai 2 (giong layout anh mau):
        Hang 1: Anh xam goc (I) | I1 - Kernel 3x3 | I2 - Kernel 5x5 | I3 - Kernel 7x7
        Hang 2: I4 - Median(I3) | I5 - Median(I1) | I6 - Nguong tu I4,I5
    Day chi la BUOC HIEN THI ket qua da tinh san bang for-loop tay; matplotlib
    khong tham gia vao bat ky phep tinh convolution/median/threshold nao.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
 
    axes[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Anh xam goc (I)")
    axes[0, 0].axis("off")
 
    axes[0, 1].imshow(i1, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("I1 - Kernel 3x3, pad=1")
    axes[0, 1].axis("off")
 
    axes[0, 2].imshow(i2, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("I2 - Kernel 5x5, pad=2")
    axes[0, 2].axis("off")
 
    axes[0, 3].imshow(i3, cmap="gray", vmin=0, vmax=255)
    axes[0, 3].set_title("I3 - Kernel 7x7, pad=3, stride=2")
    axes[0, 3].axis("off")
 
    axes[1, 0].imshow(i4, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("I4 - Median(I3), 3x3")
    axes[1, 0].axis("off")
 
    axes[1, 1].imshow(i5, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("I5 - Median(I1), 5x5")
    axes[1, 1].axis("off")
 
    axes[1, 2].imshow(i6, cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title("I6 - Nguong tu I4, I5")
    axes[1, 2].axis("off")
 
    axes[1, 3].axis("off")  # o trong, khong dung den
 
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
 

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
    print("Done bt2.")


if __name__ == "__main__":
    main()
