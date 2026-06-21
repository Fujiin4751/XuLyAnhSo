from __future__ import annotations

import shutil
import time
from pathlib import Path

import bt1
import bt2
import bt3
from imgcore import list_image_files

OUTPUT_DIR = Path("images/result_bt4")


def process_one_image(image_path: Path, img_dir: Path) -> None:
    """Chay bai 1, bai 2, bai 3 cho 1 anh, luu vao 3 thu muc con cua img_dir."""
    print(f"   -> Bai 1 (Histogram / Equalize / Shrink)")
    bt1.process_into_dir(image_path, img_dir / "Bai1")

    print(f"   -> Bai 2 (Convolution / Median / Threshold)")
    bt2.process_into_dir(image_path, img_dir / "Bai2")

    print(f"   -> Bai 3 (Local Binary Pattern)")
    bt3.process_into_dir(image_path, img_dir / "Bai3")


def main() -> None:
    if not bt1.INPUT_DIR.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc: {bt1.INPUT_DIR}")

    image_paths = list_image_files(bt1.INPUT_DIR)
    if not image_paths:
        raise FileNotFoundError(f"Khong co anh nao trong {bt1.INPUT_DIR}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()

    for index, image_path in enumerate(image_paths, start=1):
        img_label = f"Img_{index:02d}"
        img_dir = OUTPUT_DIR / img_label

        print(f"== {img_label}  (anh goc: {image_path.name}) ==")
        process_one_image(image_path, img_dir)

    elapsed = time.time() - start
    print(f"Done bt4: da xu ly {len(image_paths)} anh, ket qua trong {OUTPUT_DIR} ({elapsed:.1f}s).")


if __name__ == "__main__":
    main()
