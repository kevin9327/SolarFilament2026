"""
Kaggle: Solar Filament Instance Segmentation Challenge 2026 (`filament-segmentation-2026`)
Self-Contained Kaggle Notebook Submission Script (Panoptic Quality `PQ = SQ * RQ` Optimized)

Implements:
1. Auto-discovery of `/kaggle/input/filament-segmentation-2026` (or any mounted Kaggle input folder),
   `sample_submission.csv`, and test solar H-alpha full-disk images.
2. Radial Solar Limb-Darkening Normalization:
   Divides out the large-scale quiet-Sun radial intensity profile via Gaussian low-pass background
   estimation within the solar disk (`r < 0.94 * R_sun`).
3. Contrast-Adaptive Dark Filament Extraction + Morphological Bridging:
   Segments dark H-alpha chromospheric absorption filaments (`z < -2.35`), bridges thin barb gaps
   while keeping adjacent filaments separated, and labels strictly disjoint connected components
   (guaranteeing zero pixel overlap across instances).
4. Schema-Exact RLE Formatting:
   Preserves the exact column names and row structure of `sample_submission.csv` and writes
   `/kaggle/working/submission.csv`.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage


def rle_encode(mask: np.ndarray, transpose: bool = True) -> str:
    """1-indexed Run-Length Encoding (column-major Fortran order by default)."""
    pixels = mask.T.flatten() if transpose else mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def segment_solar_filaments(
    img_gray: np.ndarray,
    min_area: int = 35,
    z_thresh: float = -2.35,
) -> list[np.ndarray]:
    """
    Input: 2D grayscale H-alpha solar disk image.
    Output: List of strictly non-overlapping binary instance masks sorted by area descending.
    """
    img = img_gray.astype(np.float32)
    h, w = img.shape

    # Detect solar disk center and radius
    p25 = np.percentile(img, 25)
    disk_rough = img > p25
    if np.sum(disk_rough) > 0.05 * h * w:
        cy, cx = ndimage.center_of_mass(disk_rough)
        r_sun = np.sqrt(np.sum(disk_rough) / np.pi)
    else:
        cy, cx = h / 2.0, w / 2.0
        r_sun = min(h, w) * 0.45

    yy, xx = np.ogrid[:h, :w]
    dist_ratio = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / max(r_sun, 1.0)
    valid_disk = dist_ratio < 0.93

    # Remove solar limb darkening via large-scale Gaussian background division
    sigma_bg = max(min(h, w) / 16.0, 8.0)
    bg = ndimage.gaussian_filter(img, sigma=sigma_bg)
    norm_img = img / (bg + 1e-6)

    disk_vals = norm_img[valid_disk]
    if len(disk_vals) > 100:
        med = float(np.median(disk_vals))
        mad = float(np.median(np.abs(disk_vals - med))) * 1.4826 + 1e-6
    else:
        med = float(np.median(norm_img))
        mad = float(np.std(norm_img)) + 1e-6

    z_score = (norm_img - med) / mad
    binary = (z_score < z_thresh) & valid_disk

    # Morphological closing to connect thin filament barbs, then opening to remove sunspot noise
    struct = ndimage.generate_binary_structure(2, 2)
    closed = ndimage.binary_closing(binary, structure=struct, iterations=1)
    cleaned = ndimage.binary_opening(closed, structure=struct, iterations=1)

    labeled, num_features = ndimage.label(cleaned, structure=struct)
    if num_features == 0:
        return []

    objs = ndimage.find_objects(labeled)
    instances: list[tuple[int, np.ndarray]] = []
    for idx, slc in enumerate(objs, start=1):
        if slc is None:
            continue
        sub_mask = labeled[slc] == idx
        area = int(np.sum(sub_mask))
        if area < min_area:
            continue
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[slc] = sub_mask.astype(np.uint8)
        instances.append((area, full_mask))

    instances.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in instances[:60]]


def load_image_2d(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
        return arr if arr.ndim == 2 else arr[..., 0]
    with Image.open(path) as im:
        im_gray = im.convert("L")
        return np.asarray(im_gray, dtype=np.float32)


def find_kaggle_assets() -> tuple[Path | None, dict[str, Path]]:
    sample_sub: Path | None = None
    image_map: dict[str, Path] = {}
    img_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}

    kaggle_in = Path("/kaggle/input")
    if not kaggle_in.is_dir():
        return None, {}

    for p in kaggle_in.rglob("*"):
        if p.is_file():
            if p.name.lower() in {"sample_submission.csv", "test.csv"} and sample_sub is None:
                sample_sub = p
            elif p.suffix.lower() in img_exts:
                image_map[p.stem] = p
                image_map[p.name] = p

    return sample_sub, image_map


def main() -> None:
    sample_sub, image_map = find_kaggle_assets()
    work_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(__file__).resolve().parent
    out_csv = work_dir / "submission.csv"

    if sample_sub is None and not image_map:
        print("Kaggle input not mounted locally; verifying local preview submission.csv...")
        if out_csv.exists():
            df = pd.read_csv(out_csv)
            print(f"Verified local {out_csv} ({len(df):,} rows, cols={list(df.columns)})")
        return

    if sample_sub is not None:
        sub_df = pd.read_csv(sample_sub)
        id_col = sub_df.columns[0]
        pred_col = sub_df.columns[1] if len(sub_df.columns) > 1 else "EncodedPixels"
        print(f"Loaded template {sample_sub} ({len(sub_df):,} rows, cols={list(sub_df.columns)})")

        # Determine delimiter from sample_submission if present
        sample_vals = sub_df[pred_col].dropna().astype(str).tolist()
        use_pipe = any("|" in s for s in sample_vals)
        is_multi_row = sub_df[id_col].duplicated().any()

        if not is_multi_row:
            new_preds = []
            for img_id in sub_df[id_col].astype(str):
                img_path = image_map.get(img_id) or image_map.get(Path(img_id).stem)
                if img_path is None:
                    new_preds.append("")
                    continue
                img = load_image_2d(img_path)
                masks = segment_solar_filaments(img)
                if not masks:
                    new_preds.append("")
                elif use_pipe:
                    new_preds.append("|".join(rle_encode(m) for m in masks))
                else:
                    # Merge disjoint instances into single non-overlapping mask if single RLE string expected
                    union_mask = np.zeros_like(masks[0], dtype=np.uint8)
                    for m in masks:
                        union_mask |= m
                    new_preds.append(rle_encode(union_mask))
            sub_df[pred_col] = new_preds
            sub_df.to_csv(out_csv, index=False)
            print(f"Wrote {len(sub_df):,} rows to {out_csv}")
            return

    # Fallback: 1 row per detected filament instance across all test images
    rows = []
    seen_paths = sorted(set(image_map.values()))
    for img_path in seen_paths:
        img = load_image_2d(img_path)
        masks = segment_solar_filaments(img)
        if not masks:
            rows.append({"image_id": img_path.stem, "EncodedPixels": ""})
        else:
            for m in masks:
                rows.append({"image_id": img_path.stem, "EncodedPixels": rle_encode(m)})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(out_df):,} rows to {out_csv}")


if __name__ == "__main__":
    main()
