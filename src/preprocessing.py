import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

RGB_DIRNAME = "RGBImages"
DEPTH_DIRNAME = "DepthImages"
RGB_PREFIX = "RGB_"
DEPTH_PREFIX = "Depth_"

DEFAULT_IMG_SIZE = 224
# Keep 0.80 to ensure we capture the full plant even if it's large
DEFAULT_CROP_SCALE = 0.85
DEFAULT_MAX_HEIGHT_MM = 300.0


def center_square_crop(img: np.ndarray, scale: float) -> np.ndarray:
    """Center-crop to a square region, then keep only `scale` of that square."""
    h, w = img.shape[:2]
    side = min(h, w)
    crop = max(2, int(side * scale))

    cy, cx = h // 2, w // 2
    y1 = max(0, cy - crop // 2)
    x1 = max(0, cx - crop // 2)
    y2 = min(h, y1 + crop)
    x2 = min(w, x1 + crop)

    return img[y1:y2, x1:x2]


def estimate_background_depth_from_corners(depth_mm: np.ndarray, corner_ratio: float = 0.16) -> float | None:
    """
    Estimate background depth (tray level) from full-frame corners before cropping.
    """
    h, w = depth_mm.shape[:2]
    ch = max(16, int(h * corner_ratio))
    cw = max(16, int(w * corner_ratio))

    patches = [
        depth_mm[:ch, :cw],      # top-left
        depth_mm[:ch, -cw:],     # top-right
        depth_mm[-ch:, :cw],     # bottom-left
        depth_mm[-ch:, -cw:],    # bottom-right
    ]

    vals = np.concatenate([p.reshape(-1) for p in patches]).astype(np.float32)
    # Filter valid depth only
    vals = vals[np.isfinite(vals) & (vals > 0)]
    
    if vals.size < 200:
        return None

    # Remove extreme outliers
    lo, hi = np.percentile(vals, [10, 99])
    vals = vals[(vals >= lo) & (vals <= hi)]
    
    if vals.size < 50:
        return None

    # We use a high percentile (85th) to represent the "floor" / tray bottom
    # because the tray is the furthest thing away.
    return float(np.percentile(vals, 85))


def fill_depth_holes_basic(depth_mm: np.ndarray) -> np.ndarray:
    """
    Simple hole filling for depth maps without requiring a plant mask.
    Uses a small closing operation to fill black speckles (noise).
    """
    # Create a validity mask (pixels > 0 are valid)
    mask = (depth_mm > 0).astype(np.uint8)
    
    # If the image is mostly empty/invalid, skip
    if cv2.countNonZero(mask) == 0:
        return depth_mm

    # Inpaint only small holes
    # We use a simple closing morph instead of slow Navier-Stokes inpainting 
    # because we don't have a reliable mask to guide the inpaint.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    depth_closed = cv2.morphologyEx(depth_mm, cv2.MORPH_CLOSE, kernel)
    
    # Only replace 0-values with closed values
    filled = np.where((depth_mm == 0) & (depth_closed > 0), depth_closed, depth_mm)
    return filled


def depth_to_height_map(
    depth_mm: np.ndarray,
    max_height_mm: float,
    fixed_bg: float | None = None,
) -> np.ndarray:
    """
    Convert raw depth to normalized 'Height above Tray'.
    Tray/Background becomes 0.0. Plant parts become > 0.0.
    """
    if fixed_bg is None:
        valid_all = depth_mm > 0
        if not np.any(valid_all):
            return np.zeros_like(depth_mm, dtype=np.float32)
        # Fallback: assume the deepest 5% of pixels are the floor
        background_depth = float(np.percentile(depth_mm[valid_all], 95))
    else:
        background_depth = float(fixed_bg)

    # Height = TrayDepth - PixelDepth
    # Example: Tray is at 800mm. Leaf is at 600mm. Height = 200mm.
    height_mm = background_depth - depth_mm
    
    # Clip negative values (noise where pixel is 'deeper' than the calculated floor)
    height_mm[height_mm < 0.0] = 0.0
    
    # Clip to max expected height (e.g. 300mm)
    height_mm = np.clip(height_mm, 0.0, float(max_height_mm))

    # Normalize to [0, 1]
    height_norm = (height_mm / float(max_height_mm)).astype(np.float32)

    return height_norm


def preprocess_pair(
    image_id: int,
    raw_rgb_dir: Path,
    raw_depth_dir: Path,
    out_rgb_dir: Path,
    out_depth_dir: Path,
    img_size: int,
    crop_scale: float,
    max_height_mm: float,
) -> bool:
    rgb_path = raw_rgb_dir / f"{RGB_PREFIX}{image_id}.png"
    depth_path = raw_depth_dir / f"{DEPTH_PREFIX}{image_id}.png"

    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if rgb_bgr is None or depth_raw is None:
        return False

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. Estimate background from full image CORNERS (before crop)
    full_depth_mm = depth_raw.astype(np.float32)
    fixed_bg = estimate_background_depth_from_corners(full_depth_mm)

    # 2. Crop
    rgb = center_square_crop(rgb, scale=crop_scale)
    depth_crop = center_square_crop(full_depth_mm, scale=crop_scale)

    # 3. Resize
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    depth_crop = cv2.resize(depth_crop, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

    # 4. Fill Holes (Basic)
    depth_filled = fill_depth_holes_basic(depth_crop)

    # 5. Convert Depth -> Height Map
    # This naturally "masks" the background because the tray height becomes 0.0
    height_norm = depth_to_height_map(
        depth_filled,
        max_height_mm=max_height_mm,
        fixed_bg=fixed_bg,
    )

    # 6. Save (No RGB masking!)
    out_rgb = out_rgb_dir / f"{image_id}.png"
    out_depth = out_depth_dir / f"{image_id}.npy"

    cv2.imwrite(str(out_rgb), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(out_depth, height_norm.astype(np.float32))

    return True


def process_dataset(
    csv_path: Path,
    raw_root: Path,
    processed_root: Path,
    mode: str,
    img_size: int,
    crop_scale: float,
    max_height_mm: float,
) -> None:
    print(f"Processing {mode} set from {csv_path}...")

    df = pd.read_csv(csv_path)

    mode_root = processed_root / mode
    rgb_out = mode_root / "RGB"
    depth_out = mode_root / "Depth"
    # Removed Mask directory creation since we aren't using them

    rgb_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    df.to_csv(mode_root / f"{mode}.csv", index=False)

    raw_rgb = raw_root / RGB_DIRNAME
    raw_depth = raw_root / DEPTH_DIRNAME

    missing = []
    for image_id in tqdm(df["image_id"].astype(int).tolist(), desc=f"{mode} preprocessing"):
        ok = preprocess_pair(
            image_id=image_id,
            raw_rgb_dir=raw_rgb,
            raw_depth_dir=raw_depth,
            out_rgb_dir=rgb_out,
            out_depth_dir=depth_out,
            img_size=img_size,
            crop_scale=crop_scale,
            max_height_mm=max_height_mm,
        )
        if not ok:
            missing.append(image_id)

    if missing:
        preview = ", ".join(map(str, missing[:10]))
        suffix = "..." if len(missing) > 10 else ""
        print(f"[Warning] Missing {len(missing)} image pairs in {mode}: {preview}{suffix}")

    print(f"Finished {mode}. Saved to {mode_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess lettuce RGB-D data.")
    parser.add_argument("--train-csv", type=Path, default=Path("data/Training/Train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/Test/Test.csv"))
    parser.add_argument("--train-root", type=Path, default=Path("data/Training"))
    parser.add_argument("--test-root", type=Path, default=Path("data/Test"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/Processed"))
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--crop-scale", type=float, default=DEFAULT_CROP_SCALE)
    parser.add_argument("--max-height-mm", type=float, default=DEFAULT_MAX_HEIGHT_MM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.processed_root.mkdir(parents=True, exist_ok=True)

    if args.train_csv.exists():
        process_dataset(
            csv_path=args.train_csv,
            raw_root=args.train_root,
            processed_root=args.processed_root,
            mode="Train",
            img_size=args.img_size,
            crop_scale=args.crop_scale,
            max_height_mm=args.max_height_mm,
        )
    else:
        print(f"[Info] Train CSV not found at {args.train_csv}; skipping train preprocessing.")

    if args.test_csv.exists():
        process_dataset(
            csv_path=args.test_csv,
            raw_root=args.test_root,
            processed_root=args.processed_root,
            mode="Test",
            img_size=args.img_size,
            crop_scale=args.crop_scale,
            max_height_mm=args.max_height_mm,
        )
    else:
        print(f"[Info] Test CSV not found at {args.test_csv}; skipping test preprocessing.")


if __name__ == "__main__":
    main()