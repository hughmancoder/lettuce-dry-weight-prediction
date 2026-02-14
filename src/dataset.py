from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TARGET_COLUMNS = ("DryWeightShoot", "FreshWeightShoot", "Height", "LeafArea")


class LettuceDataset(Dataset):
    """RGB-D lettuce dataset with optional training augmentations."""

    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        root_dir: Union[str, Path],
        mode: str = "train",
        augment: bool = False,
    ) -> None:
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        self.root_dir = Path(root_dir)
        self.mode = mode
        self.augment = bool(augment and mode == "train")

        rows: List[Dict] = []
        missing_ids: List[int] = []

        for _, row in df.iterrows():
            image_id = int(row["image_id"])
            rgb_path = self.root_dir / "RGB" / f"{image_id}.png"
            depth_path = self.root_dir / "Depth" / f"{image_id}.npy"
            mask_path = self.root_dir / "Mask" / f"{image_id}.png"

            if rgb_path.exists() and depth_path.exists():
                record = {
                    "image_id": image_id,
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "mask_path": mask_path,
                }

                if self.mode == "train":
                    for col in TARGET_COLUMNS:
                        record[col] = float(row[col])

                rows.append(record)
            else:
                missing_ids.append(image_id)

        if missing_ids:
            preview = ", ".join(map(str, missing_ids[:10]))
            suffix = "..." if len(missing_ids) > 10 else ""
            print(
                f"[Dataset] Skipping {len(missing_ids)} samples with missing processed files: {preview}{suffix}"
            )

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def _load_triplet(self, row: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb_bgr = cv2.imread(str(row["rgb_path"]), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = np.load(str(row["depth_path"])).astype(np.float32)
        depth = np.clip(depth, 0.0, 1.0)

        if row["mask_path"].exists():
            mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = (depth > 0).astype(np.float32)

        return rgb, depth, mask

    def _apply_augmentation(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = depth.shape

        if np.random.rand() < 0.5:
            rgb = cv2.flip(rgb, 1)
            depth = cv2.flip(depth, 1)
            mask = cv2.flip(mask, 1)

        if np.random.rand() < 0.5:
            rgb = cv2.flip(rgb, 0)
            depth = cv2.flip(depth, 0)
            mask = cv2.flip(mask, 0)

        if np.random.rand() < 0.8:
            angle = float(np.random.uniform(0.0, 360.0))
            mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)

            rgb = cv2.warpAffine(
                rgb,
                mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            depth = cv2.warpAffine(
                depth,
                mat,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            mask = cv2.warpAffine(
                mask,
                mat,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )

        if np.random.rand() < 0.8:
            brightness = float(np.random.uniform(-0.08, 0.08))
            contrast = float(np.random.uniform(0.8, 1.2))
            rgb = np.clip(rgb * contrast + brightness, 0.0, 1.0)

        return rgb, depth, mask

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_id = row["image_id"]

        rgb, depth, mask = self._load_triplet(row)

        if self.augment:
            rgb, depth, mask = self._apply_augmentation(rgb, depth, mask)

        rgb *= mask[..., None]
        depth *= mask

        rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
        depth_tensor = torch.from_numpy(depth[None, ...]).float()

        if self.mode == "train":
            targets = {
                col: torch.tensor(row[col], dtype=torch.float32)
                for col in TARGET_COLUMNS
            }
            return rgb_tensor, depth_tensor, targets, image_id

        return rgb_tensor, depth_tensor, image_id
