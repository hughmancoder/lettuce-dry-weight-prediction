import os
from pathlib import Path

import torch
import numpy as np
import cv2
import pandas as pd
from torch.utils.data import Dataset

class LettuceDataset(Dataset):
    def __init__(self, csv_file, root_dir, mode='train'):
        """
        Args:
            csv_file: Path to the PROCESSED csv file.
            root_dir: Path to the PROCESSED data directory (e.g., data/Processed/Train)
        """
        df = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.mode = mode

        valid_rows = []
        missing_ids = []
        for _, row in df.iterrows():
            img_id = str(int(row['image_id']))
            rgb_path = self.root_dir / 'RGB' / f"{img_id}.png"
            depth_path = self.root_dir / 'Depth' / f"{img_id}.npy"
            if rgb_path.exists() and depth_path.exists():
                valid_rows.append(row)
            else:
                missing_ids.append(img_id)

        if missing_ids:
            print(f"[Dataset] Skipping {len(missing_ids)} samples with missing files: {missing_ids[:5]}{'...' if len(missing_ids) > 5 else ''}")

        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    # called by torch utils DataLoader
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = str(int(row['image_id']))
        
        # Load Processed RGB (PNG)
        rgb_path = self.root_dir / 'RGB' / f"{img_id}.png"
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0 # RGB to 0-1

        # Load Processed Depth (.npy)
        # Note: We load .npy because preprocessing saved it as floats [0,1]
        depth_path = self.root_dir / 'Depth' / f"{img_id}.npy"
        depth = np.load(str(depth_path)) # Already [0,1] float32

        # Stack (H, W, 4)
        depth = np.expand_dims(depth, axis=-1)
        image = np.concatenate((rgb, depth), axis=-1)

        # To Tensor
        image = torch.tensor(image).permute(2, 0, 1).float()

        # Return
        if self.mode == 'train':
            target = row['DryWeightShoot']
            return image, torch.tensor(target, dtype=torch.float32)
        else:
            return image, img_id
