import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from src.dataset import LettuceDataset
from src.models import LettuceNet


def filter_existing_samples(df: pd.DataFrame, processed_root: Path) -> pd.DataFrame:
    valid_rows = []
    missing = []

    for _, row in df.iterrows():
        image_id = int(row["image_id"])
        rgb_path = processed_root / "RGB" / f"{image_id}.png"
        depth_path = processed_root / "Depth" / f"{image_id}.npy"
        if rgb_path.exists() and depth_path.exists():
            valid_rows.append(row)
        else:
            missing.append(image_id)

    if missing:
        preview = ", ".join(map(str, missing[:10]))
        suffix = "..." if len(missing) > 10 else ""
        print(f"[Warning] Excluding {len(missing)} rows with missing processed files: {preview}{suffix}")

    return pd.DataFrame(valid_rows).reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_df = pd.read_csv(args.train_csv)
    train_df = filter_existing_samples(train_df, args.processed_root)

    if len(train_df) < args.folds:
        raise ValueError(
            f"Not enough valid samples after filtering ({len(train_df)}) for {args.folds} folds."
        )

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    oof_preds = np.zeros(len(train_df))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df), 1):
        print(f"Fold {fold} Training...")
        
        train_ds = LettuceDataset(train_df.iloc[tr_idx], args.processed_root, is_training=True)
        val_ds = LettuceDataset(train_df.iloc[va_idx], args.processed_root, is_training=False)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        model = LettuceNet().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.L1Loss() # MAE

        best_mae = float('inf')

        for epoch in range(args.epochs):
            model.train()
            for imgs, lbls in train_loader:
                imgs = imgs.to(device)
                lbls = {k: v.to(device) for k, v in lbls.items()}
                
                optimizer.zero_grad()
                out = model(imgs)
                
                # Combined Loss (Multi-task weighting)
                loss = criterion(out['dry'], lbls['dry']) + \
                       0.3 * criterion(out['fresh'], lbls['fresh']) + \
                       0.1 * criterion(out['height'], lbls['height'])
                
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            preds = []
            with torch.no_grad():
                for imgs, _ in val_loader:
                    out = model(imgs.to(device))
                    preds.extend(out['dry'].cpu().numpy())
            
            val_mae = mean_absolute_error(train_df.iloc[va_idx]['DryWeightShoot'], preds)
            if val_mae < best_mae:
                best_mae = val_mae
                torch.save(model.state_dict(), args.weights_dir / f"model_fold_{fold}.pth")
                oof_preds[va_idx] = preds

        print(f"Fold {fold} Result: {best_mae:.4f}")

    # Save outputs
    train_df['pred_dry'] = oof_preds
    train_df.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    print(f"Final OOF MAE: {mean_absolute_error(train_df['DryWeightShoot'], oof_preds):.4f}")

if __name__ == "__main__":
    main()
