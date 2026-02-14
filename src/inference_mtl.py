import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_lettuce_net_mtl import LettuceDataset, LettuceNet, filter_existing_samples


def detect_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_fold_files(weights_dir: Path) -> List[Path]:
    return sorted(weights_dir.glob("model_fold_*.pth"))


def predict_one_fold(model_path: Path, loader: DataLoader, device: torch.device) -> np.ndarray:
    model = LettuceNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            out = model(images)
            preds.append(out["dry"].detach().cpu().numpy())

    if not preds:
        return np.array([], dtype=np.float32)
    return np.concatenate(preds, axis=0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LettuceNet MTL fold-ensemble inference.")
    parser.add_argument("--test-csv", type=Path, default=Path("data/Test/final.csv"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/Processed/Test"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights/lettuce_net_mtl"))
    parser.add_argument("--output-file", type=Path, default=Path("outputs/lettuce_net_mtl/prediction.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    print(f"Device: {device}")

    raw_test_df = pd.read_csv(args.test_csv)
    if "image_id" not in raw_test_df.columns:
        raise ValueError(f"Column 'image_id' not found in {args.test_csv}")

    test_df = filter_existing_samples(raw_test_df, args.processed_root)
    if len(test_df) == 0:
        raise RuntimeError("No test samples found with processed files. Run preprocessing first.")

    test_ds = LettuceDataset(test_df, args.processed_root, is_training=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    fold_paths = list_fold_files(args.weights_dir)
    if not fold_paths:
        raise FileNotFoundError(
            f"No MTL fold models found in {args.weights_dir}. Expected files like model_fold_1.pth."
        )

    preds_per_fold = []
    for fold_path in fold_paths:
        print(f"Running MTL fold model: {fold_path.name}")
        preds_per_fold.append(predict_one_fold(fold_path, test_loader, device))

    final_preds = np.mean(np.stack(preds_per_fold, axis=0), axis=0)
    final_preds = np.clip(final_preds, 0.0, None)

    pred_map: Dict[int, float] = {
        int(image_id): float(pred)
        for image_id, pred in zip(test_df["image_id"].astype(int).tolist(), final_preds.tolist())
    }
    fallback = float(np.mean(final_preds)) if len(final_preds) else 0.0

    submission_df = raw_test_df[["image_id"]].copy()
    submission_df["DryWeightShoot"] = [
        pred_map.get(int(image_id), fallback)
        for image_id in submission_df["image_id"].astype(int).tolist()
    ]

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(args.output_file, index=False)

    summary_path = args.weights_dir / "training_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        print(f"OOF MAE from training summary: {summary.get('overall_oof_mae', 'n/a')}")

    print(f"Saved predictions to: {args.output_file}")
    print(submission_df.head())


if __name__ == "__main__":
    main()
