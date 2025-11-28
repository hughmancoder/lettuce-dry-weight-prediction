import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import dataset
import models.RGBDResNet as RGBDResNet
import os

# --- Config ---
TEST_CSV_PATH = 'data/Processed/Test/Test.csv'
TEST_ROOT_DIR = 'data/Processed/Test'
MODEL_PATH = 'weights/best_model.pth'
OUTPUT_FILE = 'prediction.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("Starting Inference...")
    
    # Load Data
    # Note: Using mode='test' so it returns (image, image_id)
    test_ds = dataset.LettuceDataset(TEST_CSV_PATH, TEST_ROOT_DIR, mode='test')
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # Load Model
    model = RGBDResNet.RGBDResNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded.")
    else:
        print("Error: No model found! Run 'make train' first.")
        return

    model.eval()
    
    # Predict
    results = []
    
    with torch.no_grad():
        for image, img_id in tqdm(test_loader):
            image = image.to(DEVICE)
            
            # Predict
            output = model(image).item()
            
            # Ensure non-negative predictions (weight can't be negative)
            prediction = max(0.0, output)
            
            results.append({
                'image_id': int(img_id[0]), # Extract ID from tuple
                'DryWeightShoot': prediction
            })
            
    # Save 
    df = pd.DataFrame(results)
    # Ensure correct column order
    df = df[['image_id', 'DryWeightShoot']] 
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Submission saved to {OUTPUT_FILE}")
    print(df.head())

if __name__ == "__main__":
    main()