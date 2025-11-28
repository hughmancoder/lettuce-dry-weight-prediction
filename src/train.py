import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import dataset  # Import the new dataset file
import models.RGBDResNet as RGBDResNet

PROCESSED_CSV = 'data/Processed/Train/Train.csv'
PROCESSED_ROOT = 'data/Processed/Train'
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 16
EPOCHS = 50
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Training on {DEVICE}")
    
    # 1. Prepare Data
    full_dataset = dataset.LettuceDataset(PROCESSED_CSV, PROCESSED_ROOT, mode='train')
    
    # Simple split for validation
    train_size = int((1 - VALIDATION_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Model
    model = RGBDResNet.RGBDResNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss() # MAE

    # 3. Train Loop
    best_mae = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(imgs).squeeze()
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                preds = model(imgs).squeeze()
                val_loss += criterion(preds, targets).item()
        
        train_mae = train_loss / len(train_loader)
        val_mae = val_loss / len(val_loader)
        
        print(f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")
        
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model")

if __name__ == "__main__":
    main()