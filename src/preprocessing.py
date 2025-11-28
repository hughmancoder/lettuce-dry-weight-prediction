import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

"""
Preprocessing
- Crops RGB and depth images to a standard region.
- Normalizes depth values to [0,1].
- Saves processed images into structured directories.
- Supports both full dataset and train/test subset processing.
"""

RAW_DATA_DIR = 'data'
PROCESSED_DATA_DIR = os.path.join(RAW_DATA_DIR, 'Processed')
RGB_DIRNAME = 'RGBImages'
DEPTH_DIRNAME = 'DepthImages'
RGB_PREFIX = 'RGB_'
DEPTH_PREFIX = 'Depth_'
IMG_SIZE = 224 # Standard input resolution for many well-known image classification models. Useful for transfer learning
CROP_SCALE = 0.7 # found through visualise.ipynb script

def center_square_crop(img, scale=1.0):
    h, w = img.shape[:2]
    base = min(h, w)
    crop = int(base * scale)

    cy, cx = h//2, w//2
    y1 = cy - crop//2
    x1 = cx - crop//2

    return img[y1:y1+crop, x1:x1+crop]

def preprocess_image(img_id, source_dir, save_dir, prefix, is_depth=False):
    """
    Reads, crops, normalises (if depth), and saves.
    """
    ext = 'png'
    filename = f"{img_id}.{ext}"
    input_filename = f"{prefix}{img_id}.{ext}"
    input_path = os.path.join(source_dir, input_filename)
    
    # Read Image
    if is_depth:
        # Load 16-bit depth (or similar)
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False

        # Get range from original dtype *before* casting
        info = np.iinfo(img.dtype)
        img = img.astype(np.float32) / float(info.max)   # normalise  depth image  to [0,1]
    else:
        # Load RGB (OpenCV gives BGR)
        img = cv2.imread(input_path)
        if img is None:
            return False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = center_square_crop(img, scale=CROP_SCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Save processed image into structured directories
    save_path = os.path.join(save_dir, filename)
    
    if is_depth:
        # Save normalised depth as .npy to preserve [0,1] float precision after normalisation
        np.save(save_path.replace('.png', '.npy'), img)
    else:
        # Save RGB as standard PNG (convert back to BGR for OpenCV)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)
        
    return True


def process_dataset(csv_path, raw_root, processed_root, mode='Train'):
    print(f"Processing {mode} set...")
    df = pd.read_csv(csv_path)
    
    # Setup directories
    rgb_out = os.path.join(processed_root, mode, 'RGB')
    depth_out = os.path.join(processed_root, mode, 'Depth')
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(depth_out, exist_ok=True)
    
    # Copy CSV to processed folder
    df.to_csv(os.path.join(processed_root, mode, f'{mode}.csv'), index=False)

    # Loop through data
    raw_rgb = os.path.join(raw_root, RGB_DIRNAME)
    raw_depth = os.path.join(raw_root, DEPTH_DIRNAME)

    for img_id in tqdm(df['image_id']):
        img_id = str(int(img_id))
        
        # Process RGB
        preprocess_image(img_id, raw_rgb, rgb_out, prefix=RGB_PREFIX, is_depth=False)
        # Process Depth
        preprocess_image(img_id, raw_depth, depth_out, prefix=DEPTH_PREFIX, is_depth=True)

if __name__ == "__main__":
    # Ensure directories exist
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    TRAIN_CSV = 'data/Training/Train.csv'
    TEST_CSV = 'data/Test/Test.csv'

    if os.path.exists(TRAIN_CSV):
        process_dataset(TRAIN_CSV, 'data/Training', PROCESSED_DATA_DIR, mode='Train')
        train_processed_csv = os.path.join(PROCESSED_DATA_DIR, 'Train', 'Train.csv')
    
    if os.path.exists(TEST_CSV):
        process_dataset(TEST_CSV, 'data/Test', PROCESSED_DATA_DIR, mode='Test')
