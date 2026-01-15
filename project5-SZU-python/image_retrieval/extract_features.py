"""
Gallery Feature Extraction Script
Extract features from all gallery images using ViT
"""
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add assignments path
SCRIPT_DIR = Path(__file__).resolve().parent
ASSIGNMENTS_PATH = SCRIPT_DIR.parent / 'assignments'
sys.path.insert(0, str(ASSIGNMENTS_PATH))

from dinov2_numpy import Dinov2Numpy
from preprocess_image import center_crop

def extract_features():
    # Paths
    weights_path = ASSIGNMENTS_PATH / 'vit-dinov2-base.npz'
    gallery_dir = SCRIPT_DIR / 'static' / 'gallery'
    features_dir = SCRIPT_DIR / 'features'
    output_path = features_dir / 'gallery_features.npz'

    # Create directories
    features_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading ViT model...")
    weights = np.load(str(weights_path))
    vit = Dinov2Numpy(weights)

    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in gallery_dir.iterdir()
                   if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No images found in {gallery_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Extract features
    features = []
    paths = []

    for img_path in tqdm(image_files, desc="Extracting features"):
        try:
            pixel_values = center_crop(str(img_path))
            feat = vit(pixel_values)[0]
            features.append(feat)
            paths.append(str(img_path))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Save features
    features = np.array(features)
    paths = np.array(paths)

    np.savez(str(output_path), features=features, paths=paths)
    print(f"Saved {len(features)} features to {output_path}")

if __name__ == '__main__':
    extract_features()
