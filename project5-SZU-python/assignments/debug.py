"""
DINOv2 NumPy Implementation Verification Script

This script verifies the correctness of the NumPy ViT implementation by:
1. Extracting features from test images (cat.jpg, dog.jpg)
2. Comparing with reference features from PyTorch DINOv2
3. Computing multiple error metrics to ensure numerical alignment
"""

import numpy as np
from dinov2_numpy import Dinov2Numpy
from preprocess_image import center_crop


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))


def compute_metrics(pred, ref, name):
    """Compute and display various error metrics."""
    # Flatten for comparison
    pred_flat = pred.flatten()
    ref_flat = ref.flatten()

    # Compute metrics
    max_abs_error = np.abs(pred_flat - ref_flat).max()
    mean_abs_error = np.abs(pred_flat - ref_flat).mean()
    mse = np.mean((pred_flat - ref_flat) ** 2)
    rmse = np.sqrt(mse)
    cos_sim = cosine_similarity(pred_flat, ref_flat)

    print(f"\n{'='*50}")
    print(f"  {name} Feature Verification")
    print(f"{'='*50}")
    print(f"  Feature shape:        {pred.shape}")
    print(f"  Max Absolute Error:   {max_abs_error:.6e}")
    print(f"  Mean Absolute Error:  {mean_abs_error:.6e}")
    print(f"  MSE:                  {mse:.6e}")
    print(f"  RMSE:                 {rmse:.6e}")
    print(f"  Cosine Similarity:    {cos_sim:.6f}")

    return cos_sim, max_abs_error


def main():
    print("\n" + "="*60)
    print("  DINOv2 NumPy Implementation - Verification Script")
    print("="*60)

    # Step 1: Load model
    print("\n[1/4] Loading ViT-DINOv2 weights...")
    weights = np.load("vit-dinov2-base.npz")
    vit = Dinov2Numpy(weights)
    print("      Model loaded successfully!")

    # Step 2: Extract features
    print("\n[2/4] Extracting features from test images...")

    cat_pixel_values = center_crop("./demo_data/cat.jpg")
    cat_feat = vit(cat_pixel_values)
    print("      - cat.jpg: feature extracted")

    dog_pixel_values = center_crop("./demo_data/dog.jpg")
    dog_feat = vit(dog_pixel_values)
    print("      - dog.jpg: feature extracted")

    # Step 3: Load reference features
    print("\n[3/4] Loading reference features (PyTorch DINOv2)...")
    ref_feats = np.load("./demo_data/cat_dog_feature.npy")
    cat_ref = ref_feats[0]
    dog_ref = ref_feats[1]
    print(f"      Reference shape: {ref_feats.shape}")

    # Step 4: Compare and verify
    print("\n[4/4] Comparing with reference features...")

    cat_cos, cat_err = compute_metrics(cat_feat, cat_ref, "Cat")
    dog_cos, dog_err = compute_metrics(dog_feat, dog_ref, "Dog")

    # Final verdict
    print("\n" + "="*60)
    print("  VERIFICATION SUMMARY")
    print("="*60)

    # Thresholds
    cos_threshold = 0.999

    cat_pass = cat_cos >= cos_threshold
    dog_pass = dog_cos >= cos_threshold
    all_pass = cat_pass and dog_pass

    print(f"\n  Cosine Similarity Threshold: {cos_threshold}")
    print(f"  Cat: {cat_cos:.6f} {'[PASS]' if cat_pass else '[FAIL]'}")
    print(f"  Dog: {dog_cos:.6f} {'[PASS]' if dog_pass else '[FAIL]'}")

    print("\n" + "-"*60)
    if all_pass:
        print("  [PASSED] VERIFICATION PASSED!")
        print("  Your NumPy ViT implementation is correctly aligned")
        print("  with PyTorch DINOv2.")
    else:
        print("  [FAILED] VERIFICATION FAILED!")
        print("  Please check your implementation.")
    print("-"*60 + "\n")

    return all_pass


if __name__ == "__main__":
    success = main()