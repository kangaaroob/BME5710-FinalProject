import os
import numpy as np
from PIL import Image
from tqdm import tqdm # Optional: for progress bar

def check_image_value_range(directories):
    """
    Checks the minimum and maximum pixel values across all TIFF images
    in the specified directories.

    Args:
        directories (list): A list of directory paths to scan.
    """
    overall_min = float('inf')
    overall_max = float('-inf')
    num_images_processed = 0
    num_images_failed = 0

    all_files = []
    for directory in directories:
        if not os.path.isdir(directory):
            print(f"Warning: Directory not found - {directory}")
            continue
        for filename in os.listdir(directory):
            if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
                all_files.append(os.path.join(directory, filename))

    if not all_files:
        print("No TIFF images found in the specified directories.")
        return

    print(f"Checking {len(all_files)} images...")
    pbar = tqdm(all_files, desc="Checking Images")
    for img_path in pbar:
        try:
            with Image.open(img_path) as img:
                # Convert image to NumPy array.
                # Ensure it handles different modes appropriately (e.g., grayscale 'L', 'I')
                img_array = np.array(img, dtype=np.float32) # Use float32 for calculations

                current_min = np.min(img_array)
                current_max = np.max(img_array)

                if current_min < overall_min:
                    overall_min = current_min
                if current_max > overall_max:
                    overall_max = current_max
                num_images_processed += 1
                pbar.set_postfix({'Current Min': f'{current_min:.4f}', 'Current Max': f'{current_max:.4f}'})

        except Exception as e:
            print(f"\nError processing image {img_path}: {e}")
            num_images_failed += 1
            continue # Skip to the next image if one fails

    print("\n--- Value Range Check Complete ---")
    if num_images_processed > 0:
        print(f"Processed {num_images_processed} images.")
        print(f"Overall Minimum Value Found: {overall_min}")
        print(f"Overall Maximum Value Found: {overall_max}")
    else:
        print("No images were successfully processed.")
    if num_images_failed > 0:
        print(f"Failed to process {num_images_failed} images.")

# --- Specify Directories to Check ---
# Make sure these paths are correct relative to where you run the script
# (Based on your codebase.txt paths)
high_res_train_dir = 'data/train/high-res'
low_res_train_dir = 'data/train/low-res'
high_res_val_dir = 'data/val/high-res'
low_res_val_dir = 'data/val/low-res'

dirs_to_check = [
    # high_res_train_dir,
    low_res_train_dir,
    # high_res_val_dir,
    low_res_val_dir
]

# --- Run the Check ---
check_image_value_range(dirs_to_check)