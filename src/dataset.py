import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class TIFFDataset(Dataset):
    def __init__(self, high_res_dir, low_res_dir, transform=None, augment=False, dataset_name="Dataset"):
        self.high_res_dir = high_res_dir
        self.low_res_dir = low_res_dir
        self.transform = transform
        self.augment = augment
        self.dataset_name = dataset_name
        self.original_filenames = sorted([f for f in os.listdir(high_res_dir) if f.endswith('.tif')])
        self.num_original_images = len(self.original_filenames)

        self.low_res_images = []
        self.high_res_images = []

        for filename in self.original_filenames:
            high_res_path = os.path.join(self.high_res_dir, filename)
            low_res_path = os.path.join(self.low_res_dir, filename)

            # Load original images
            try:
                hr_img = Image.open(high_res_path)
                lr_img = Image.open(low_res_path)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
                continue

            # Resize low-res to 128x128
            lr_img = lr_img.resize((128, 128), Image.BICUBIC)

            original_lr, original_hr = lr_img, hr_img
            processed_lr = []
            processed_hr = []

            if self.augment:
                # --- Apply transformations ---
                # 1) Original
                processed_lr.append(original_lr)
                processed_hr.append(original_hr)
                # 2) Rotated by 90 degrees
                processed_lr.append(original_lr.rotate(90))
                processed_hr.append(original_hr.rotate(90))
                # 3) Rotated by 180 degrees
                processed_lr.append(original_lr.rotate(180))
                processed_hr.append(original_hr.rotate(180))
                # 4) Rotated by 270 degrees
                processed_lr.append(original_lr.rotate(270))
                processed_hr.append(original_hr.rotate(270))

                # Apply flips
                lr_v_flip = original_lr.transpose(Image.FLIP_LEFT_RIGHT)
                hr_v_flip = original_hr.transpose(Image.FLIP_LEFT_RIGHT)
                lr_h_flip = original_lr.transpose(Image.FLIP_TOP_BOTTOM)
                hr_h_flip = original_hr.transpose(Image.FLIP_TOP_BOTTOM)

                # 5) Mirrored on the vertical axis (Flip Left/Right)
                processed_lr.append(lr_v_flip)
                processed_hr.append(hr_v_flip)
                # 6) Mirrored on the horizontal axis (Flip Top/Bottom)
                processed_lr.append(lr_h_flip)
                processed_hr.append(hr_h_flip)

                # Apply flips combined with rotation (unique combinations)
                # 7) Rotated by 90 degrees and mirrored on the vertical axis
                processed_lr.append(lr_v_flip.rotate(90))
                processed_hr.append(hr_v_flip.rotate(90))
                # 8) Rotated by 90 degrees and mirrored on the horizontal axis
                processed_lr.append(lr_h_flip.rotate(90))
                processed_hr.append(hr_h_flip.rotate(90))
            else:
                # If not augmenting, just use the original
                processed_lr.append(original_lr)
                processed_hr.append(original_hr)

            # Apply the final transform (ToTensor) and store
            if self.transform:
                for lr, hr in zip(processed_lr, processed_hr):
                    self.low_res_images.append(torch.clamp(self.transform(lr), min=0.0, max=1.0))
                    self.high_res_images.append(torch.clamp(self.transform(hr), min=0.0, max=1.0))
            else:
                 # If no transform, store PIL images (not recommended for training/eval)
                self.low_res_images.extend(processed_lr)
                self.high_res_images.extend(processed_hr)

    # Get the number of samples in the dataset (original or augmented)
    def __len__(self):
        return len(self.low_res_images)

    # Get the sample at the given index
    def __getitem__(self, idx):
        # Return pre-processed tensors
        return self.low_res_images[idx], self.high_res_images[idx]
    
    def verify_clamped_values(self):
        """
        Verifies the min/max values of the loaded and clamped tensors.
        """
        print(f"\n--- Verifying Clamped Values for {self.dataset_name} ---")
        if not self.low_res_images or not self.high_res_images:
            print("No images loaded to verify.")
            return

        min_lr, max_lr = float('inf'), float('-inf')
        min_hr, max_hr = float('inf'), float('-inf')

        # Check low-resolution images
        for tensor in self.low_res_images:
            current_min_lr = torch.min(tensor).item()
            current_max_lr = torch.max(tensor).item()
            if current_min_lr < min_lr: min_lr = current_min_lr
            if current_max_lr > max_lr: max_lr = current_max_lr

        # Check high-resolution images
        for tensor in self.high_res_images:
            current_min_hr = torch.min(tensor).item()
            current_max_hr = torch.max(tensor).item()
            if current_min_hr < min_hr: min_hr = current_min_hr
            if current_max_hr > max_hr: max_hr = current_max_hr

        print(f"Low-Res Images (Clamped): Min={min_lr:.8f}, Max={max_lr:.8f}")
        print(f"High-Res Images (Clamped): Min={min_hr:.8f}, Max={max_hr:.8f}")
        print("----------------------------------------------")


def create_loader(dataset, batch_size, shuffle_data=True, torch_seed=0):
    """Creates a DataLoader."""
    if torch_seed != -1:
        torch.manual_seed(torch_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_data
    )