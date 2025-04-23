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

        print(f"Loading {self.dataset_name}...")
        for filename in self.original_filenames:
            high_res_path = os.path.join(self.high_res_dir, filename)
            low_res_path = os.path.join(self.low_res_dir, filename)

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

            ### NORMALIZE ALL IMAGES AGAIN ###
            # # Apply the final transform (ToTensor), normalize each image individually, and store
            # if self.transform:
            #     for lr, hr in zip(processed_lr, processed_hr):
            #         # Convert PIL images to tensors
            #         lr_tensor = self.transform(lr)
            #         hr_tensor = self.transform(hr)

            #         # Normalize Low-Resolution Tensor
            #         min_val_lr = torch.min(lr_tensor)
            #         max_val_lr = torch.max(lr_tensor)
            #         if max_val_lr > min_val_lr:
            #             lr_tensor_normalized = (lr_tensor - min_val_lr) / (max_val_lr - min_val_lr)
            #         else:
            #             # Handle constant image (set to 0 or handle as appropriate)
            #             lr_tensor_normalized = torch.zeros_like(lr_tensor)
            #         self.low_res_images.append(lr_tensor_normalized) # Store normalized tensor

            #         # Normalize High-Resolution Tensor
            #         min_val_hr = torch.min(hr_tensor)
            #         max_val_hr = torch.max(hr_tensor)
            #         if max_val_hr > min_val_hr:
            #             hr_tensor_normalized = (hr_tensor - min_val_hr) / (max_val_hr - min_val_hr)
            #         else:
            #             # Handle constant image
            #             hr_tensor_normalized = torch.zeros_like(hr_tensor)
            #         self.high_res_images.append(hr_tensor_normalized) # Store normalized tensor
            # else:
            #      # If no transform, store PIL images (normalization not applied)
            #     self.low_res_images.extend(processed_lr)
            #     self.high_res_images.extend(processed_hr)

            # Apply the final transform (ToTensor) and store
            if self.transform:
                for lr, hr in zip(processed_lr, processed_hr):
                    # Clamp tensors to [0, 1] after transformation
                    self.low_res_images.append(torch.clamp(self.transform(lr), min=0.0, max=1.0))
                    self.high_res_images.append(torch.clamp(self.transform(hr), min=0.0, max=1.0))
            else:
                 # If no transform, store PIL images (not recommended for training/eval)
                 # Note: Clamping is not applied here as they are PIL images
                self.low_res_images.extend(processed_lr)
                self.high_res_images.extend(processed_hr)
        print(f"Finished loading {self.dataset_name}. Total samples: {len(self.low_res_images)}")

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        return self.low_res_images[idx], self.high_res_images[idx]

    def verify_clamped_values(self):
        """Verifies the min/max values of the loaded and clamped tensors."""
        print(f"\n--- Verifying Clamped Values for {self.dataset_name} ---")
        if not self.low_res_images or not self.high_res_images or not isinstance(self.low_res_images[0], torch.Tensor):
            print("No tensors loaded or transform not applied, cannot verify.")
            return

        min_lr, max_lr = float('inf'), float('-inf')
        min_hr, max_hr = float('inf'), float('-inf')

        for tensor in self.low_res_images:
            current_min_lr = torch.min(tensor).item()
            current_max_lr = torch.max(tensor).item()
            if current_min_lr < min_lr: min_lr = current_min_lr
            if current_max_lr > max_lr: max_lr = current_max_lr

        for tensor in self.high_res_images:
            current_min_hr = torch.min(tensor).item()
            current_max_hr = torch.max(tensor).item()
            if current_min_hr < min_hr: min_hr = current_min_hr
            if current_max_hr > max_hr: max_hr = current_max_hr

        # Check if values are actually clamped close to 0 and 1
        lr_clamped_ok = abs(min_lr - 0.0) < 1e-6 and abs(max_lr - 1.0) < 1e-6
        hr_clamped_ok = abs(min_hr - 0.0) < 1e-6 and abs(max_hr - 1.0) < 1e-6

        print(f"Low-Res Images (Clamped): Min={min_lr:.8f}, Max={max_lr:.8f} -> Clamped OK: {lr_clamped_ok}")
        print(f"High-Res Images (Clamped): Min={min_hr:.8f}, Max={max_hr:.8f} -> Clamped OK: {hr_clamped_ok}")
        print("----------------------------------------------")


def create_loader(dataset, batch_size, shuffle_data=True, num_workers=0, pin_memory=False):
    """Creates a DataLoader."""
    # Note: Reproducibility is primarily handled by set_seed in main.py
    # The manual_seed here might be redundant if set globally, but harmless.
    # Consider num_workers > 0 for performance, but check compatibility with multiprocessing needs.
    # torch.manual_seed(0) # This might interfere with global seed setting, removed for now.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_data,
        num_workers=num_workers, # Adjust based on system capabilities and multiprocessing plans
        pin_memory=pin_memory # Set to True if using GPU for potential speedup
    )