import os
import torch
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt # Keep import here if main directly calls plotting

# Import modules from src directory
from src.dataset import TIFFDataset, create_loader
from src.model import SuperResolutionNet
from src.losses import CombinedLoss
from src.training import train_model
from src.evaluation import evaluate_model, visualize_single_result
from src.utils import set_seed, get_device, get_timestamp # Import necessary utils

def main():
    # --- Configuration Variables ---
    # --- Data Paths ---
    TRAIN_HIGH_RES_DIR = 'data/train/high-res'
    TRAIN_LOW_RES_DIR = 'data/train/low-res'
    VAL_HIGH_RES_DIR = 'data/val/high-res'
    VAL_LOW_RES_DIR = 'data/val/low-res'

    # --- Model Saving & Results ---
    SAVE_DIR = 'saved_models'  # Directory to save model checkpoints
    RESULTS_DIR = 'results'    # Directory to save plots and evaluation outputs

    # --- Reproducibility ---
    SEED = 0  # Seed for random number generators (-1 for random initialization)

    # --- Training Hyperparameters ---
    BATCH_SIZE = 8            # Number of samples per batch
    LEARNING_RATE = 1e-4      # Initial learning rate for the optimizer
    NUM_EPOCHS = 2          # Maximum number of training epochs
    PATIENCE_EARLY_STOPPING = 15 # Epochs to wait for improvement before stopping
    SCHEDULER_PATIENCE = 5    # Epochs to wait for improvement before reducing LR
    SCHEDULER_FACTOR = 0.1    # Factor by which to reduce LR (new_lr = lr * factor)
    LOSS_ALPHA = 0.5          # Weight for MSE component of the combined loss
    LOSS_BETA = 0.5           # Weight for SSIM component of the combined loss

    # --- Hardware & Performance ---
    USE_AMP = True            # Use Automatic Mixed Precision (True/False)
    NUM_WORKERS = 0           # Number of worker processes for DataLoader (0 for main process)
                              # Increase for potential speedup, check system compatibility
    PIN_MEMORY = True         # Use pinned memory for DataLoader (potentially faster data transfer to GPU)

    # --- Evaluation ---
    VISUALIZATION_INDEX = 1   # Index of the validation image to visualize after training


    # --- Setup ---
    print("--- Initializing Setup ---")
    set_seed(SEED)
    device = get_device()
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Define image transform (applies ToTensor and ensures output is float)
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts PIL image to FloatTensor in range [0, 1]
    ])

    # Create datasets
    train_dataset = TIFFDataset(
        high_res_dir=TRAIN_HIGH_RES_DIR,
        low_res_dir=TRAIN_LOW_RES_DIR,
        transform=transform,
        augment=True, # Enable augmentation for training
        dataset_name="Training Set"
    )
    val_dataset = TIFFDataset(
        high_res_dir=VAL_HIGH_RES_DIR,
        low_res_dir=VAL_LOW_RES_DIR,
        transform=transform,
        augment=False, # No augmentation for validation
        dataset_name="Validation Set"
    )

    # Verify data clamping (should be [0, 1] after ToTensor)
    train_dataset.verify_clamped_values()
    val_dataset.verify_clamped_values()

    # Create DataLoaders
    train_loader = create_loader(
        train_dataset, BATCH_SIZE, shuffle_data=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY and (device.type == 'cuda')
    )
    val_loader = create_loader(
        val_dataset, BATCH_SIZE, shuffle_data=False, # No shuffling for validation
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY and (device.type == 'cuda')
    )

    # Print dataset sizes
    print(f'\nNumber of original training images: {getattr(train_dataset, "num_original_images", "N/A")}')
    print(f'Number of training samples (after augmentation): {len(train_dataset)}')
    print(f'Number of original validation images: {getattr(val_dataset, "num_original_images", "N/A")}')
    print(f'Number of validation samples (no augmentation): {len(val_dataset)}')
    print(f'Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}')

    # Initialize Model
    print("\n--- Initializing Model ---")
    model = SuperResolutionNet().to(device)
    print(f"Model initialized on {device}.")
    # print(model) # Optional: Print model summary

    # Optimizer
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Loss Function (Criterion) - Ensure it uses the correct device
    criterion = CombinedLoss(alpha=LOSS_ALPHA, beta=LOSS_BETA, device=device).to(device)

    # Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(
        opt,
        mode='max', # Monitors the validation score (higher is better)
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE
    )

    # --- Start Training ---
    print("\n--- Starting Training ---")
    print(f"Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"Early Stopping Patience: {PATIENCE_EARLY_STOPPING}")
    print(f"Scheduler Patience: {SCHEDULER_PATIENCE}, Factor: {SCHEDULER_FACTOR}")
    print(f"Loss Weights (MSE Alpha, SSIM Beta): {LOSS_ALPHA}, {LOSS_BETA}")
    print(f"Using Automatic Mixed Precision: {USE_AMP}")

    best_score = train_model(
        model=model,
        opt=opt,
        criterion=criterion,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epoch=NUM_EPOCHS,
        patience=PATIENCE_EARLY_STOPPING,
        device=device,
        save_dir=SAVE_DIR,
        results_dir=RESULTS_DIR,
        use_amp=USE_AMP
    )

    print(f"\n--- Training Finished ---")
    print(f"Best validation score achieved during training: {best_score:.4f}")

    # --- Final Evaluation on Validation Set ---
    # The train_model function loads the best model state automatically if found.
    print("\n--- Starting Final Evaluation on Validation Set ---")
    final_sr_score, final_interp_score = evaluate_model(
        model=model,
        data_loader=val_loader,
        device=device,
        use_amp=USE_AMP # Use same AMP setting for consistency
    )

    # --- Visualize a Single Result ---
    visualize_single_result(
        model=model,
        dataset=val_dataset, # Use the validation dataset
        device=device,
        results_dir=RESULTS_DIR,
        index=VISUALIZATION_INDEX,
        use_amp=USE_AMP
    )

    print("\n--- Script Finished ---")

if __name__ == '__main__':
    # This ensures the main function runs only when the script is executed directly
    main()