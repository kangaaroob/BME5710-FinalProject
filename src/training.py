import os
import torch
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm # Use standard tqdm
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from .utils import get_timestamp, save_plot # Import from utils

# Helper function for one training epoch
def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp):
    model.train() # Set model to training mode
    total_train_loss = 0.0
    # Use tqdm for progress bar
    pbar = tqdm(train_loader, desc="Training Epoch", leave=False)
    for x_tr_batch, y_tr_batch in pbar:
        x_tr_batch, y_tr_batch = x_tr_batch.to(device), y_tr_batch.to(device)

        optimizer.zero_grad()

        # --- Mixed Precision: Forward pass with autocast ---
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            y_hat_tr_batch = model(x_tr_batch)
            loss = criterion(y_hat_tr_batch, y_tr_batch)

        if torch.isnan(loss):
             print(f"NaN loss detected during training. Skipping batch.")
             print("x_tr_batch stats:", torch.min(x_tr_batch), torch.max(x_tr_batch), torch.mean(x_tr_batch))
             print("y_tr_batch stats:", torch.min(y_tr_batch), torch.max(y_tr_batch), torch.mean(y_tr_batch))
             print("y_hat_tr_batch stats:", torch.min(y_hat_tr_batch), torch.max(y_hat_tr_batch), torch.mean(y_hat_tr_batch))
             continue # Skip this batch

        # --- Mixed Precision: Scale loss and backward pass ---
        # scaler.scale(loss).backward() will handle scaling if use_amp is True
        scaler.scale(loss).backward()

        # --- Mixed Precision: Scaler step and update ---
        # scaler.step() and scaler.update() handle the optimizer step correctly with AMP
        scaler.step(optimizer)
        scaler.update()

        batch_loss = loss.item()
        total_train_loss += batch_loss
        pbar.set_postfix({'batch_loss': f'{batch_loss:.4f}'}) # Update progress bar description

    avg_train_loss = total_train_loss / len(train_loader)
    return avg_train_loss

# Helper function for one validation epoch
def validate_one_epoch(model, val_loader, criterion, device, use_amp):
    model.eval() # Set model to evaluation mode
    total_val_loss = 0.0
    total_val_psnr = 0.0
    total_val_ssim = 0.0
    num_samples = 0

    # Initialize metrics here for each validation run
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    pbar = tqdm(val_loader, desc="Validation Epoch", leave=False)
    with torch.no_grad():
        for x_val_batch, y_val_batch in pbar:
            x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)
            batch_size = x_val_batch.size(0)

            # --- Mixed Precision: Validation forward pass with autocast ---
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                y_hat_val_batch = model(x_val_batch)
                val_loss = criterion(y_hat_val_batch, y_val_batch)

            if torch.isnan(val_loss):
                print(f"NaN loss detected during validation. Skipping batch.")
                continue # Skip this batch

            # Accumulate total loss correctly, considering potential variations in last batch size
            total_val_loss += val_loss.item() * batch_size

            # Calculate metrics
            # Ensure metrics handle potential float16 inputs if use_amp is True
            # TorchMetrics generally handles this, but explicit casting can be added if needed:
            # psnr = psnr_metric(y_hat_val_batch.float(), y_val_batch.float())
            # ssim = ssim_metric(y_hat_val_batch.float(), y_val_batch.float())
            psnr = psnr_metric(y_hat_val_batch, y_val_batch)
            ssim = ssim_metric(y_hat_val_batch, y_val_batch)

            if torch.isnan(psnr) or torch.isnan(ssim):
                 print(f"NaN metric detected during validation. Skipping batch metrics.")
                 # Optionally investigate inputs/outputs
                 continue # Skip metrics for this batch

            total_val_psnr += psnr.item() * batch_size
            total_val_ssim += ssim.item() * batch_size
            num_samples += batch_size # Increment sample count correctly

    if num_samples == 0:
        print("Warning: No samples processed during validation.")
        return 0.0, 0.0, 0.0, 0.0 # Avoid division by zero

    avg_val_loss = total_val_loss / num_samples
    avg_val_psnr = total_val_psnr / num_samples
    avg_val_ssim = total_val_ssim / num_samples
    avg_val_score = avg_val_psnr + (40 * avg_val_ssim)

    return avg_val_loss, avg_val_score, avg_val_psnr, avg_val_ssim


# Main training loop function
def train_model(model, opt, criterion, scheduler, train_loader, val_loader, num_epoch, patience, device, save_dir, results_dir, use_amp):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True) # Ensure results directory exists
    epoch_nums, avg_train_losses, avg_val_losses, avg_val_scores = [], [], [], []
    best_val_score = -float('inf')
    epochs_no_improve = 0
    best_model_state = None

    # Initialize GradScaler - enabled based on use_amp flag
    # If use_amp is False, scaler operations are no-ops.
    scaler = torch.amp.GradScaler(enabled=use_amp)

    overall_pbar = tqdm(range(num_epoch), desc="Overall Training Progress")

    for epoch in overall_pbar:
        # --- Training ---
        avg_train_loss = train_one_epoch(model, train_loader, criterion, opt, device, scaler, use_amp)

        # --- Validation ---
        avg_val_loss, avg_val_score, avg_val_psnr, avg_val_ssim = validate_one_epoch(
            model, val_loader, criterion, device, use_amp
        )

        # Store metrics for plot
        epoch_nums.append(epoch + 1)
        avg_train_losses.append(avg_train_loss)
        avg_val_losses.append(avg_val_loss)
        avg_val_scores.append(avg_val_score)

        # --- Update Progress Bar ---
        current_lr = opt.param_groups[0]['lr']
        overall_pbar.set_description(
            f"LR: {current_lr:.1e} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Score: {avg_val_score:.2f} (PSNR:{avg_val_psnr:.2f}, SSIM:{avg_val_ssim:.4f}) | "
            f"Best Score: {best_val_score:.2f} | Epochs Since Best: {epochs_no_improve}"
        )

        # --- Learning Rate Scheduler Step ---
        # Step based on the validation score
        scheduler.step(avg_val_score)

        # --- Checkpointing and Early Stopping ---
        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"\nEpoch {epoch+1}: New best validation score: {best_val_score:.2f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            break

    if best_model_state:
        timestamp = get_timestamp()
        score_str = f"{best_val_score:.2f}".replace('.', 'p')
        filename = f"best_model_score_{score_str}_epoch_{epoch+1}_time_{timestamp}.pth"
        save_path = os.path.join(save_dir, filename)
        torch.save(best_model_state, save_path)
        print(f"Best model saved to '{save_path}'")
    
    # --- Final Plot ---
    print("\nGenerating final training plot...")
    fig_final, ax1_final = plt.subplots(figsize=(12, 7)) # Adjusted size
    ax2_final = ax1_final.twinx()

    # Plot losses
    line1_final, = ax1_final.plot(epoch_nums, avg_train_losses, 'r-', label='Training Loss')
    line2_final, = ax1_final.plot(epoch_nums, avg_val_losses, 'r--', label='Validation Loss')
    ax1_final.set_xlabel('Epochs')
    ax1_final.set_ylabel('Loss', color='tab:red')
    ax1_final.tick_params(axis='y', labelcolor='tab:red')
    ax1_final.grid(True, axis='y', linestyle='--', alpha=0.7) # Add grid for loss axis

    # Plot validation score
    line3_final, = ax2_final.plot(epoch_nums, avg_val_scores, 'b-', label='Validation Score')
    ax2_final.set_ylabel('Validation Score (PSNR + 40*SSIM)', color='tab:blue')
    ax2_final.tick_params(axis='y', labelcolor='tab:blue')

    # Combine legends
    lines = [line1_final, line2_final, line3_final]
    labels = [l.get_label() for l in lines]
    ax1_final.legend(lines, labels, loc='center right') # Adjust location

    plt.title('Final Training and Validation Progress')
    fig_final.tight_layout()

    # Save and show plot
    timestamp = get_timestamp()
    score_str = f"{best_val_score:.2f}".replace('.', 'p') if best_val_score > -float('inf') else "no_score"
    plot_filename = f"training_plot_score_{score_str}_time_{timestamp}.png"
    save_plot(fig_final, results_dir, plot_filename) # Use save_plot from utils
    plt.show() # Show the plot

    # Load the best model state back into the model if it exists
    if best_model_state:
        print(f"\nLoading best model state (Score: {best_val_score:.2f}) for final evaluation.")
        model.load_state_dict(best_model_state)
    else:
        print("\nWarning: No best model state found. Using the model's current state for final evaluation.")

    return best_val_score