import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Use standard tqdm
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from .utils import get_timestamp, save_plot # Import from utils

def evaluate_model(model, data_loader, device, use_amp):
    """Computes PSNR, SSIM, and Score over a dataset using a DataLoader."""
    model.eval() # Set model to evaluation mode
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_psnr_sr = 0.0
    total_ssim_sr = 0.0
    total_psnr_interp = 0.0
    total_ssim_interp = 0.0
    num_samples = 0

    eval_pbar = tqdm(data_loader, desc="Evaluating Model")
    with torch.no_grad():
        for x_batch, y_batch in eval_pbar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device) # x_batch is 128x128
            batch_size_actual = x_batch.size(0)

            # Interpolate low-res for baseline comparison (outside autocast)
            # Ensure interpolation happens on the correct device
            x_batch_interpolated = torch.nn.functional.interpolate(
                x_batch, scale_factor=2, mode='bicubic', align_corners=False
            )

            # --- Mixed Precision: Evaluation forward pass with autocast ---
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                y_hat_batch = model(x_batch) # Model outputs 256x256

            # Calculate metrics for super-resolved
            # Ensure metrics handle potential float16 inputs if use_amp is True
            psnr_sr = psnr_metric(y_hat_batch, y_batch)
            ssim_sr = ssim_metric(y_hat_batch, y_batch)

            if torch.isnan(psnr_sr) or torch.isnan(ssim_sr):
                 print(f"NaN metric detected during SR evaluation. Skipping batch metrics.")
                 continue # Skip metrics for this batch

            total_psnr_sr += psnr_sr.item() * batch_size_actual
            total_ssim_sr += ssim_sr.item() * batch_size_actual

            # Calculate metrics for interpolated (baseline)
            psnr_interp = psnr_metric(x_batch_interpolated, y_batch)
            ssim_interp = ssim_metric(x_batch_interpolated, y_batch)

            if torch.isnan(psnr_interp) or torch.isnan(ssim_interp):
                 print(f"NaN metric detected during Interp evaluation. Skipping batch metrics.")
                 # Need to ensure num_samples isn't incremented if skipping SR metrics too
            else:
                total_psnr_interp += psnr_interp.item() * batch_size_actual
                total_ssim_interp += ssim_interp.item() * batch_size_actual
                num_samples += batch_size_actual # Increment samples only if all metrics are valid for the batch

            # Update progress bar description with running averages (optional)
            if num_samples > 0:
                running_psnr_sr = total_psnr_sr / num_samples
                running_ssim_sr = total_ssim_sr / num_samples
                eval_pbar.set_postfix({'PSNR_SR': f'{running_psnr_sr:.2f}', 'SSIM_SR': f'{running_ssim_sr:.4f}'})

    if num_samples == 0:
        print("\nWarning: No samples processed during final evaluation.")
        return 0.0, 0.0 # Avoid division by zero

    # Calculate averages
    avg_psnr_sr = total_psnr_sr / num_samples
    avg_ssim_sr = total_ssim_sr / num_samples
    avg_psnr_interp = total_psnr_interp / num_samples
    avg_ssim_interp = total_ssim_interp / num_samples

    # Calculate scores
    avg_score_sr = avg_psnr_sr + (40 * avg_ssim_sr)
    avg_score_interp = avg_psnr_interp + (40 * avg_ssim_interp)

    # Print results
    print("\n--- Final Evaluation Results ---")
    print(f'Device Used: {device}, Mixed Precision Enabled: {use_amp}')
    print(f'Evaluated on {num_samples} samples.')
    print(f'Average PSNR (interpolated): {avg_psnr_interp:.2f} dB')
    print(f'Average PSNR (super-resolved): {avg_psnr_sr:.2f} dB')
    print(f'Average SSIM (interpolated): {avg_ssim_interp:.4f}')
    print(f'Average SSIM (super-resolved): {avg_ssim_sr:.4f}')
    print(f'Average Score (interpolated): {avg_score_interp:.2f}')
    print(f'Average Score (super-resolved): {avg_score_sr:.2f}')
    print("---------------------------------")

    return avg_score_sr, avg_score_interp

def visualize_single_result(model, dataset, device, results_dir, index=1, use_amp=False):
    """Loads a single image pair, applies the model, and visualizes/saves results."""
    print(f"\n--- Visualizing Result for Validation Image Index: {index} ---")
    model.eval()
    os.makedirs(results_dir, exist_ok=True)

    try:
        val_low_res, val_high_res = dataset[index]  # Input (128x128), Ground truth (256x256)
    except IndexError:
        print(f"Error: Index {index} out of bounds for the dataset (size {len(dataset)}). Using index 0.")
        index = 0
        val_low_res, val_high_res = dataset[index]

    val_low_res, val_high_res = val_low_res.to(device), val_high_res.to(device)

    # Keep the interpolated version for visualization comparison
    # Ensure interpolation happens on the correct device
    val_low_res_interpolated = torch.nn.functional.interpolate(
        val_low_res.unsqueeze(0), scale_factor=2, mode='bicubic', align_corners=False
    ).squeeze(0)

    # Apply the trained model to the original low-res image
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            # Add batch dimension for model, then remove it
            val_super_res = model(val_low_res.unsqueeze(0)).squeeze(0)

    # Move tensors to CPU and convert to numpy for visualization
    val_low_res_np = val_low_res_interpolated.squeeze().cpu().numpy()
    val_high_res_np = val_high_res.squeeze().cpu().numpy()
    # Detach super-res tensor before moving to CPU
    val_super_res_np = val_super_res.detach().squeeze().cpu().numpy()

    # Clamp super-resolved output just before visualization/error calc for safety
    val_super_res_np = np.clip(val_super_res_np, 0.0, 1.0)

    # Plot an example image and error maps
    fig, ax = plt.subplots(2, 3, figsize=(12, 8)) # Adjusted size

    # Plot images
    im_args = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}
    ax[0, 0].imshow(val_high_res_np, **im_args)
    ax[0, 0].set_title('Ground Truth (High-Res)')
    ax[0, 0].axis('off')

    ax[0, 1].imshow(val_low_res_np, **im_args)
    ax[0, 1].set_title('Bicubic Interpolated Low-Res')
    ax[0, 1].axis('off')

    ax[0, 2].imshow(val_super_res_np, **im_args)
    ax[0, 2].set_title('Super-Resolved Image')
    ax[0, 2].axis('off')

    # Error maps (multiplied by 5 for visibility)
    error_args = {'cmap': 'gray', 'vmin': 0, 'vmax': 1} # Display error map scaled 0-1
    err_mult = 5
    text_props = {'fontsize': 12, 'va': 'top', 'ha': 'left', 'color': 'white'}

    err_gt_interp = err_mult * np.abs(val_high_res_np - val_low_res_np)
    ax[1, 1].imshow(np.clip(err_gt_interp, 0, 1), **error_args)
    ax[1, 1].set_title('Error (GT - Interp)')
    ax[1, 1].axis('off')
    ax[1, 1].text(0.02, 0.98, fr'$\times {err_mult}$', transform=ax[1, 1].transAxes, **text_props)

    err_gt_sr = err_mult * np.abs(val_high_res_np - val_super_res_np)
    ax[1, 2].imshow(np.clip(err_gt_sr, 0, 1), **error_args)
    ax[1, 2].set_title('Error (GT - SR)')
    ax[1, 2].axis('off')
    ax[1, 2].text(0.02, 0.98, fr'$\times {err_mult}$', transform=ax[1, 2].transAxes, **text_props)

    # Placeholder for the first error map (GT - GT = 0)
    ax[1, 0].imshow(np.zeros_like(val_high_res_np), **error_args)
    ax[1, 0].set_title('Error (GT - GT)')
    ax[1, 0].axis('off')

    plt.suptitle(f'Visual Comparison for Validation Image Index {index}', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Save and show plot
    timestamp = get_timestamp()
    plot_filename = f"visual_comparison_index_{index}_time_{timestamp}.png"
    save_plot(fig, results_dir, plot_filename) # Use save_plot from utils
    plt.show() # Show the plot

    print(f"--- Finished Visualizing Result for Index: {index} ---")