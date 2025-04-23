import torch
import torch.nn as nn
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, device='cpu'):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha # Weight for MSE (related to PSNR) loss
        self.beta = beta # Weight for SSIM loss
        self.mse = nn.MSELoss()
        # Note: data_range=1.0 assumes inputs are normalized to [0, 1]
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.device = device # Store device for input tensor placement

    def forward(self, y_pred, y_true):
        # Ensure inputs are on the same device as the ssim metric module
        y_pred_dev = y_pred.to(self.device)
        y_true_dev = y_true.to(self.device)

        # Calculate MSE Loss
        mse_loss = self.mse(y_pred_dev, y_true_dev)

        # Calculate SSIM Loss (1 - SSIM, as SSIM higher is better)
        # Clamp prediction to [0, 1] before SSIM calculation as required by the metric
        y_pred_clamped = torch.clamp(y_pred_dev, 0.0, 1.0)

        # Ensure target is also clamped just in case, though it should already be
        y_true_clamped = torch.clamp(y_true_dev, 0.0, 1.0)

        # Calculate SSIM value
        ssim_val = self.ssim(y_pred_clamped, y_true_clamped)

        # Clamp SSIM value to [-1, 1] range as per documentation/potential outputs
        ssim_val_clamped = torch.clamp(ssim_val, min=-1.0, max=1.0)

        # Calculate SSIM loss (higher SSIM is better, so loss = 1 - SSIM)
        ssim_loss = 1.0 - ssim_val_clamped

        # Combine losses
        combined_loss = (self.alpha * mse_loss) + (self.beta * ssim_loss)

        return combined_loss