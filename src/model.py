import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperResolutionNet(nn.Module):
    def __init__(self):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        upscale_factor = 2
        channels_mult = upscale_factor * upscale_factor # = 4

        # --- Initial Feature Extraction ---
        # Input: 1 x 128 x 128
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=80, kernel_size=3, padding=1) # 80 x 128 x 128
        self.conv1b = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, padding=1) # 80 x 128 x 128

        # --- Upscaling Path 1 (128 -> 256) using PixelShuffle ---
        # Main Path
        self.upscale1_conv = nn.Conv2d(in_channels=80, out_channels=80 * channels_mult, kernel_size=3, padding=1) # 80*4 x 128 x 128
        self.pixel_shuffle1 = nn.PixelShuffle(upscale_factor) # -> 80 x 256 x 256
        self.conv2 = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, padding=1) # 80 x 256 x 256
        self.conv2b = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, padding=1) # 80 x 256 x 256
        # Skip Path 1 (Interpolate features from 128x128 stage)
        self.bskip1 = nn.Upsample(scale_factor=upscale_factor, mode='bicubic', align_corners=False) # Interpolates 80 channels to 256x256
        # Convolution after combining skip connection 1 (Input channels = 80 from main path + 80 from skip = 160)
        self.conv_after_skip1 = nn.Conv2d(in_channels=160, out_channels=80, kernel_size=3, padding=1) # 160x256x256 -> 80x256x256
        self.conv_after_skip1b = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, padding=1) # 80x256x256

        # --- Upscaling Path 2 (256 -> 512) using PixelShuffle & Widened Middle ---
        # Main Path
        self.upscale2_conv = nn.Conv2d(in_channels=80, out_channels=80 * channels_mult, kernel_size=3, padding=1) # 80*4 x 256 x 256 (Widened)
        self.pixel_shuffle2 = nn.PixelShuffle(upscale_factor) # -> 80 x 512 x 512 (Widened)
        self.conv3 = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, padding=1) # 80 x 512 x 512 (Widened)
        self.conv3b = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, padding=1) # 80 x 512 x 512 (Widened)
        # Skip Path 2 (Interpolate features from 256x256 stage after conv_after_skip1b)
        self.bskip2 = nn.Upsample(scale_factor=upscale_factor, mode='bicubic', align_corners=False) # Interpolates 80 channels to 512x512
        # Convolution after combining skip connection 2 (Input channels = 80 from main path + 80 from skip = 160) (Widened)
        self.conv_after_skip2 = nn.Conv2d(in_channels=160, out_channels=80, kernel_size=3, padding=1) # 160x512x512 -> 80x512x512 (Widened)
        self.conv_after_skip2b = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, padding=1) # 80x512x512 (Widened)

        # --- Downscaling Path (512 -> 256) ---
        # Input channel matches the widened middle stage (80)
        self.downscale = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=2, stride=2) # 80x512x512 -> 80x256x256
        self.conv4 = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, padding=1) # 80x256x256
        self.conv4b = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, padding=1) # 80x256x256

        # --- Final Output Layer ---
        # Takes the output of the downscaling path (80 channels) and maps to 1 channel
        self.conv_out_final = nn.Conv2d(in_channels=80, out_channels=1, kernel_size=3, padding=1) # 80x256x256 -> 1x256x256

    def forward(self, x):
        # x: Input 1 x 128 x 128

        # --- Initial Feature Extraction ---
        features128 = self.relu(self.conv1(x))
        features128 = self.relu(self.conv1b(features128)) # 80 x 128 x 128

        # --- Upscaling Path 1 + Skip Connection 1 ---
        skip1_interpolated = self.bskip1(features128) # 80 x 256 x 256
        up1_conv_out = self.upscale1_conv(features128) # 80*4 x 128 x 128
        up1_out = self.pixel_shuffle1(up1_conv_out)    # 80 x 256 x 256
        features256_main = self.relu(self.conv2(up1_out))
        features256_main = self.relu(self.conv2b(features256_main)) # 80 x 256 x 256
        concat1 = torch.cat((features256_main, skip1_interpolated), dim=1) # 160 x 256 x 256
        features256_processed = self.relu(self.conv_after_skip1(concat1))
        features256_processed = self.relu(self.conv_after_skip1b(features256_processed)) # 80 x 256 x 256

        # --- Upscaling Path 2 + Skip Connection 2 ---
        skip2_interpolated = self.bskip2(features256_processed) # 80 x 512 x 512
        up2_conv_out = self.upscale2_conv(features256_processed) # 80*4 x 256 x 256
        up2_out = self.pixel_shuffle2(up2_conv_out)    # 80 x 512 x 512
        features512_main = self.relu(self.conv3(up2_out))
        features512_main = self.relu(self.conv3b(features512_main)) # 80 x 512 x 512
        concat2 = torch.cat((features512_main, skip2_interpolated), dim=1) # 160 x 512 x 512
        features512_processed = self.relu(self.conv_after_skip2(concat2))
        features512_processed = self.relu(self.conv_after_skip2b(features512_processed)) # 80 x 512 x 512

        # --- Downscaling Path ---
        down_out = self.downscale(features512_processed) # 80 x 256 x 256
        features256_final = self.relu(self.conv4(down_out))
        features256_final = self.relu(self.conv4b(features256_final)) # 80 x 256 x 256

        # --- Final Output Layer ---
        output = self.conv_out_final(features256_final) # 1 x 256 x 256

        return output