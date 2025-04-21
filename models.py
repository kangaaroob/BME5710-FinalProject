### ROB V1:
# class SuperResolutionNet(nn.Module):
#     def __init__(self):
#         super(SuperResolutionNet, self).__init__()

#         # --- Encoder / Initial Feature Extraction ---
#         # Input: 1 x 128 x 128
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
#         self.relu1 = nn.ReLU(inplace=True)

#         # --- Upscaling Path 1 (128 -> 256) ---
#         # Feature map size: 64 x 128 x 128
#         self.upscale1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2) # 64x128x128 -> 32x256x256
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1) # 32x256x256
#         self.relu2 = nn.ReLU(inplace=True)

#         # --- Upscaling Path 2 (256 -> 512) ---
#         # Feature map size: 32 x 256 x 256
#         self.upscale2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2) # 32x256x256 -> 16x512x512
#         self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) # 16x512x512
#         self.relu3 = nn.ReLU(inplace=True)

#         # --- Downscaling Path (512 -> 256) ---
#         # Feature map size: 16 x 512 x 512
#         self.downscale = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2) # 16x512x512 -> 32x256x256
#         self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1) # 32x256x256
#         self.relu4 = nn.ReLU(inplace=True)

#         # --- Output Layer ---
#         # Feature map size: 32 x 256 x 256
#         self.conv_out = nn.Conv2d(in_channels=32 + 32, out_channels=1, kernel_size=3, padding=1) # Concatenated features + skip -> 1x256x256

#     def forward(self, x):
#         # x: Input 1 x 128 x 128

#         # --- Initial Feature Extraction ---
#         features128 = self.relu1(self.conv1(x)) # 64 x 128 x 128

#         # --- Upscaling Path 1 + Skip Connection 1 ---
#         up1_out = self.upscale1(features128) # 32 x 256 x 256
#         skip1_bicubic = F.interpolate(x, size=(256, 256), mode='bicubic', align_corners=False) # 1 x 256 x 256 (Original input upscaled)
#         # Note: The description asked for skip connections performing bicubic interpolation.
#         # This is interpreted as interpolating the *input* for the skip connection.
#         # We'll need another conv layer to match channel dimensions if we were to add/cat features here.
#         # For now, let's focus on the main path and add the skip connection later.
#         features256 = self.relu2(self.conv2(up1_out)) # 32 x 256 x 256

#         # --- Upscaling Path 2 ---
#         up2_out = self.upscale2(features256) # 16 x 512 x 512
#         features512 = self.relu3(self.conv3(up2_out)) # 16 x 512 x 512

#         # --- Downscaling Path ---
#         down_out = self.downscale(features512) # 32 x 256 x 256
#         features256_down = self.relu4(self.conv4(down_out)) # 32 x 256 x 256

#         # --- Skip Connection 2 (Concatenate features from first upscale path) ---
#         # Concatenate along the channel dimension
#         combined_features = torch.cat((features256_down, features256), dim=1) # (32+32) x 256 x 256

#         # --- Output Layer ---
#         output = self.conv_out(combined_features) # 1 x 256 x 256

#         return output

### LUKA:
# class SuperResolutionNet(nn.Module):
#     def __init__(self):
#         super(SuperResolutionNet, self).__init__()
#         self.gelu = nn.GELU()

#         # 128 x 128 input
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

#         # upscale to 256 x 256
#         self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

#         # bicubic interpolation to 256 x 256 skip connection
#         self.bskip1 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

#         # convolution to 256 x 256 combine with skip connection
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

#         # upscale to 512 x 512
#         self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
#         self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

#         # bicubic interpolation to 512 x 512 skip connection
#         self.bskip2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

#         # convolution to 512 x 512 combine with skip connection
#         self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

#         self.conv7 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
#         self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

#         # downsample to 256 x 256
#         self.downconv1 = nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1)
#         self.conv9 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.conv10 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        

#     def forward(self, x):
#         x = self.gelu(self.conv1(x))
#         x = self.gelu(self.conv2(x))

#         x_prev = self.bskip1(x)
#         x = self.gelu(self.upconv1(x))
#         x = self.gelu(self.conv3(x))
#         x = torch.cat((x, x_prev), dim=1)

#         x = self.gelu(self.conv4(x))
#         x_prev = self.bskip2(x)
#         x = self.gelu(self.upconv2(x))
#         x = self.gelu(self.conv5(x))
#         x = torch.cat((x, x_prev), dim=1)

#         x = self.gelu(self.conv6(x))
#         x = self.gelu(self.conv7(x))
#         x = self.gelu(self.conv8(x))
#         x = self.gelu(self.downconv1(x))
#         x = self.gelu(self.conv9(x))
#         x = self.conv10(x)
#         return x


### ROB V2:
# class SuperResolutionNet(nn.Module):
#     def __init__(self):
#         super(SuperResolutionNet, self).__init__()

#         # --- Activation Function ---
#         # Keeping ReLU as per original 'Our Network' design
#         self.relu = nn.ReLU(inplace=True)

#         # --- Initial Feature Extraction ---
#         # Input: 1 x 128 x 128
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)

#         # --- Upscaling Path 1 (128 -> 256) ---
#         # Main Path
#         self.upscale1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2) # 64x128x128 -> 32x256x256
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1) # 32x256x256
#         # Skip Path 1 (Interpolate features from 128x128 stage)
#         self.bskip1 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False) # Interpolates 64 channels to 256x256
#         # Convolution after combining skip connection 1 (Input channels = 32 from main path + 64 from skip = 96)
#         # Outputting 32 channels to match input requirement of upscale2
#         self.conv_after_skip1 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, padding=1)

#         # --- Upscaling Path 2 (256 -> 512) ---
#         # Main Path
#         self.upscale2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2) # 32x256x256 -> 16x512x512
#         self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) # 16x512x512
#         # Skip Path 2 (Interpolate features from 256x256 stage)
#         self.bskip2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False) # Interpolates 32 channels to 512x512
#         # Convolution after combining skip connection 2 (Input channels = 16 from main path + 32 from skip = 48)
#         # Outputting 16 channels to match input requirement of downscale
#         self.conv_after_skip2 = nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, padding=1)

#         # --- Downscaling Path (512 -> 256) ---
#         self.downscale = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2) # 16x512x512 -> 32x256x256
#         self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1) # 32x256x256

#         # --- Final Output Layer ---
#         # Takes the output of the downscaling path (32 channels) and maps to 1 channel
#         self.conv_out_final = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1) # 32x256x256 -> 1x256x256

#     def forward(self, x):
#         # x: Input 1 x 128 x 128

#         # --- Initial Feature Extraction ---
#         features128 = self.relu(self.conv1(x)) # 64 x 128 x 128

#         # --- Upscaling Path 1 + Skip Connection 1 ---
#         # Calculate interpolated skip features first
#         skip1_interpolated = self.bskip1(features128) # 64 x 256 x 256
#         # Main path upscale
#         up1_out = self.upscale1(features128) # 32 x 256 x 256
#         features256_main = self.relu(self.conv2(up1_out)) # 32 x 256 x 256
#         # Concatenate main path features and interpolated skip features
#         concat1 = torch.cat((features256_main, skip1_interpolated), dim=1) # 96 x 256 x 256
#         # Process combined features
#         features256_processed = self.relu(self.conv_after_skip1(concat1)) # 32 x 256 x 256

#         # --- Upscaling Path 2 + Skip Connection 2 ---
#         # Calculate interpolated skip features
#         skip2_interpolated = self.bskip2(features256_processed) # 32 x 512 x 512
#         # Main path upscale
#         up2_out = self.upscale2(features256_processed) # 16 x 512 x 512
#         features512_main = self.relu(self.conv3(up2_out)) # 16 x 512 x 512
#         # Concatenate main path features and interpolated skip features
#         concat2 = torch.cat((features512_main, skip2_interpolated), dim=1) # 48 x 512 x 512
#         # Process combined features
#         features512_processed = self.relu(self.conv_after_skip2(concat2)) # 16 x 512 x 512

#         # --- Downscaling Path ---
#         down_out = self.downscale(features512_processed) # 32 x 256 x 256
#         features256_final = self.relu(self.conv4(down_out)) # 32 x 256 x 256

#         # --- Final Output Layer ---
#         # Map final features to the output image (No final activation here, typical for regression tasks like SR)
#         output = self.conv_out_final(features256_final) # 1 x 256 x 256

#         return output