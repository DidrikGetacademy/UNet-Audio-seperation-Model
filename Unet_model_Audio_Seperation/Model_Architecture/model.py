import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.Logger import setup_logger
Model_logger = setup_logger('Model', r'C:\Users\didri\Desktop\UNet-Models\Unet_model_Audio_Seperation\Model_performance_logg\log\Model_Training_logg.txt')





#Frequency Channel Attention Module (FCAM). Leverage attention mechanisms specifically designed for frequency-domain processing.
class SpectralAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpectralAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))  # Pool over frequency dimension
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Attention map along frequency dimension
       
        freq_attention = self.avg_pool(x)
        freq_attention = self.fc(freq_attention)
        return x * freq_attention


#Squeeze-and-Excitation (SE) Blocks enhance feature maps by modeling inter-channel dependencies
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# Attention Block: Applies an attention mechanism to enhance the feature representation.
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # Ensure spatial dimensions match
        if x.size(2) != g.size(2) or x.size(3) != g.size(3):
            g = F.interpolate(g, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))

        out = x * psi

      

        Model_logger.debug(f"[AttentionBlock] Output shape: {out.shape}, psi min={psi.min().item():.4f}, max={psi.max().item():.4f}")
        return out


# MultiScaleDecoderBlock: Upsamples and concatenates with skip connections, followed by convolution.
class MultiScaleDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleDecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels), 
            SpectralAttentionBlock(out_channels),
        )

    def forward(self, x, skip):
        x = self.up(x)

        # Ensure dimensions match between upsampled x and skip connection
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=False)

        x = torch.cat((x, skip), dim=1)
        out = self.conv(x)
        Model_logger.debug(f"[MultiScaleDecoderBlock] Output shape: {out.shape}")
        return out


# UNet Model: Includes encoder, bottleneck, decoder, and attention blocks.
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        prev_channels = in_channels

        # Encoder
        for feature in features:
            self.encoder.append(self.conv_block(prev_channels, feature))
            prev_channels = feature

        # Bottleneck
        self.bottleneck = self.conv_block(features[-1], features[-1] * 2)

        # Decoder
        self.decoder = nn.ModuleList()
        reversed_features = list(reversed(features))
        for idx, feature in enumerate(reversed_features):
            input_channels = features[-1] * 2 if idx == 0 else reversed_features[idx - 1]
            self.decoder.append(MultiScaleDecoderBlock(input_channels, feature))
            self.decoder.append(AttentionBlock(feature, feature, feature // 2))

        # Final convolution to predict the mask
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels),  
            SpectralAttentionBlock(out_channels), 
            nn.Dropout(p=0.4),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        skip_connections = []

        # Encoder
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections and attention
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            dec_block = self.decoder[idx]
            att_block = self.decoder[idx + 1]

            x = dec_block(x, skip_connections[idx // 2])
            x = att_block(skip_connections[idx // 2], x)

        # Predict mask
        mask = self.final_conv(x)

        # Apply mask to the input mixture
        output = mask * x

        # Ensure output matches input dimensions
        if output.size() != x.size():
            output = F.interpolate(output, size=x.size()[2:], mode='bilinear', align_corners=False)
            mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear', align_corners=False)

        Model_logger.debug(f"Final mask shape: {mask.shape}, Final output shape: {output.shape}")

        return mask, output  # Return both mask and output for debugging or loss computation
