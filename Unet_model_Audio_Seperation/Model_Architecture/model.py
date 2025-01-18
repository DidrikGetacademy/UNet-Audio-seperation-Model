import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Training.Externals.Logger import setup_logger
Model_logger = setup_logger('Model', r'C:\Users\didri\Desktop\UNet Models\UNet_vocal_isolation_model\Model_performance_logg\log\Model_Training_logg.txt')


#AttentionBlock: This module applies an attention mechanism to enhance the feature representation by combining gating features and encoder features.
class AttentionBlock(nn.Module):
    #Applies an attention mechanism to combine gating and encoder features.
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        Model_logger.debug(f"[AttentionBlock] Input shapes -> x: {x.shape}, g: {g.shape}")
        print(f"[AttentionBlock] Input shapes -> x: {x.shape}, g: {g.shape}")

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))

        #logging shape and min/max
        out = x * psi
        Model_logger.debug( f"[AttentionBlock] Output shape: {out.shape},"  f"psi min={psi.min().item():.4f}, psi max={psi.max().item():.4f}"  )
        print( f"[AttentionBlock] Output shape: {out.shape},"  f"psi min={psi.min().item():.4f}, psi max={psi.max().item():.4f}"  )
        return out



#MultiScaleDecoderBlock: This block is part of the decoder in the UNet model. It performs upsampling and concatenation with skip connections, followed by convolution to refine the output features.
class MultiScaleDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleDecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        Model_logger.debug( f"[MultiScaleDecoderBlock] Input shapes -> x: {x.shape}, skip: {skip.shape}" )

        x = self.up(x)
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=False)
            Model_logger.debug(f"[MultiScaleDecoderBlock] Resized upsampled x to: {x.shape}")

      
        x = torch.cat((x, skip), dim=1)
        Model_logger.debug(f"[MultiScaleDecoderBlock] After concat: {x.shape}")

        out = self.conv(x)
        Model_logger.debug(f"[MultiScaleDecoderBlock] Output shape: {out.shape}")
        return out





#The core UNet-Model architecture itself, which consists of an encoder, bottleneck, decoder, and the final output layer. It uses the previously defined blocks
class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        prev_channels = in_channels

        for feature in features:
            self.encoder.append(self.conv_block(prev_channels, feature))
            prev_channels = feature
            Model_logger.debug(f"[UNet] Added encoder block with out_channels: {feature}")
            print(f"[UNet] Added encoder block with out_channels: {feature}")

        self.bottleneck = self.conv_block(features[-1], features[-1] * 2)
        Model_logger.debug(f"[UNet] Bottleneck channels: {features[-1] * 2}")
        print(f"[UNet] Bottleneck channels: {features[-1] * 2}")

       
        self.decoder = nn.ModuleList()
        reversed_features = list(reversed(features))

        for idx, feature in enumerate(reversed_features):

            input_channels = features[-1] * 2 if idx == 0 else reversed_features[idx - 1]
            self.decoder.append(MultiScaleDecoderBlock(input_channels, feature))
            self.decoder.append(AttentionBlock(feature, feature, feature // 2))
            Model_logger.debug( f"[UNet] Added decoder & attention block -> in:{input_channels}, out:{feature}" )
            print( f"[UNet] Added decoder & attention block -> in:{input_channels}, out:{feature}" )

     
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        Model_logger.info(f"[UNet] Final conv initialized: in={features[0]}, out={out_channels}")
        print(f"[UNet] Final conv initialized: in={features[0]}, out={out_channels}")


 
    def conv_block(self, in_channels, out_channels):
 
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )



    def forward(self, x):
        x = x.to(dtype=torch.float32)
        Model_logger.debug(f"[UNet forward] Input shape: {x.shape}")
        print(f"[UNet forward] Input shape: {x.shape}")

        skip_connections = []

        for idx, enc in enumerate(self.encoder):
            x = enc(x)
            skip_connections.append(x)
            Model_logger.debug(f"[UNet forward] Encoder[{idx}] output: {x.shape}")
            print(f"[UNet forward] Encoder[{idx}] output: {x.shape}")
            x = F.max_pool2d(x, kernel_size=2, stride=2)

    
        x = self.bottleneck(x)
        Model_logger.debug(f"[UNet forward] Bottleneck output: {x.shape}")
        print(f"[UNet forward] Bottleneck output: {x.shape}")

    
        skip_connections = skip_connections[::-1]

  
        for idx in range(0, len(self.decoder), 2):
            dec_block = self.decoder[idx]
            att_block = self.decoder[idx + 1]

     
            x = dec_block(x, skip_connections[idx // 2])
  
            x = att_block(skip_connections[idx // 2], x)
            Model_logger.debug(f"[UNet forward] Decoder stage {idx//2}, output: {x.shape}")
            print(f"[UNet forward] Decoder stage {idx//2}, output: {x.shape}")

        out = self.final_conv(x)
        Model_logger.debug(f"[UNet forward] Final output shape: {out.shape}")
        print(f"[UNet forward] Final output shape: {out.shape}")
        return out
    