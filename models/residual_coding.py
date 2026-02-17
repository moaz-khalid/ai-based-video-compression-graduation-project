"""
residual_coding.py
Complete PyTorch implementation of resnet.py from OpenDVC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Basic Residual Block
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ResidualEncoder(nn.Module):
    """
    Residual Encoder
    """
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=128, num_blocks=6):
        super(ResidualEncoder, self).__init__()
        
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1)
        self.bn_in = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(ResidualBlock(hidden_channels))
        
        # Downsampling
        self.downsample_layers = nn.ModuleList()
        current_channels = hidden_channels
        for _ in range(3):  # 8x reduction
            down = nn.Sequential(
                nn.Conv2d(current_channels, current_channels * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(current_channels * 2),
                nn.ReLU(inplace=True)
            )
            self.downsample_layers.append(down)
            current_channels *= 2
        
        self.conv_out = nn.Conv2d(current_channels, out_channels, 3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        for down in self.downsample_layers:
            x = down(x)
        
        x = self.conv_out(x)
        return x


class ResidualDecoder(nn.Module):
    """
    Residual Decoder
    """
    def __init__(self, in_channels=128, hidden_channels=256, out_channels=3, num_blocks=6):
        super(ResidualDecoder, self).__init__()
        
        self.upsample_layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(3):  # 8x upsampling
            up = nn.Sequential(
                nn.ConvTranspose2d(current_channels, current_channels // 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(current_channels // 2),
                nn.ReLU(inplace=True)
            )
            self.upsample_layers.append(up)
            current_channels = current_channels // 2
        
        self.conv_init = nn.Conv2d(current_channels, hidden_channels, 3, stride=1, padding=1)
        self.bn_init = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.res_blocks.append(ResidualBlock(hidden_channels))
        
        self.conv_out = nn.Conv2d(hidden_channels, out_channels, 3, stride=1, padding=1)
        
    def forward(self, x):
        for up in self.upsample_layers:
            x = up(x)
        
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = self.relu(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        x = self.conv_out(x)
        return x


class ResidualCoder(nn.Module):
    """
    Complete Residual Coder
    """
    def __init__(self, in_channels=3, latent_channels=128, hidden_channels=64):
        super(ResidualCoder, self).__init__()
        
        self.encoder = ResidualEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=latent_channels
        )
        
        self.decoder = ResidualDecoder(
            in_channels=latent_channels,
            hidden_channels=hidden_channels * 8,
            out_channels=in_channels
        )
        
    def forward(self, residual):
        latent = self.encoder(residual)
        recon_residual = self.decoder(latent)
        return recon_residual, latent
    
    def encode(self, residual):
        return self.encoder(residual)
    
    def decode(self, latent):
        return self.decoder(latent)