"""
residual_coding.py
Complete PyTorch implementation of resnet.py from OpenDVC
Contains residual coding networks with ResNet-style architecture
"""
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """
    Basic Residual Block
    Matches the residual block architecture in resnet.py
    """
    def __init__(self, in_channels, out_channels=None, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        if out_channels is None:
            out_channels = in_channels
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class BottleneckBlock(nn.Module):
    """
    Bottleneck Residual Block for deeper networks
    Used in some variants of resnet.py
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResidualEncoder(nn.Module):
    """
    Residual Encoder for coding residuals
    Matches the encoder architecture in resnet.py
    """
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=128, num_blocks=6):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            hidden_channels: Number of hidden channels
            out_channels: Number of output channels (latent dimension)
            num_blocks: Number of residual blocks
        """
        super(ResidualEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1)
        self.bn_in = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.res_blocks.append(
                ResidualBlock(hidden_channels, hidden_channels)
            )
        
        # Downsampling layers (for spatial reduction)
        self.downsample_layers = nn.ModuleList()
        current_channels = hidden_channels
        
        # Progressive downsampling (2x reduction each time)
        for i in range(3):  # 8x reduction total (2^3)
            down = nn.Sequential(
                nn.Conv2d(current_channels, current_channels * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(current_channels * 2),
                nn.ReLU(inplace=True)
            )
            self.downsample_layers.append(down)
            current_channels *= 2
        
        # Output projection
        self.conv_out = nn.Conv2d(current_channels, out_channels, 3, stride=1, padding=1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Encode residual to latent representation
        
        Args:
            x: Input residual (N, C, H, W)
            
        Returns:
            Latent representation (N, out_channels, H//8, W//8)
        """
        # Initial convolution
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu(x)
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Apply downsampling
        for down in self.downsample_layers:
            x = down(x)
        
        # Output projection
        x = self.conv_out(x)
        
        return x
    
    def get_features(self, x):
        """Get intermediate features for visualization or analysis"""
        features = []
        
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu(x)
        features.append(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
            features.append(x)
        
        for down in self.downsample_layers:
            x = down(x)
            features.append(x)
        
        return features


class ResidualDecoder(nn.Module):
    """
    Residual Decoder for reconstructing residuals
    Matches the decoder architecture in resnet.py
    """
    def __init__(self, in_channels=128, hidden_channels=256, out_channels=3, num_blocks=6):
        """
        Args:
            in_channels: Number of input channels (latent dimension)
            hidden_channels: Number of hidden channels
            out_channels: Number of output channels (3 for RGB)
            num_blocks: Number of residual blocks
        """
        super(ResidualDecoder, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        
        current_channels = in_channels
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        for i in range(3):  # 8x upsampling total (2^3)
            up = nn.Sequential(
                nn.ConvTranspose2d(current_channels, current_channels // 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(current_channels // 2),
                nn.ReLU(inplace=True)
            )
            self.upsample_layers.append(up)
            current_channels = current_channels // 2
        
        # Initial convolution after upsampling
        self.conv_init = nn.Conv2d(current_channels, hidden_channels, 3, stride=1, padding=1)
        self.bn_init = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.res_blocks.append(
                ResidualBlock(hidden_channels, hidden_channels)
            )
        
        # Output convolution
        self.conv_out = nn.Conv2d(hidden_channels, out_channels, 3, stride=1, padding=1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Decode latent representation to residual
        
        Args:
            x: Latent representation (N, in_channels, H, W)
            
        Returns:
            Reconstructed residual (N, out_channels, H*8, W*8)
        """
        # Upsample
        for up in self.upsample_layers:
            x = up(x)
        
        # Initial convolution
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = self.relu(x)
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Output convolution
        x = self.conv_out(x)
        
        return x


class ResidualCoder(nn.Module):
    """
    Complete Residual Coder (Encoder + Decoder)
    Matches the full architecture in resnet.py
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
            hidden_channels=hidden_channels * 8,  # After upsampling
            out_channels=in_channels
        )
        
    def forward(self, residual):
        """
        Encode and decode residual
        
        Args:
            residual: Input residual (N, C, H, W)
            
        Returns:
            Reconstructed residual and latent representation
        """
        latent = self.encoder(residual)
        recon_residual = self.decoder(latent)
        
        return recon_residual, latent
    
    def encode(self, residual):
        """Only encode"""
        return self.encoder(residual)
    
    def decode(self, latent):
        """Only decode"""
        return self.decoder(latent)


class DeepResidualEncoder(nn.Module):
    """
    Deeper Residual Encoder with more layers
    Used for higher quality coding
    """
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=128):
        super(DeepResidualEncoder, self).__init__()
        
        # Initial layers
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(hidden_channels, hidden_channels, 3)
        self.layer2 = self._make_layer(hidden_channels, hidden_channels * 2, 4, stride=2)
        self.layer3 = self._make_layer(hidden_channels * 2, hidden_channels * 4, 6, stride=2)
        self.layer4 = self._make_layer(hidden_channels * 4, hidden_channels * 8, 3, stride=2)
        
        # Output projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels * 8, out_channels)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class ResidualPredictor(nn.Module):
    """
    Residual Predictor for motion compensation
    Predicts residual between warped and target frames
    """
    def __init__(self, in_channels=3+2, hidden_channels=64):
        """
        Args:
            in_channels: Input channels (warped frame + flow)
            hidden_channels: Hidden channels
        """
        super(ResidualPredictor, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_channels, hidden_channels),
            ResidualBlock(hidden_channels, hidden_channels),
            ResidualBlock(hidden_channels, hidden_channels),
            ResidualBlock(hidden_channels, hidden_channels)
        )
        
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        
        self.conv_out = nn.Conv2d(hidden_channels, 3, 3, stride=1, padding=1)
        
    def forward(self, warped_frame, flow):
        """
        Predict residual
        
        Args:
            warped_frame: Warped reference frame (N, 3, H, W)
            flow: Optical flow (N, 2, H, W)
            
        Returns:
            Predicted residual
        """
        x = torch.cat([warped_frame, flow], dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.res_blocks(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        residual = self.conv_out(x)
        
        return residual


class MultiScaleResidualCoder(nn.Module):
    """
    Multi-scale Residual Coder
    Codes residuals at multiple scales for better quality
    """
    def __init__(self, in_channels=3, latent_channels=[64, 128, 256]):
        super(MultiScaleResidualCoder, self).__init__()
        
        self.num_scales = len(latent_channels)
        
        # Encoders for each scale
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        prev_channels = in_channels
        for i, l_channels in enumerate(latent_channels):
            # Encoder for this scale
            encoder = nn.Sequential(
                nn.Conv2d(prev_channels, 64, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, l_channels, 3, stride=1, padding=1)
            )
            self.encoders.append(encoder)
            
            # Decoder for this scale
            decoder = nn.Sequential(
                nn.Conv2d(l_channels, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, prev_channels, 4, stride=2, padding=1)
            )
            self.decoders.append(decoder)
            
            prev_channels = l_channels
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList()
        for i in range(self.num_scales - 1):
            fusion = nn.Conv2d(latent_channels[i] + latent_channels[i+1], latent_channels[i], 1)
            self.fusion_layers.append(fusion)
        
    def forward(self, residual):
        """
        Multi-scale residual coding
        
        Args:
            residual: Input residual (N, 3, H, W)
            
        Returns:
            Reconstructed residual and multi-scale latents
        """
        # Encode at multiple scales
        latents = []
        x = residual
        for encoder in self.encoders:
            x = encoder(x)
            latents.append(x)
        
        # Decode with fusion
        recon = None
        for i in range(self.num_scales - 1, -1, -1):
            if recon is None:
                recon = self.decoders[i](latents[i])
            else:
                # Fuse with lower scale
                fused = self.fusion_layers[i](torch.cat([latents[i], recon], dim=1))
                recon = self.decoders[i](fused)
        
        return recon, latents


class ResidualCoderWithAttention(nn.Module):
    """
    Residual Coder with Attention Mechanism
    Enhanced version with spatial and channel attention
    """
    def __init__(self, in_channels=3, latent_channels=128, hidden_channels=64):
        super(ResidualCoderWithAttention, self).__init__()
        
        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        # Channel attention module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Encoder with attention
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels * 2, latent_channels, 3, stride=2, padding=1)
        )
        
        # Decoder with attention
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, hidden_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(hidden_channels, in_channels, 4, stride=2, padding=1)
        )
        
    def forward(self, residual):
        """
        Encode and decode residual with attention
        
        Args:
            residual: Input residual (N, 3, H, W)
            
        Returns:
            Reconstructed residual and attention maps
        """
        # Apply attention
        spatial_att = self.spatial_attention(residual)
        channel_att = self.channel_attention(residual)
        
        # Apply attention to residual
        attended = residual * spatial_att * channel_att
        
        # Encode
        latent = self.encoder(attended)
        
        # Decode
        recon = self.decoder(latent)
        
        return recon, latent, spatial_att, channel_att


# Utility functions for residual coding
def compute_residual_metrics(original_residual, reconstructed_residual):
    """
    Compute metrics for residual coding quality
    
    Args:
        original_residual: Original residual
        reconstructed_residual: Reconstructed residual
        
    Returns:
        Dictionary of metrics
    """
    mse = F.mse_loss(reconstructed_residual, original_residual)
    mae = F.l1_loss(reconstructed_residual, original_residual)
    
    # Signal-to-Noise Ratio
    snr = 10 * torch.log10((original_residual ** 2).mean() / mse)
    
    # Compression ratio estimation (simplified)
    original_entropy = estimate_entropy(original_residual)
    recon_entropy = estimate_entropy(reconstructed_residual)
    
    return {
        'mse': mse.item(),
        'mae': mae.item(),
        'snr': snr.item(),
        'original_entropy': original_entropy.item(),
        'recon_entropy': recon_entropy.item()
    }


def estimate_entropy(tensor, bins=256):
    """
    Estimate entropy of a tensor
    
    Args:
        tensor: Input tensor
        bins: Number of bins for histogram
        
    Returns:
        Estimated entropy in bits
    """
    # Flatten tensor
    flat = tensor.flatten().detach().cpu().numpy()
    
    # Compute histogram
    hist, _ = np.histogram(flat, bins=bins, density=True)
    
    # Remove zeros
    hist = hist[hist > 0]
    
    # Compute entropy
    entropy = -np.sum(hist * np.log2(hist))
    
    return torch.tensor(entropy)


class ResidualLoss(nn.Module):
    """
    Composite loss function for residual coding
    Combines reconstruction loss with rate estimation
    """
    def __init__(self, lambda_rate=0.1, lambda_distortion=1.0):
        super(ResidualLoss, self).__init__()
        
        self.lambda_rate = lambda_rate
        self.lambda_distortion = lambda_distortion
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def estimate_rate(self, latent):
        """
        Estimate bit rate from latent representation
        
        Args:
            latent: Latent representation
            
        Returns:
            Estimated rate
        """
        # Simple entropy estimation
        # In practice, this would be a proper entropy model
        return torch.mean(torch.log(torch.abs(latent) + 1))
    
    def forward(self, recon_residual, target_residual, latent):
        """
        Compute composite loss
        
        Args:
            recon_residual: Reconstructed residual
            target_residual: Target residual
            latent: Latent representation
            
        Returns:
            Total loss and individual components
        """
        # Distortion losses
        mse = self.mse_loss(recon_residual, target_residual)
        l1 = self.l1_loss(recon_residual, target_residual)
        
        # Rate estimation
        rate = self.estimate_rate(latent)
        
        # Combined loss
        total_loss = (self.lambda_distortion * mse + 
                     0.1 * l1 + 
                     self.lambda_rate * rate)
        
        return {
            'total': total_loss,
            'mse': mse,
            'l1': l1,
            'rate': rate
        }


# Unit test
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 2
    height, width = 128, 128
    
    # Create test data (residuals)
    residual = torch.randn(batch_size, 3, height, width).to(device) * 0.1
    
    print("\n=== Testing Residual Coding Networks ===\n")
    
    # Test Residual Encoder
    print("1. Testing Residual Encoder...")
    res_encoder = ResidualEncoder(in_channels=3, hidden_channels=64, out_channels=128).to(device)
    latent = res_encoder(residual)
    print(f"   Input shape: {residual.shape}")
    print(f"   Latent shape: {latent.shape}")
    print(f"   Expected latent shape: ({batch_size}, 128, {height//8}, {width//8})")
    
    # Test Residual Decoder
    print("\n2. Testing Residual Decoder...")
    res_decoder = ResidualDecoder(in_channels=128, hidden_channels=256, out_channels=3).to(device)
    recon = res_decoder(latent)
    print(f"   Latent shape: {latent.shape}")
    print(f"   Reconstructed shape: {recon.shape}")
    print(f"   Expected recon shape: ({batch_size}, 3, {height}, {width})")
    
    # Test Complete Residual Coder
    print("\n3. Testing Complete Residual Coder...")
    res_coder = ResidualCoder(in_channels=3, latent_channels=128, hidden_channels=64).to(device)
    recon_full, latent_full = res_coder(residual)
    print(f"   Input shape: {residual.shape}")
    print(f"   Reconstructed shape: {recon_full.shape}")
    print(f"   Latent shape: {latent_full.shape}")
    print(f"   Reconstruction error: {F.mse_loss(recon_full, residual).item():.6f}")
    
    # Test Residual Predictor
    print("\n4. Testing Residual Predictor...")
    warped = torch.randn(batch_size, 3, height, width).to(device)
    flow = torch.randn(batch_size, 2, height, width).to(device)
    res_predictor = ResidualPredictor(in_channels=3+2, hidden_channels=64).to(device)
    predicted_res = res_predictor(warped, flow)
    print(f"   Warped shape: {warped.shape}")
    print(f"   Flow shape: {flow.shape}")
    print(f"   Predicted residual shape: {predicted_res.shape}")
    
    # Test Multi-Scale Residual Coder
    print("\n5. Testing Multi-Scale Residual Coder...")
    ms_coder = MultiScaleResidualCoder(in_channels=3, latent_channels=[64, 128, 256]).to(device)
    recon_ms, latents_ms = ms_coder(residual)
    print(f"   Input shape: {residual.shape}")
    print(f"   Reconstructed shape: {recon_ms.shape}")
    print(f"   Number of scales: {len(latents_ms)}")
    for i, l in enumerate(latents_ms):
        print(f"   Scale {i} latent shape: {l.shape}")
    
    # Test Residual Coder with Attention
    print("\n6. Testing Residual Coder with Attention...")
    att_coder = ResidualCoderWithAttention(in_channels=3, latent_channels=128, hidden_channels=64).to(device)
    recon_att, latent_att, spatial_att, channel_att = att_coder(residual)
    print(f"   Input shape: {residual.shape}")
    print(f"   Reconstructed shape: {recon_att.shape}")
    print(f"   Spatial attention shape: {spatial_att.shape}")
    print(f"   Spatial attention range: [{spatial_att.min():.3f}, {spatial_att.max():.3f}]")
    print(f"   Channel attention shape: {channel_att.shape}")
    print(f"   Channel attention range: [{channel_att.min():.3f}, {channel_att.max():.3f}]")
    
    # Test Deep Residual Encoder
    print("\n7. Testing Deep Residual Encoder...")
    deep_encoder = DeepResidualEncoder(in_channels=3, hidden_channels=64, out_channels=128).to(device)
    deep_latent = deep_encoder(residual)
    print(f"   Input shape: {residual.shape}")
    print(f"   Deep latent shape: {deep_latent.shape}")
    
    # Test Gradient Flow
    print("\n8. Testing Gradient Flow...")
    loss = recon_full.mean()
    loss.backward()
    print("   Gradient flow test passed!")
    
    # Test Residual Loss
    print("\n9. Testing Residual Loss...")
    criterion = ResidualLoss(lambda_rate=0.1, lambda_distortion=1.0).to(device)
    losses = criterion(recon_full, residual, latent_full)
    print(f"   Total loss: {losses['total'].item():.6f}")
    print(f"   MSE loss: {losses['mse'].item():.6f}")
    print(f"   L1 loss: {losses['l1'].item():.6f}")
    print(f"   Rate estimate: {losses['rate'].item():.6f}")
    
    # Test Residual Metrics
    print("\n10. Computing Residual Metrics...")
    metrics = compute_residual_metrics(residual, recon_full)
    for key, value in metrics.items():
        print(f"   {key}: {value:.6f}")
    
    # Count Parameters
    print("\n11. Model Parameters:")
    print(f"   Residual Coder: {sum(p.numel() for p in res_coder.parameters()):,} parameters")
    print(f"   Attention Coder: {sum(p.numel() for p in att_coder.parameters()):,} parameters")
    print(f"   Multi-Scale Coder: {sum(p.numel() for p in ms_coder.parameters()):,} parameters")
    
    # Test encode/decode separately
    print("\n12. Testing Separate Encode/Decode...")
    encoded = res_coder.encode(residual)
    decoded = res_coder.decode(encoded)
    print(f"   Encode only shape: {encoded.shape}")
    print(f"   Decode only shape: {decoded.shape}")
    print(f"   Encode+Decode error: {F.mse_loss(decoded, residual).item():.6f}")
    
    print("\n✅ All residual coding tests passed!")