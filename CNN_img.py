"""
image_compression.py
Complete PyTorch implementation of CNN_img.py from OpenDVC
Contains Analysis and Synthesis transforms with GDN/IGDN layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GDN(nn.Module):
    """
    Generalized Divisive Normalization layer
    Exactly matches the TensorFlow implementation in CNN_img.py
    """
    def __init__(self, num_features, inverse=False, epsilon=1e-6):
        super(GDN, self).__init__()
        self.num_features = num_features
        self.inverse = inverse
        self.epsilon = epsilon
        
        # Learnable parameters: beta (offset) and gamma (scale matrix)
        self.beta = nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.1)
        self.gamma = nn.Parameter(torch.eye(num_features).view(1, num_features, num_features, 1, 1) * 0.1)
        
    def forward(self, x):
        """
        Args:
            x: tensor of shape (N, C, H, W)
        Returns:
            normalized tensor of same shape
        """
        _, C, H, W = x.shape
        assert C == self.num_features, f"Expected {self.num_features} channels, got {C}"
        
        # Reshape for normalization: (N, C, H*W)
        x_flat = x.view(-1, self.num_features, H * W)
        
        # Compute normalization factor
        gamma = self.gamma.view(self.num_features, self.num_features)
        gamma_squared = gamma ** 2
        
        # Compute weighted sum of squares: sum(gamma_ij^2 * x_j^2) over j
        x_squared = x_flat ** 2
        norm_factor = torch.einsum('ij,njw->niw', gamma_squared, x_squared)
        
        if self.inverse:
            # Inverse GDN: multiply by sqrt(beta + sum)
            norm_factor = torch.sqrt(norm_factor + self.beta.view(1, self.num_features, 1))
            output = x_flat * norm_factor
        else:
            # Forward GDN: divide by sqrt(beta + sum)
            norm_factor = torch.sqrt(norm_factor + self.beta.view(1, self.num_features, 1))
            output = x_flat / (norm_factor + self.epsilon)
        
        # Reshape back to (N, C, H, W)
        output = output.view(-1, self.num_features, H, W)
        
        return output


class AnalysisTransform(nn.Module):
    """
    Analysis transform (Encoder)
    Matches the architecture in CNN_img.py for OpenDVC
    """
    def __init__(self, N=128, M=192):
        """
        Args:
            N: Number of filters in intermediate layers
            M: Number of filters in output layer (latent representation)
        """
        super(AnalysisTransform, self).__init__()
        self.N = N
        self.M = M
        
        # Layer 1: 3x5x5 conv, stride 2 -> GDN
        self.conv1 = nn.Conv2d(3, N, 5, stride=2, padding=2)
        self.gdn1 = GDN(N)
        
        # Layer 2: Nx5x5 conv, stride 2 -> GDN
        self.conv2 = nn.Conv2d(N, N, 5, stride=2, padding=2)
        self.gdn2 = GDN(N)
        
        # Layer 3: Nx5x5 conv, stride 2 -> GDN
        self.conv3 = nn.Conv2d(N, N, 5, stride=2, padding=2)
        self.gdn3 = GDN(N)
        
        # Layer 4: Nx5x5 conv, stride 2 -> (output M channels, no GDN)
        self.conv4 = nn.Conv2d(N, M, 5, stride=2, padding=2)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using He initialization (matches TF default)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (N, 3, H, W)
        Returns:
            Latent representation y of shape (N, M, H//16, W//16)
        """
        # Normalize input to [0, 1] if not already
        if x.max() > 1.0:
            x = x / 255.0
            
        x = self.conv1(x)
        x = self.gdn1(x)
        
        x = self.conv2(x)
        x = self.gdn2(x)
        
        x = self.conv3(x)
        x = self.gdn3(x)
        
        y = self.conv4(x)
        
        return y


class SynthesisTransform(nn.Module):
    """
    Synthesis transform (Decoder)
    Matches the architecture in CNN_img.py for OpenDVC
    """
    def __init__(self, N=128, M=192):
        """
        Args:
            N: Number of filters in intermediate layers
            M: Number of filters in input layer (latent representation)
        """
        super(SynthesisTransform, self).__init__()
        self.N = N
        self.M = M
        
        # Layer 1: Mx5x5 deconv, stride 2 -> IGDN
        self.deconv1 = nn.ConvTranspose2d(M, N, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = GDN(N, inverse=True)
        
        # Layer 2: Nx5x5 deconv, stride 2 -> IGDN
        self.deconv2 = nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = GDN(N, inverse=True)
        
        # Layer 3: Nx5x5 deconv, stride 2 -> IGDN
        self.deconv3 = nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1)
        self.igdn3 = GDN(N, inverse=True)
        
        # Layer 4: Nx5x5 deconv, stride 2 -> (output 3 channels, no IGDN)
        self.deconv4 = nn.ConvTranspose2d(N, 3, 5, stride=2, padding=2, output_padding=1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, y):
        """
        Args:
            y: Latent tensor of shape (N, M, H//16, W//16)
        Returns:
            Reconstructed image x_hat of shape (N, 3, H, W)
        """
        x = self.deconv1(y)
        x = self.igdn1(x)
        
        x = self.deconv2(x)
        x = self.igdn2(x)
        
        x = self.deconv3(x)
        x = self.igdn3(x)
        
        x_hat = self.deconv4(x)
        
        # Clip to valid range
        x_hat = torch.clamp(x_hat, 0, 1)
        
        return x_hat


class HyperpriorAnalysis(nn.Module):
    """
    Hyperprior analysis transform (for entropy modeling)
    Additional network from CNN_img.py for side information
    """
    def __init__(self, M=192, N=128):
        super(HyperpriorAnalysis, self).__init__()
        
        self.conv1 = nn.Conv2d(M, N, 3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(N, N, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(N, N, 5, stride=2, padding=2)
        
    def forward(self, y):
        z = self.leaky_relu(self.conv1(y))
        z = self.leaky_relu(self.conv2(z))
        z = self.conv3(z)
        return z


class HyperpriorSynthesis(nn.Module):
    """
    Hyperprior synthesis transform (for entropy modeling)
    Generates Gaussian parameters for the latent representation
    """
    def __init__(self, M=192, N=128):
        super(HyperpriorSynthesis, self).__init__()
        
        self.deconv1 = nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.deconv2 = nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(N, M * 2, 3, stride=1, padding=1)
        
    def forward(self, z):
        gaussian_params = self.leaky_relu(self.deconv1(z))
        gaussian_params = self.leaky_relu(self.deconv2(gaussian_params))
        gaussian_params = self.deconv3(gaussian_params)
        
        # Split into mean and scale
        means, scales = gaussian_params.chunk(2, dim=1)
        scales = torch.abs(scales) + 1e-6  # Ensure positive scales
        
        return means, scales


# Unit test to verify the implementation
if __name__ == "__main__":
    # Test the networks
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample input
    batch_size = 4
    height, width = 256, 256
    x = torch.rand(batch_size, 3, height, width).to(device)
    
    # Test Analysis Transform
    analysis = AnalysisTransform(N=128, M=192).to(device)
    y = analysis(x)
    print(f"Analysis transform: {x.shape} -> {y.shape}")
    print(f"Expected: ({batch_size}, 192, {height//16}, {width//16})")
    
    # Test Synthesis Transform
    synthesis = SynthesisTransform(N=128, M=192).to(device)
    x_hat = synthesis(y)
    print(f"Synthesis transform: {y.shape} -> {x_hat.shape}")
    print(f"Expected: ({batch_size}, 3, {height}, {width})")
    
    # Test Hyperprior networks
    hyper_analysis = HyperpriorAnalysis(M=192, N=128).to(device)
    z = hyper_analysis(y)
    print(f"Hyperprior analysis: {y.shape} -> {z.shape}")
    
    hyper_synthesis = HyperpriorSynthesis(M=192, N=128).to(device)
    means, scales = hyper_synthesis(z)
    print(f"Hyperprior synthesis: {z.shape} -> means {means.shape}, scales {scales.shape}")
    
    # Check gradient flow
    loss = F.mse_loss(x_hat, x)
    loss.backward()
    print("Gradient flow test passed!")
    
    # Count parameters
    total_params = sum(p.numel() for p in analysis.parameters())
    print(f"Analysis transform parameters: {total_params:,}")