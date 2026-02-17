"""
image_compression.py
Complete PyTorch implementation of CNN_img.py from OpenDVC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GDN(nn.Module):
    """
    Generalized Divisive Normalization layer
    """
    def __init__(self, num_features, inverse=False, epsilon=1e-6):
        super(GDN, self).__init__()
        self.num_features = num_features
        self.inverse = inverse
        self.epsilon = epsilon
        
        # Learnable parameters
        self.beta = nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.1)
        self.gamma = nn.Parameter(torch.eye(num_features).view(1, num_features, num_features, 1, 1) * 0.1)
        
    def forward(self, x):
        _, C, H, W = x.shape
        assert C == self.num_features, f"Expected {self.num_features} channels, got {C}"
        
        # Reshape for normalization
        x_flat = x.view(-1, self.num_features, H * W)
        
        # Compute normalization factor
        gamma = self.gamma.view(self.num_features, self.num_features)
        gamma_squared = gamma ** 2
        
        x_squared = x_flat ** 2
        norm_factor = torch.einsum('ij,njw->niw', gamma_squared, x_squared)
        
        if self.inverse:
            norm_factor = torch.sqrt(norm_factor + self.beta.view(1, self.num_features, 1))
            output = x_flat * norm_factor
        else:
            norm_factor = torch.sqrt(norm_factor + self.beta.view(1, self.num_features, 1))
            output = x_flat / (norm_factor + self.epsilon)
        
        output = output.view(-1, self.num_features, H, W)
        return output


class AnalysisTransform(nn.Module):
    """
    Analysis transform (Encoder)
    """
    def __init__(self, N=128, M=192):
        super(AnalysisTransform, self).__init__()
        self.N = N
        self.M = M
        
        self.conv1 = nn.Conv2d(3, N, 5, stride=2, padding=2)
        self.gdn1 = GDN(N)
        
        self.conv2 = nn.Conv2d(N, N, 5, stride=2, padding=2)
        self.gdn2 = GDN(N)
        
        self.conv3 = nn.Conv2d(N, N, 5, stride=2, padding=2)
        self.gdn3 = GDN(N)
        
        self.conv4 = nn.Conv2d(N, M, 5, stride=2, padding=2)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
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
    """
    def __init__(self, N=128, M=192):
        super(SynthesisTransform, self).__init__()
        self.N = N
        self.M = M
        
        self.deconv1 = nn.ConvTranspose2d(M, N, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = GDN(N, inverse=True)
        
        self.deconv2 = nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = GDN(N, inverse=True)
        
        self.deconv3 = nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1)
        self.igdn3 = GDN(N, inverse=True)
        
        self.deconv4 = nn.ConvTranspose2d(N, 3, 5, stride=2, padding=2, output_padding=1)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, y):
        x = self.deconv1(y)
        x = self.igdn1(x)
        
        x = self.deconv2(x)
        x = self.igdn2(x)
        
        x = self.deconv3(x)
        x = self.igdn3(x)
        
        x_hat = self.deconv4(x)
        x_hat = torch.clamp(x_hat, 0, 1)
        
        return x_hat