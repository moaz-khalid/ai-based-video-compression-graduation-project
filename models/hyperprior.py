"""
models/hyperprior.py - Complete Scale Hyperprior from Ballé et al. 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HyperAnalysis(nn.Module):
    """Hyper-analysis transform (ha) - compresses y to z"""
    def __init__(self, M=192, N=128):
        super().__init__()
        self.conv1 = nn.Conv2d(M, N, 3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(N, N, 5, stride=2, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(N, N, 5, stride=2, padding=2)
        
    def forward(self, y):
        x = self.relu1(self.conv1(y))
        x = self.relu2(self.conv2(x))
        z = self.conv3(x)
        return z


class HyperSynthesis(nn.Module):
    """Hyper-synthesis transform (hs) - decompresses z to entropy parameters"""
    def __init__(self, M=192, N=128):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(N, M, 3, stride=1, padding=1)
        
    def forward(self, z):
        x = self.relu1(self.deconv1(z))
        x = self.relu2(self.deconv2(x))
        sigma = torch.exp(self.deconv3(x))  # Scale parameter (positive)
        return sigma


class FactorizedEntropyModel(nn.Module):
    """Factorized entropy model for hyper-latent z"""
    def __init__(self, N=128, L=64):
        super().__init__()
        self.N = N
        self.L = L
        self.bound = 10.0
        
        # Learnable cumulative distribution parameters
        self.cdf_params = nn.Parameter(torch.randn(N, L) * 0.1)
        
        # Register bins
        bins = torch.linspace(-self.bound, self.bound, L + 1)
        self.register_buffer('bins', bins)
        
    def forward(self, z, training=True):
        """Estimate bits for z"""
        if training:
            z_q = z + torch.rand_like(z) - 0.5
        else:
            z_q = torch.round(z)
        
        # Clip to bounds
        z_q = torch.clamp(z_q, -self.bound, self.bound)
        
        # Find bin indices
        bin_width = 2 * self.bound / self.L
        indices = ((z_q + self.bound) / bin_width).long()
        indices = torch.clamp(indices, 0, self.L - 1)
        
        # Get probabilities from CDF
        cdf = torch.softmax(self.cdf_params, dim=1)
        
        # Gather probabilities
        B, C, H, W = z_q.shape
        probs = cdf[0, indices].view(B, C, H, W)
        
        # Bits = -log2(prob)
        bits = -torch.log2(probs + 1e-9)
        
        return bits.sum(), z_q


class ScaleHyperprior(nn.Module):
    """Complete Scale Hyperprior Module"""
    def __init__(self, M=192, N=128):
        super().__init__()
        self.ha = HyperAnalysis(M, N)
        self.hs = HyperSynthesis(M, N)
        self.entropy_model = FactorizedEntropyModel(N)
        
    def forward(self, y, training=True):
        """Forward pass - returns entropy parameters and z"""
        z = self.ha(y)
        z_bits, z_q = self.entropy_model(z, training)
        sigma = self.hs(z_q)
        
        return {
            'z': z,
            'z_q': z_q,
            'sigma': sigma,
            'z_bits': z_bits
        }
    
    def compress_z(self, z_q):
        """Compress z to bytes"""
        import zlib
        import struct
        import json
        
        z_np = z_q.cpu().numpy()
        z_min = float(z_np.min())
        z_max = float(z_np.max())
        
        if z_max - z_min < 1e-6:
            quantized = np.zeros_like(z_np, dtype=np.uint8)
        else:
            quantized = np.round((z_np - z_min) / (z_max - z_min) * 255)
            quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        
        raw = quantized.tobytes()
        compressed = zlib.compress(raw, level=9)
        
        meta = json.dumps({
            'min': z_min,
            'max': z_max,
            'shape': list(z_np.shape)
        }).encode()
        
        return compressed, meta
    
    def decompress_z(self, compressed, meta, device='cuda'):
        """Decompress z from bytes"""
        import zlib
        import json
        import numpy as np
        
        meta_dict = json.loads(meta.decode())
        raw = zlib.decompress(compressed)
        quantized = np.frombuffer(raw, dtype=np.uint8).reshape(meta_dict['shape'])
        z_np = quantized.astype(np.float32) / 255.0 * (meta_dict['max'] - meta_dict['min']) + meta_dict['min']
        
        return torch.from_numpy(z_np).to(device)