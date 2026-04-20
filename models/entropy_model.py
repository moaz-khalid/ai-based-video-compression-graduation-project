"""
models/entropy_model.py - Complete entropy model combining hyperprior and arithmetic coding
"""

import torch
import torch.nn as nn
from models.hyperprior import ScaleHyperprior
from models.arithmetic_coder import ArithmeticCoder


class FullEntropyModel(nn.Module):
    """Complete entropy model for true compression"""
    
    def __init__(self, M=192, N=128, quality=8):
        super().__init__()
        self.M = M
        self.N = N
        self.quality = quality
        
        self.hyperprior = ScaleHyperprior(M, N)
        self.coder = ArithmeticCoder(quality)
        
    def forward(self, y, training=True):
        """Training forward pass - returns rate estimate"""
        hyper_out = self.hyperprior(y, training)
        
        if training:
            y_q = y + torch.rand_like(y) - 0.5
        else:
            y_q = torch.round(y)
        
        # Rate estimation: Gaussian entropy
        sigma = hyper_out['sigma']
        y_bits = torch.mean(((y_q) / (sigma + 1e-9)) ** 2) * 0.5 * y.numel()
        z_bits = hyper_out['z_bits']
        
        total_bits = y_bits + z_bits
        
        return {
            'y_q': y_q,
            'sigma': sigma,
            'z_q': hyper_out['z_q'],
            'y_bits': y_bits,
            'z_bits': z_bits,
            'total_bits': total_bits
        }
    
    def compress(self, y):
        """Compression mode - returns actual bytes"""
        hyper_out = self.hyperprior(y, training=False)
        y_q = torch.round(y)
        sigma = hyper_out['sigma']
        z_q = hyper_out['z_q']
        
        # Compress y
        y_channels, y_meta, y_bytes = self.coder.compress_latent(y_q, sigma)
        
        # Compress z
        z_compressed, z_meta = self.hyperprior.compress_z(z_q)
        
        return {
            'y_channels': y_channels,
            'y_meta': y_meta,
            'y_bytes': y_bytes,
            'z_compressed': z_compressed,
            'z_meta': z_meta,
            'z_bytes': len(z_compressed),
            'total_bytes': y_bytes + len(z_compressed)
        }
    
    def decompress(self, compressed, device='cuda'):
        """Decompression mode"""
        # Decompress z
        z_q = self.hyperprior.decompress_z(compressed['z_compressed'], compressed['z_meta'], device)
        
        # Get sigma
        sigma = self.hyperprior.hs(z_q)
        
        # Decompress y
        y_q = self.coder.decompress_latent(compressed['y_channels'], compressed['y_meta'], device)
        
        return y_q