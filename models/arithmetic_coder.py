"""
models/arithmetic_coder.py - Arithmetic coding for latent compression
"""

import torch
import numpy as np
import zlib
import json
import struct


class ArithmeticCoder:
    """Entropy coder using zlib with learned parameters"""
    
    def __init__(self, quality=8):
        self.quality = quality
        self.levels = 2 ** quality
    
    def compress_latent(self, y_q, sigma):
        """
        Compress latent using Gaussian entropy model
        Returns: compressed_bytes, metadata
        """
        y_np = y_q.cpu().numpy()
        sigma_np = sigma.cpu().numpy()
        
        # Per-channel quantization bounds
        C = y_np.shape[1]
        channels = []
        total_bytes = 0
        
        for c in range(C):
            y_c = y_np[0, c].flatten()
            s_c = sigma_np[0, c].flatten()
            
            # Adaptive quantization based on sigma
            y_min = float(np.percentile(y_c, 1))
            y_max = float(np.percentile(y_c, 99))
            
            if y_max - y_min < 1e-6:
                quantized = np.zeros_like(y_c, dtype=np.uint8)
                ch_meta = {'min': y_min, 'max': y_max, 'constant': True}
            else:
                quantized = np.round((y_c - y_min) / (y_max - y_min) * (self.levels - 1))
                quantized = np.clip(quantized, 0, self.levels - 1).astype(np.uint8)
                ch_meta = {'min': y_min, 'max': y_max, 'constant': False}
            
            # Compress with zlib
            raw = quantized.tobytes()
            compressed = zlib.compress(raw, level=9)
            
            channels.append({
                'data': compressed,
                'meta': ch_meta
            })
            total_bytes += len(compressed)
        
        metadata = {
            'shape': list(y_np.shape),
            'quality': self.quality,
            'channels': len(channels)
        }
        
        return channels, metadata, total_bytes
    
    def decompress_latent(self, channels, metadata, device='cuda'):
        """Decompress latent from bytes"""
        C = metadata['channels']
        shape = metadata['shape']
        
        y_recon = np.zeros(shape, dtype=np.float32)
        
        for c in range(C):
            ch_data = channels[c]['data']
            ch_meta = channels[c]['meta']
            
            raw = zlib.decompress(ch_data)
            quantized = np.frombuffer(raw, dtype=np.uint8)
            
            if ch_meta.get('constant', False):
                y_recon[0, c] = ch_meta['min']
            else:
                dequant = quantized.astype(np.float32) / (self.levels - 1)
                dequant = dequant * (ch_meta['max'] - ch_meta['min']) + ch_meta['min']
                y_recon[0, c] = dequant.reshape(shape[2], shape[3])
        
        return torch.from_numpy(y_recon).to(device)
    
    def estimate_bpp(self, channels, total_pixels):
        """Estimate actual bits per pixel"""
        total_bits = sum(len(ch['data']) for ch in channels) * 8
        return total_bits / total_pixels if total_pixels > 0 else 0