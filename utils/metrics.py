# pylint: disable=not-callable, no-member, E1101, E1102
"""
metrics.py
Complete PyTorch implementation of ms_ssim_np.py
"""
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """Generate 1D Gaussian kernel"""
    coords = torch.arange(size).float() - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def create_window(window_size: int, sigma: float, channel: int, device: torch.device) -> torch.Tensor:
    """Create 2D Gaussian window"""
    g1 = gaussian_kernel(window_size, sigma).to(device)
    g2 = gaussian_kernel(window_size, sigma).to(device)
    window = g1[:, None] * g2[None, :]
    window = window[None, None, :, :].repeat(channel, 1, 1, 1)
    return window


class SSIM(nn.Module):
    """Structural Similarity Index"""
    def __init__(self, window_size: int = 11, sigma: float = 1.5, size_average: bool = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.size_average = size_average
        self.channel = 3
        self.window = None
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        device = img1.device
        channel = img1.size(1)
        
        if self.window is None or self.window.shape[1] != channel or self.window.device != device:
            self.window = create_window(self.window_size, self.sigma, channel, device)
            self.window = self.window.to(device)
        
        # These F.conv2d calls are valid - Pylint warnings are false positives
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(dim=(1, 2, 3))


class MS_SSIM(nn.Module):
    """Multi-Scale Structural Similarity Index"""
    def __init__(self, window_size: int = 11, sigma: float = 1.5, levels: int = 5, 
                 size_average: bool = True, weights: Optional[List[float]] = None):
        super(MS_SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.levels = levels
        self.size_average = size_average
        
        if weights is None:
            self.weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        else:
            self.weights = torch.tensor(weights)
            
        self.ssim = SSIM(window_size=window_size, sigma=sigma, size_average=False)
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        device = img1.device
        weights = self.weights.to(device)
        
        if img1.max() > 1.0 or img2.max() > 1.0:
            img1 = img1 / 255.0
            img2 = img2 / 255.0
        
        mcs = []
        
        for _ in range(self.levels):
            ssim_map = self.ssim(img1, img2)
            
            C2 = (0.03) ** 2
            window = create_window(self.window_size, self.sigma, img1.size(1), device)
            
            mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=img1.size(1))
            mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=img1.size(1))
            
            sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=img1.size(1)) - mu1 ** 2
            sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=img1.size(1)) - mu2 ** 2
            sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=img1.size(1)) - mu1 * mu2
            
            cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
            mcs.append(cs_map.mean(dim=(1, 2, 3)))
            
            if _ < self.levels - 1:
                img1 = F.avg_pool2d(img1, 2)
                img2 = F.avg_pool2d(img2, 2)
        
        mcs = torch.stack(mcs, dim=0)
        ms_ssim_val = torch.prod(mcs ** weights.view(-1, 1), dim=0)
        
        if self.size_average:
            return ms_ssim_val.mean()
        else:
            return ms_ssim_val


class PSNR(nn.Module):
    """Peak Signal-to-Noise Ratio"""
    def __init__(self, max_val: float = 1.0):
        super(PSNR, self).__init__()
        self.max_val = max_val
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(img1, img2)
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse + 1e-8))
        return psnr


class VideoQualityMetrics(nn.Module):
    """Comprehensive video quality metrics"""
    def __init__(self):
        super(VideoQualityMetrics, self).__init__()
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.ms_ssim = MS_SSIM()
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> dict:
        return {
            'psnr': self.psnr(img1, img2),
            'ssim': self.ssim(img1, img2),
            'ms_ssim': self.ms_ssim(img1, img2)
        }