"""
metrics.py
Complete PyTorch implementation of ms_ssim_np.py from OpenDVC
Contains MS-SSIM and other quality metrics for video compression evaluation
"""
from math import exp
from typing import Optional, Tuple, List, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Suppress Pylint warnings for torch.nn.functional
# pylint: disable=not-callable, no-member

def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """
    Generate 1D Gaussian kernel
    
    Args:
        size: Kernel size
        sigma: Gaussian sigma
        
    Returns:
        1D Gaussian kernel tensor
    """
    coords = torch.arange(size).float() - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def create_window(window_size: int, sigma: float, channel: int, device: torch.device) -> torch.Tensor:
    """
    Create 2D Gaussian window for SSIM
    
    Args:
        window_size: Size of the window
        sigma: Gaussian sigma
        channel: Number of channels
        device: Device to place tensor on
        
    Returns:
        2D Gaussian window tensor of shape (channel, 1, window_size, window_size)
    """
    g1 = gaussian_kernel(window_size, sigma).to(device)
    g2 = gaussian_kernel(window_size, sigma).to(device)
    window = g1[:, None] * g2[None, :]
    window = window[None, None, :, :].repeat(channel, 1, 1, 1)
    return window


class SSIM(nn.Module):
    """
    Structural Similarity Index (SSIM)
    Matches the implementation in ms_ssim_np.py
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5, size_average: bool = True):
        """
        Args:
            window_size: Size of Gaussian window
            sigma: Sigma for Gaussian window
            size_average: If True, average over all pixels
        """
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.size_average = size_average
        self.channel = 3
        self.window = None
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM between two images
        
        Args:
            img1: First image (N, C, H, W) in range [0, 1]
            img2: Second image (N, C, H, W) in range [0, 1]
            
        Returns:
            SSIM value(s)
        """
        device = img1.device
        channel = img1.size(1)
        
        # Create window if needed
        if self.window is None or self.window.shape[1] != channel or self.window.device != device:
            self.window = create_window(self.window_size, self.sigma, channel, device)
            self.window = self.window.to(device)
        
        # Calculate means
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        # Constants for numerical stability
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        
        # Compute SSIM map
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(dim=(1, 2, 3))


class MS_SSIM(nn.Module):
    """
    Multi-Scale Structural Similarity Index (MS-SSIM)
    Matches exactly the implementation in ms_ssim_np.py
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5, levels: int = 5, 
                 size_average: bool = True, weights: Optional[List[float]] = None):
        """
        Args:
            window_size: Size of Gaussian window
            sigma: Sigma for Gaussian window
            levels: Number of scales
            size_average: If True, average over batch
            weights: Weights for each scale (default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        """
        super(MS_SSIM, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.levels = levels
        self.size_average = size_average
        
        # Default weights from original MS-SSIM paper
        if weights is None:
            self.weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        else:
            self.weights = torch.tensor(weights)
            
        self.ssim = SSIM(window_size=window_size, sigma=sigma, size_average=False)
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute MS-SSIM between two images
        
        Args:
            img1: First image (N, C, H, W) in range [0, 1]
            img2: Second image (N, C, H, W) in range [0, 1]
            
        Returns:
            MS-SSIM value(s)
        """
        device = img1.device
        weights = self.weights.to(device)
        
        # Ensure images are in [0, 1] range
        if img1.max() > 1.0 or img2.max() > 1.0:
            img1 = img1 / 255.0
            img2 = img2 / 255.0
        
        mcs = []
        ssims = []
        
        # Multi-scale computation
        for _ in range(self.levels):
            # Compute SSIM and contrast at current scale
            ssim_map = self.ssim(img1, img2)
            
            # For MS-SSIM, we need the contrast map (CS) as well
            # CS = (2*sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
            # We can get this from the SSIM computation
            C2 = (0.03) ** 2
            
            # Compute local statistics for CS
            window = create_window(self.window_size, self.sigma, img1.size(1), device)
            
            mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=img1.size(1))
            mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=img1.size(1))
            
            sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=img1.size(1)) - mu1 ** 2
            sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=img1.size(1)) - mu2 ** 2
            sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=img1.size(1)) - mu1 * mu2
            
            cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
            
            mcs.append(cs_map.mean(dim=(1, 2, 3)))
            ssims.append(ssim_map)
            
            # Downsample
            if _ < self.levels - 1:
                img1 = F.avg_pool2d(img1, 2)
                img2 = F.avg_pool2d(img2, 2)
        
        # Combine scales according to MS-SSIM formula
        mcs = torch.stack(mcs[:-1], dim=0)
        ssims = torch.stack(ssims, dim=0)
        
        # MS-SSIM = product of mcs^weights * ssim_last^weight_last
        ms_ssim_val = torch.prod(mcs ** weights[:-1].view(-1, 1), dim=0) * (ssims[-1] ** weights[-1])
        
        if self.size_average:
            return ms_ssim_val.mean()
        else:
            return ms_ssim_val


class PSNR(nn.Module):
    """
    Peak Signal-to-Noise Ratio (PSNR)
    """
    def __init__(self, max_val: float = 1.0):
        """
        Args:
            max_val: Maximum pixel value (1.0 for [0,1], 255 for [0,255])
        """
        super(PSNR, self).__init__()
        self.max_val = max_val
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute PSNR between two images
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            PSNR value in dB
        """
        mse = F.mse_loss(img1, img2)
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return psnr


class VideoQualityMetrics(nn.Module):
    """
    Comprehensive video quality metrics
    Combines multiple metrics into one class
    """
    def __init__(self):
        super(VideoQualityMetrics, self).__init__()
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.ms_ssim = MS_SSIM()
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> dict:
        """
        Compute all quality metrics
        
        Args:
            img1: First image/frame
            img2: Second image/frame
            
        Returns:
            Dictionary with all metrics
        """
        return {
            'psnr': self.psnr(img1, img2),
            'ssim': self.ssim(img1, img2),
            'ms_ssim': self.ms_ssim(img1, img2)
        }