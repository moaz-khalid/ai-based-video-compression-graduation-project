"""
motion_compensation.py
Complete PyTorch implementation of MC_network.py from OpenDVC
"""
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WarpingLayer(nn.Module):
    """
    Differentiable warping layer using grid_sample
    """
    def __init__(self):
        super(WarpingLayer, self).__init__()
        
    def forward(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp image x using optical flow
        """
        B, C, H, W = x.shape
        
        # Create mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat([xx, yy], 1).float().to(x.device)
        
        # Add flow to grid
        vgrid = grid + flow
        
        # Normalize grid to [-1, 1] for grid_sample
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        # Resample
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='border', align_corners=False)
        
        return output


class RefinementNetwork(nn.Module):
    """
    Refinement network for motion compensation
    """
    def __init__(self, in_channels: int = 3+2, hidden_channels: int = 64):
        super(RefinementNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(hidden_channels, 3, 3, stride=1, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, warped_frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        x = torch.cat([warped_frame, flow], dim=1)
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        residual = self.conv4(x)
        
        return residual


class FullMotionCompensation(nn.Module):
    """
    Complete motion compensation network
    """
    def __init__(self, hidden_channels: int = 64, use_context: bool = True, use_multi_scale: bool = True):
        super(FullMotionCompensation, self).__init__()
        
        self.use_context = use_context
        self.use_multi_scale = use_multi_scale
        
        # Basic components
        self.warping = WarpingLayer()
        
        if use_multi_scale:
            # Multi-scale refinement
            self.refinement_coarse = RefinementNetwork(in_channels=3+2, hidden_channels=hidden_channels)
            self.refinement_medium = RefinementNetwork(in_channels=3+2, hidden_channels=hidden_channels)
            
            # Downsampling for multi-scale
            self.downsample = nn.AvgPool2d(2, stride=2)
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.refinement = RefinementNetwork(in_channels=3+2, hidden_channels=hidden_channels)
        
        if use_context:
            # Context networks
            self.context_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
        
        # Confidence prediction
        self.confidence = nn.Sequential(
            nn.Conv2d(3 + 2, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, ref_frame: torch.Tensor, flow: torch.Tensor, 
                return_all: bool = False) -> Dict[str, torch.Tensor]:
        """
        Full motion compensation pipeline
        """
        B, C, H, W = ref_frame.shape
        
        if self.use_multi_scale:
            # Multi-scale processing
            # Coarse level
            ref_coarse = self.downsample(ref_frame)
            flow_coarse = self.downsample(flow)
            
            warped_coarse = self.warping(ref_coarse, flow_coarse)
            refined_coarse = self.refinement_coarse(warped_coarse, flow_coarse)
            comp_coarse = warped_coarse + refined_coarse
            
            # Medium level
            warped_medium = self.warping(ref_frame, flow)
            refined_medium = self.refinement_medium(warped_medium, flow)
            comp_medium = warped_medium + refined_medium
            
            # Blend based on confidence
            conf_medium = self.confidence(torch.cat([warped_medium, flow], dim=1))
            compensated = comp_medium * conf_medium + self.upsample(comp_coarse) * (1 - conf_medium)
            
        else:
            # Single scale processing
            warped = self.warping(ref_frame, flow)
            
            if self.use_context:
                context = self.context_encoder(ref_frame)
                warped_context = self.warping(context, flow)
                refined = self.refinement(torch.cat([warped, flow, warped_context], dim=1))
            else:
                refined = self.refinement(warped, flow)
            
            compensated = warped + refined
        
        # Final blending
        confidence = self.confidence(torch.cat([compensated, flow], dim=1))
        warped_final = self.warping(ref_frame, flow)
        compensated = compensated * confidence + warped_final * (1 - confidence)
        
        compensated = torch.clamp(compensated, 0, 1)
        
        if return_all:
            return {
                'compensated': compensated,
                'confidence': confidence,
                'warped': warped_final,
                'refined': refined_medium if self.use_multi_scale else refined,
                'flow': flow
            }
        else:
            return {'compensated': compensated}