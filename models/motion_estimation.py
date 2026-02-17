"""
motion_estimation.py
Complete PyTorch implementation of motion.py from OpenDVC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FlowEstimator(nn.Module):
    """
    Basic flow estimation network
    """
    def __init__(self, input_channels=6):
        super(FlowEstimator, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.conv4 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(32, 2, 7, stride=1, padding=3)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        flow = self.conv5(x)
        flow = flow * 0.5
        return flow


class MultiScaleMotionEstimation(nn.Module):
    """
    Multi-scale optical flow estimation
    """
    def __init__(self, num_levels=3):
        super(MultiScaleMotionEstimation, self).__init__()
        
        self.num_levels = num_levels
        
        self.flow_estimators = nn.ModuleList([
            FlowEstimator(input_channels=6) for _ in range(num_levels)
        ])
        
        self.downsample = nn.AvgPool2d(2, stride=2)
        
    def build_pyramid(self, img):
        pyramid = [img]
        for _ in range(self.num_levels - 1):
            img = self.downsample(img)
            pyramid.append(img)
        return pyramid[::-1]
    
    def forward(self, ref_frame, cur_frame):
        # Normalize
        ref_frame = ref_frame * 2 - 1
        cur_frame = cur_frame * 2 - 1
        
        # Build pyramids
        ref_pyramid = self.build_pyramid(ref_frame)
        cur_pyramid = self.build_pyramid(cur_frame)
        
        flow = None
        
        for level, (ref_level, cur_level) in enumerate(zip(ref_pyramid, cur_pyramid)):
            if flow is not None:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=False)
                flow = flow * 2
                
                # Warp reference
                warped_ref = self.warp_image(ref_level, flow)
                inputs = torch.cat([warped_ref, cur_level], dim=1)
                
                flow_res = self.flow_estimators[level](inputs)
                flow = flow + flow_res
            else:
                inputs = torch.cat([ref_level, cur_level], dim=1)
                flow = self.flow_estimators[level](inputs)
        
        return flow
    
    def warp_image(self, image, flow):
        B, C, H, W = image.shape
        
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat([xx, yy], 1).float().to(image.device)
        
        vgrid = grid + flow
        
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        vgrid = vgrid.permute(0, 2, 3, 1)
        warped = F.grid_sample(image, vgrid, mode='bilinear', padding_mode='border', align_corners=False)
        
        return warped