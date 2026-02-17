"""
motion_estimation.py
Complete PyTorch implementation of motion.py from OpenDVC
Contains optical flow estimation network (SpyNet architecture)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FlowEstimator(nn.Module):
    """
    Basic flow estimation network for a single scale
    Matches the architecture in motion.py
    """
    def __init__(self, input_channels=6):
        super(FlowEstimator, self).__init__()
        
        # 5-layer CNN as in the original implementation
        self.conv1 = nn.Conv2d(input_channels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.conv4 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(32, 2, 7, stride=1, padding=3)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: Concatenated reference and current frames (N, 6, H, W)
        Returns:
            Optical flow (N, 2, H, W)
        """
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        flow = self.conv5(x)
        
        # Scale flow to pixel coordinates (original implementation scales by 0.5)
        flow = flow * 0.5
        
        return flow


def warp_image(image, flow):
    """
    Differentiable warping using grid_sample
    Matches the warping operation in motion.py
    """
    B, C, H, W = image.shape
    
    # Create mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat([xx, yy], 1).float().to(image.device)
    
    # Add flow
    vgrid = grid + flow
    
    # Normalize to [-1, 1] for grid_sample
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    
    # Resample
    vgrid = vgrid.permute(0, 2, 3, 1)
    warped = F.grid_sample(image, vgrid, mode='bilinear', padding_mode='border', align_corners=False)
    
    return warped


class MultiScaleMotionEstimation(nn.Module):
    """
    Multi-scale optical flow estimation (SpyNet architecture)
    Matches the full motion.py implementation with pyramid processing
    """
    def __init__(self, num_levels=3):
        super(MultiScaleMotionEstimation, self).__init__()
        
        self.num_levels = num_levels
        
        # Create flow estimators for each scale
        self.flow_estimators = nn.ModuleList([
            FlowEstimator(input_channels=6) for _ in range(num_levels)
        ])
        
        # Downsampling layers for pyramid
        self.downsample = nn.AvgPool2d(2, stride=2)
        
    def build_pyramid(self, img):
        """Build image pyramid"""
        pyramid = [img]
        for _ in range(self.num_levels - 1):
            img = self.downsample(img)
            pyramid.append(img)
        return pyramid[::-1]  # Reverse to have coarse-to-fine order
    
    def forward(self, ref_frame, cur_frame):
        """
        Multi-scale flow estimation
        
        Args:
            ref_frame: Reference frame (N, 3, H, W)
            cur_frame: Current frame (N, 3, H, W)
            
        Returns:
            Estimated optical flow (N, 2, H, W)
        """
        # Normalize to [-1, 1] for better stability
        ref_frame = ref_frame * 2 - 1
        cur_frame = cur_frame * 2 - 1
        
        # Build pyramids
        ref_pyramid = self.build_pyramid(ref_frame)
        cur_pyramid = self.build_pyramid(cur_frame)
        
        # Coarse-to-fine flow estimation
        flow = None
        
        for level, (ref_level, cur_level) in enumerate(zip(ref_pyramid, cur_pyramid)):
            if flow is not None:
                # Upscale flow to current level
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=False)
                flow = flow * 2  # Scale flow accordingly
                
                # Warp reference frame using current flow estimate
                warped_ref = warp_image(ref_level, flow)
                
                # Concatenate warped reference and current frame
                inputs = torch.cat([warped_ref, cur_level], dim=1)
                
                # Estimate residual flow
                flow_res = self.flow_estimators[level](inputs)
                flow = flow + flow_res
            else:
                # First level (coarsest): estimate flow from original frames
                inputs = torch.cat([ref_level, cur_level], dim=1)
                flow = self.flow_estimators[level](inputs)
        
        return flow


class MotionEstimationWithContext(nn.Module):
    """
    Enhanced motion estimation with context features
    Matches the full motion.py implementation including context extraction
    """
    def __init__(self, num_levels=3, context_features=64):
        super(MotionEstimationWithContext, self).__init__()
        
        self.num_levels = num_levels
        self.context_features = context_features
        
        # Context extraction networks
        self.ref_context = nn.Conv2d(3, context_features, 3, stride=1, padding=1)
        self.cur_context = nn.Conv2d(3, context_features, 3, stride=1, padding=1)
        
        # Flow estimators with context
        self.flow_estimators = nn.ModuleList([
            nn.Conv2d(2 * context_features + 2, 32, 7, stride=1, padding=3) for _ in range(num_levels)
        ])
        self.flow_outputs = nn.ModuleList([
            nn.Conv2d(32, 2, 7, stride=1, padding=3) for _ in range(num_levels)
        ])
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool2d(2, stride=2)
        
    def build_pyramid(self, img):
        """Build image pyramid"""
        pyramid = [img]
        for _ in range(self.num_levels - 1):
            img = self.downsample(img)
            pyramid.append(img)
        return pyramid[::-1]
    
    def extract_context_pyramid(self, img):
        """Extract context features at multiple scales"""
        # Extract context at original scale
        context = self.leaky_relu(self.ref_context(img))
        
        # Build pyramid
        context_pyramid = [context]
        for _ in range(self.num_levels - 1):
            context = self.downsample(context)
            context_pyramid.append(context)
        
        return context_pyramid[::-1]
    
    def forward(self, ref_frame, cur_frame):
        """
        Motion estimation with context features
        
        Args:
            ref_frame: Reference frame (N, 3, H, W)
            cur_frame: Current frame (N, 3, H, W)
            
        Returns:
            Estimated optical flow (N, 2, H, W)
        """
        # Normalize
        ref_frame = ref_frame * 2 - 1
        cur_frame = cur_frame * 2 - 1
        
        # Extract context features
        ref_context_pyramid = self.extract_context_pyramid(ref_frame)
        cur_context_pyramid = self.extract_context_pyramid(cur_frame)
        
        # Build image pyramids
        ref_pyramid = self.build_pyramid(ref_frame)
        cur_pyramid = self.build_pyramid(cur_frame)
        
        # Coarse-to-fine flow estimation
        flow = None
        
        for level, (ref_img, cur_img, ref_ctx, cur_ctx) in enumerate(zip(
            ref_pyramid, cur_pyramid, ref_context_pyramid, cur_context_pyramid
        )):
            if flow is not None:
                # Upscale flow
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=False)
                flow = flow * 2
                
                # Warp reference context using current flow
                warped_ctx = warp_image(ref_ctx, flow)
                
                # Concatenate features and flow
                inputs = torch.cat([warped_ctx, cur_ctx, flow], dim=1)
            else:
                # First level: no flow yet
                inputs = torch.cat([ref_ctx, cur_ctx], dim=1)
                # Pad with zeros for flow dimension
                zeros = torch.zeros_like(ref_ctx[:, :2])
                inputs = torch.cat([inputs, zeros], dim=1)
            
            # Estimate flow residual
            features = self.leaky_relu(self.flow_estimators[level](inputs))
            flow_res = self.flow_outputs[level](features)
            
            if flow is not None:
                flow = flow + flow_res
            else:
                flow = flow_res
        
        return flow


# Unit test
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test data
    batch_size = 2
    height, width = 256, 256
    ref_frame = torch.rand(batch_size, 3, height, width).to(device)
    cur_frame = torch.rand(batch_size, 3, height, width).to(device)
    
    # Test basic flow estimator
    basic_estimator = FlowEstimator().to(device)
    inputs = torch.cat([ref_frame, cur_frame], dim=1)
    flow_basic = basic_estimator(inputs)
    print(f"Basic flow estimator: {flow_basic.shape}")
    
    # Test multi-scale estimator
    multi_scale = MultiScaleMotionEstimation(num_levels=3).to(device)
    flow_ms = multi_scale(ref_frame, cur_frame)
    print(f"Multi-scale flow: {flow_ms.shape}")
    
    # Test context-based estimator
    context_estimator = MotionEstimationWithContext().to(device)
    flow_context = context_estimator(ref_frame, cur_frame)
    print(f"Context-based flow: {flow_context.shape}")
    
    # Test gradient flow
    loss = flow_context.mean()
    loss.backward()
    print("Gradient flow test passed!")
    
    # Count parameters
    total_params = sum(p.numel() for p in multi_scale.parameters())
    print(f"Multi-scale motion estimation parameters: {total_params:,}")