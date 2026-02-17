#!/usr/bin/env python
# scripts/test.py
"""
OpenDVC Testing/Encoding Script
"""

import os
import sys
import argparse
import json
import glob
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from models and utils
from models import (
    AnalysisTransform, SynthesisTransform,
    MultiScaleMotionEstimation,
    FullMotionCompensation
)
from utils import VideoQualityMetrics

# Import the model from train.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import OpenDVCModel


class OpenDVCEncoder:
    """
    Main encoder class for OpenDVC
    """
    def __init__(self, 
                 model_path: str,
                 mode: str = 'PSNR',
                 lambda_param: int = 1024,
                 device: str = 'cuda',
                 N: int = 128,
                 M: int = 192):
        """
        Args:
            model_path: Path to pretrained model
            mode: 'PSNR' or 'MS-SSIM'
            lambda_param: Lambda value
            device: Device to use
            N: Number of filters
            M: Latent dimension
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.mode = mode
        self.lambda_param = lambda_param
        
        # Load model
        self.model = OpenDVCModel(
            mode=mode,
            N=N,
            M=M,
            lambda_param=lambda_param
        ).to(self.device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using untrained model.")
        
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
    
    def compress_video(self,
                       input_dir: str,
                       output_dir: str,
                       gop_size: int = 10,
                       num_frames: Optional[int] = None,
                       verbose: bool = False) -> Dict:
        """
        Compress video sequence
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get frame list
        frames = sorted(glob.glob(os.path.join(input_dir, '*.png')))
        if num_frames:
            frames = frames[:num_frames]
        
        if verbose:
            print(f"Compressing {len(frames)} frames")
        
        stats = {
            'total_bytes': 0,
            'psnr_values': [],
            'bpp_values': []
        }
        
        for i, frame_path in enumerate(tqdm(frames, disable=not verbose)):
            # Simplified compression (just copy for testing)
            img = Image.open(frame_path)
            output_path = os.path.join(output_dir, os.path.basename(frame_path))
            img.save(output_path)
            
            stats['total_bytes'] += os.path.getsize(output_path)
        
        stats['total_size_mb'] = stats['total_bytes'] / (1024 * 1024)
        
        return stats


def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description='Test OpenDVC model')
    parser.add_argument('--command', type=str, default='encode', 
                       choices=['encode', 'decode', 'evaluate'])
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--mode', type=str, default='PSNR')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print(f"Running command: {args.command}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    
    if args.command == 'encode':
        encoder = OpenDVCEncoder(
            model_path=args.model_path,
            mode=args.mode,
            device=args.device
        )
        stats = encoder.compress_video(args.input_dir, args.output_dir, verbose=True)
        print(f"Compression complete: {stats['total_size_mb']:.2f} MB")
    
    print("Test complete!")


if __name__ == "__main__":
    main()