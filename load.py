"""
data_loader.py
Complete PyTorch implementation of load.py from OpenDVC
Contains data loading utilities for Vimeo90k dataset and video frame handling
"""
import os
import fnmatch
import glob
from typing import Optional, Callable, List, Tuple, Dict, Any
import subprocess
import warnings
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import numpy as np
from PIL import Image

# Suppress Pylint warnings
# pylint: disable=no-member, not-callable

# Define constants for PIL
Image.FLIP_LEFT_RIGHT = Image.Transpose.FLIP_LEFT_RIGHT
Image.BICUBIC = Image.Resampling.BICUBIC


class VimeoDataset(Dataset):
    """
    Vimeo90k dataset loader
    Matches the dataset preparation in load.py
    """
    def __init__(self, 
                 folder_list_path: str,
                 root_dir: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 frame_count: int = 7,
                 patch_size: Optional[int] = 256,
                 random_crop: bool = True,
                 random_flip: bool = True):
        """
        Args:
            folder_list_path: Path to folder.npy containing directory list
            root_dir: Root directory of Vimeo90k dataset (if folder_list contains relative paths)
            transform: Optional transform to apply
            frame_count: Number of frames per sample (7 for Vimeo90k)
            patch_size: Size of random crops (None for full frames)
            random_crop: Whether to apply random cropping
            random_flip: Whether to apply random horizontal flipping
        """
        super(VimeoDataset, self).__init__()
        self.folder_list = np.load(folder_list_path)
        self.root_dir = root_dir
        self.frame_count = frame_count
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
            
        # Validate folders
        self.valid_folders = []
        for folder in self.folder_list:
            if self._check_folder_valid(folder):
                self.valid_folders.append(folder)
        
        print(f"Loaded {len(self.valid_folders)} valid folders out of {len(self.folder_list)}")
        self.last_crop_params = None
        
    def _check_folder_valid(self, folder: str) -> bool:
        """Check if folder contains all required frames"""
        for i in range(1, self.frame_count + 1):
            img_path = os.path.join(folder, f'im{i}.png')
            if self.root_dir:
                img_path = os.path.join(self.root_dir, img_path)
            if not os.path.exists(img_path):
                return False
        return True
    
    def _get_frame_path(self, folder: str, idx: int) -> str:
        """Get path to frame with given index"""
        img_path = os.path.join(folder, f'im{idx}.png')
        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)
        return img_path
    
    def __len__(self) -> int:
        return len(self.valid_folders)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            Tensor of shape (frame_count, C, H, W)
        """
        folder = self.valid_folders[idx]
        
        # Load all frames
        frames = []
        for i in range(1, self.frame_count + 1):
            img_path = self._get_frame_path(folder, i)
            img = Image.open(img_path).convert('RGB')
            frames.append(img)
        
        # Apply augmentations consistently across all frames
        if self.patch_size is not None and self.random_crop:
            # Get random crop parameters
            w, h = frames[0].size
            i = random.randint(0, h - self.patch_size)
            j = random.randint(0, w - self.patch_size)
            self.last_crop_params = (i, j)
            
            # Crop all frames
            frames = [f.crop((j, i, j + self.patch_size, i + self.patch_size)) for f in frames]
        
        if self.random_flip and random.random() > 0.5:
            frames = [f.transpose(Image.FLIP_LEFT_RIGHT) for f in frames]
        
        # Apply transform and stack
        frames_tensor = torch.stack([self.transform(f) for f in frames])
        
        return frames_tensor


def find_folders(pattern: str, path: str) -> List[str]:
    """
    Find folders containing files matching pattern
    Matches the find() function in load.py
    
    Args:
        pattern: File pattern to match (e.g., 'im1.png')
        path: Root path to search
        
    Returns:
        List of folder paths
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(root)
                break
    return result


def create_folder_list(data_path: str, output_path: str = 'folder.npy'):
    """
    Create folder.npy file for Vimeo90k dataset
    Matches the folder creation code in load.py
    
    Args:
        data_path: Path to Vimeo90k dataset
        output_path: Output path for folder.npy
    """
    folders = find_folders('im1.png', data_path)
    np.save(output_path, folders)
    print(f"Found {len(folders)} folders, saved to {output_path}")
    return folders