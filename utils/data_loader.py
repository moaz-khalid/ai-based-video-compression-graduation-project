"""
data_loader.py
Complete PyTorch implementation of load.py from OpenDVC
Contains data loading utilities for Vimeo90k dataset and video frame handling
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import os
import fnmatch
from PIL import Image
import random
import glob
from typing import Optional, Callable, List, Tuple, Dict, Any
import subprocess
import warnings

# Suppress Pylint warnings
# pylint: disable=no-member, not-callable

# Fix for PIL constants across different versions
try:
    # For PIL >= 10.0.0
    from PIL import Image
    FLIP_LEFT_RIGHT = Image.Transpose.FLIP_LEFT_RIGHT
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    try:
        # For older PIL versions
        FLIP_LEFT_RIGHT = Image.FLIP_LEFT_RIGHT
        BICUBIC = Image.BICUBIC
    except AttributeError:
        # Fallback values
        FLIP_LEFT_RIGHT = 0
        BICUBIC = 3


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
            frames = [f.transpose(FLIP_LEFT_RIGHT) for f in frames]
        
        # Apply transform and stack
        frames_tensor = torch.stack([self.transform(f) for f in frames])
        
        return frames_tensor


class VideoFrameDataset(Dataset):
    """
    Dataset for loading video frames from a directory
    Used for testing/encoding
    """
    def __init__(self, 
                 frame_dir: str,
                 transform: Optional[Callable] = None,
                 frame_limit: Optional[int] = None,
                 start_frame: int = 0,
                 extension: str = 'png'):
        """
        Args:
            frame_dir: Directory containing frame images
            transform: Optional transform to apply
            frame_limit: Maximum number of frames to load
            start_frame: Starting frame index
            extension: File extension of frames
        """
        self.frame_dir = frame_dir
        self.transform = transform or transforms.ToTensor()
        
        # Get all frame files
        self.frame_files = sorted(glob.glob(os.path.join(frame_dir, f'*.{extension}')))
        
        if frame_limit:
            self.frame_files = self.frame_files[start_frame:start_frame + frame_limit]
        elif start_frame > 0:
            self.frame_files = self.frame_files[start_frame:]
            
        print(f"Loaded {len(self.frame_files)} frames from {frame_dir}")
        
    def __len__(self) -> int:
        return len(self.frame_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns:
            Frame tensor and filename
        """
        img_path = self.frame_files[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        
        return img_tensor, os.path.basename(img_path)


class VideoPairDataset(Dataset):
    """
    Dataset for loading pairs of frames (reference and current)
    Used for motion estimation training
    """
    def __init__(self, 
                 folder_list_path: str,
                 root_dir: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 gap: int = 1,
                 patch_size: Optional[int] = 256):
        """
        Args:
            folder_list_path: Path to folder.npy
            root_dir: Root directory
            transform: Optional transform
            gap: Frame gap between reference and current
            patch_size: Size for random crops
        """
        self.base_dataset = VimeoDataset(
            folder_list_path=folder_list_path,
            root_dir=root_dir,
            transform=transform,
            patch_size=patch_size,
            random_crop=True,
            random_flip=True
        )
        self.gap = gap
        
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (reference frame, current frame)
        """
        frames = self.base_dataset[idx]  # (7, C, H, W)
        
        # Randomly select reference and current frames with given gap
        max_start = 7 - self.gap
        start_idx = random.randint(0, max_start - 1)
        
        ref_frame = frames[start_idx]
        cur_frame = frames[start_idx + self.gap]
        
        return ref_frame, cur_frame


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


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Datasets
    train_dataset = VimeoDataset(
        folder_list_path=config['train_folder_list'],
        root_dir=config.get('data_root', None),
        transform=train_transform,
        patch_size=config.get('patch_size', 256),
        random_crop=True,
        random_flip=True
    )
    
    val_dataset = VimeoDataset(
        folder_list_path=config.get('val_folder_list', config['train_folder_list']),
        root_dir=config.get('data_root', None),
        transform=val_transform,
        patch_size=None,  # Use full frames for validation
        random_crop=False,
        random_flip=False
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader