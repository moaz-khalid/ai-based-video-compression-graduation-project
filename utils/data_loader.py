# pylint: disable=no-member, E1101
"""
data_loader.py
Complete PyTorch implementation of load.py from OpenDVC
"""

import os
import fnmatch
import random
import glob

from typing import Optional, Callable, List, Tuple, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

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
    """
    def __init__(self, 
                 folder_list_path: str,
                 root_dir: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 frame_count: int = 7,
                 patch_size: Optional[int] = 256,
                 random_crop: bool = True,
                 random_flip: bool = True):
        
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
        for i in range(1, self.frame_count + 1):
            img_path = os.path.join(folder, f'im{i}.png')
            if self.root_dir:
                img_path = os.path.join(self.root_dir, img_path)
            if not os.path.exists(img_path):
                return False
        return True
    
    def _get_frame_path(self, folder: str, idx: int) -> str:
        img_path = os.path.join(folder, f'im{idx}.png')
        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)
        return img_path
    
    def __len__(self) -> int:
        return len(self.valid_folders)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        folder = self.valid_folders[idx]
        
        # Load all frames
        frames = []
        for i in range(1, self.frame_count + 1):
            img_path = self._get_frame_path(folder, i)
            img = Image.open(img_path).convert('RGB')
            frames.append(img)
        
        # Apply augmentations
        if self.patch_size is not None and self.random_crop:
            w, h = frames[0].size
            i = random.randint(0, h - self.patch_size)
            j = random.randint(0, w - self.patch_size)
            self.last_crop_params = (i, j)
            
            frames = [f.crop((j, i, j + self.patch_size, i + self.patch_size)) for f in frames]
        
        if self.random_flip and random.random() > 0.5:
            frames = [f.transpose(FLIP_LEFT_RIGHT) for f in frames]
        
        frames_tensor = torch.stack([self.transform(f) for f in frames])
        return frames_tensor


def find_folders(pattern: str, path: str) -> List[str]:
    """
    Find folders containing files matching pattern
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(root)
                break
    return result


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = VimeoDataset(
        folder_list_path=config['train_folder_list'],
        root_dir=config.get('data_root', None),
        transform=transform,
        patch_size=config.get('patch_size', 256),
        random_crop=True,
        random_flip=True
    )
    
    val_dataset = VimeoDataset(
        folder_list_path=config.get('val_folder_list', config['train_folder_list']),
        root_dir=config.get('data_root', None),
        transform=transform,
        patch_size=None,
        random_crop=False,
        random_flip=False
    )
    
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