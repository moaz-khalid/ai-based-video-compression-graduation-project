#!/usr/bin/env python
"""
train_hyperprior.py - Complete training with hyperprior for true compression
"""

import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

torch.backends.cudnn.benchmark = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    AnalysisTransform, SynthesisTransform,
    MultiScaleMotionEstimation, FullMotionCompensation,
    FullEntropyModel
)
from utils import MS_SSIM


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Vimeo90KDataset(Dataset):
    def __init__(self, root_dir, crop_size=256, mode='train', val_split=0.1):
        self.crop_size = crop_size
        self.mode = mode
        
        all_samples = []
        if os.path.exists(root_dir):
            for root, dirs, files in os.walk(root_dir):
                if "im1.png" in files and "im2.png" in files:
                    all_samples.append(root)
        
        random.shuffle(all_samples)
        val_size = int(len(all_samples) * val_split)
        
        if mode == 'train':
            self.samples = all_samples[val_size:]
        else:
            self.samples = all_samples[:val_size]
        
        print(f"✅ {mode.capitalize()}: {len(self.samples)} sequences")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        
        im1 = cv2.imread(os.path.join(path, "im1.png"))
        im2 = cv2.imread(os.path.join(path, "im2.png"))
        
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
        
        h, w, _ = im1.shape
        if h >= self.crop_size and w >= self.crop_size:
            x = random.randint(0, w - self.crop_size)
            y = random.randint(0, h - self.crop_size)
            im1 = im1[y:y+self.crop_size, x:x+self.crop_size]
            im2 = im2[y:y+self.crop_size, x:x+self.crop_size]
        else:
            im1 = cv2.resize(im1, (self.crop_size, self.crop_size))
            im2 = cv2.resize(im2, (self.crop_size, self.crop_size))
        
        if self.mode == 'train' and random.random() > 0.5:
            im1 = cv2.flip(im1, 1)
            im2 = cv2.flip(im2, 1)
        
        im1 = torch.tensor(im1).permute(2, 0, 1).float() / 255.0
        im2 = torch.tensor(im2).permute(2, 0, 1).float() / 255.0
        
        return im1, im2


class OpenDVCWithHyperprior(nn.Module):
    """Complete OpenDVC with hyperprior for true compression"""
    
    def __init__(self, mode='PSNR', N=128, M=192, lambda_param=2048.0):
        super().__init__()
        self.mode = mode
        self.lambda_param = lambda_param
        
        self.motion_net = MultiScaleMotionEstimation(num_levels=3)
        self.compensation_net = FullMotionCompensation(
            hidden_channels=64, use_context=True, use_multi_scale=True
        )
        self.residual_encoder = AnalysisTransform(N=N, M=M)
        self.residual_decoder = SynthesisTransform(N=N, M=M)
        
        # Entropy model with hyperprior
        self.entropy_model = FullEntropyModel(M=M, N=128, quality=8)
        
        self.mse_loss = nn.MSELoss()
        self.ms_ssim_loss = MS_SSIM() if mode == 'MS-SSIM' else None
    
    def forward(self, ref_frame, cur_frame, training=True):
        flow = self.motion_net(ref_frame, cur_frame)
        pred_frame = self.compensation_net(ref_frame, flow)['compensated']
        
        residual = cur_frame - pred_frame
        y = self.residual_encoder(residual)
        
        # Entropy model forward
        entropy_out = self.entropy_model(y, training)
        y_q = entropy_out['y_q']
        
        residual_recon = self.residual_decoder(y_q)
        recon_frame = torch.clamp(pred_frame + residual_recon, 0, 1)
        
        if training:
            if self.mode == 'PSNR':
                distortion = self.mse_loss(recon_frame, cur_frame)
            else:
                distortion = 1 - self.ms_ssim_loss(recon_frame, cur_frame)
            
            total_bits = entropy_out['total_bits']
            bpp = total_bits / cur_frame.numel()
            
            loss = bpp + self.lambda_param * distortion
            
            with torch.no_grad():
                psnr = 20 * torch.log10(1.0 / torch.sqrt(distortion + 1e-8))
            
            return {
                'loss': loss,
                'bpp': bpp.item(),
                'psnr': psnr.item(),
                'distortion': distortion.item(),
                'recon': recon_frame,
                'y_bits': entropy_out['y_bits'].item(),
                'z_bits': entropy_out['z_bits'].item()
            }
        else:
            return recon_frame, entropy_out
    
    def compress(self, ref_frame, cur_frame):
        """True compression - returns compressed bytes"""
        with torch.no_grad():
            flow = self.motion_net(ref_frame, cur_frame)
            pred_frame = self.compensation_net(ref_frame, flow)['compensated']
            
            residual = cur_frame - pred_frame
            y = self.residual_encoder(residual)
            
            compressed = self.entropy_model.compress(y)
            compressed['pred_frame'] = pred_frame
            
            return compressed
    
    def decompress(self, compressed, ref_frame):
        """Decompress from bytes - FIXED VERSION"""
        with torch.no_grad():
            # Handle missing pred_frame key (for compatibility with different encoder versions)
            if 'pred_frame' in compressed:
                pred_frame = compressed['pred_frame']
            else:
                # Use ref_frame as prediction if pred_frame not available
                pred_frame = ref_frame
            
            y_q = self.entropy_model.decompress(compressed, ref_frame.device)
            residual_recon = self.residual_decoder(y_q)
            
            # Ensure dimensions match (fix for 1080 vs 1088 error)
            if pred_frame.shape[2] != residual_recon.shape[2] or pred_frame.shape[3] != residual_recon.shape[3]:
                residual_recon = torch.nn.functional.interpolate(
                    residual_recon, 
                    size=(pred_frame.shape[2], pred_frame.shape[3]), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            recon_frame = torch.clamp(pred_frame + residual_recon, 0, 1)
            
            return recon_frame


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_bpp = 0
    total_psnr = 0
    
    with torch.no_grad():
        for ref_frame, cur_frame in dataloader:
            ref_frame = ref_frame.to(device)
            cur_frame = cur_frame.to(device)
            
            outputs = model(ref_frame, cur_frame, training=True)
            
            total_loss += outputs['loss'].item()
            total_bpp += outputs['bpp']
            total_psnr += outputs['psnr']
    
    model.train()
    n = len(dataloader)
    return total_loss/n, total_bpp/n, total_psnr/n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='checkpoints_hyperprior')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='PSNR')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"🧠 OpenDVC + Hyperprior Training (TRUE COMPRESSION)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Data: {args.data_root}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch: {args.batch_size}")
    print(f"{'='*70}\n")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    train_dataset = Vimeo90KDataset(args.data_root, mode='train')
    val_dataset = Vimeo90KDataset(args.data_root, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")
    
    model = OpenDVCWithHyperprior(mode=args.mode).to(device)
    
    # Load pretrained base model if available
    base_checkpoint = 'checkpoints/best_model.pth'
    if os.path.exists(base_checkpoint):
        print(f"📦 Loading pretrained base model...")
        checkpoint = torch.load(base_checkpoint, map_location=device)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model_dict = model.state_dict()
        matched = 0
        for k in model_dict:
            if k in state_dict and model_dict[k].shape == state_dict[k].shape:
                model_dict[k] = state_dict[k]
                matched += 1
        model.load_state_dict(model_dict, strict=False)
        print(f"   Loaded {matched}/{len(model_dict)} layers\n")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    
    # Lambda schedule for rate-distortion
    lambda_schedule = {0: 256, 25: 512, 50: 1024, 75: 2048}
    
    start_epoch = 0
    best_psnr = 0.0
    history = []
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_psnr = checkpoint.get('psnr', 0.0)
        history = checkpoint.get('history', [])
        print(f"📦 Resumed from epoch {start_epoch}\n")
    
    for epoch in range(start_epoch, args.epochs):
        # Update lambda
        for threshold, lam in lambda_schedule.items():
            if epoch >= threshold:
                model.lambda_param = lam
        
        model.train()
        total_loss = 0
        total_bpp = 0
        total_psnr = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for ref_frame, cur_frame in loop:
            ref_frame = ref_frame.to(device)
            cur_frame = cur_frame.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(ref_frame, cur_frame, training=True)
                loss = outputs['loss']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            total_bpp += outputs['bpp']
            total_psnr += outputs['psnr']
            
            loop.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bpp': f'{outputs["bpp"]:.3f}',
                'psnr': f'{outputs["psnr"]:.1f}',
                'λ': f'{model.lambda_param:.0f}'
            })
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        avg_bpp = total_bpp / len(train_loader)
        avg_psnr = total_psnr / len(train_loader)
        
        val_loss, val_bpp, val_psnr = validate(model, val_loader, device)
        
        print(f"\n📈 Epoch {epoch+1}: Loss={avg_loss:.4f}, BPP={avg_bpp:.4f}, PSNR={avg_psnr:.2f} dB")
        print(f"   Val: Loss={val_loss:.4f}, BPP={val_bpp:.4f}, PSNR={val_psnr:.2f} dB, λ={model.lambda_param:.0f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_bpp': avg_bpp,
            'train_psnr': avg_psnr,
            'val_loss': val_loss,
            'val_bpp': val_bpp,
            'val_psnr': val_psnr,
            'lambda': model.lambda_param
        })
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'psnr': val_psnr,
                'bpp': val_bpp,
                'history': history
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"   🏆 Best model! PSNR={best_psnr:.2f} dB")
        
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history
            }, os.path.join(args.save_dir, f'checkpoint_{epoch+1}.pth'))
    
    torch.save({
        'epoch': args.epochs,
        'model': model.state_dict(),
        'history': history
    }, os.path.join(args.save_dir, 'last.pth'))
    
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete! Best PSNR: {best_psnr:.2f} dB")
    print(f"   Models saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()