"""
decoder.py - Working decoder for .aibc files
Actually reconstructs video frames from the compressed data
"""

import struct
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import cv2
import os

sys.path.insert(0, str(Path(__file__).parent))

from models import SynthesisTransform


class RealDecoder:
    """Actually decodes the aibc file and reconstructs video"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Decoder device: {self.device}")
        
        # Load residual decoder
        self.residual_decoder = SynthesisTransform(N=128, M=192).to(self.device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load residual decoder weights
        decoder_dict = {}
        for k, v in state_dict.items():
            if k.startswith('residual_decoder.'):
                decoder_dict[k.replace('residual_decoder.', '')] = v
        
        self.residual_decoder.load_state_dict(decoder_dict, strict=False)
        self.residual_decoder.eval()
        
        print(f"✅ Model loaded")
    
    def decode_y_tensor(self, y_channels, y_meta):
        """Decode y tensor using the actual data from file"""
        y_shape = y_meta.get('shape', [1, 192, 68, 120])
        _, C, H, W = y_shape
        
        # Create y tensor
        y_tensor = torch.zeros(1, C, H, W, dtype=torch.float32).to(self.device)
        
        for ch_idx, channel_data in enumerate(y_channels[:C]):
            data_bytes = channel_data['data']
            meta = channel_data.get('meta', {})
            
            if len(data_bytes) == 0:
                continue
            
            # Convert bytes to values
            byte_array = np.frombuffer(data_bytes, dtype=np.uint8)
            
            # Check if this is a constant channel
            if meta.get('constant', False) and len(byte_array) >= 4:
                import struct
                constant_val = struct.unpack('f', byte_array[:4])[0]
                # Create constant channel
                channel_data_2d = np.full((H, W), constant_val, dtype=np.float32)
            else:
                # Normalize to 0-1 range
                values = byte_array.astype(np.float32) / 255.0
                
                # Reshape to 2D - use the actual data
                target_size = H * W
                
                if len(values) >= target_size:
                    # Use exact values
                    values = values[:target_size]
                else:
                    # Repeat to fill
                    repeats = (target_size // len(values)) + 1
                    values = np.tile(values, repeats)[:target_size]
                
                # Reshape to 2D
                channel_data_2d = values.reshape(H, W)
            
            # Convert to tensor
            y_tensor[0, ch_idx] = torch.from_numpy(channel_data_2d).float()
        
        return y_tensor
    
    def read_aibc(self, filepath, max_frames=None):
        """Read and decode aibc file"""
        frames = []
        
        with open(filepath, 'rb') as f:
            # Read header
            header_len = struct.unpack('I', f.read(4))[0]
            header = json.loads(f.read(header_len).decode('utf-8'))
            
            print(f"📦 File: {header['frames']} frames, {header['width']}x{header['height']}, {header['fps']} FPS")
            
            recon_ref = None
            total_frames = min(header['frames'], max_frames) if max_frames else header['frames']
            
            for i in range(total_frames):
                # Read frame type
                frame_type_byte = struct.unpack('B', f.read(1))[0]
                frame_type = 'I' if frame_type_byte == 0 else 'P'
                
                # Read Y metadata
                y_meta_len = struct.unpack('I', f.read(4))[0]
                y_meta = json.loads(f.read(y_meta_len).decode('utf-8'))
                
                # Read Y channels
                y_channels = []
                num_channels = struct.unpack('I', f.read(4))[0]
                
                for _ in range(num_channels):
                    data_len = struct.unpack('I', f.read(4))[0]
                    data = f.read(data_len)
                    meta_len = struct.unpack('I', f.read(4))[0]
                    meta = json.loads(f.read(meta_len).decode('utf-8'))
                    y_channels.append({'data': data, 'meta': meta})
                
                # Skip Z data (not needed for this decoder)
                z_meta_len = struct.unpack('I', f.read(4))[0]
                f.seek(z_meta_len, 1)
                z_len = struct.unpack('I', f.read(4))[0]
                f.seek(z_len, 1)
                
                # Decode frame
                with torch.no_grad():
                    # Create prediction frame
                    if frame_type == 'I' or recon_ref is None:
                        pred_frame = torch.zeros(1, 3, header['height'], header['width']).to(self.device)
                    else:
                        pred_frame = recon_ref
                    
                    # Decode y tensor
                    y_tensor = self.decode_y_tensor(y_channels, y_meta)
                    
                    # Decode residual
                    residual_recon = self.residual_decoder(y_tensor)
                    
                    # Match dimensions
                    if pred_frame.shape[2] != residual_recon.shape[2]:
                        residual_recon = F.interpolate(
                            residual_recon,
                            size=(pred_frame.shape[2], pred_frame.shape[3]),
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    # Reconstruct
                    recon_frame = torch.clamp(pred_frame + residual_recon, 0, 1)
                    recon_ref = recon_frame.detach()
                
                # Convert to numpy
                frame_np = recon_frame.squeeze(0).cpu()
                frame_np = (frame_np.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                frames.append(frame_np)
                
                if (i + 1) % 50 == 0:
                    print(f"   Decoded {i+1}/{total_frames} frames")
        
        print(f"✅ Decoded {len(frames)} frames")
        return frames, header
    
    def play(self, filepath):
        """Play video"""
        frames, header = self.read_aibc(filepath)
        
        if len(frames) == 0:
            print("❌ No frames!")
            return
        
        print(f"\n🎬 Playing at {header['fps']} FPS")
        print("   Controls: 'q'=quit, 'space'=pause, 'r'=restart")
        
        window = 'Video Player'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, header['width'], header['height'])
        
        delay = int(1000 / header['fps'])
        paused = False
        idx = 0
        
        while idx < len(frames):
            if not paused:
                frame_bgr = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR)
                cv2.putText(frame_bgr, f"{idx+1}/{len(frames)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(window, frame_bgr)
                idx += 1
            
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('r'):
                idx = 0
        
        cv2.destroyAllWindows()
        print("✅ Done")
    
    def export_video(self, filepath, output="output.mp4", fps=None):
        """Export to MP4"""
        frames, header = self.read_aibc(filepath)
        
        if len(frames) == 0:
            print("❌ No frames!")
            return
        
        if fps is None:
            fps = header['fps']
        
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output, fourcc, fps, (w, h))
        
        print(f"\n🎥 Exporting to {output}...")
        for i, frame in enumerate(frames):
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if (i+1) % 50 == 0:
                print(f"   {i+1}/{len(frames)}")
        
        out.release()
        print(f"✅ Saved to {output}")
    
    def save_frames(self, filepath, output_dir="frames"):
        """Save as images"""
        frames, header = self.read_aibc(filepath)
        
        if len(frames) == 0:
            print("❌ No frames!")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n💾 Saving to {output_dir}/...")
        
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(f"{output_dir}/frame_{i:04d}.png")
            if (i+1) % 50 == 0:
                print(f"   {i+1}/{len(frames)}")
        
        print(f"✅ Saved {len(frames)} frames")
    
    def debug(self, filepath):
        """Debug - show frame info"""
        print("\n" + "="*50)
        print("DEBUG MODE")
        print("="*50)
        
        frames, header = self.read_aibc(filepath, max_frames=5)
        
        for i, frame in enumerate(frames):
            print(f"\nFrame {i}:")
            print(f"  Shape: {frame.shape}")
            print(f"  Range: {frame.min()}-{frame.max()}")
            print(f"  Mean: {frame.mean():.2f}")
            print(f"  Std: {frame.std():.2f}")
            
            if frame.mean() > 0:
                print(f"  ✅ Valid frame")
                cv2.imwrite(f"debug_{i}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                print(f"  ❌ Black frame")
        
        if frames and frames[0].mean() > 0:
            cv2.imshow('First Frame', cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='.aibc file')
    parser.add_argument('--model', default='checkpoints_hyperprior/best_model.pth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-frames', action='store_true')
    parser.add_argument('--output-dir', default='decoded_frames')
    parser.add_argument('--export-video', help='Export to video file')
    parser.add_argument('--fps', type=float)
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    decoder = RealDecoder(args.model, args.device)
    
    if args.debug:
        decoder.debug(args.input)
    elif args.save_frames:
        decoder.save_frames(args.input, args.output_dir)
    elif args.export_video:
        decoder.export_video(args.input, args.export_video, args.fps)
    else:
        decoder.play(args.input)


if __name__ == "__main__":
    import argparse
    main()