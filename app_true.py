"""
app_true.py - TRUE AI COMPRESSION - MAXIMUM SPEED + PSNR + TIMING
No H.264 reliance - No GUI changes - Full metrics
"""

import os
import sys
import tempfile
import gradio as gr
from PIL import Image, ImageDraw
import cv2
import time
import subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import json
import struct
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_hyperprior import OpenDVCWithHyperprior

# ============================================================
# ABSOLUTE MAXIMUM GPU OPTIMIZATIONS
# ============================================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.98)
    
    # Use persistent caches
    torch.cuda.memory.empty_cache()


def find_trained_models():
    models = []
    for folder in ['checkpoints_hyperprior', 'checkpoints']:
        path = Path(folder)
        if path.exists():
            for pattern in ["*.pth", "*.pt"]:
                models.extend(path.rglob(pattern))
    models = [m for m in models if '.venv' not in str(m) and m.is_file()]
    models = list(set(models))
    models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return models


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 30, 640, 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames, fps, width, height


def calculate_psnr(original, reconstructed):
    """Calculate PSNR between two numpy arrays"""
    mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


class MaxSpeedAICompressor:
    """MAXIMUM SPEED - Parallel video creation + PSNR + Timing"""
    
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n🚀 Device: {self.device}")
        print(f"⚡ MAX OPTIMIZATION: Full FP16 + Preload + Parallel Encode")
        
        if model_path and Path(model_path).exists() and model_path != "No models found":
            self.model_path = model_path
        else:
            models = find_trained_models()
            self.model_path = str(models[0]) if models else None
        
        self.model = None
        self.has_weights = False
        
        if self.model_path:
            try:
                self.model = OpenDVCWithHyperprior().to(self.device).half()
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                model_dict = self.model.state_dict()
                matched = sum(1 for k in model_dict if k in state_dict and model_dict[k].shape == state_dict[k].shape)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()
                self.has_weights = True
                
                print(f"✅ Loaded {matched}/{len(model_dict)} layers (FP16)")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                self.model = None
        
        self.transform = transforms.ToTensor()
        self._warmup_done = False
        
        # Thread pool for parallel encoding
        self.executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
    
    def _warmup_model(self):
        if self._warmup_done:
            return
        print("⚡ Warming up GPU...")
        dummy = torch.randn(2, 3, 256, 256).half().to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            _ = self.model.compress(dummy[0:1], dummy[0:1])
        self._warmup_done = True
        print("   ✅ Ready")
    
    def encode_frame_parallel(self, args):
        """Parallel frame encoding for video creation"""
        frame, path = args
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        return path
    
    def create_video_fast(self, frames, output_path, fps):
        """Ultra-fast video creation using parallel encoding and FFmpeg pipe"""
        temp_dir = Path(tempfile.gettempdir()) / f"frames_{int(time.time()*1000)}"
        temp_dir.mkdir(exist_ok=True)
        
        # Parallel frame saving
        args = [(frames[i], str(temp_dir / f"frame_{i:06d}.jpg")) for i in range(len(frames))]
        list(self.executor.map(self.encode_frame_parallel, args))
        
        # Use FFmpeg with JPEG input (faster than PNG)
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(temp_dir / "frame_%06d.jpg"),
            "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
            "-tune", "fastdecode", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-threads", str(multiprocessing.cpu_count()),
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        import shutil
        shutil.rmtree(temp_dir)
        return output_path
    
    def process(self, video_path, quality, process_type, max_frames):
        if not video_path:
            return "⚠️ Upload a video", None, None, None, [], [], []
        
        if not self.has_weights:
            return "❌ No trained model!", None, None, None, [], [], []
        
        try:
            self._warmup_model()
            
            total_start = time.time()
            timestamp = str(int(time.time()))
            total_frames, fps, orig_width, orig_height = get_video_info(video_path)
            original_size_mb = Path(video_path).stat().st_size / 1024 / 1024
            
            print(f"\n{'='*70}")
            print(f"🧠 TRUE AI COMPRESSION - MAX SPEED")
            print(f"{'='*70}")
            print(f"📹 {Path(video_path).name}")
            print(f"   Original: {original_size_mb:.2f} MB")
            print(f"   Total frames: {total_frames}")
            print(f"{'='*70}\n")
            
            if process_type == "full":
                frames_to_process = total_frames
                mode_text = f"Full ({total_frames} frames)"
            else:
                frames_to_process = min(max_frames, total_frames)
                mode_text = f"Partial ({frames_to_process} frames)"
            
            gop_size = 10
            pad_width = ((orig_width + 63) // 64) * 64
            pad_height = ((orig_height + 63) // 64) * 64
            
            temp_base = Path(tempfile.gettempdir()) / f"max_speed_{timestamp}"
            temp_base.mkdir(exist_ok=True)
            
            frames_dir = temp_base / "input_frames"
            preview_dir = temp_base / "preview"
            frames_dir.mkdir(exist_ok=True)
            preview_dir.mkdir(exist_ok=True)
            
            # Extract frames
            extract_start = time.time()
            print(f"📤 Extracting {frames_to_process} frames...")
            extract_cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"fps={fps},pad={pad_width}:{pad_height}:(ow-iw)/2:(oh-ih)/2:black",
                "-vframes", str(frames_to_process), "-threads", "4",
                str(frames_dir / "frame_%06d.jpg")
            ]
            subprocess.run(extract_cmd, check=True, capture_output=True, text=True)
            extract_time = time.time() - extract_start
            
            frame_files = sorted(frames_dir.glob("frame_*.jpg"))
            print(f"   ✅ {len(frame_files)} frames in {extract_time:.1f}s\n")
            
            # AI Compression
            print("🧠 AI Compression (MAX GPU)")
            print("="*70)
            
            ai_start = time.time()
            recon_ref = None
            total_bytes = total_y = total_z = 0
            i_frames = p_frames = 0
            
            original_gallery = []
            compressed_gallery = []
            strips = []
            reconstructed_frames = []
            aibc_frames = []
            psnr_values = []
            
            # Pre-load to GPU
            all_tensors = []
            original_frames = []
            for fp in frame_files:
                img = Image.open(fp).convert('RGB')
                original_frames.append(np.array(img))
                all_tensors.append(self.transform(img).unsqueeze(0).half().to(self.device))
            
            for i, (img_tensor, orig_np) in enumerate(zip(all_tensors, original_frames)):
                is_iframe = (i % gop_size == 0)
                
                with torch.no_grad(), torch.cuda.amp.autocast():
                    if is_iframe:
                        compressed = self.model.compress(img_tensor, img_tensor)
                        out_tensor = self.model.decompress(compressed, img_tensor)
                        recon_ref = out_tensor.detach()
                        i_frames += 1
                        ftype = "I"
                    else:
                        compressed = self.model.compress(recon_ref, img_tensor)
                        out_tensor = self.model.decompress(compressed, recon_ref)
                        recon_ref = out_tensor.detach()
                        p_frames += 1
                        ftype = "P"
                
                total_bytes += compressed['total_bytes']
                total_y += compressed['y_bytes']
                total_z += compressed['z_bytes']
                
                aibc_frames.append({
                    'type': ftype, 'y_channels': compressed['y_channels'],
                    'y_meta': compressed['y_meta'], 'z_compressed': compressed['z_compressed'],
                    'z_meta_bytes': compressed['z_meta']
                })
                
                out_np = out_tensor.float().squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                out_np = (out_np * 255).astype(np.uint8)
                
                if pad_width != orig_width:
                    crop_h = (pad_height - orig_height) // 2
                    crop_w = (pad_width - orig_width) // 2
                    out_np = out_np[crop_h:crop_h+orig_height, crop_w:crop_w+orig_width]
                    orig_cropped = orig_np[crop_h:crop_h+orig_height, crop_w:crop_w+orig_width]
                else:
                    orig_cropped = orig_np
                
                reconstructed_frames.append(out_np)
                psnr_values.append(calculate_psnr(orig_cropped, out_np))
                
                if i < 5:
                    orig_path = preview_dir / f"orig_{i}.jpg"
                    Image.fromarray(orig_cropped).save(orig_path)
                    original_gallery.append(str(orig_path))
                    
                    comp_path = preview_dir / f"comp_{i}.jpg"
                    Image.fromarray(out_np).save(comp_path)
                    compressed_gallery.append(str(comp_path))
                    
                    w, h = orig_cropped.shape[1], orig_cropped.shape[0]
                    strip = Image.new('RGB', (w*2 + 40, h + 50), color=(20, 20, 30))
                    strip.paste(Image.fromarray(orig_cropped), (15, 35))
                    strip.paste(Image.fromarray(out_np), (w + 25, 35))
                    draw = ImageDraw.Draw(strip)
                    draw.text((15, 10), "ORIGINAL", fill=(100, 255, 100))
                    draw.text((w+25, 10), "AI COMPRESSED", fill=(255, 100, 100))
                    strip_path = preview_dir / f"strip_{i}.jpg"
                    strip.save(strip_path)
                    strips.append(str(strip_path))
                
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - ai_start
                    print(f"   Frame {i+1:4d} ({ftype}): {compressed['total_bytes']:8d} bytes | {len(frame_files)/elapsed:.1f} FPS")
            
            ai_time = time.time() - ai_start
            avg_psnr = np.mean(psnr_values)
            print("="*70)
            print(f"✅ AI: {len(frame_files)} frames in {ai_time:.1f}s ({len(frame_files)/ai_time:.1f} FPS)")
            print(f"   PSNR: {avg_psnr:.2f} dB | I:{i_frames} P:{p_frames}")
            
            # Save .aibc
            aibc_start = time.time()
            aibc_file = temp_base / f"compressed_{timestamp}.aibc"
            with open(aibc_file, 'wb') as f:
                header = json.dumps({'frames': len(aibc_frames), 'width': orig_width, 'height': orig_height, 'fps': fps}).encode()
                f.write(struct.pack('I', len(header))); f.write(header)
                for fr in aibc_frames:
                    f.write(struct.pack('B', 0 if fr['type']=='I' else 1))
                    ym = json.dumps(fr['y_meta']).encode()
                    f.write(struct.pack('I', len(ym))); f.write(ym)
                    f.write(struct.pack('I', len(fr['y_channels'])))
                    for ch in fr['y_channels']:
                        f.write(struct.pack('I', len(ch['data']))); f.write(ch['data'])
                        f.write(struct.pack('I', len(json.dumps(ch['meta']).encode())))
                        f.write(json.dumps(ch['meta']).encode())
                    f.write(struct.pack('I', len(fr['z_meta_bytes']))); f.write(fr['z_meta_bytes'])
                    f.write(struct.pack('I', len(fr['z_compressed']))); f.write(fr['z_compressed'])
            aibc_time = time.time() - aibc_start
            
            # Video creation
            video_start = time.time()
            print(f"\n🎬 Creating video...")
            output_video_path = temp_base / f"reconstructed_{timestamp}.mp4"
            self.create_video_fast(reconstructed_frames, str(output_video_path), fps)
            video_time = time.time() - video_start
            print(f"   ✅ Video in {video_time:.1f}s")
            
            total_time = time.time() - total_start
            ai_mb = total_bytes / 1024 / 1024
            ratio = original_size_mb / ai_mb if ai_mb > 0 else 1
            saved = original_size_mb - ai_mb
            rate = (1 - ai_mb / original_size_mb) * 100
            total_pixels = frames_to_process * orig_width * orig_height * 3
            bpp = (total_bytes * 8) / total_pixels if total_pixels > 0 else 0
            
            print(f"\n📊 RESULTS:")
            print(f"   Original: {original_size_mb:.2f} MB → AI: {ai_mb:.2f} MB ({ratio:.2f}x)")
            print(f"   PSNR: {avg_psnr:.2f} dB | BPP: {bpp:.4f}")
            print(f"   Extract: {extract_time:.1f}s | AI: {ai_time:.1f}s | Video: {video_time:.1f}s | Total: {total_time:.1f}s")
            
            result_html = f"""
            <div style="padding:25px; background:linear-gradient(135deg,#1a1a2e,#16213e); border-radius:15px; color:white; text-align:center; margin-bottom:20px;">
                <h2 style="color:#00d4ff; margin:0 0 10px 0; font-size:28px;">✅ TRUE AI COMPRESSION COMPLETE</h2>
                <p style="color:#ffffff; font-size:16px; margin:5px 0;">{mode_text} • MAX GPU Optimized</p>
            </div>
            
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:15px; margin-bottom:20px;">
                <div style="background:#1e293b; padding:15px; border-radius:10px; text-align:center;">
                    <p style="color:#94a3b8; font-size:12px;">📹 EXTRACT</p>
                    <p style="color:#60a5fa; font-size:22px; font-weight:bold;">{extract_time:.1f}s</p>
                </div>
                <div style="background:#1e293b; padding:15px; border-radius:10px; text-align:center;">
                    <p style="color:#94a3b8; font-size:12px;">🧠 AI COMPRESSION</p>
                    <p style="color:#34d399; font-size:22px; font-weight:bold;">{ai_time:.1f}s</p>
                </div>
                <div style="background:#1e293b; padding:15px; border-radius:10px; text-align:center;">
                    <p style="color:#94a3b8; font-size:12px;">🎬 TOTAL</p>
                    <p style="color:#fbbf24; font-size:22px; font-weight:bold;">{total_time:.1f}s</p>
                </div>
            </div>
            
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px;">
                <div style="background:#1e293b; padding:25px; border-radius:15px; text-align:center; border:2px solid #3b82f6;">
                    <h3 style="color:#94a3b8; margin:0 0 15px 0;">📁 ORIGINAL</h3>
                    <p style="font-size:48px; font-weight:bold; color:#60a5fa; margin:10px 0;">{original_size_mb:.2f}</p>
                    <p style="color:#94a3b8;">MB</p>
                </div>
                <div style="background:#1e293b; padding:25px; border-radius:15px; text-align:center; border:2px solid #10b981;">
                    <h3 style="color:#94a3b8; margin:0 0 15px 0;">🧠 AI BITSTREAM</h3>
                    <p style="font-size:48px; font-weight:bold; color:#34d399; margin:10px 0;">{ai_mb:.2f}</p>
                    <p style="color:#94a3b8;">MB (Pure AI)</p>
                </div>
            </div>
            
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:10px; margin-bottom:20px;">
                <div style="background:#1e293b; padding:15px; border-radius:10px; text-align:center;">
                    <p style="color:#94a3b8; font-size:12px;">💾 SAVED</p>
                    <p style="color:#fbbf24; font-size:24px; font-weight:bold;">{saved:.2f} MB</p>
                </div>
                <div style="background:#1e293b; padding:15px; border-radius:10px; text-align:center;">
                    <p style="color:#94a3b8; font-size:12px;">📊 RATE</p>
                    <p style="color:#fbbf24; font-size:24px; font-weight:bold;">{rate:.1f}%</p>
                </div>
                <div style="background:#1e293b; padding:15px; border-radius:10px; text-align:center;">
                    <p style="color:#94a3b8; font-size:12px;">📈 BPP</p>
                    <p style="color:#fbbf24; font-size:24px; font-weight:bold;">{bpp:.4f}</p>
                </div>
                <div style="background:#1e293b; padding:15px; border-radius:10px; text-align:center;">
                    <p style="color:#94a3b8; font-size:12px;">🎯 PSNR</p>
                    <p style="color:#fbbf24; font-size:24px; font-weight:bold;">{avg_psnr:.1f} dB</p>
                </div>
            </div>
            
            <div style="background:#0f172a; padding:15px; border-radius:10px; border-left:4px solid #10b981;">
                <p style="color:#e2e8f0; margin:0; font-size:14px;">
                    <strong style="color:#34d399;">✅ {len(frame_files)} frames ({i_frames} I, {p_frames} P) • PSNR: {avg_psnr:.2f} dB • Y: {total_y:,} • Z: {total_z:,}</strong>
                </p>
            </div>
            """
            
            return result_html, video_path, str(output_video_path), str(aibc_file), original_gallery, compressed_gallery, strips
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error: {str(e)}", None, None, None, [], [], []


# Gradio Interface - COMPLETELY UNCHANGED
css = """
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; padding: 20px !important; background: #0f172a !important; }
.video-player { border-radius: 15px !important; overflow: hidden !important; box-shadow: 0 10px 30px rgba(0,0,0,0.5) !important; border: 2px solid #334155 !important; }
.gallery { border-radius: 10px !important; overflow: hidden !important; box-shadow: 0 5px 15px rgba(0,0,0,0.3) !important; }
"""

with gr.Blocks(title="OpenDVC - True AI Compression", theme=gr.themes.Soft(), css=css) as demo:
    
    models = find_trained_models()
    model_choices = [str(m) for m in models] if models else ["No models found - Train first!"]
    
    gr.HTML("""
        <div style="text-align:center; margin-bottom:30px;">
            <h1 style="font-size:52px; background:linear-gradient(135deg,#667eea,#764ba2); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">
                🧠 OpenDVC
            </h1>
            <p style="font-size:18px; color:#94a3b8; margin-top:10px;">TRUE AI Compression • MAX Speed • PSNR Metrics</p>
        </div>
    """)
    
    compressor_state = gr.State()
    
    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Video(label="📹 Upload Video", elem_classes="video-player")
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            
            model_dropdown = gr.Dropdown(
                choices=model_choices, value=model_choices[0] if model_choices else None, label="📦 Trained Model"
            )
            
            quality = gr.Radio(["low", "medium", "high"], value="medium", label="🎯 Quality")
            process_type = gr.Radio(["full", "partial"], value="partial", label="📊 Process Type")
            max_frames = gr.Slider(5, 500, value=20, label="🎬 Max Frames", step=5)
            
            init_btn = gr.Button("🔄 Load Model", variant="secondary", size="sm")
            process_btn = gr.Button("🧠 True AI Compression", variant="primary", size="lg")
    
    init_status = gr.Textbox(label="Status", interactive=False)
    
    def init_compressor(model_path):
        if not model_path or model_path == "No models found - Train first!":
            return None, "❌ No model found"
        compressor = MaxSpeedAICompressor(model_path)
        if compressor.has_weights:
            return compressor, f"✅ Model loaded! {compressor.device} | ⚡ MAX SPEED"
        return None, "❌ Failed"
    
    init_btn.click(fn=init_compressor, inputs=[model_dropdown], outputs=[compressor_state, init_status])
    
    result_html = gr.HTML()
    
    with gr.Row():
        original_video = gr.Video(label="🎬 Original", elem_classes="video-player")
        compressed_video = gr.Video(label="🧠 AI Reconstructed", elem_classes="video-player")
    
    aibc_output = gr.File(label="📦 AI Bitstream (.aibc)", file_types=[".aibc"])
    
    with gr.Row():
        original_gallery = gr.Gallery(columns=5, label="Original Frames")
        compressed_gallery = gr.Gallery(columns=5, label="AI Frames")
    
    comparison_gallery = gr.Gallery(columns=5, label="Comparison")
    
    def process_wrapper(compressor, video, quality, process_type, max_frames):
        if compressor is None:
            return "❌ Load model first!", None, None, None, [], [], []
        return compressor.process(video, quality, process_type, int(max_frames))
    
    process_btn.click(
        fn=process_wrapper,
        inputs=[compressor_state, video_input, quality, process_type, max_frames],
        outputs=[result_html, original_video, compressed_video, aibc_output, original_gallery, compressed_gallery, comparison_gallery]
    )
    
    demo.load(fn=init_compressor, inputs=[model_dropdown], outputs=[compressor_state, init_status])

if __name__ == "__main__":
    demo.launch(share=True, server_name="127.0.0.1", server_port=7860)
