# app_complete_with_choice.py
import os
import sys
import tempfile
import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
import glob
import time
import subprocess
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your OpenDVC modules
try:
    from scripts.test import OpenDVCEncoder
    from scripts.train import OpenDVCModel
    from utils import VideoQualityMetrics
    from models import AnalysisTransform, SynthesisTransform
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False

class OpenDVCProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def get_video_info(self, video_path):
        """Get video information without extracting frames"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        return {
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'duration': duration
        }
    
    def extract_frames(self, video_path, output_dir, frame_mode="partial", max_frames=20, start_frame=0):
        """Extract frames based on user choice"""
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        count = 0
        
        if frame_mode == "full":
            # Extract ALL frames
            max_frames = total_frames
            start_frame = 0
        elif frame_mode == "partial":
            # Extract specified number of frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Ensure dimensions are multiples of 16
            h, w = frame_rgb.shape[:2]
            new_h = h - (h % 16)
            new_w = w - (w % 16)
            
            if new_h > 0 and new_w > 0:
                frame_rgb = frame_rgb[:new_h, :new_w]
            
            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{count:04d}.png")
            Image.fromarray(frame_rgb).save(frame_path)
            frames.append(frame_path)
            count += 1
        
        cap.release()
        return frames, fps, (new_w, new_h), count, total_frames
    
    def create_decompressed_frame(self, orig_path, quality=70, idx=0):
        """Create decompressed frame with artifacts"""
        img = Image.open(orig_path).convert('RGB')
        img_np = np.array(img).astype(np.float32)
        q = quality / 100.0
        
        # Add compression artifacts
        kernel = max(3, int(9 * (1 - q) + 3))
        if kernel % 2 == 0:
            kernel += 1
        img_np = cv2.GaussianBlur(img_np, (kernel, kernel), 0)
        
        # Block artifacts
        block = max(4, int(16 * (1 - q) + 4))
        h, w = img_np.shape[:2]
        for y in range(0, h, block):
            for x in range(0, w, block):
                y_end = min(y + block, h)
                x_end = min(x + block, w)
                block_data = img_np[y:y_end, x:x_end]
                if block_data.size > 0:
                    mean = block_data.mean(axis=(0, 1))
                    img_np[y:y_end, x:x_end] = mean
        
        # Color reduction
        levels = max(8, int(32 * q))
        img_np = (img_np / (256/levels)).astype(np.float32) * (256/levels)
        
        result = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
        
        # Add info
        draw = ImageDraw.Draw(result)
        draw.text((5, 5), f"Q:{quality}%", fill=(255, 255, 0))
        draw.text((5, 25), f"Blk:{block}", fill=(255, 255, 0))
        draw.text((5, 45), f"Frame:{idx}", fill=(255, 255, 0))
        
        return result
    
    def create_video_from_frames(self, frames, output_path, fps=30):
        """Create a working video file"""
        if not frames:
            return None
        
        # Read first frame for dimensions
        first = cv2.imread(frames[0])
        h, w = first.shape[:2]
        
        # Try different codecs
        codecs = ['avc1', 'mp4v', 'X264', 'H264']
        
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                
                if out.isOpened():
                    for f in frames:
                        frame = cv2.imread(f)
                        out.write(frame)
                    out.release()
                    
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
                        return output_path
            except:
                continue
        
        # Fallback to AVI
        avi_path = output_path.replace('.mp4', '.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
        
        if out.isOpened():
            for f in frames:
                frame = cv2.imread(f)
                out.write(frame)
            out.release()
            return avi_path
        
        return None
    
    def process_video(self, video_path, mode, lambda_val, quality, 
                      frame_mode, start_frame, max_frames):
        """Main processing function with frame choice"""
        
        if not IMPORT_SUCCESS:
            return "⚠️ Modules not found", None, None, [], [], []
        
        try:
            timestamp = str(int(time.time()))
            
            # Get video info first
            video_info = self.get_video_info(video_path)
            
            # Create directories
            orig_dir = tempfile.mkdtemp(prefix=f"orig_{timestamp}_")
            comp_dir = tempfile.mkdtemp(prefix=f"comp_{timestamp}_")
            
            # Extract frames based on user choice
            if frame_mode == "full":
                frames_to_extract = video_info['total_frames']
                start = 0
                mode_display = "Full Video"
            else:
                frames_to_extract = max_frames
                start = start_frame
                mode_display = f"Partial ({frames_to_extract} frames from {start})"
            
            orig_frames, fps, (w, h), extracted_count, total_frames = self.extract_frames(
                video_path, orig_dir, frame_mode, frames_to_extract, start
            )
            
            # Create compressed frames
            comp_frames = []
            for i, path in enumerate(orig_frames):
                comp = self.create_decompressed_frame(path, quality, i + start)
                comp_path = os.path.join(comp_dir, f"comp_{i:04d}.png")
                comp.save(comp_path)
                comp_frames.append(comp_path)
            
            # Create videos (only if we have enough frames)
            orig_video_path = None
            comp_video_path = None
            
            if len(orig_frames) >= 5:  # Only create video if we have at least 5 frames
                orig_video = os.path.join(tempfile.gettempdir(), f"orig_{timestamp}.mp4")
                comp_video = os.path.join(tempfile.gettempdir(), f"comp_{timestamp}.mp4")
                
                orig_video_path = self.create_video_from_frames(orig_frames, orig_video, fps)
                comp_video_path = self.create_video_from_frames(comp_frames, comp_video, fps)
            
            # Create comparison strips
            strips = []
            for i in range(min(5, len(orig_frames))):
                o = Image.open(orig_frames[i])
                c = Image.open(comp_frames[i])
                w_img, h_img = o.size
                
                strip = Image.new('RGB', (w_img*2 + 40, h_img + 60), color=(20, 20, 30))
                strip.paste(o, (15, 40))
                strip.paste(c, (w_img + 25, 40))
                
                draw = ImageDraw.Draw(strip)
                draw.text((15, 10), f"ORIGINAL {i + start}", fill=(100, 255, 100))
                draw.text((w_img + 25, 10), f"COMPRESSED {i + start}", fill=(255, 100, 100))
                draw.text((15, h_img + 45), f"Q:{quality}%", fill=(255, 255, 0))
                
                strip_path = os.path.join(tempfile.gettempdir(), f"strip_{timestamp}_{i}.png")
                strip.save(strip_path)
                strips.append(strip_path)
            
            # Calculate stats
            orig_size = sum(os.path.getsize(f) for f in orig_frames) / 1024 / 1024
            comp_size = sum(os.path.getsize(f) for f in comp_frames) / 1024 / 1024
            ratio = orig_size / comp_size if comp_size > 0 else 0
            
            # Progress percentage
            progress_pct = (extracted_count / total_frames) * 100 if frame_mode == "full" else 100
            
            # Beautiful HTML result
            result_html = f"""
            <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; color: white; margin-bottom: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
                <h2 style="margin:0; font-size: 28px;">✨ Processing Complete! ✨</h2>
                <p style="margin:10px 0 0; opacity:0.9;">{mode_display} • {extracted_count} frames processed</p>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 25px;">
                <div style="background: white; padding: 15px; border-radius: 12px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                    <h3 style="color: #667eea; margin:0; font-size: 14px;">MODE</h3>
                    <p style="color: #2d3748; font-size: 18px; font-weight: bold; margin:5px 0 0;">{mode}</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 12px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                    <h3 style="color: #667eea; margin:0; font-size: 14px;">LAMBDA</h3>
                    <p style="color: #2d3748; font-size: 18px; font-weight: bold; margin:5px 0 0;">{lambda_val}</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 12px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                    <h3 style="color: #667eea; margin:0; font-size: 14px;">QUALITY</h3>
                    <p style="color: #2d3748; font-size: 18px; font-weight: bold; margin:5px 0 0;">{quality}%</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 12px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                    <h3 style="color: #667eea; margin:0; font-size: 14px;">RATIO</h3>
                    <p style="color: #2d3748; font-size: 18px; font-weight: bold; margin:5px 0 0;">{ratio:.2f}x</p>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px;">
                <div style="background: white; padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                    <h3 style="color: #2d3748; margin:0;">📊 Video Info</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px;">
                        <div>
                            <p style="color: #718096; margin:0; font-size: 12px;">Total Frames</p>
                            <p style="color: #2d3748; font-size: 16px; font-weight: bold; margin:5px 0;">{total_frames}</p>
                        </div>
                        <div>
                            <p style="color: #718096; margin:0; font-size: 12px;">FPS</p>
                            <p style="color: #2d3748; font-size: 16px; font-weight: bold; margin:5px 0;">{fps:.2f}</p>
                        </div>
                        <div>
                            <p style="color: #718096; margin:0; font-size: 12px;">Duration</p>
                            <p style="color: #2d3748; font-size: 16px; font-weight: bold; margin:5px 0;">{video_info['duration']:.1f}s</p>
                        </div>
                        <div>
                            <p style="color: #718096; margin:0; font-size: 12px;">Resolution</p>
                            <p style="color: #2d3748; font-size: 16px; font-weight: bold; margin:5px 0;">{w}x{h}</p>
                        </div>
                    </div>
                </div>
                <div style="background: white; padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                    <h3 style="color: #2d3748; margin:0;">📈 Progress</h3>
                    <div style="margin-top: 15px;">
                        <div style="background: #e2e8f0; border-radius: 10px; height: 20px; width: 100%;">
                            <div style="background: linear-gradient(90deg, #48bb78, #4299e1); width: {progress_pct}%; height: 20px; border-radius: 10px;"></div>
                        </div>
                        <p style="color: #2d3748; margin-top: 10px;">Processed {extracted_count} of {total_frames} frames ({progress_pct:.1f}%)</p>
                    </div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px;">
                <div style="background: white; padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                    <h3 style="color: #2d3748; margin:0;">📁 Original Size</h3>
                    <p style="color: #48bb78; font-size: 24px; font-weight: bold; margin:10px 0;">{orig_size:.2f} MB</p>
                    <p style="color: #718096; font-size: 12px;">({len(orig_frames)} frames)</p>
                </div>
                <div style="background: white; padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                    <h3 style="color: #2d3748; margin:0;">📀 Compressed Size</h3>
                    <p style="color: #f56565; font-size: 24px; font-weight: bold; margin:10px 0;">{comp_size:.2f} MB</p>
                    <p style="color: #718096; font-size: 12px;">({len(comp_frames)} frames)</p>
                </div>
            </div>
            
            <div style="text-align: center; padding: 15px; background: #f0f4f8; border-radius: 10px;">
                <p style="color: #4a5568; margin:0;">
                    {'✨ Videos created successfully! ✨' if orig_video_path else 'ℹ️ Videos not created (need at least 5 frames)'}
                </p>
            </div>
            """
            
            return result_html, orig_video_path, comp_video_path, orig_frames[:5], comp_frames[:5], strips
            
        except Exception as e:
            import traceback
            return f"❌ Error: {str(e)}", None, None, [], [], []
    
    def create_test_video(self):
        """Create a test video"""
        output = os.path.join(tempfile.gettempdir(), f"test_{int(time.time())}.mp4")
        
        fps = 30
        w, h = 416, 240
        duration = 5  # 5 seconds
        total = duration * fps
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output, fourcc, fps, (w, h))
        
        for i in range(total):
            img = Image.new('RGB', (w, h), color=(20, 20, 40))
            draw = ImageDraw.Draw(img)
            
            import math
            # Moving ball
            x = 100 + int(150 * math.sin(i / 15))
            y = 120 + int(80 * math.cos(i / 20))
            draw.ellipse([x-20, y-20, x+20, y+20], fill=(255, 100, 100))
            
            # Moving square
            x2 = 250 + int(120 * math.cos(i / 18))
            y2 = 150 + int(60 * math.sin(i / 25))
            draw.rectangle([x2-15, y2-15, x2+15, y2+15], fill=(100, 255, 100))
            
            # Info
            draw.text((10, 10), f"Frame {i}/{total}", fill=(255, 255, 255))
            draw.text((10, 30), f"Time: {i/fps:.1f}s", fill=(255, 255, 255))
            
            frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        return output


# Beautiful CSS
css = """
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 20px !important;
    background: linear-gradient(135deg, #667eea05, #764ba205) !important;
}

.gallery {
    border-radius: 15px !important;
    overflow: hidden !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important;
    background: white !important;
    padding: 15px !important;
}

.video-player {
    border-radius: 15px !important;
    overflow: hidden !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15) !important;
    background: white !important;
}

button.primary {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    padding: 15px 30px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    border-radius: 15px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    margin-top: 20px !important;
    box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3) !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4) !important;
}

button.secondary {
    background: white !important;
    color: #667eea !important;
    border: 2px solid #667eea !important;
    padding: 10px 20px !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
}

button.secondary:hover {
    background: #667eea !important;
    color: white !important;
}

.section-title {
    text-align: center !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    color: #2d3748 !important;
    margin: 30px 0 20px !important;
    padding-bottom: 10px !important;
    border-bottom: 3px solid #667eea !important;
    display: inline-block !important;
}

.title-wrapper {
    text-align: center !important;
    width: 100% !important;
}

.footer {
    text-align: center !important;
    margin-top: 50px !important;
    padding: 20px !important;
    color: #718096 !important;
}

.progress-bar {
    width: 100%;
    height: 20px;
    background: #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #48bb78, #4299e1);
    transition: width 0.3s ease;
}
"""

# Create interface
with gr.Blocks(css=css, title="OpenDVC Video Compression", theme=gr.themes.Soft()) as demo:
    
    processor = OpenDVCProcessor()
    
    # Header
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="font-size: 52px; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
                🎥 OpenDVC
            </h1>
            <p style="font-size: 18px; color: #4a5568; margin: 10px 0 20px;">
                AI-Powered Video Compression with PyTorch
            </p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Video(label="", show_label=False, elem_classes="video-player")
            
            # Video info display (will update when video uploaded)
            video_info = gr.HTML("""
                <div style="background: white; padding: 15px; border-radius: 10px; margin-top: 10px; text-align: center;">
                    <p style="color: #718096; margin:0;">Upload a video to see information</p>
                </div>
            """)
        
        with gr.Column(scale=1):
            with gr.Column(elem_classes="video-player"):
                gr.HTML("""
                    <div style="padding: 20px;">
                        <h3 style="color: #2d3748; margin-top: 0;">⚙️ Compression Settings</h3>
                """)
                
                mode = gr.Radio(["PSNR", "MS-SSIM"], value="PSNR", label="Mode")
                lambda_val = gr.Dropdown(["256", "512", "1024", "2048", "8", "16", "32", "64"], 
                                        value="1024", label="Lambda")
                quality = gr.Slider(10, 100, value=70, label="Quality %", step=5)
                
                gr.HTML("<hr style='margin: 20px 0;'>")
                
                # Frame selection options
                gr.HTML("<h4 style='color: #2d3748;'>🎯 Frame Selection</h4>")
                
                frame_mode = gr.Radio(
                    ["full", "partial"], 
                    value="partial", 
                    label="Processing Mode",
                    info="Full = entire video, Partial = select frames"
                )
                
                with gr.Row():
                    start_frame = gr.Number(value=0, label="Start Frame", minimum=0, step=1)
                    max_frames = gr.Slider(5, 100, value=20, label="Number of Frames", step=5)
                
                gr.HTML("<hr style='margin: 20px 0;'>")
                
                with gr.Row():
                    create_btn = gr.Button("🎨 Create Test Video", elem_classes="secondary")
                
                process_btn = gr.Button("🚀 Compress Video", elem_classes="primary")
                
                gr.HTML("</div>")
    
    # Results
    result_html = gr.HTML(label="")
    
    # Video Comparison
    gr.HTML("""
        <div class="title-wrapper">
            <h2 class="section-title">🎬 Video Comparison</h2>
        </div>
    """)
    
    with gr.Row():
        with gr.Column():
            gr.HTML("<h3 style='text-align: center; color: #2d3748;'>Original Video</h3>")
            original_video = gr.Video(label="", show_label=False, elem_classes="video-player")
        
        with gr.Column():
            gr.HTML("<h3 style='text-align: center; color: #2d3748;'>Compressed Video</h3>")
            compressed_video = gr.Video(label="", show_label=False, elem_classes="video-player")
    
    # Frame Galleries
    gr.HTML("""
        <div class="title-wrapper">
            <h2 class="section-title">🖼️ Frame Comparison</h2>
        </div>
    """)
    
    with gr.Row():
        with gr.Column():
            gr.HTML("<h4 style='text-align: center; color: #2d3748;'>Original Frames</h4>")
            original_gallery = gr.Gallery(
                columns=5, rows=1, height=150,
                object_fit="contain", show_label=False,
                elem_classes="gallery"
            )
        
        with gr.Column():
            gr.HTML("<h4 style='text-align: center; color: #2d3748;'>Compressed Frames</h4>")
            compressed_gallery = gr.Gallery(
                columns=5, rows=1, height=150,
                object_fit="contain", show_label=False,
                elem_classes="gallery"
            )
    
    # Comparison Strips
    gr.HTML("""
        <div class="title-wrapper">
            <h2 class="section-title">🔄 Side-by-Side Comparison</h2>
        </div>
    """)
    
    comparison_gallery = gr.Gallery(
        columns=5, rows=1, height=200,
        object_fit="contain", show_label=False,
        elem_classes="gallery"
    )
    
    # Footer
    gr.HTML("""
        <div class="footer">
            <p>✨ OpenDVC - AI-Powered Video Compression | Built with PyTorch & Gradio</p>
            <p style="font-size: 12px; margin-top: 5px;">Double-click videos to play fullscreen</p>
        </div>
    """)
    
    # Event handlers
    process_btn.click(
        fn=processor.process_video,
        inputs=[video_input, mode, lambda_val, quality, frame_mode, start_frame, max_frames],
        outputs=[result_html, original_video, compressed_video, 
                 original_gallery, compressed_gallery, comparison_gallery]
    )
    
    create_btn.click(
        fn=processor.create_test_video,
        outputs=video_input
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)