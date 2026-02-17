# prepare_video_for_opendvc.py
import os
import subprocess
import sys
from PIL import Image

def prepare_video_for_opendvc(video_path, output_dir="opendvc_frames", target_fps=30, max_frames=None):
    """
    Prepare a video for OpenDVC compression
    """
    
    print("="*60)
    print("OpenDVC Video Preparation Tool")
    print("="*60)
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build ffmpeg command
    output_pattern = os.path.join(output_dir, 'frame_%06d.png')
    
    cmd = ['ffmpeg', '-i', video_path]
    
    # Add filters
    filters = [f'fps={target_fps}']
    cmd.extend(['-vf', ','.join(filters)])
    
    if max_frames:
        cmd.extend(['-vframes', str(max_frames)])
    
    cmd.append(output_pattern)
    
    print(f"\nRunning: {' '.join(cmd)}")
    print("\nExtracting frames...")
    
    try:
        subprocess.run(cmd, check=True)
        
        # Count extracted frames
        frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
        frame_count = len(frame_files)
        
        print(f"\n✅ Successfully extracted {frame_count} frames!")
        print(f"   Output directory: {output_dir}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg first.")
        return False

if __name__ == "__main__":
    # If no arguments, ask for input
    if len(sys.argv) == 1:
        print("\nOpenDVC Video Preparation")
        print("-" * 40)
        video_path = input("Enter video file path: ").strip()
        output_dir = input("Enter output directory [opendvc_frames]: ").strip() or "opendvc_frames"
        fps_input = input("Target FPS [30]: ").strip()
        target_fps = int(fps_input) if fps_input else 30
        max_input = input("Max frames (press Enter for all): ").strip()
        max_frames = int(max_input) if max_input else None
        
        prepare_video_for_opendvc(video_path, output_dir, target_fps, max_frames)
    else:
        # Use command line arguments
        video_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "opendvc_frames"
        target_fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else None
        
        prepare_video_for_opendvc(video_path, output_dir, target_fps, max_frames)