# check_frames.py
import os
import sys
from PIL import Image

def check_frames(frame_dir):
    """Check if frames are ready for OpenDVC"""
    
    print("="*50)
    print("OpenDVC Frame Checker")
    print("="*50)
    
    if not os.path.exists(frame_dir):
        print(f"❌ Directory not found: {frame_dir}")
        return False
    
    # Get all PNG files
    frames = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith('.png')])
    
    if not frames:
        print(f"❌ No PNG files found in {frame_dir}")
        return False
    
    print(f"\n✅ Found {len(frames)} PNG frames")
    print(f"First frame: {frames[0]}")
    print(f"Last frame: {frames[-1]}")
    
    # Check first frame
    sample_path = os.path.join(frame_dir, frames[0])
    try:
        img = Image.open(sample_path)
        width, height = img.size
        mode = img.mode
        
        print(f"\nSample frame info:")
        print(f"  Dimensions: {width}x{height}")
        print(f"  Mode: {mode}")
        print(f"  Format: {img.format}")
        
        # Check if dimensions are multiples of 16
        if width % 16 == 0 and height % 16 == 0:
            print(f"  ✅ Dimensions are multiples of 16 (good for OpenDVC)")
        else:
            new_width = width - (width % 16)
            new_height = height - (height % 16)
            print(f"  ⚠️  Dimensions are NOT multiples of 16")
            print(f"     OpenDVC will crop to: {new_width}x{new_height}")
        
        # Check color mode
        if mode == 'RGB':
            print(f"  ✅ RGB color mode (good)")
        else:
            print(f"  ⚠️  Mode is {mode}, should be RGB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading sample frame: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        frame_dir = sys.argv[1]
    else:
        frame_dir = input("Enter path to your frames directory: ").strip()
    
    if check_frames(frame_dir):
        print("\n✅ Your frames are ready for OpenDVC!")
        print("\nNext step: Compress them with:")
        print(f'python scripts\\test.py --command encode --input_dir "{frame_dir}" --output_dir compressed --model_path ""')
    else:
        print("\n❌ Please fix the issues above before running OpenDVC")