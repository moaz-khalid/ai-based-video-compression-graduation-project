# create_test_frames.py
from PIL import Image
import os

os.makedirs("test_frames", exist_ok=True)

for i in range(10):
    img = Image.new('RGB', (416, 240), color=(i*25, i*25, i*25))
    img.save(f"test_frames/frame_{i:03d}.png")
    
print("✅ Created 10 test frames in 'test_frames' folder")