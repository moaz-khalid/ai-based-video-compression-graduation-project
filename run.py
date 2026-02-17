# run.py - Simple test script
import os
import sys

print("="*50)
print("OpenDVC PyTorch Import Test")
print("="*50)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test imports
tests = [
    ("models.image_compression", ["AnalysisTransform", "SynthesisTransform", "GDN"]),
    ("models.motion_estimation", ["MultiScaleMotionEstimation"]),
    ("models.motion_compensation", ["FullMotionCompensation"]),
    ("models.residual_coding", ["ResidualCoder"]),
    ("utils.metrics", ["SSIM", "MS_SSIM", "PSNR", "VideoQualityMetrics"]),
    ("utils.data_loader", ["VimeoDataset"]),
]

for module_name, class_names in tests:
    try:
        module = __import__(module_name, fromlist=class_names)
        print(f"✓ {module_name}")
    except ImportError as e:
        print(f"✗ {module_name}: {e}")

print("\n" + "="*50)