# ABVC -- A PyTorch implementation of the DVC Video Compression Method

ABVC is a PyTorch-based video compression project built from the OpenDVC reference implementation of the paper:

Lu, Guo, et al. "DVC: An end-to-end deep video compression framework." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2019.

The original DVC method is only optimized for PSNR. ABVC provides the PSNR-optimized model and also the MS-SSIM-optimized model, following the OpenDVC reference variants denoted as OpenDVC (PSNR) and OpenDVC (MS-SSIM).

If the OpenDVC reference code is helpful for your research, especially if you compare with the MS-SSIM model of OpenDVC in your paper, please cite the OpenDVC technical report:

@article{yang2020opendvc,
title={Open{DVC}: An Open Source Implementation of the {DVC} Video Compression Method},
author={Yang, Ren and Van Gool, Luc and Timofte, Radu},
journal={arXiv preprint arXiv:2006.15862},
year={2020}
}

If you have any questions or find any bugs, please feel free to contact:

Ren Yang @ ETH Zurich, Switzerland
Email: r.yangchn@gmail.com



RELATED WORKS

Hierarchical Learned Video Compression (HLVC) (CVPR 2020) \[Paper] \[Codes]
Recurrent Learned Video Compression (RLVC) (IEEE J-STSP 2021) \[Paper] \[Codes]
Perceptual Learned Video Compression (PLVC) (IJCAI 2022) \[Paper] \[Codes]
Advanced Learned Video Compression (ALVC) (IEEE T-CSVT 2022) \[Paper] \[Codes]



DEPENDENCIES

Python 3.7+
PyTorch 1.9+
torchvision
numpy
PIL (Pillow)
OpenCV (for video processing)
tqdm (for progress bars)

Optional (for I-frame compression):
BPG encoder (for PSNR models) - Download link: https://bellard.org/bpg/
Context-adaptive image compression model (for MS-SSIM models) - Paper: https://openreview.net/forum?id=rkxa6jC5FX



REPOSITORY STRUCTURE

ABVC/
models/
image\_compression.py      Analysis/Synthesis transforms with GDN
motion\_estimation.py       Optical flow estimation network
motion\_compensation.py     Motion compensation with refinement
residual\_coding.py         Residual coding network
utils/
metrics.py                 SSIM, MS-SSIM, PSNR metrics
data\_loader.py             Vimeo90k dataset loader
scripts/
train.py                    Training script
test.py                     Testing/encoding script
pretrained/                   Pre-trained models (download separately)
README.md
ABVC_SETUP.txt              Environment setup guide



QUICK START

Testing with Pre-trained Models

For full environment setup instructions, see ABVC_SETUP.txt.

1. Download pre-trained models from the official OpenDVC reference repository and place them in the pretrained/ directory.
2. Prepare your test video as PNG frames (ensure dimensions are multiples of 16).
3. Run the encoder:

For PSNR-optimized model:
python scripts/test.py --command encode --input\_dir /path/to/png\_frames --output\_dir compressed\_output --model\_path pretrained/opendvc\_psnr\_l1024.pth --mode PSNR --lambda\_param 1024 --gop 10

For MS-SSIM-optimized model:
python scripts/test.py --command encode --input\_dir /path/to/png\_frames --output\_dir compressed\_output --model\_path pretrained/opendvc\_msssim\_l32.pth --mode MS-SSIM --lambda\_param 32 --gop 10

Decoding

python scripts/test.py --command decode --bitstream\_dir compressed\_output --output\_dir decoded\_frames --model\_path pretrained/opendvc\_psnr\_l1024.pth --height 240 --width 416

Evaluating Quality

python scripts/test.py --command evaluate --original\_dir /path/to/original\_frames --reconstructed\_dir decoded\_frames



KEY PARAMETERS

Parameter: --mode
Description: Optimization mode
Values: PSNR or MS-SSIM

Parameter: --lambda\_param
Description: Rate-distortion trade-off
Values: PSNR: 256,512,1024,2048 | MS-SSIM: 8,16,32,64

Parameter: --gop
Description: Group of Pictures (I-frame interval)
Values: 10 (default)

Parameter: --N
Description: Number of filters in CNN
Values: 128 (do not change)

Parameter: --M
Description: Latent representation channels
Values: 192 (do not change)



TRAINING YOUR OWN MODELS

Data Preparation

1. Download the Vimeo90k dataset (82GB) from: http://data.csail.mit.edu/tofu/dataset/vimeo\_septuplet.zip
2. Generate the folder list:

from utils.data\_loader import find\_folders, create\_folder\_list
create\_folder\_list('/path/to/vimeo90k/vimeo\_septuplet/sequences/', 'folder.npy')

3. Pre-compress I-frames for training:

For PSNR models: Use BPG 444 with QP values matching lambda:
Lambda 256 → QP 37
Lambda 512 → QP 32
Lambda 1024 → QP 27
Lambda 2048 → QP 22

Example:
bpgenc -f 444 -m 9 im1.png -o im1\_QP27.bpg -q 27
bpgdec im1\_QP27.bpg -o im1\_bpg444\_QP27.png

For MS-SSIM models: Use Lee et al.'s CA model with quality levels:
Lambda 8 → Quality level 2
Lambda 16 → Quality level 3
Lambda 32 → Quality level 5
Lambda 64 → Quality level 7

Example:
python path\_to\_CA\_model/encode.py --model\_type 1 --input\_path im1.png --compressed\_file\_path im1\_level5.bin --quality\_level 5
python path\_to\_CA\_model/decode.py --compressed\_file\_path im1\_level5.bin --recon\_path im1\_level5\_ssim.png

4. Download pre-trained optical flow models and place in motion\_flow/ directory.

Training PSNR Models

python scripts/train.py --mode PSNR --lambda\_param 1024 --data\_root /path/to/vimeo90k --batch\_size 8

Training MS-SSIM Models (fine-tuned from PSNR models)

python scripts/train.py --mode MS-SSIM --lambda\_param 32 --data\_root /path/to/vimeo90k --batch\_size 8



PERFORMANCE

As shown in the original OpenDVC paper, the OpenDVC (PSNR) reference model achieves comparable PSNR performance with the reported results in Lu et al., DVC (PSNR optimized), and the OpenDVC (MS-SSIM) reference model significantly outperforms DVC in terms of MS-SSIM.

Model: Original DVC
PSNR Performance: Baseline
MS-SSIM Performance: Baseline

Model: ABVC / OpenDVC reference (PSNR)
PSNR Performance: Comparable to DVC
MS-SSIM Performance: Comparable to DVC

Model: ABVC / OpenDVC reference (MS-SSIM)
PSNR Performance: Slightly lower than PSNR model
MS-SSIM Performance: Significantly better than DVC



IMPORTANT NOTES

The code currently only supports frames with height and width as multiples of 16.

For YUV videos, first convert to PNG frames using ffmpeg:

ffmpeg -pix\_fmt yuv420p -s WidthxHeight -i Name.yuv -vframes Frame path\_to\_PNG/f%03d.png

Ensure frames are cropped to multiples of 16:

ffmpeg -pix\_fmt yuv420p -s 1920x1080 -i Name.yuv -vframes Frame -filter:v "crop=1920:1072:0:0" path\_to\_PNG/f%03d.png

The provided BasketballPass sequence (first 100 frames) can be used as a test demo.



CITATION

If you use the OpenDVC reference implementation for your research, please cite:

@article{yang2020opendvc,
title={Open{DVC}: An Open Source Implementation of the {DVC} Video Compression Method},
author={Yang, Ren and Van Gool, Luc and Timofte, Radu},
journal={arXiv preprint arXiv:2006.15862},
year={2020}
}



LICENSE

This project is licensed under the MIT License - see the LICENSE file for details.



ACKNOWLEDGMENTS

ABVC is based on the original TensorFlow OpenDVC reference code by Ren Yang. We thank the authors for making their work publicly available.

