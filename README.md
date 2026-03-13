# OpenDVC -- An open source PyTorch implementation of the DVC Video Compression Method

A PyTorch reimplementation of the paper:

Lu, Guo, et al. "DVC: An end-to-end deep video compression framework." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2019.

The original DVC method is only optimized for PSNR. This implementation provides the PSNR-optimized model and also the MS-SSIM-optimized model, denoted as OpenDVC (PSNR) and OpenDVC (MS-SSIM).

If this open source code is helpful for your research, especially if you compare with the MS-SSIM model of OpenDVC in your paper, please cite our technical report:

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

Hierarchical Learned Video Compression (HLVC) (CVPR 2020) [Paper] [Codes]
Recurrent Learned Video Compression (RLVC) (IEEE J-STSP 2021) [Paper] [Codes]
Perceptual Learned Video Compression (PLVC) (IJCAI 2022) [Paper] [Codes]
Advanced Learned Video Compression (ALVC) (IEEE T-CSVT 2022) [Paper] [Codes]


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

OpenDVC-PyTorch/
  models/
    image_compression.py      Analysis/Synthesis transforms with GDN
    motion_estimation.py       Optical flow estimation network
    motion_compensation.py     Motion compensation with refinement
    residual_coding.py         Residual coding network
  utils/
    metrics.py                 SSIM, MS-SSIM, PSNR metrics
    data_loader.py             Vimeo90k dataset loader
  scripts/
    train.py                    Training script
    test.py                     Testing/encoding script
  pretrained/                   Pre-trained models (download separately)
  README.md


QUICK START

Testing with Pre-trained Models

1. Download pre-trained models from the official OpenDVC repository and place them in the pretrained/ directory.

2. Prepare your test video as PNG frames (ensure dimensions are multiples of 16).

3. Run the encoder:

For PSNR-optimized model:
python scripts/test.py --command encode --input_dir /path/to/png_frames --output_dir compressed_output --model_path pretrained/opendvc_psnr_l1024.pth --mode PSNR --lambda_param 1024 --gop 10

For MS-SSIM-optimized model:
python scripts/test.py --command encode --input_dir /path/to/png_frames --output_dir compressed_output --model_path pretrained/opendvc_msssim_l32.pth --mode MS-SSIM --lambda_param 32 --gop 10

Decoding

python scripts/test.py --command decode --bitstream_dir compressed_output --output_dir decoded_frames --model_path pretrained/opendvc_psnr_l1024.pth --height 240 --width 416

Evaluating Quality

python scripts/test.py --command evaluate --original_dir /path/to/original_frames --reconstructed_dir decoded_frames


KEY PARAMETERS

Parameter: --mode
Description: Optimization mode
Values: PSNR or MS-SSIM

Parameter: --lambda_param
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

1. Download the Vimeo90k dataset (82GB) from: http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip

2. Generate the folder list:

from utils.data_loader import find_folders, create_folder_list
create_folder_list('/path/to/vimeo90k/vimeo_septuplet/sequences/', 'folder.npy')

3. Pre-compress I-frames for training:

For PSNR models: Use BPG 444 with QP values matching lambda:
  Lambda 256 → QP 37
  Lambda 512 → QP 32
  Lambda 1024 → QP 27
  Lambda 2048 → QP 22

Example:
bpgenc -f 444 -m 9 im1.png -o im1_QP27.bpg -q 27
bpgdec im1_QP27.bpg -o im1_bpg444_QP27.png

For MS-SSIM models: Use Lee et al.'s CA model with quality levels:
  Lambda 8 → Quality level 2
  Lambda 16 → Quality level 3
  Lambda 32 → Quality level 5
  Lambda 64 → Quality level 7

Example:
python path_to_CA_model/encode.py --model_type 1 --input_path im1.png --compressed_file_path im1_level5.bin --quality_level 5
python path_to_CA_model/decode.py --compressed_file_path im1_level5.bin --recon_path im1_level5_ssim.png

4. Download pre-trained optical flow models and place in motion_flow/ directory.

Training PSNR Models

python scripts/train.py --mode PSNR --lambda_param 1024 --data_root /path/to/vimeo90k --batch_size 8

Training MS-SSIM Models (fine-tuned from PSNR models)

python scripts/train.py --mode MS-SSIM --lambda_param 32 --data_root /path/to/vimeo90k --batch_size 8


PERFORMANCE

As shown in the original OpenDVC paper, our OpenDVC (PSNR) model achieves comparable PSNR performance with the reported results in Lu et al., DVC (PSNR optimized), and our OpenDVC (MS-SSIM) model significantly outperforms DVC in terms of MS-SSIM.

Model: Original DVC
PSNR Performance: Baseline
MS-SSIM Performance: Baseline

Model: OpenDVC (PSNR)
PSNR Performance: Comparable to DVC
MS-SSIM Performance: Comparable to DVC

Model: OpenDVC (MS-SSIM)
PSNR Performance: Slightly lower than PSNR model
MS-SSIM Performance: Significantly better than DVC


IMPORTANT NOTES

The code currently only supports frames with height and width as multiples of 16.

For YUV videos, first convert to PNG frames using ffmpeg:

ffmpeg -pix_fmt yuv420p -s WidthxHeight -i Name.yuv -vframes Frame path_to_PNG/f%03d.png

Ensure frames are cropped to multiples of 16:

ffmpeg -pix_fmt yuv420p -s 1920x1080 -i Name.yuv -vframes Frame -filter:v "crop=1920:1072:0:0" path_to_PNG/f%03d.png

The provided BasketballPass sequence (first 100 frames) can be used as a test demo.


CITATION

If you use this code for your research, please cite:

@article{yang2020opendvc,
  title={Open{DVC}: An Open Source Implementation of the {DVC} Video Compression Method},
  author={Yang, Ren and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint arXiv:2006.15862},
  year={2020}
}


LICENSE

This project is licensed under the MIT License - see the LICENSE file for details.


ACKNOWLEDGMENTS

This implementation is based on the original TensorFlow OpenDVC code by Ren Yang. We thank the authors for making their work publicly available.