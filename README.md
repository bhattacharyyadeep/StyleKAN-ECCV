# StyleKAN: Salient Object Detection via Stylized Fusion and KAN Decoding

> **Anonymous ECCV Submission**  
> This repository contains the training and evaluation code for StyleKAN, our image-only saliency model that generalizes to VSOD benchmarks without any video supervision.

---

## 1. Environment Setup

We recommend Python 3.10+ and CUDA 11.8+.

```bash
# create and activate a virtual environment (conda example)
conda create -y -n stylekan python=3.10
conda activate stylekan

# (optional) set CUDA if needed, e.g. export CUDA_HOME=/usr/local/cuda
# install PyTorch that matches your CUDA
# see https://pytorch.org for the exact command for your system
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install other requirements
pip install -r requirements.txt
