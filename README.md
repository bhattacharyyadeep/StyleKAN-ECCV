# StyleKAN: Salient Object Detection via Stylized Fusion and KAN Decoding

> **Anonymous ECCV Submission**  
> This repository contains the training and evaluation code for StyleKAN, our image-only saliency model that generalizes to VSOD benchmarks without any video supervision.

---

## 1. Environment Setup

We recommend Python 3.10+ and CUDA 11.8+.

```bash
pyenv install 3.10.13

# create a virtual environment
pyenv virtualenv 3.10.13 stylekan

# activate the environment
pyenv activate stylekan

# install PyTorch compatible with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install other dependencies
pip install -r requirements.txt
```

## 2. Training

We train our model just of DUTS-Training datasets to save the checkpoints you use the DUTS-TE to evaluate and save the best checkpoint.

data_path/
└── DUTS/
    ├── train/
    │   ├── image/
    │   │   ├── 0001.jpg
    │   │   ├── 0002.jpg
    │   │   └── ...
    │   └── mask/
    │       ├── 0001.png
    │       ├── 0002.png
    │       └── ...
