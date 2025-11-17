# SCAFNet
SCAFNet: Semantic Compensated Adaptive Fusion Network for Remote Sensing Change Detection
https://img.shields.io/badge/Paper-IEEE%2520GRSL-blue
https://img.shields.io/badge/License-MIT-green
https://img.shields.io/badge/Python-3.8%252B-blue
https://img.shields.io/badge/PyTorch-1.9%252B-orange

This is the official implementation of our paper "SCAFNet: A Semantic Compensated Adaptive Fusion Network for Remote Sensing Images Change Detection".

Abstract
Current CNN-Transformer hybrid methods for remote sensing change detection suffer from semantic misalignment and non-adaptive fusion between dual branches, resulting in persistent sensitivity to pseudo-changes. SCAFNet addresses these issues through three innovative components:

Semantic Compensation Module (SCM): Aligns local-global features via cross-attention

CNN-Transformer Feature Adaptive Fusion (CTFAF): Dynamically balances local details and global context

Change Feature Identification Module (CFIM): Enhances true changes while suppressing seasonal variations

Experiments on CDD and WHU-CD datasets demonstrate SCAFNet's superior robustness and accuracy.

Model Architecture
SCAFNet employs an encoder-decoder architecture with three core modules:

text
Input → Siamese Encoder (CNN + Transformer) → SCM → CTFAF → CFIM → Output Change Map
Key Components
SCM (Semantic Compensation Module): Bridges semantic gap through cross-attention with self-correlation masking

CTFAF (CNN-Transformer Feature Adaptive Fusion): Adaptive fusion with learnable parameters for dynamic feature recalibration

CFIM (Change Feature Identification Module): Multi-stage feature interaction for robust change identification

Installation
Requirements
Python 3.8+

PyTorch 1.9+

CUDA 11.0+

bash
# Create conda environment
conda create -n scafnet python=3.8
conda activate scafnet

# Install PyTorch
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install dependencies
pip install -r requirements.txt
Dependencies
txt
# requirements.txt
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
opencv-python>=4.5.0
pillow>=8.3.0
scikit-learn>=0.24.0
tqdm>=4.60.0
tensorboard>=2.5.0
albumentations>=0.5.0
Data Preparation
Download Datasets
CDD Dataset: Download Link

WHU-CD Dataset: Download Link

Directory Structure
Organize your data as follows:

text
data/
├── CDD/
│   ├── train/
│   │   ├── A/          # Time1 images (10,000)
│   │   ├── B/          # Time2 images (10,000)
│   │   └── label/      # Change masks (10,000)
│   ├── val/            # 3,000 images each
│   └── test/           # 3,000 images each
└── WHU-CD/
    ├── train/          # 5,947 images
    ├── val/            # 743 images  
    └── test/           # 744 images
Training
Single GPU Training
bash
# Train on CDD dataset
python train.py --dataset CDD --data_dir ./data/CDD --batch_size 8 --epochs 200 --lr 0.0001 --output_dir ./outputs/cdd

# Train on WHU-CD dataset  
python train.py --dataset WHU-CD --data_dir ./data/WHU-CD --batch_size 8 --epochs 200 --lr 0.0001 --output_dir ./outputs/whu
Multi-GPU Training
bash
# Train with 4 GPUs
python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset CDD --batch_size 32 --epochs 200 --distributed
Training Script
python
# Example training code structure
from models.scafnet import SCAFNet
from data.datasets import get_dataloader
from utils.trainer import Trainer

# Initialize model
model = SCAFNet(
    encoder_channels=[64, 128, 256, 512],
    decoder_channels=[256, 128, 64, 32]
)

# Get data loaders
train_loader = get_dataloader('CDD', 'train', batch_size=8)
val_loader = get_dataloader('CDD', 'val', batch_size=8)

# Train model
trainer = Trainer(model, train_loader, val_loader)
trainer.train(epochs=200)
Evaluation
Test Pre-trained Model
bash
# Evaluate on test set
python test.py --dataset CDD --checkpoint ./checkpoints/scafnet_cdd_best.pth --save_dir ./results/cdd

# Generate visualization
python test.py --dataset WHU-CD --checkpoint ./checkpoints/scafnet_whu_best.pth --visualize
Evaluation Metrics
The code automatically computes:

F1-Score

IoU (Intersection over Union)

Precision

Recall

Overall Accuracy

Pre-trained Models
Download our pre-trained models:

Dataset	F1-Score	Download
CDD	94.2%	scafnet_cdd.pth
WHU-CD	92.8%	scafnet_whu.pth
Results
Quantitative Results
Method	CDD (F1)	WHU-CD (F1)	Params (M)	FLOPs (G)	Inference Time (ms)
SCAFNet	94.2%	92.8%	45.6	128.3	25.4
BIT	91.0%	89.5%	38.2	115.7	22.1
ChangeFormer	90.8%	88.9%	41.3	121.5	23.8
DMINet	89.5%	87.2%	43.1	119.2	24.3
Qualitative Results
https://./assets/visualization.png

Repository Structure
text
SCAFNet/
├── configs/                 # Configuration files
│   ├── train_cdd.yaml
│   └── train_whu.yaml
├── data/                    # Data loaders and preprocessing
│   ├── __init__.py
│   ├── datasets.py
│   └── transforms.py
├── models/                  # Model definitions
│   ├── __init__.py
│   ├── backbone/           # Encoder backbones
│   ├── scm.py             # Semantic Compensation Module
│   ├── ctfaf.py           # Adaptive Fusion Module  
│   ├── cfim.py            # Change Identification Module
│   └── scafnet.py         # Complete SCAFNet
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── losses.py          # Loss functions
│   ├── metrics.py         # Evaluation metrics
│   ├── trainer.py         # Training logic
│   └── visualization.py   # Visualization tools
├── analysis/              # Supplementary analysis
│   ├── complexity.py      # Model complexity analysis
│   └── ablation.py        # Ablation study scripts
├── checkpoints/           # Model checkpoints
├── outputs/               # Training outputs
├── results/               # Evaluation results
├── train.py              # Main training script
├── test.py               # Main testing script
├── requirements.txt      # Python dependencies
└── README.md            # This file
Citation
If you use this code in your research, please cite our paper:

bibtex
@article{zhang2025scafnet,
  title={SCAFNet: A Semantic Compensated Adaptive Fusion Network for Remote Sensing Images Change Detection},
  author={Zhang, Yunzuo and Zhen, Jiawen and Sun, Shibo and Liu, Ting and Huo, Lei and Wang, Tong},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2025},
  publisher={IEEE}
}
License
This project is licensed under the MIT License - see the LICENSE file for details.
