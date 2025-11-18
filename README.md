# SCAFNet
SCAFNet: Semantic Compensated Adaptive Fusion Network for Remote Sensing Change Detection

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

# Create conda environment
conda create -n scafnet python=3.8
conda activate scafnet

# Install PyTorch
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install requirements
pip install -r requirements.txt

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

# Directory Structure
Organize your data as follows:
text
data/
- CDD/
  - train/
    - A/       # Time1 images (10,000)
    - B/       # Time2 images (10,000)
    - label/   # Change masks (10,000)
  - val/       # 3,000 images each
  - test/      # 3,000 images each
- WHU-CD/
  - train/     # 5,947 images
  - val/       # 743 images
  - test/      # 744 images

# Train 
python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr} --optimizer ${optimizer}

# Evaluation Metrics
The code automatically computes:
F1-Score

IoU (Intersection over Union)

Precision

Recall

Overall Accuracy

# Repository Structure
SCAFNet/
- data_preprocess/
  - compute_mean_std.py
- datasets/
  - CD_dataset.py
  - data_utils.py
- models/
  - __init__.py
  - basic_model.py
  - BitemporalFusion.py
  - cc_attention.py
  - DualFusion.py
  - evaluator.py
  - FCTransformer.py
  - help_funcs.py
  - HybridModel.py
  - losses.py
  - networks.py
  - resnet.py
  - trainer.py
- scripts/
  - eval.sh
  - eval_HY_CDD.sh
  - eval_HY_WHU.sh
  - run_baseline_SCM_CTFF_CDD.sh
  - run_baseline_SCM_CTFF_WHU.sh
  - run_hy_CDD.sh
  - run_hy_WHU.sh
# Citation
If you use this code in your research, please cite our paper:
bibtex
@article{zhang2025scafnet,
  title={SCAFNet: A Semantic Compensated Adaptive Fusion Network for Remote Sensing Images Change Detection},
  author={Zhang, Yunzuo and Zhen, Jiawen and Sun, Shibo and Liu, Ting and Huo, Lei and Wang, Tong},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2025},
  publisher={IEEE}
}
# License
This project is licensed under the MIT License - see the LICENSE file for details.
