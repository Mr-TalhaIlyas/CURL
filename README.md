# 🏥 CURL: Contrastive Ultrasound Video Representation Learning

<div align="center">
    
![Badge](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2FMr-TalhaIlyas%2FCURL%2F&label=CURL%2Fvisitor&icon=github&color=%23198754)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


*A novel self-supervised framework for fetal movement detection from extended ultrasound video recordings*

</div>

---

## 🎯 Overview [Paper](https://arxiv.org/abs/2510.20214)

**CURL (Contrastive Ultrasound Video Representation Learning)** is a cutting-edge self-supervised framework designed specifically for fetal movement assessment from ultrasound videos. Our method employs a **dual-contrastive loss** that captures both spatial (anatomical) and temporal (motion-based) features, enabling robust representation learning for fetal movement dynamics.

### 🔬 Key Innovations

- **🎭 Dual-Contrastive Learning**: Combines spatial (SimCLR-style NT-Xent) and temporal contrastive objectives
- **🎯 Task-Specific Sampling**: Intelligent sampling strategy for movement vs. non-movement segments
- **🔄 Flexible Inference**: Supports ultrasound recordings of arbitrary length through probabilistic fine-tuning
- **🏗️ Modular Architecture**: Support for both SlowFast and Vision Transformer (ViT) backbones

<div align="center">
<img src="/screens/summary.jpg" alt="CURL Framework Overview" width="80%">
</div>

**Pipeline Overview**: Starting from expertly annotated ultrasound videos (A), CURL splits clips into spatiotemporal patches (B), uses transformer backbones with dual-contrastive learning to extract robust features, fine-tunes with lightweight classifiers (C), and delivers clinically reliable fetal movement detection (D).

---

## 🚀 Quick Start

### 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM for video processing

### 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/Mr-TalhaIlyas/CURL.git
cd CURL
```

2. **Create virtual environment**
```bash
# Using conda (recommended)
conda create -n curl python=3.8
conda activate curl

# Or using pip
python -m venv curl
source curl/bin/activate  # Linux/Mac
curl\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda create --name curl --file requirements.txt
```

### 📁 Data Preparation

1. **Organize your data structure**:
```
data/
├── videos/           # Raw ultrasound videos (.mp4)
├── optical_flow/     # Optical flow videos (.mp4) 
├── labels/           # Label files (.npy)
└── folds/           # Train/test split files
    ├── train_fold_1.txt
    ├── test_fold_1.txt
    └── ...
```

2. **Update configuration**:
```python
# In configs/config.py
config = dict(
    vid_dir = "path/to/videos/",
    flow_dir = "path/to/optical_flow/", 
    lbl_dir = "path/to/labels/",
    folds = "path/to/folds/"
)
```

---

## 🎓 Training

### 1. 🔄 Self-Supervised Pre-training

Choose between two backbone architectures:

#### **SlowFast + Dual Contrastive Loss**
```bash
# Train with both spatial and temporal losses
python dual_contrastive_main.py \
    --enable_temporal_loss \
    --spatial_loss_weight 1.0 \
    --temporal_loss_weight 0.5 \
    --dual_loss_mode both \
    --epochs 100

# Spatial-only training
python dual_contrastive_main.py \
    --spatial_loss_weight 1.0 \
    --dual_loss_mode spatial_only
```

#### **Vision Transformer (ViT) + Dual Contrastive Loss**
```bash
# MAE-style contrastive learning
python run_mae_contrastive.py \
    --enable_temporal_loss \
    --spatial_loss_weight 1.0 \
    --temporal_loss_weight 0.7 \
    --embed_dim 1024 \
    --depth 24
```

### 2. 🎯 Fine-tuning for Classification

```bash
# Fine-tune pre-trained contrastive model
python run_finetune.py \
    --model_type contrastive_mae \
    --checkpoint_path /path/to/pretrained_model.pth \
    --epochs 30 \
    --lr 2e-4 \
    --loss_type focal

# Fine-tune standard MAE model
python run_finetune.py \
    --model_type standard_mae \
    --checkpoint_path /path/to/mae_model.pth \
    --epochs 30
```

---

## 🏗️ Architecture

### 🎭 Dual-Contrastive Loss Framework

```python
# Spatial Contrastive Loss (NT-Xent)
spatial_loss = NT_XentLoss(spatial_features_i, spatial_features_j)

# Temporal Contrastive Loss (TC)
temporal_loss = temporal_contrastive_loss(
    temporal_features_i, 
    temporal_features_j, 
    temperature, 
    clusters=8
)

# Combined Loss
total_loss = α * spatial_loss + β * temporal_loss
```

### 🔧 Supported Architectures

| Model | Backbone | Key Features |
|-------|----------|--------------|
| **SimCLR + SlowFast** | SlowFast ResNet | Two-stream processing for spatial-temporal features |
| **Contrastive MAE** | Vision Transformer | Patch-based processing with attention mechanisms |
| **Hybrid Models** | Custom | Combine benefits of both approaches |

---


## ⚙️ Configuration

### 🔧 Key Parameters

```python
# Dual contrastive learning
enable_temporal_loss = True
spatial_loss_weight = 1.0
temporal_loss_weight = 0.5
temperature_spatial = 0.5
temperature_temporal = 0.1

# Temporal contrastive loss
tc_clusters = 8
tc_num_iters = 10
tc_do_entro = True  # Enable IID regularization

# Model architecture  
mae_contrastive = dict(
    embed_dim = 1024,
    depth = 24,
    num_heads = 16,
    projection_dim = 256,
    temporal_projection_dim = 128
)
```

---

## 📁 Project Structure

```
CURL/
├── 📄 README.md
├── 📋 requirements.txt
├── 📂 scripts/
│   ├── 🔧 configs/
│   │   └── config.py
│   ├── 📊 data/
│   │   ├── simclr_loader.py
│   │   ├── dataloader.py
│   │   └── utils.py
│   ├── 🏗️ models/
│   │   ├── mae/
│   │   ├── slowfast/
│   │   └── contrastive_mae.py
│   ├── 🛠️ tools/
│   │   ├── nt_xnet.py           # Spatial contrastive loss
│   │   ├── tc_loss.py           # Temporal contrastive loss  
│   │   └── simclr_training.py
│   ├── 🎓 Training Scripts
│   │   ├── main_simclr.py
│   │   ├── main_mae_contrastive.py
│   │   └── finetune_contrastive_mae.py
│   └── 🚀 Run Scripts
│       ├── dual_contrastive_main.py
│       ├── run_mae_contrastive.py
│       └── run_finetune.py
└── 📸 screens/
    └── summary.jpg
```

---

## 🔬 Technical Details

### 🎯 Loss Functions

#### **Spatial Contrastive Loss (NT-Xent)**
- Based on SimCLR framework
- Learns anatomical feature representations
- Temperature-scaled InfoNCE loss

#### **Temporal Contrastive Loss (TC)**
- Novel clustering-based approach
- Learns motion dynamics
- Combines Cross-Level Distillation (CLD) and IID regularization

### 🎭 Data Augmentation

- **Spatial**: Random cropping, color jittering, Gaussian blur
- **Temporal**: Frame dropping, temporal jittering
- **Domain-specific**: Ultrasound-aware transformations

---

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
Paper is currently under review.
```

---


### 🐛 Issues

Found a bug? Please open an [issue](https://github.com/Mr-TalhaIlyas/CURL/issues) with:
- Detailed description
- Steps to reproduce
- Environment details
- Expected vs actual behavior

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Thanks to the medical imaging community for inspiration
- Built upon excellent work in self-supervised learning
- Special thanks to [SimCLR](https://github.com/google-research/simclr) and [MAE](https://github.com/facebookresearch/mae) teams

---

<div align="center">

