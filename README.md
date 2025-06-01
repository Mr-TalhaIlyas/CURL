# Contrastive Representation Learning for Fetal Movement Assessment

Contrastive Ultrasound Video Representation Learning (CURL) is a novel self-supervised framework designed for fetal movement detection from extended ultrasound video recordings. Our method employs a dual-contrastive loss that captures both spatial and temporal features, enabling the model to learn robust motion representations. We also introduce a task-specific sampling strategy that effectively distinguishes between movement and non-movement segments during training. Additionally, CURL supports flexible inference on ultrasound recordings of arbitrary length through a probabilistic fine-tuning approach.

**Starting from a curated, expertly annotated antenatal ultrasound video dataset (A). The CURL framework splits each clip into spatiotemporal patches (B), using a transformer backbone with dual-contrastive self-supervised learning to extract robust anatomical and motion features. A lightweight linear classifier is fine-tuned on augmented clips for task-specific adaptation (C). The resulting model delivers clinically reliable fetal movement detection with high sensitivity and AUROC (D).**
![alt text](/screens/summary.jpg)

## Setup

If you want to create an identical virtual environment from requirements.txt, you can either:
```shell
# using pip
pip install -r requirements.txt

# using Conda
conda create --name <env_name> --file requirements.txt
```
## Data Preparation