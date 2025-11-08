# Domain-Adaptive Mamba for Cross-Scene Hyperspectral Image Classification

## Overview

This repository implements Domain-Adaptive Mamba for hyperspectral image classification across different scenes, addressing domain shift challenges with the efficient Mamba architecture.

## 🔧Requirements

```tex
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data
Place your hyperspectral datasets in the folder:
```
├── DAMamba_main
├── DAMamba_model
├── DAMamba_basenet
└── Houston/
    ├── Houston13.mat
    ├── Houston13_7gt.mat
    ├── Houston18.mat
    └── Houston18_7gt.mat
```

### 2. Train Model

```bash
CUDA_VISIBLE_DEVICES=0 python DAMamba_main.py --config DeepCoral/DeepCoral.yaml --data_dir Houston/ --src_domain Houston13 --tgt_domain Houston18 --lr 0.2 --n_epoch 500  --start_iter 0  --cls_weight 0 --save_path Results/ |tee ./Results/test/test_ori.log
```

- Results

Classification accuracies of Houston dataset:

OA:79.51, AA: 63.64, Kappa: 65.81

## Citation



## Contact

For questions, please contact: 