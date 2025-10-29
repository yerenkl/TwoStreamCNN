# Two-Stream CNN for Video Classification

This repository implements dual stream convolutional neural networks for video action recognition.  
The architecture follows the **Two-Stream CNN** introduced by *Simonyan & Zisserman (2014)* — where RGB frames and optical flows are learned in separate networks and later fused during inference.

## Project Structure

```bash
$ tree
TwoStreamCNN/
│
├── main.py           → Main training entry point
├── test.py           → Evaluate trained models (spatial, temporal, or fused)
│
├── model.py          → SpatialStream and TemporalStream architectures
├── datasets.py       → Dataset loaders for RGB and Optical Flow inputs
├── trainer.py        → PyTorch Trainer (training + validation)
├── utils.py          → Metrics, checkpointing, and helper functions
├── config.py         → Global constants (paths, hyperparameters)
│
└── results/          → Output directory for checkpoints and logs
```

## Quick Start

### 1-Train the Spatial Stream (RGB)
```bash
python train.py --stream spatial --epochs 25 --batch_size 64 --lr 1e-4
```
### 2-Train the Temporal Stream (Optical Flow)
```bash
python train.py --stream temporal --epochs 25 --batch_size 8 --lr 1e-4
```
### Command-Line Arguments
| Argument         | Description                                     | Default                               |
| ---------------- | ----------------------------------------------- | ------------------------------------- |
| `--stream`       | Which stream to train (`spatial` or `temporal`) | **Required**                          |
| `--epochs`       | Number of training epochs                       | `DEFAULT_EPOCHS` *(from `config.py`)* |
| `--batch_size`   | Batch size per iteration                        | `DEFAULT_BATCH_SIZE`                  |
| `--lr`           | Learning rate                                   | `DEFAULT_LR`                          |
| `--weight_decay` | L2 regularization                               | `DEFAULT_WEIGHT_DECAY`                |
| `--resume`       | Resume training if checkpoint exists            | `False`                               |
| `--device`       | Device to train on (`cuda` or `cpu`)            | *Auto-detected*                       |
| `--root_dir`     | Root directory of dataset                       | `ROOT_DIR`                            |
| `--save_dir`     | Directory to save results                       | `SAVE_DIR`                            |
| `--num_workers`  | Number of DataLoader workers                    | `4`                                   |

##  Dataset Layout
Your dataset folder should be organized as:
```bash
ucf101_noleakage/
├── rgb/
│   ├── train/
│   │   ├── class1/
│   │   │   ├── video1/
│   │   │   │   ├── frame0001.jpg
│   │   │   │   ├── frame0002.jpg
│   │   │   │   └── ...
│   ├── val/
│   │   └── ...
│
├── flows/
│   ├── train/
│   │   ├── class1/
│   │   │   ├── video1/
│   │   │   │   ├── flow_1_2.npy
│   │   │   │   ├── flow_2_3.npy
│   │   │   │   └── ...
│   ├── val/
│   │   └── ...
│
└── metadata/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

##  Model Details
- **Spatial Stream:** Standard 2D CNN (e.g., *ResNet-18* backbone) trained on RGB frames.  
- **Temporal Stream:** 2D CNN trained on stacked optical flow maps *(u, v)*.  
- **Fusion:** Performed offline by averaging logits.

##  Citation
If you use this code, please cite:

> **Simonyan, K., & Zisserman, A. (2014).**
> *Two-Stream Convolutional Networks for Action Recognition in Videos.*
> *Advances in Neural Information Processing Systems (NeurIPS 27).*


