# CorrNet+ Implementation for PHOENIX-2014

Implementation of CorrNet+ (Spatial-Temporal Correlation Network) for Continuous Sign Language Recognition.

**Paper**: [CorrNet+: Sign Language Recognition and Translation via Spatial-Temporal Correlation](https://arxiv.org/abs/2404.11111)

**Original Code**: https://github.com/hulianyuyy/CorrNet_Plus

## Performance

CorrNet+ achieves state-of-the-art results on PHOENIX-2014:

| Method | Dev WER | Test WER |
|--------|---------|----------|
| Previous SOTA | ~19.5% | ~20.1% |
| **CorrNet+** | **18.0%** | **18.2%** |

## Architecture

```
Video Input (B, C, T, H, W)
         │
         ▼
┌─────────────────────────────────────┐
│ ResNet-18 with Correlation Modules  │
│   ├── Layer 1 (64 channels)         │
│   ├── Layer 2 (128 channels)        │
│   │   └── GetCorrelation (n=1)      │
│   │   └── TemporalWeighting         │
│   ├── Layer 3 (256 channels)        │
│   │   └── GetCorrelation (n=3)      │
│   │   └── TemporalWeighting         │
│   └── Layer 4 (512 channels)        │
│       └── GetCorrelation (n=5)      │
│       └── TemporalWeighting         │
└─────────────────────────────────────┘
         │
         ▼ (B*T, 512)
┌─────────────────────────────────────┐
│ TemporalConv with LiftPooling       │
│   K5 → P2 → K5 → P2                 │
│   (Temporal downsampling 4x)        │
│   └── ConvCTC Head                  │
└─────────────────────────────────────┘
         │
         ▼ (T/4, B, 1024)
┌─────────────────────────────────────┐
│ BiLSTM (2 layers, bidirectional)    │
│   └── SeqCTC Head                   │
└─────────────────────────────────────┘
         │
         ▼
    CTC Decoding
```

## Key Components

### 1. Spatial-Temporal Correlation (`GetCorrelation`)
- Computes cross-frame body trajectories
- Uses attention pooling + multi-scale spatial aggregation
- No pose estimation required

### 2. Temporal LiftPooling
- Learnable temporal downsampling
- Uses predict-update lifting scheme
- Preserves important temporal details

### 3. Dual CTC Heads with Distillation
- ConvCTC: On TemporalConv output
- SeqCTC: On BiLSTM output
- Knowledge distillation from SeqCTC to ConvCTC

## Usage

### Training

```bash
python train.py --config configs/phoenix2014.yaml
```

### Configuration

Edit `configs/phoenix2014.yaml` to adjust:
- Data paths
- Model architecture
- Loss weights
- Training hyperparameters

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 2 | Per-GPU batch size |
| `accumulation_steps` | 8 | Gradient accumulation |
| `learning_rate` | 1e-4 | Initial learning rate |
| `SeqCTC` weight | 1.0 | Main CTC loss |
| `ConvCTC` weight | 1.0 | Auxiliary CTC loss |
| `Dist` weight | 25.0 | Distillation loss |

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (recommended)
- 12GB+ GPU memory

## Data Preparation

1. Download PHOENIX-2014 dataset
2. Extract to `data/phoenix2014-release`
3. Update `data.root` in config file

## Citation

```bibtex
@article{hu2024corrnetplus,
  title={CorrNet+: Sign Language Recognition and Translation via Spatial-Temporal Correlation},
  author={Hu, Lianyu and others},
  journal={arXiv preprint arXiv:2404.11111},
  year={2024}
}
```
