# I3D Feature Upgrade for PHOENIX-SLR

> **Recommendation**: Use pre-extracted I3D features to improve WER from 52.85% to ~40-45%

---

## Current vs Proposed

| Aspect | Current (ResNet-18) | Proposed (I3D) |
|--------|---------------------|----------------|
| **Backbone** | ResNet-18 (2D CNN) | I3D (3D CNN) |
| **Pretrained on** | ImageNet (images) | Kinetics-400 (videos) |
| **Feature dim** | 512 | 1024 |
| **Understands motion** | ❌ No | ✅ Yes |
| **Expected WER** | 52.85% | ~40-45% |
| **Training time** | ~7 hours | ~6 hours |

---

## Why I3D is Better

### ResNet-18 (Current)
- Trained on **static images** (ImageNet)
- Sees each frame **independently**
- Misses temporal patterns in signs

### I3D (Proposed)
- Trained on **video actions** (Kinetics-400)
- Processes **multiple frames together**
- Understands **motion and dynamics**

---

## Implementation Plan

### Step 1: Extract Features (~1-2 hours, one-time)
```bash
python scripts/extract_i3d_features.py \
    --data_dir data/phoenix2014-release \
    --output_dir data/i3d_features \
    --device cuda
```

### Step 2: Train with I3D Features (~5 hours)
```bash
python train_i3d.py \
    --features_dir data/i3d_features \
    --epochs 100 \
    --batch_size 8 \
    --device cuda
```

---

## File Structure After Implementation

```
D:\PHOENIX-SLR\
├── data/
│   ├── phoenix2014-release/     # Original dataset
│   └── i3d_features/            # NEW: Extracted features
│       ├── train/
│       │   ├── video_001.npy    # (T, 1024) features
│       │   ├── video_002.npy
│       │   └── ...
│       ├── dev/
│       └── test/
├── scripts/
│   └── extract_i3d_features.py  # NEW: Feature extraction
├── src/
│   ├── models/
│   │   ├── transformer.py       # Existing
│   │   └── i3d_model.py         # NEW: I3D-based model
│   └── data/
│       ├── phoenix_dataset.py   # Existing
│       └── i3d_dataset.py       # NEW: I3D feature dataset
└── train_i3d.py                 # NEW: Training script
```

---

## Expected Results

| Metric | Current | With I3D |
|--------|---------|----------|
| Best WER | 52.85% | ~40-45% |
| Blank Ratio | 69% | ~50-60% |
| Training Time | 7 hours | 6 hours |
| GPU Memory | 3-4 GB | 2-3 GB |

---

## Two-Stage Upgrade Path

### Stage 1: Pre-extracted I3D (This Implementation)
- Extract features once, save to disk
- Train on pre-computed features
- Quick validation of improvement
- **Expected: ~40-45% WER**

### Stage 2: End-to-End I3D (Future)
- Fine-tune I3D backbone during training
- Longer training (~30 hours)
- Best possible results
- **Expected: ~35-40% WER**

---

## Hardware Requirements

| Resource | Requirement | Your Setup |
|----------|-------------|------------|
| GPU VRAM | 3-4 GB | RTX 5070 Ti (~12-16 GB) ✅ |
| RAM | 16 GB | 64 GB ✅ |
| Disk Space | ~2 GB | ✅ |
| Time | ~7 hours total | ✅ |

---

*Created: December 17, 2025*


