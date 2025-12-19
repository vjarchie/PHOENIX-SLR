# PHOENIX Sign Language Recognition

> **Academic Project - Semester III**  
> **Hybrid CTC+Attention Transformer for Continuous Sign Language Recognition**

---

## 🎯 Results Achieved

| Metric | Value |
|--------|-------|
| **Test WER** | **50.91%** |
| Dev WER | 51.44% |
| Model | Hybrid CTC+Attention |
| Backbone | ResNet-18 (pretrained) |
| Training Time | ~20 hours (100 epochs) |

---

## Project Goal

Build a **continuous sign language recognition system** using the PHOENIX-2014 dataset:

- **Input**: Video of sign language
- **Output**: Gloss sequence (text representation of signs)
- **Demo**: Real-time camera recognition with English translation

---

## Why This Dataset?

[RWTH-PHOENIX-Weather 2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) is the standard benchmark for CSLR:

| Reason | Details |
|--------|---------|
| **Published Benchmarks** | Best: 26.8% WER (Koller et al., CVPR 2017) |
| **Research Standard** | Most CSLR papers use this dataset |
| **Professional Quality** | Studio recording, consistent lighting |
| **Clean Annotations** | Gloss-level annotations |

---

## Architecture: Hybrid CTC + Attention

We discovered that **pure CTC collapses** (outputs only blanks). Our solution:

```
Video Frames (B, T, 3, 260, 210)
        ↓
CNN Backbone (ResNet-18, pretrained)
        ↓
Transformer Encoder (6 layers, 8 heads)
        ↓
    ┌───┴───┐
    ↓       ↓
CTC Head   Attention Decoder
(30%)      (70%)
    ↓       ↓
    └───┬───┘
        ↓
Joint Loss → Gloss Prediction
```

### Why Hybrid Works

| Component | Purpose |
|-----------|---------|
| **CTC Head (30%)** | Alignment signal, handles variable-length output |
| **Attention Decoder (70%)** | Models output dependencies, **prevents collapse** |
| **Joint Loss** | Balances alignment and generation |

---

## Quick Start

### 1. Setup Environment

```bash
cd D:\PHOENIX-SLR
python -m venv venv
.\venv\Scripts\Activate  # Windows
pip install -r requirements.txt
```

### 2. Download Dataset (~53 GB)

```bash
python scripts/download_phoenix.py
```

Or download manually from: https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014.v3.tar.gz

### 3. Train Model

```bash
# End-to-End training (recommended, ~20 hours)
python train_e2e.py --epochs 100 --batch_size 4 --device cuda

# Or with I3D features (faster, ~6 hours)
python scripts/extract_i3d_features.py --device cuda
python train_i3d.py --epochs 100 --batch_size 8 --device cuda
```

### 4. Test Model

```bash
python test_model.py --checkpoint checkpoints/e2e/best.pth --device cuda
```

### 5. Demo

```bash
# Live camera demo
python demo_camera.py --checkpoint checkpoints/e2e/best.pth --device cuda

# Process video/frames from dataset
python demo_video.py --folder "path/to/frames" --device cuda
```

---

## Project Structure

```
PHOENIX-SLR/
├── src/
│   ├── models/
│   │   ├── transformer.py      # Hybrid CTC+Attention model
│   │   └── i3d_model.py        # I3D-based model
│   ├── data/
│   │   ├── phoenix_dataset.py  # Main dataset loader
│   │   ├── i3d_dataset.py      # I3D features loader
│   │   └── augmentation.py     # Video augmentation
│   └── translation/
│       └── gloss_to_english.py # German→English translation
├── scripts/
│   ├── download_phoenix.py     # Dataset download
│   └── extract_i3d_features.py # I3D feature extraction
├── train_e2e.py               # End-to-end training (recommended)
├── train_i3d.py               # I3D feature training
├── train.py                   # Original CTC training
├── test_model.py              # Evaluation script
├── demo_camera.py             # Live camera demo
├── demo_video.py              # Video file demo
├── dashboard.py               # Streamlit training dashboard
├── configs/
│   └── transformer_ctc.yaml   # Training configuration
├── requirements.txt
└── README.md
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [E2E_APPROACH_GUIDE.md](E2E_APPROACH_GUIDE.md) | **Complete implementation guide** for hybrid approach |
| [ARCHITECTURE_EVOLUTION.md](ARCHITECTURE_EVOLUTION.md) | Why pure CTC failed, evolution to hybrid |
| [HYBRID_APPROACH.md](HYBRID_APPROACH.md) | Detailed hybrid architecture explanation |
| [I3D_UPGRADE.md](I3D_UPGRADE.md) | I3D feature extraction approach |
| [TECHNICAL_PAPER_NOTES.md](TECHNICAL_PAPER_NOTES.md) | Academic paper notes |
| [TRAINING_SUMMARY.md](TRAINING_SUMMARY.md) | Final training results |

---

## Key Learnings

### What Worked ✅

- Hybrid CTC + Attention (prevents collapse)
- Pretrained ResNet-18 backbone
- Data augmentation (NO horizontal flip!)
- Label smoothing (0.1)
- Learning rate warmup + cosine annealing
- Gradient clipping (max_norm=5.0)

### What Didn't Work ❌

- Pure CTC → collapsed to blank-only
- SimpleCNN backbone → too weak
- High CTC weight (>0.5) → collapse
- No warmup → unstable training

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| d_model | 512 |
| Encoder layers | 6 |
| Decoder layers | 3 |
| Attention heads | 8 |
| Dropout | 0.2 |
| CTC weight | 0.3 |
| Learning rate | 1e-4 |
| Batch size | 4 |
| Max frames | 64 |

---

## Dataset Information

### RWTH-PHOENIX-Weather 2014

| Property | Value |
|----------|-------|
| Size | 53 GB |
| Language | German Sign Language (DGS) |
| Vocabulary | ~1,236 glosses |
| Train samples | 5,672 |
| Dev samples | 540 |
| Test samples | 629 |
| Frame size | 210 × 260 pixels |
| FPS | 25 |

---

## Training Tips

1. **Monitor blank ratio** - if >90%, CTC is collapsing
2. **Use lower CTC weight** (0.1-0.3) for stability
3. **Don't use horizontal flip** - changes sign meaning
4. **Early stopping** with patience=15 prevents overfitting
5. **Use Streamlit dashboard** for real-time monitoring:
   ```bash
   python -m streamlit run dashboard.py
   ```

---

## Citation

```bibtex
@article{koller2015continuous,
  title={Continuous sign language recognition: Towards large vocabulary 
         statistical recognition systems handling multiple signers},
  author={Koller, Oscar and Forster, Jens and Ney, Hermann},
  journal={Computer Vision and Image Understanding},
  volume={141},
  pages={108--125},
  year={2015}
}
```

---

## License

This project is for academic purposes. The PHOENIX dataset has its own license terms.

---

*Created: December 2025*  
*Final WER: 50.91% (Hybrid CTC+Attention with ResNet-18)*
