# Technical Documentation for Academic Paper & Presentation

> **Project Title**: An Integrated Transformer Architecture for Real-Time Bidirectional Continuous Sign Language Interpretation
>
> **Academic Period**: Semester III
>
> **Dataset**: RWTH-PHOENIX-Weather 2014 (German Sign Language)
>
> **Target Metric**: Word Error Rate (WER) < 30%

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction & Motivation](#2-introduction--motivation)
3. [Related Work](#3-related-work)
4. [Methodology](#4-methodology)
5. [Implementation Details](#5-implementation-details)
6. [Challenges & Solutions](#6-challenges--solutions)
7. [Experimental Setup](#7-experimental-setup)
8. [Results](#8-results)
9. [Discussion](#9-discussion)
10. [Future Work](#10-future-work)
11. [Conclusion](#11-conclusion)
12. [References](#12-references)

---

## 1. Abstract

This project presents a Transformer-based architecture with Connectionist Temporal Classification (CTC) for Continuous Sign Language Recognition (CSLR). We utilize the RWTH-PHOENIX-Weather 2014 dataset, a benchmark dataset for German Sign Language recognition. Our approach combines a CNN backbone for spatial feature extraction with a Transformer encoder for temporal modeling. We address key challenges including CTC collapse, memory optimization, and efficient video processing. The system targets a Word Error Rate below 30%, demonstrating the viability of attention-based architectures for sign language interpretation.

**Keywords**: Sign Language Recognition, Transformer, CTC, Deep Learning, Computer Vision, Sequence-to-Sequence Learning

---

## 2. Introduction & Motivation

### 2.1 Problem Statement

Sign language is the primary mode of communication for approximately 70 million deaf individuals worldwide. However, communication barriers persist due to the limited availability of sign language interpreters. Automated sign language recognition systems can bridge this gap by providing real-time translation between sign language and spoken/written language.

### 2.2 Project Evolution

This project represents a **continuation and refinement** of an earlier ASL (American Sign Language) recognition attempt that encountered significant challenges:

| Previous Approach | Issue | Current Solution |
|-------------------|-------|------------------|
| TCN Architecture | Limited receptive field | Transformer (global attention) |
| How2Sign/OpenASL Dataset | No benchmarks available | PHOENIX-2014 (published benchmarks) |
| 6,068 word vocabulary | CTC collapse | 1,234 gloss vocabulary |
| Keypoint-based input | Feature mismatch issues | Raw video frames |

### 2.3 Why RWTH-PHOENIX-Weather 2014?

| Criterion | Benefit |
|-----------|---------|
| **Published Benchmarks** | State-of-the-art WER: 26.8% (Koller et al., 2017) |
| **Research Standard** | Most CSLR papers use this dataset |
| **Professional Quality** | Studio recording with consistent lighting |
| **Clean Annotations** | Gloss-level annotations verified by deaf signers |
| **Manageable Size** | ~6K training samples, 1,234 glosses |

### 2.4 Objectives

1. **Primary**: Achieve < 30% Word Error Rate on PHOENIX-2014 test set
2. **Secondary**: Develop a modular, extensible architecture for future improvements
3. **Tertiary**: Document learnings for academic contribution

---

## 3. Related Work

### 3.1 Evolution of Sign Language Recognition

| Era | Approach | Limitation |
|-----|----------|------------|
| **Pre-2015** | HMM + Handcrafted Features | Limited expressiveness |
| **2015-2018** | CNN + LSTM | Vanishing gradients |
| **2018-2020** | TCN + CTC | Limited global context |
| **2020-Present** | Transformer + CTC | Current state-of-the-art |

### 3.2 Key Papers on PHOENIX Dataset

| Paper | Year | Architecture | WER (Test) |
|-------|------|--------------|------------|
| Koller et al. | 2015 | CNN-LSTM-HMM | 55.6% |
| Koller et al. | 2017 | CNN-LSTM-HMM (improved) | 26.8% |
| Camgöz et al. | 2020 | Sign Language Transformers | ~24% |
| Hu et al. (CorrNet) | 2023 | Correlation Network | 17.8% |

### 3.3 Why Transformer + CTC?

The Transformer architecture offers several advantages for CSLR:

1. **Global Context**: Self-attention captures dependencies across the entire sequence
2. **Parallelization**: Unlike RNNs, Transformers process sequences in parallel
3. **Proven Success**: Sign Language Transformers (Camgöz et al., 2020) achieved SOTA
4. **CTC Compatibility**: Handles variable-length alignment without frame-level labels

---

## 4. Methodology

### 4.1 Problem Formulation

**Input**: Video sequence \( X = \{x_1, x_2, ..., x_T\} \) where \( x_t \in \mathbb{R}^{H \times W \times 3} \)

**Output**: Gloss sequence \( Y = \{y_1, y_2, ..., y_N\} \) where \( y_n \in \mathcal{V} \) (vocabulary)

**Objective**: Learn mapping \( f: X \rightarrow Y \) minimizing Word Error Rate

### 4.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT VIDEO FRAMES                        │
│                   (T × 210 × 260 × 3)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CNN BACKBONE                              │
│            (SimpleCNN or ResNet18)                           │
│                                                              │
│    Conv(3→64) → BN → ReLU → MaxPool                         │
│    Conv(64→128) → BN → ReLU                                  │
│    Conv(128→256) → BN → ReLU                                 │
│    Conv(256→512) → BN → ReLU → AdaptiveAvgPool               │
│    Linear(512→512)                                           │
│                                                              │
│    Output: (T × 512) features                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 POSITIONAL ENCODING                          │
│           Sinusoidal positional embeddings                   │
│           PE(pos, 2i) = sin(pos / 10000^(2i/d))             │
│           PE(pos, 2i+1) = cos(pos / 10000^(2i/d))           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               TRANSFORMER ENCODER                            │
│                                                              │
│    ┌──────────────────────────────────────────────────┐     │
│    │  Multi-Head Self-Attention (8 heads)             │ ×6  │
│    │  Layer Normalization                              │     │
│    │  Feed-Forward Network (512 → 2048 → 512)         │     │
│    │  Layer Normalization                              │     │
│    │  Residual Connections                             │     │
│    └──────────────────────────────────────────────────┘     │
│                                                              │
│    Output: (T × 512) contextualized features                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  OUTPUT PROJECTION                           │
│              Linear(512 → 1234 glosses)                      │
│              Log Softmax                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CTC DECODER                               │
│                                                              │
│    Training: CTC Loss (handles alignment)                    │
│    Inference: Greedy Decoding (collapse repeats + blanks)   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   GLOSS SEQUENCE                             │
│            ["WOLKE", "REGEN", "MORGEN", ...]                │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Model Components

#### 4.3.1 CNN Backbone

Two options implemented:

**SimpleCNNBackbone** (Default):
- Lightweight 4-layer CNN
- ~2M parameters
- Faster training

**ResNet18 Backbone** (Optional):
- Pretrained on ImageNet
- ~11M parameters
- Better feature extraction

#### 4.3.2 Transformer Encoder

| Parameter | Value | Justification |
|-----------|-------|---------------|
| d_model | 512 | Standard for medium-scale tasks |
| n_heads | 8 | d_model / n_heads = 64 (good head dim) |
| n_layers | 6 | Balance between capacity and training time |
| d_ff | 2048 | 4× d_model (standard ratio) |
| dropout | 0.1 | Regularization |

#### 4.3.3 CTC Loss

CTC (Connectionist Temporal Classification) enables training without frame-level alignment:

```
P(Y|X) = Σ P(π|X)  for all valid alignments π
```

Where valid alignments allow:
- Repeated characters (collapsed during decoding)
- Blank tokens (for temporal padding)

### 4.4 Training Strategy

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| **Optimizer** | AdamW | Better generalization than Adam |
| **Learning Rate** | 0.0001 | Conservative for stability |
| **Weight Decay** | 0.01 | Regularization |
| **Scheduler** | Cosine Annealing | Smooth LR decay |
| **Gradient Clipping** | max_norm=5.0 | Prevents gradient explosion |
| **Batch Size** | 2-4 | Memory constraint |

---

## 5. Implementation Details

### 5.1 Data Pipeline

```python
# Temporal Subsampling (Memory Optimization)
if total_frames > max_frames:
    indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    frames = [all_frames[i] for i in indices]
```

**Frame Processing**:
1. Load PNG frames from disk
2. Convert BGR → RGB
3. Normalize to [0, 1] float32
4. Stack into tensor (T × H × W × C)

### 5.2 Memory Optimization

| Optimization | Before | After | Savings |
|--------------|--------|-------|---------|
| max_frames | 300 | 128 | ~57% |
| batch_size | 4 | 2 | 50% |
| num_workers | 4 | 0 | Prevents Windows issues |
| Subsampling | None | Uniform | Variable |

**Memory Calculation**:
- Per frame: 210 × 260 × 3 × 4 bytes = 655 KB
- Per sample (128 frames): 84 MB
- Per batch (2 samples): 168 MB

### 5.3 CTC Collapse Monitoring

```python
def compute_blank_ratio(log_probs, lengths, blank_idx):
    """Monitor for CTC collapse - if > 80%, model is failing"""
    total_blank = 0
    total_frames = 0
    for b in range(batch_size):
        best_path = torch.argmax(log_probs[:, b, :], dim=-1)
        blank_count = (best_path == blank_idx).sum()
        total_blank += blank_count
        total_frames += lengths[b]
    return total_blank / total_frames
```

**Warning Threshold**: blank_ratio > 80% indicates CTC collapse

### 5.4 Model Parameters

```
SignLanguageTransformer:
  CNN backbone: SimpleCNN
  Input dim: 512
  Model dim: 512
  Heads: 8
  Layers: 6
  Vocab size: 1234
  Total Parameters: 21,635,282 (~21.6M)
```

---

## 6. Challenges & Solutions

### 6.1 CTC Collapse (Critical Issue)

**Problem**: Model outputs mostly blank tokens, minimizing loss without learning.

**Symptoms**:
- Loss decreases but WER stays high
- Blank ratio > 95%
- Predictions are empty or single words

**Causes**:
1. Vocabulary too large relative to data
2. Learning rate too high
3. Insufficient training data

**Solutions Implemented**:
1. ✅ Monitor blank ratio during training
2. ✅ Use appropriate vocabulary size (1,234 vs 6,068)
3. ✅ Gradient clipping (max_norm=5.0)
4. ✅ Conservative learning rate (0.0001)
5. ✅ CTC loss with `zero_infinity=True`

### 6.2 Memory Exhaustion

**Problem**: Loading 300 frames at 210×260×3 resolution exhausts RAM.

**Solution**:
```python
# Before: 300 frames × 655KB = 196MB per sample
max_frames = 300

# After: 128 frames × 655KB = 84MB per sample
max_frames = 128

# Added: Uniform temporal subsampling
indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
```

### 6.3 Windows Multiprocessing Issues

**Problem**: DataLoader with `num_workers > 0` crashes silently on Windows.

**Solution**:
```python
# Windows-compatible DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=0,  # Disable multiprocessing on Windows
    pin_memory=True
)
```

### 6.4 Dataset Path Format

**Problem**: PHOENIX CSV contains glob patterns in folder column.

**Before**: `folder = "01April_2010_Thursday_heute_default-0/1/*.png"`

**Solution**:
```python
folder_path = sample['folder'].replace('/*.png', '')
```

### 6.5 Previous Project Failures (Lessons Learned)

| Issue | Previous Project | Current Solution |
|-------|------------------|------------------|
| CTC Collapse | 95% blank outputs | Vocabulary reduction + monitoring |
| Feature Mismatch | OpenPose vs MediaPipe | Use raw frames directly |
| No Benchmarks | Couldn't validate results | PHOENIX has published WER |
| TCN Limitations | Poor long-range modeling | Transformer architecture |

---

## 7. Experimental Setup

### 7.1 Hardware Configuration

| Component | Specification |
|-----------|---------------|
| **CPU** | AMD Ryzen 9 9950X (16 cores / 32 threads) |
| **RAM** | 64 GB DDR5 |
| **GPU** | NVIDIA RTX 5070 Ti (16 GB VRAM) |
| **Storage** | NVMe SSD |
| **OS** | Windows 10/11 |

### 7.2 Dataset Statistics

| Split | Samples | Vocabulary | Avg. Frames | Avg. Glosses |
|-------|---------|------------|-------------|--------------|
| Train | 5,672 | 1,234 | ~120 | ~7 |
| Dev | 540 | (shared) | ~115 | ~7 |
| Test | 629 | (shared) | ~118 | ~7 |

### 7.3 Training Configuration

```yaml
# Final Training Configuration
model:
  type: transformer
  d_model: 512
  nhead: 8
  num_layers: 6
  dropout: 0.1

training:
  epochs: 100
  batch_size: 2
  learning_rate: 0.0001
  weight_decay: 0.01
  gradient_clip: 5.0
  
data:
  max_frames: 128
  feature_type: fullFrame-210x260px
```

### 7.4 Evaluation Metric

**Word Error Rate (WER)**:

```
WER = (S + I + D) / N × 100%

Where:
  S = Substitutions
  I = Insertions
  D = Deletions
  N = Total words in reference
```

---

## 8. Results

> **Note**: Fill in after training completes

### 8.1 Training Curves

| Epoch | Train Loss | Dev Loss | Dev WER | Blank Ratio |
|-------|------------|----------|---------|-------------|
| 1 | ___ | ___ | ___% | ___% |
| 10 | ___ | ___ | ___% | ___% |
| 25 | ___ | ___ | ___% | ___% |
| 50 | ___ | ___ | ___% | ___% |
| 100 | ___ | ___ | ___% | ___% |

### 8.2 Final Results

| Metric | Dev Set | Test Set |
|--------|---------|----------|
| WER | ___% | ___% |
| Sentence Accuracy | ___% | ___% |
| Blank Ratio | ___% | ___% |

### 8.3 Sample Predictions

```
[1] Target: WOLKE REGEN MORGEN KOMMEN
    Pred:   ___

[2] Target: HEUTE ABEND WETTER BESSER
    Pred:   ___

[3] Target: TEMPERATUR STEIGEN GRAD ZEHN
    Pred:   ___
```

### 8.4 Comparison with Published Results

| Method | Year | WER (Test) | Our Result |
|--------|------|------------|------------|
| CNN-LSTM-HMM | 2017 | 26.8% | |
| Transformer + CTC | 2020 | ~24% | |
| CorrNet (SOTA) | 2023 | 17.8% | |
| **Ours** | 2025 | | ___% |

---

## 9. Discussion

### 9.1 Analysis of Results

> Fill in after training

### 9.2 Comparison with Baseline

| Aspect | Expected | Achieved | Analysis |
|--------|----------|----------|----------|
| WER | <30% | ___% | |
| Training Time | 24-48h | ___h | |
| Memory Usage | <16GB | ___GB | |

### 9.3 Ablation Studies (Future)

| Variant | Modification | Expected Impact |
|---------|--------------|-----------------|
| ResNet backbone | Replace SimpleCNN | -2-3% WER |
| More frames | 128 → 256 | -1-2% WER |
| Larger batch | 2 → 4 | Faster training |
| More layers | 6 → 8 | -1% WER |

### 9.4 Limitations

1. **Single modality**: Uses only RGB frames (no optical flow, pose)
2. **Limited augmentation**: No temporal/spatial augmentation
3. **Greedy decoding**: No beam search or language model
4. **German only**: Trained on DGS, not generalizable to other sign languages

---

## 10. Future Work

### 10.1 Short-term Improvements (High Impact, Low Effort)

| Improvement | Expected Gain | Effort |
|-------------|---------------|--------|
| ResNet50 backbone | -2-3% WER | 1 day |
| Temporal augmentation | -1-2% WER | 1 day |
| Beam search decoding | -0.5-1% WER | 2 days |
| Mixed precision (FP16) | 2× training speed | 1 day |

### 10.2 Medium-term Improvements

| Improvement | Expected Gain | Effort |
|-------------|---------------|--------|
| TCN + Transformer hybrid | -2-4% WER | 1 week |
| Visual Alignment Constraint (VAC) | -1-2% WER | 1 week |
| Two-stream (RGB + Flow) | -3-5% WER | 2 weeks |

### 10.3 Long-term Research Directions

1. **Multi-modal Fusion**: Combine RGB, optical flow, and pose keypoints
2. **Self-supervised Pretraining**: Use VideoMAE or similar
3. **Cross-lingual Transfer**: Apply to ASL after DGS training
4. **Real-time Deployment**: Optimize for edge devices
5. **Sign Language Translation**: Extend from recognition to translation

---

## 11. Conclusion

This project demonstrates the application of Transformer architectures with CTC loss for Continuous Sign Language Recognition on the RWTH-PHOENIX-Weather 2014 dataset. Key contributions include:

1. **Architecture Design**: CNN backbone + Transformer encoder effectively captures both spatial and temporal features
2. **Implementation Insights**: Documented solutions for CTC collapse, memory optimization, and Windows compatibility
3. **Reproducibility**: Complete codebase with clear documentation
4. **Foundation for Future Work**: Modular design enables easy experimentation

The project validates Transformer + CTC as a viable approach for CSLR, achieving [___% WER] compared to the state-of-the-art of 17.8% and our target of <30%.

---

## 12. References

1. Koller, O., Forster, J., & Ney, H. (2015). Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers. *Computer Vision and Image Understanding*, 141, 108-125.

2. Camgöz, N. C., Hadfield, S., Koller, O., Ney, H., & Bowden, R. (2020). Sign language transformers: Joint end-to-end sign language recognition and translation. *CVPR 2020*.

3. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017*.

4. Graves, A., et al. (2006). Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks. *ICML 2006*.

5. Hu, H., et al. (2023). Continuous Sign Language Recognition with Correlation Network. *CVPR 2023*.

6. Zhu, Z., et al. (2021). VAC: Visual Alignment Constraint for Continuous Sign Language Recognition. *ICCV 2021*.

---

## Appendix A: Code Structure

```
PHOENIX-SLR/
├── src/
│   ├── models/
│   │   └── transformer.py    # SignLanguageTransformer, CNNBackbone
│   ├── data/
│   │   └── phoenix_dataset.py # PhoenixDataset, collate_fn
│   └── utils/
├── scripts/
│   └── download_phoenix.py   # Dataset download script
├── configs/
│   └── transformer_ctc.yaml  # Training configuration
├── train.py                  # Training script with CTC loss
├── evaluate.py               # Evaluation script (WER calculation)
├── checkpoints/              # Saved models
├── data/                     # Dataset directory
├── README.md                 # Project overview
├── FUTURE_SCOPE.md          # Detailed future improvements
└── TECHNICAL_PAPER_NOTES.md # This document
```

---

## Appendix B: Key Equations

### B.1 Self-Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### B.2 Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### B.3 CTC Loss

```
L_CTC = -log P(Y|X) = -log Σ_π P(π|X)
```

### B.4 Word Error Rate

```
WER = (S + I + D) / N × 100%
```

---

## Appendix C: Presentation Outline

### Slide 1: Title
- Project title, your name, semester, institution

### Slide 2: Problem Statement
- Communication barrier for deaf community
- Need for automated SLR systems

### Slide 3: Motivation
- Previous project challenges
- Why PHOENIX dataset

### Slide 4: Architecture Overview
- Diagram of CNN + Transformer + CTC

### Slide 5: Implementation Highlights
- Key challenges and solutions
- Memory optimization

### Slide 6: Dataset
- PHOENIX statistics
- Sample frames

### Slide 7: Results
- Training curves
- WER comparison table

### Slide 8: Demo (if applicable)
- Video of system in action

### Slide 9: Future Work
- Planned improvements
- Research directions

### Slide 10: Conclusion
- Summary of contributions
- Acknowledgments

---

*Document Version: 1.0*
*Last Updated: December 16, 2025*
*Author: [Your Name]*

