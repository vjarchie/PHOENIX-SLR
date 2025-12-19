# Architecture Evolution: From Pure CTC to Hybrid CTC+Attention

> **Document Purpose**: Technical record of architectural decisions and the evolution from pure CTC to Hybrid CTC+Attention approach for the PHOENIX Sign Language Recognition project.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Initial Approach: Pure CTC](#2-initial-approach-pure-ctc)
3. [CTC Collapse Problem](#3-ctc-collapse-problem)
4. [Attempted Fixes](#4-attempted-fixes)
5. [Decision to Switch](#5-decision-to-switch)
6. [Final Architecture: Hybrid CTC+Attention](#6-final-architecture-hybrid-ctcattention)
7. [Training Configuration](#7-training-configuration)
8. [Lessons Learned](#8-lessons-learned)

---

## 1. Executive Summary

### The Problem
Pure CTC (Connectionist Temporal Classification) suffered from **catastrophic collapse** - the model learned to output only blank tokens, resulting in 100% WER regardless of input.

### The Solution
Switched to **Hybrid CTC + Attention Decoder** architecture that provides stable gradients through cross-entropy loss while maintaining CTC's alignment benefits.

### Key Insight
> "CTC alone lacks output dependency modeling and is prone to collapse. The attention decoder provides regularization that prevents the encoder from collapsing to trivial solutions."

---

## 2. Initial Approach: Pure CTC

### Architecture (v1)

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT VIDEO FRAMES                        │
│                     (B, T, 210, 260, 3)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  ResNet-18 CNN Backbone                      │
│              (Pretrained on ImageNet)                        │
│                    Output: (B, T, 512)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Transformer Encoder (6 layers)                 │
│                  8 heads, d_model=512                        │
│                    Output: (B, T, 512)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      CTC Head                                │
│                 Linear(512 → vocab_size)                     │
│                      Log Softmax                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        CTC Loss
```

### Configuration
- **Model**: SignLanguageTransformer
- **Parameters**: 31.2M
- **Loss**: CTCLoss only
- **Optimizer**: AdamW (lr=0.0001)
- **Scheduler**: CosineAnnealingLR

### Initial Results
| Epoch | Train Loss | Dev Loss | WER | Blank Ratio |
|-------|------------|----------|-----|-------------|
| 1 | 5.34 | 5.16 | 91.6% | 98.3% |
| 2 | 5.06 | 4.99 | 91.4% | 97.7% |
| 3 | 5.00 | 5.11 | 91.3% | 97.7% |
| 4 | 5.04 | 5.11 | 98.2% | 97.8% |
| **5** | **5.43** | **5.66** | **100%** | **99.9%** |

**Observation**: Model learned briefly (WER: 91.6% → 91.3%) but collapsed at epoch 5.

---

## 3. CTC Collapse Problem

### What is CTC Collapse?

CTC collapse occurs when the model learns to output the **blank token** for every frame, regardless of input. This is a degenerate solution that minimizes CTC loss trivially.

### Symptoms
1. **Blank ratio → 100%**: Model predicts blank for every frame
2. **WER = 100%**: No glosses predicted after blank removal
3. **Loss plateaus**: No further learning occurs
4. **Predictions are empty**: `[]` for every input

### Why Pure CTC Collapsed

| Factor | Impact |
|--------|--------|
| **Conditional Independence** | CTC assumes output tokens are independent given input - can't model "MORGEN REGEN" dependencies |
| **Blank Dominance** | Blank token is always a valid output, easy local minimum |
| **No Output Supervision** | Only alignment loss, no token-level supervision |
| **Learning Rate Sensitivity** | Collapsed when LR reached peak (0.0003) |

### Evidence from Training Log

```
Epoch 3: WER=91.3%, blank=97.7%  ← Model was learning
Epoch 4: WER=98.2%, blank=97.8%  ← Started degrading  
Epoch 5: WER=100%, blank=99.9%   ← COLLAPSED
```

---

## 4. Attempted Fixes

### Fix 1: Learning Rate Warmup
**Hypothesis**: Gradual LR increase prevents early collapse.

```python
# Warmup for 5 epochs, then cosine decay
def lr_lambda(epoch):
    if epoch < 5:
        return 0.1 + 0.9 * (epoch / 5)
    else:
        return 0.5 * (1 + cos(π * progress))
```

**Result**: ❌ Delayed collapse but didn't prevent it (collapsed at epoch 5 instead of epoch 3).

### Fix 2: Higher Peak Learning Rate
**Hypothesis**: Higher LR helps escape local minimum.

```python
optimizer = AdamW(lr=0.0001 * 3)  # 3x higher
```

**Result**: ❌ Collapsed faster when LR reached peak.

### Fix 3: ResNet-18 Pretrained Backbone
**Hypothesis**: Better visual features prevent collapse.

```python
resnet = models.resnet18(weights='IMAGENET1K_V1')
```

**Result**: ⚠️ Partial improvement - blank ratio fluctuated (94-100%) instead of stuck at 100%, but still collapsed eventually.

### Fix 4: Reduced Sequence Length
**Hypothesis**: Shorter sequences are easier to align.

```python
max_frames = 64  # Reduced from 300
```

**Result**: ❌ Memory improved but collapse still occurred.

### Summary of Fix Attempts

| Fix | Expected Impact | Actual Result |
|-----|-----------------|---------------|
| LR Warmup | Prevent early collapse | Delayed but didn't prevent |
| Higher LR | Escape local minimum | Faster collapse |
| ResNet Backbone | Better features | Marginal improvement |
| Shorter sequences | Easier alignment | No improvement |

**Conclusion**: Pure CTC architecture is fundamentally prone to collapse for this task.

---

## 5. Decision to Switch

### External Validation

Consulted Gemini AI which confirmed:
> "CTC/Attention Joint Training: Add a CTC loss component to your existing attention-based model. The CTC branch encourages the encoder to produce monotonic and well-segmented features."

### Literature Review

| Paper | Approach | WER | Notes |
|-------|----------|-----|-------|
| Pure CTC | CTC only | 26-30% | Prone to collapse |
| Pure Attention | Seq2Seq | 28-35% | Alignment issues |
| **Sign Language Transformers** | **CTC + Attention** | **21-24%** | Joint training |
| VAC | CTC + Alignment Loss | 19% | Auxiliary supervision |

### Key Decision Factors

1. **Stability**: Decoder provides stable gradients even when CTC collapses
2. **Published Results**: SOTA papers use hybrid approach
3. **Output Modeling**: Decoder can model gloss-to-gloss dependencies
4. **Fallback**: Can use either CTC or decoder at inference

---

## 6. Final Architecture: Hybrid CTC+Attention

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT VIDEO FRAMES                          │
│                       (B, T, 210, 260, 3)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ResNet-18 CNN Backbone                          │
│                (Pretrained on ImageNet)                          │
│                     Output: (B, T, 512)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               Transformer Encoder (6 layers)                     │
│                    8 heads, d_model=512                          │
│                 **SHARED** between heads                         │
│                     Output: (B, T, 512)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│      CTC HEAD           │     │   ATTENTION DECODER     │
│                         │     │      (3 layers)         │
│  Linear(512 → vocab)    │     │                         │
│  Log Softmax            │     │  Cross-Attention to     │
│                         │     │  encoder outputs        │
│  Weight: 0.3            │     │  Weight: 0.7            │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              ▼                               ▼
         CTC Loss                    Cross-Entropy Loss
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       JOINT LOSS                                 │
│                                                                  │
│          L = 0.3 × L_CTC + 0.7 × L_CrossEntropy                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Model Components

#### 1. Shared Encoder
- **CNN Backbone**: ResNet-18 (pretrained)
- **Transformer Encoder**: 6 layers, 8 heads, d_model=512
- Processes video frames into contextualized features
- Shared between CTC and Attention heads

#### 2. CTC Head
- Simple linear projection to vocabulary
- Used for fast inference at test time
- Provides alignment signal during training
- Weight in loss: **0.3**

#### 3. Attention Decoder (GlossDecoder)
- 3-layer Transformer decoder
- Cross-attention to encoder outputs
- Autoregressive generation
- Models output dependencies
- Weight in loss: **0.7**

### Why This Works

| Component | Contribution |
|-----------|--------------|
| **CTC Head** | Encourages monotonic alignment, provides frame-level supervision |
| **Decoder** | Models output dependencies, prevents collapse via stable CE loss |
| **Shared Encoder** | Benefits from both supervision signals |
| **Joint Loss** | Balances alignment (CTC) and generation (CE) |

### Model Statistics

| Metric | Pure CTC | Hybrid |
|--------|----------|--------|
| Total Parameters | 31.2M | 45.2M |
| Trainable Parameters | 30.5M | 44.5M |
| Encoder Layers | 6 | 6 |
| Decoder Layers | 0 | 3 |
| Inference Mode | CTC only | CTC or Attention |

---

## 7. Training Configuration

### Current Setup

```python
# Model
model = HybridCTCAttentionModel(
    vocab_size=1236,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=3,
    ctc_weight=0.3,
    use_resnet=True
)

# Optimizer
optimizer = AdamW(lr=0.0001, weight_decay=0.01)

# Scheduler (Warmup + Cosine)
warmup_epochs = 3
scheduler = LambdaLR(optimizer, lr_lambda)

# Data
batch_size = 2
max_frames = 64
num_workers = 2
```

### Loss Function

```python
def compute_loss(ctc_log_probs, decoder_logits, ...):
    # CTC Loss (alignment)
    ctc_loss = F.ctc_loss(ctc_log_probs, ctc_targets, ...)
    
    # Cross-Entropy Loss (generation)
    ce_loss = F.cross_entropy(decoder_logits, decoder_targets, ...)
    
    # Joint Loss
    total_loss = 0.3 * ctc_loss + 0.7 * ce_loss
    
    return total_loss
```

### Special Tokens

| Token | Index | Purpose |
|-------|-------|---------|
| `<pad>` | 0 | Padding |
| `<unk>` | 1 | Unknown words |
| `<blank>` | 2 | CTC blank |
| `<sos>` | 3 | Start of sequence (decoder) |
| `<eos>` | 4 | End of sequence (decoder) |

---

## 8. Lessons Learned

### Technical Lessons

1. **CTC is fragile**: Without auxiliary supervision, CTC easily collapses to blank-only predictions.

2. **Pretrained backbones help but aren't enough**: ResNet-18 improved features but couldn't prevent collapse alone.

3. **Learning rate is critical**: CTC is highly sensitive to LR - collapsed when LR reached peak.

4. **Hybrid approaches are more robust**: The decoder's cross-entropy loss provides stable gradients regardless of CTC's behavior.

5. **Monitor blank ratio**: Blank ratio > 95% sustained over epochs indicates impending or ongoing collapse.

### Process Lessons

1. **Document failures**: The failed CTC attempts informed the switch to hybrid.

2. **Check literature**: Published papers (Sign Language Transformers) already solved this with hybrid approaches.

3. **Validate with peers**: External AI (Gemini) confirmed the hybrid approach.

4. **Implement monitoring**: Dashboard and blank ratio tracking caught collapse early.

### Future Recommendations

1. **Start with hybrid**: For CTC-based sequence tasks, start with hybrid architecture.

2. **Add auxiliary losses**: VAC, self-distillation, or frame-level losses can further stabilize training.

3. **Dynamic loss weighting**: Consider shifting from CE-heavy to CTC-heavy as training progresses.

4. **Curriculum learning**: Start with shorter, simpler sequences before full dataset.

---

## Appendix: Key Files

| File | Purpose |
|------|---------|
| `src/models/transformer.py` | Model definitions (HybridCTCAttentionModel, GlossDecoder) |
| `src/data/phoenix_dataset.py` | Dataset with <sos>/<eos> token support |
| `train_hybrid.py` | Hybrid training script |
| `train.py` | Original CTC-only training (deprecated) |
| `HYBRID_APPROACH.md` | Detailed hybrid implementation guide |

---

## Timeline

| Date | Event |
|------|-------|
| Dec 16, 2025 | Started with pure CTC approach |
| Dec 16, 2025 | CTC collapse observed after epoch 5 |
| Dec 16, 2025 | Attempted fixes (warmup, LR, ResNet) |
| Dec 17, 2025 | Decided to switch to hybrid approach |
| Dec 17, 2025 | Implemented HybridCTCAttentionModel |
| Dec 17, 2025 | Started hybrid training |

---

*Document created: December 17, 2025*
*Project: PHOENIX Sign Language Recognition - Semester III*

