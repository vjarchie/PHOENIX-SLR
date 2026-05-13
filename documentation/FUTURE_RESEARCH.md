# Future Research Directions

This document captures promising research directions for future exploration.

---

## 1. Wave Field LLM - O(n log n) Attention Alternative

**Source**: 
- GitHub: https://github.com/badaramoni/wave-field-llm
- Discussion: https://discuss.huggingface.co/t/wave-field-llm-o-n-log-n-attention-via-wave-equation-dynamics-within-5-of-standard-transformer/173625

**Published**: February 2026

### Overview

Wave Field LLM replaces O(n²) self-attention with wave equation dynamics, achieving O(n log n) complexity while staying within 5% of standard transformer quality.

### Key Concept

Each attention head is a damped wave oscillation:

```python
k(t) = exp(-α·t) · cos(ω·t + φ)    # for t ≥ 0 (causal)

# Parameters:
# ω (frequency) - Oscillation speed, controls attention pattern periodicity
# α (damping)   - Decay rate, controls how far back to attend
# φ (phase)     - Offset for head diversity
```

Convolution is computed via FFT in O(n log n).

### Computational Savings

| Sequence Length | Standard O(n²) | Wave O(n log n) | Savings |
|-----------------|----------------|-----------------|---------|
| 512             | 134M ops       | 14.3M ops       | 9x      |
| 2,048           | 2.1B ops       | 68M ops         | 31x     |
| 8,192           | 34B ops        | 319M ops        | 107x    |
| 32,768          | 550B ops       | 1.5B ops        | 367x    |

### Results (WikiText-2, 6M params)

| Model                | Test PPL | Test Acc | Complexity |
|----------------------|----------|----------|------------|
| Standard Transformer | 5.9      | 51.0%    | O(n²)      |
| Wave Field V3.5      | 6.2      | 50.5%    | O(n log n) |

### Architecture

```
Input tokens
    │
[Token Embedding + Sinusoidal Position Encoding]
    │
[Wave Field Layer ×N]
    │── Pre-norm
    │── Wave Field Attention:
    │     │── QKV projection
    │     │── Absolute position mapping (token_i → field_pos = i × stride)
    │     │── Bilinear scatter (deposit values onto continuous field)
    │     │── Wave convolution via FFT — O(n log n)
    │     │── Static multi-field coupling (cross-head interactions)
    │     │── Content-dependent gating
    │     │── Bilinear gather (read from field)
    │── Pre-norm FFN (GELU)
    │── Field Interference (every 3 layers)
    │
[LayerNorm → Output Projection]
    │
Output
```

### Potential for Sign Language Recognition

**Pros:**
- Video sequences are long (100s of frames) → O(n log n) helps significantly
- Wave frequencies could naturally model motion patterns in signs
- Multi-scale temporal modeling via different frequency heads
- Heads self-organize into local (hand shape) vs long-range (sentence structure)

**Cons:**
- Only tested on text, not video/vision
- Requires significant architectural changes
- Experimental (February 2026)
- Would need adaptation for visual features

### Implementation Notes

```python
# Core wave kernel implementation (from their repo)
class WaveKernel(nn.Module):
    def __init__(self, num_heads, max_len):
        super().__init__()
        # 3 learnable parameters per head
        self.omega = nn.Parameter(torch.randn(num_heads))  # frequency
        self.alpha = nn.Parameter(torch.ones(num_heads))   # damping
        self.phi = nn.Parameter(torch.zeros(num_heads))    # phase
    
    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.omega.device).float()
        # k(t) = exp(-α·t) · cos(ω·t + φ)
        kernel = torch.exp(-self.alpha.abs().unsqueeze(1) * t) * \
                 torch.cos(self.omega.unsqueeze(1) * t + self.phi.unsqueeze(1))
        return kernel  # (num_heads, seq_len)
```

### To Try Later

1. Replace Transformer encoder with Wave Field encoder in our hybrid model
2. Test if wave dynamics help with sign motion patterns
3. Compare training speed and memory usage
4. Evaluate WER on PHOENIX-2014

---

## 2. Other Research to Explore

### 2.1 Conformer (ConSignformer)
- Combines CNN + Self-Attention in each layer
- Better for capturing both local and global patterns
- Source: https://arxiv.org/abs/2405.12018

### 2.2 SignGraph (CVPR 2024)
- Graph-based representation instead of grids
- Captures cross-region features
- Source: https://github.com/gswycf/SignGraph

### 2.3 Cross-Sentence Gloss Consistency (AAAI 2024)
- Prototype-based contrastive learning
- 1.6% improvement over previous SOTA
- Easy to add to existing architecture

---

## Status

| Research Direction | Status | Priority |
|-------------------|--------|----------|
| Wave Field LLM    | Documented | Future |
| CorrNet+          | In Progress | High |
| VAC Loss          | Tested (failed with hybrid) | Low |
| Extended Temporal | Pending | Medium |

---

*Last updated: January 2026*
