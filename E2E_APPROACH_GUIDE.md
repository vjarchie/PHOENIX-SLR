# End-to-End Hybrid CTC+Attention Approach for Sign Language Recognition

> **Purpose**: This document provides a complete guide to implementing the Hybrid CTC+Attention architecture that achieved **50.91% WER** on PHOENIX-2014. Use this as a reference for implementing similar systems on other sign language datasets.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Key Components](#3-key-components)
4. [Implementation Details](#4-implementation-details)
5. [Training Strategy](#5-training-strategy)
6. [Data Augmentation](#6-data-augmentation)
7. [Common Pitfalls & Solutions](#7-common-pitfalls--solutions)
8. [Hyperparameters](#8-hyperparameters)
9. [Code Templates](#9-code-templates)

---

## 1. Executive Summary

### The Problem with Pure CTC
Pure CTC (Connectionist Temporal Classification) for sign language recognition suffers from **CTC collapse** - the model learns to output only blank tokens, resulting in empty predictions and 100% WER.

### The Solution: Hybrid CTC+Attention
Combine CTC with an Attention-based decoder:
- **CTC Head (30%)**: Provides alignment signal, fast inference
- **Attention Decoder (70%)**: Models output dependencies, prevents collapse

### Results Achieved
| Metric | Value |
|--------|-------|
| Test WER | **50.91%** |
| Dev WER | 51.44% |
| Training Time | ~20 hours (100 epochs) |
| Model Size | 45.2M parameters |

---

## 2. Architecture Overview

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT VIDEO FRAMES                          │
│                       (B, T, C, H, W)                            │
│                    e.g., (4, 64, 3, 260, 210)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CNN BACKBONE (ResNet-18)                        │
│                                                                  │
│   • Pretrained on ImageNet (frozen first few epochs optional)   │
│   • Processes each frame independently                          │
│   • Output: (B, T, 512) - 512-dim features per frame            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               TRANSFORMER ENCODER (6 layers)                     │
│                                                                  │
│   • 8 attention heads, d_model=512                              │
│   • Positional encoding added                                    │
│   • Models temporal dependencies across frames                   │
│   • Output: (B, T, 512) - contextualized features               │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│      CTC HEAD           │     │   ATTENTION DECODER     │
│                         │     │      (3 layers)         │
│  Linear(512 → vocab)    │     │                         │
│  Log Softmax            │     │  Embedding + Pos Enc    │
│                         │     │  Cross-Attention        │
│  Loss Weight: 0.3       │     │  Linear → vocab         │
│                         │     │                         │
│  For: Alignment         │     │  Loss Weight: 0.7       │
│  Inference: Fast        │     │  For: Sequence modeling │
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

### Why This Works

| Component | Purpose | Without It |
|-----------|---------|------------|
| **CNN Backbone** | Extract rich visual features | Raw pixels too high-dimensional |
| **Transformer Encoder** | Model long-range temporal dependencies | Limited context window |
| **CTC Head** | Alignment signal, handles variable-length output | Needs explicit alignment |
| **Attention Decoder** | Model output dependencies, stable gradients | CTC collapse! |
| **Joint Loss** | Balance alignment and generation | One dominates |

---

## 3. Key Components

### 3.1 CNN Backbone (ResNet-18)

```python
class CNNBackbone(nn.Module):
    def __init__(self, output_dim=512, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Project to desired dimension
        self.proj = nn.Linear(512, output_dim)
    
    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        
        # Process all frames
        x = x.view(B * T, C, H, W)
        x = self.features(x)  # (B*T, 512, 1, 1)
        x = x.view(B * T, -1)
        x = self.proj(x)  # (B*T, output_dim)
        x = x.view(B, T, -1)  # (B, T, output_dim)
        
        return x
```

**Key Points**:
- Use **pretrained** weights (ImageNet) - critical for small datasets
- Process frames independently through CNN
- Temporal modeling happens in Transformer

### 3.2 Transformer Encoder

```python
class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_ff=2048, dropout=0.1):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, x, src_key_padding_mask=None):
        x = self.pos_encoder(x)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)
```

### 3.3 Attention Decoder (GlossDecoder)

```python
class GlossDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, encoder_output, target_ids, 
                encoder_padding_mask=None, target_padding_mask=None):
        # Embed targets
        tgt = self.embedding(target_ids) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        # Causal mask for autoregressive decoding
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            target_ids.size(1)
        ).to(target_ids.device)
        
        # Decode
        output = self.decoder(
            tgt, encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=encoder_padding_mask
        )
        
        return self.output_proj(output)
```

### 3.4 Joint Loss Function

```python
def compute_hybrid_loss(ctc_log_probs, decoder_logits, 
                        ctc_targets, decoder_targets,
                        input_lengths, target_lengths,
                        ctc_weight=0.3, blank_idx=0, pad_idx=0):
    
    # CTC Loss
    ctc_loss = F.ctc_loss(
        ctc_log_probs,      # (T, B, vocab)
        ctc_targets,         # (B, max_target_len)
        input_lengths,       # (B,)
        target_lengths,      # (B,)
        blank=blank_idx,
        zero_infinity=True   # Important for stability!
    )
    
    # Cross-Entropy Loss (with label smoothing)
    ce_loss = F.cross_entropy(
        decoder_logits.view(-1, decoder_logits.size(-1)),
        decoder_targets.view(-1),
        ignore_index=pad_idx,
        label_smoothing=0.1  # Helps generalization
    )
    
    # Combined loss
    total_loss = ctc_weight * ctc_loss + (1 - ctc_weight) * ce_loss
    
    return total_loss, ctc_loss.item(), ce_loss.item()
```

---

## 4. Implementation Details

### 4.1 Special Tokens

```python
vocab = {
    '<pad>': 0,    # Padding token
    '<unk>': 1,    # Unknown token
    '<blank>': 2,  # CTC blank token
    '<sos>': 3,    # Start of sequence (decoder input)
    '<eos>': 4,    # End of sequence (decoder target)
    # ... actual vocabulary starting at index 5
}
```

### 4.2 Data Preparation

**For CTC**:
- Input: frames (B, T, C, H, W)
- Target: gloss IDs without special tokens

**For Decoder**:
- Input: `<sos>` + gloss IDs (teacher forcing)
- Target: gloss IDs + `<eos>`

```python
# In dataset __getitem__:
return {
    'frames': frames,                    # (T, C, H, W)
    'ctc_targets': gloss_ids,           # For CTC: [g1, g2, g3]
    'decoder_input': [SOS] + gloss_ids, # [<sos>, g1, g2, g3]
    'decoder_target': gloss_ids + [EOS] # [g1, g2, g3, <eos>]
}
```

### 4.3 Collate Function

```python
def collate_fn(batch):
    # Pad frames to max length
    max_frames = max(x['frames'].shape[0] for x in batch)
    
    frames_padded = []
    frame_lengths = []
    
    for x in batch:
        T = x['frames'].shape[0]
        pad_len = max_frames - T
        if pad_len > 0:
            padding = torch.zeros(pad_len, *x['frames'].shape[1:])
            frames_padded.append(torch.cat([x['frames'], padding]))
        else:
            frames_padded.append(x['frames'])
        frame_lengths.append(T)
    
    # Similar padding for targets...
    
    return {
        'frames': torch.stack(frames_padded),
        'frame_lengths': torch.tensor(frame_lengths),
        'ctc_targets': padded_ctc_targets,
        'decoder_input': padded_decoder_input,
        'decoder_target': padded_decoder_target,
        # ... masks
    }
```

---

## 5. Training Strategy

### 5.1 Learning Rate Schedule

```python
# Warmup + Cosine Annealing
warmup_epochs = 5
total_epochs = 100

def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs  # Linear warmup
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))  # Cosine decay

scheduler = LambdaLR(optimizer, lr_lambda)
```

### 5.2 Optimizer Configuration

```python
# Separate LR for backbone (pretrained) vs rest (randomly initialized)
optimizer = AdamW([
    {'params': backbone_params, 'lr': base_lr * 0.1},  # Lower LR for pretrained
    {'params': other_params, 'lr': base_lr}
], weight_decay=0.05)
```

### 5.3 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

### 5.4 Early Stopping

```python
patience = 15
no_improve_count = 0

for epoch in range(epochs):
    # ... training ...
    
    if dev_wer < best_wer:
        best_wer = dev_wer
        no_improve_count = 0
        save_checkpoint(model)
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print("Early stopping!")
            break
```

---

## 6. Data Augmentation

### 6.1 Spatial Augmentation

```python
class VideoAugmentation:
    def __init__(self, training=True):
        self.training = training
        
        # Parameters (consistent across all frames)
        self.crop_scale = (0.85, 1.0)
        self.rotation = 8.0  # degrees
        self.brightness = 0.15
        self.contrast = 0.15
        self.noise_std = 0.01
    
    def __call__(self, frames):
        if not self.training:
            return frames
        
        # Sample parameters ONCE for entire video
        scale = random.uniform(*self.crop_scale)
        angle = random.uniform(-self.rotation, self.rotation)
        brightness = 1.0 + random.uniform(-self.brightness, self.brightness)
        
        # Apply to all frames consistently
        for i in range(len(frames)):
            frames[i] = self.augment_frame(frames[i], scale, angle, brightness)
        
        return frames
```

**Important**: Do NOT use horizontal flip for sign language - it changes the meaning!

### 6.2 Temporal Augmentation

```python
class TemporalSampling:
    def __init__(self, target_frames=64, speed_range=(0.8, 1.2)):
        self.target_frames = target_frames
        self.speed_range = speed_range
    
    def __call__(self, frames):
        T = len(frames)
        
        # Random speed perturbation
        speed = random.uniform(*self.speed_range)
        effective_frames = int(self.target_frames * speed)
        
        # Uniform sampling
        indices = np.linspace(0, T-1, effective_frames).astype(int)
        frames = frames[indices]
        
        # Resize to target
        if len(frames) != self.target_frames:
            indices = np.linspace(0, len(frames)-1, self.target_frames).astype(int)
            frames = frames[indices]
        
        return frames
```

### 6.3 Time Masking (SpecAugment-style)

```python
def time_mask(frames, mask_prob=0.15, mask_ratio=0.08):
    if random.random() > mask_prob:
        return frames
    
    T = len(frames)
    mask_len = max(1, int(T * mask_ratio))
    mask_start = random.randint(0, T - mask_len)
    
    frames[mask_start:mask_start + mask_len] = 0
    return frames
```

---

## 7. Common Pitfalls & Solutions

### 7.1 CTC Collapse

**Symptoms**:
- Blank ratio > 95%
- WER = 100%
- Loss plateaus

**Solutions**:
1. ✅ Use Hybrid CTC+Attention (this approach!)
2. ✅ Lower CTC weight (0.1-0.3)
3. ✅ Add label smoothing to CE loss
4. ✅ Learning rate warmup

### 7.2 Overfitting

**Symptoms**:
- Train loss decreasing, dev loss increasing
- WER plateaus or increases

**Solutions**:
1. ✅ Data augmentation (spatial + temporal)
2. ✅ Dropout (0.1-0.2)
3. ✅ Weight decay (0.01-0.05)
4. ✅ Label smoothing (0.1)
5. ✅ Early stopping

### 7.3 Out of Memory

**Solutions**:
1. Reduce batch size (4 → 2)
2. Reduce max_frames (128 → 64)
3. Use gradient accumulation
4. Use mixed precision (torch.cuda.amp)

### 7.4 Slow Training

**Solutions**:
1. Use pretrained CNN backbone
2. Freeze backbone for first few epochs
3. Use DataLoader with num_workers > 0
4. Pre-extract features if using I3D/S3D

---

## 8. Hyperparameters

### Final Configuration (PHOENIX-SLR)

```python
# Model
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 3
dim_feedforward = 2048
dropout = 0.2
vocab_size = 1236

# Training
batch_size = 4
max_frames = 64
learning_rate = 1e-4
weight_decay = 0.05
epochs = 100
warmup_epochs = 5
patience = 15

# Loss
ctc_weight = 0.3
ce_weight = 0.7
label_smoothing = 0.1

# Augmentation
crop_scale = (0.85, 1.0)
rotation = 8.0
speed_range = (0.8, 1.2)
time_mask_prob = 0.15
```

### Recommended Starting Points for Other Datasets

| Dataset Size | batch_size | max_frames | epochs | patience |
|--------------|------------|------------|--------|----------|
| < 1,000 | 2 | 32 | 200 | 20 |
| 1,000 - 5,000 | 4 | 64 | 100 | 15 |
| 5,000 - 20,000 | 8 | 64 | 50 | 10 |
| > 20,000 | 16 | 128 | 30 | 10 |

---

## 9. Code Templates

### 9.1 Complete Model Class

```python
class HybridCTCAttentionModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=3,
                 dropout=0.1, ctc_weight=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight
        
        # CNN Backbone
        self.backbone = CNNBackbone(output_dim=d_model, pretrained=True)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=2048,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # CTC Head
        self.ctc_proj = nn.Linear(d_model, vocab_size)
        
        # Attention Decoder
        self.decoder = GlossDecoder(
            vocab_size, d_model, nhead, num_decoder_layers, dropout
        )
    
    def encode(self, frames, src_mask=None):
        x = self.backbone(frames)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=src_mask)
        return x
    
    def forward(self, frames, target_ids, src_mask=None, tgt_mask=None):
        # Encode
        encoder_output = self.encode(frames, src_mask)
        
        # CTC output
        ctc_logits = self.ctc_proj(encoder_output)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
        
        # Decoder output
        decoder_logits = self.decoder(
            encoder_output, target_ids,
            encoder_padding_mask=src_mask,
            target_padding_mask=tgt_mask
        )
        
        return ctc_log_probs, decoder_logits
    
    @torch.no_grad()
    def greedy_decode(self, frames, max_len=50, sos_idx=3, eos_idx=4):
        self.eval()
        encoder_output = self.encode(frames)
        
        batch_size = frames.size(0)
        device = frames.device
        
        # Start with <sos>
        predictions = torch.full((batch_size, 1), sos_idx, device=device)
        
        for _ in range(max_len - 1):
            decoder_output = self.decoder(encoder_output, predictions)
            next_token = decoder_output[:, -1, :].argmax(dim=-1, keepdim=True)
            predictions = torch.cat([predictions, next_token], dim=1)
            
            if (next_token == eos_idx).all():
                break
        
        return predictions
```

### 9.2 Training Loop Template

```python
def train_epoch(model, dataloader, optimizer, device, ctc_weight=0.3):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        
        frames = batch['frames'].to(device)
        ctc_targets = batch['ctc_targets'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        decoder_target = batch['decoder_target'].to(device)
        frame_lengths = batch['frame_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)
        
        # Forward
        ctc_log_probs, decoder_logits = model(frames, decoder_input)
        
        # Loss
        loss, ctc_loss, ce_loss = compute_hybrid_loss(
            ctc_log_probs, decoder_logits,
            ctc_targets, decoder_target,
            frame_lengths, target_lengths,
            ctc_weight=ctc_weight
        )
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

---

## Summary

### Key Takeaways

1. **Always use Hybrid CTC+Attention** for sign language recognition - pure CTC will collapse
2. **Use pretrained CNN backbone** (ResNet-18, I3D, S3D) - don't train from scratch
3. **Data augmentation is essential** - but NO horizontal flip!
4. **Monitor blank ratio** - if > 90%, CTC is collapsing
5. **Lower CTC weight** (0.1-0.3) helps stability
6. **Label smoothing** (0.1) improves generalization
7. **Early stopping** prevents overfitting

### Expected Results by Dataset

| Dataset | Expected WER | Notes |
|---------|--------------|-------|
| PHOENIX-2014 | 45-55% | Weather domain, studio quality |
| How2Sign | 60-70% | Instructional videos |
| WLASL | 50-60% | Isolated words (easier) |
| OpenASL | 70-80% | YouTube videos (harder) |

---

*Document created from PHOENIX-SLR project achieving 50.91% WER*
*Author: AI Assistant (Claude)*
*Date: December 2024*


