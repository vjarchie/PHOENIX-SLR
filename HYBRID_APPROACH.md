# Hybrid CTC + Attention Decoder Approach

> **Alternative Architecture for Sign Language Recognition**
> 
> Use this approach if pure CTC continues to collapse (blank ratio >95%)

---

## Table of Contents

1. [Why Hybrid?](#1-why-hybrid)
2. [Architecture Overview](#2-architecture-overview)
3. [Implementation Guide](#3-implementation-guide)
4. [Training Strategy](#4-training-strategy)
5. [Expected Results](#5-expected-results)
6. [Code Implementation](#6-code-implementation)

---

## 1. Why Hybrid?

### Problems with Pure CTC

| Issue | Description | Impact |
|-------|-------------|--------|
| **Blank Collapse** | Model outputs mostly blank tokens | WER = 100% |
| **Conditional Independence** | CTC assumes outputs are independent | Can't model "MORGEN REGEN" dependencies |
| **No Output Modeling** | Doesn't learn gloss-to-gloss relationships | Poor sequence coherence |

### Why Hybrid Works Better

| Benefit | CTC | Attention Decoder | Hybrid |
|---------|-----|-------------------|--------|
| Fast inference | ✅ | ❌ | ✅ (use CTC) |
| Output dependencies | ❌ | ✅ | ✅ |
| Monotonic alignment | ✅ | ❌ | ✅ |
| No collapse | ❌ | ✅ | ✅ |
| Published SOTA | ❌ | ❌ | ✅ |

### Evidence from Literature

| Paper | Approach | WER | Notes |
|-------|----------|-----|-------|
| Pure CTC | CTC only | 26-30% | Prone to collapse |
| Pure Attention | Seq2Seq | 28-35% | Alignment issues |
| **Sign Language Transformers** | CTC + Attention | **21-24%** | Joint training |
| **VAC** | CTC + Alignment Loss | **19%** | Auxiliary supervision |

---

## 2. Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT VIDEO FRAMES                          │
│                       (T × H × W × 3)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CNN BACKBONE (ResNet18)                      │
│                    Extracts visual features                      │
│                       Output: (T × 512)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TRANSFORMER ENCODER (Shared)                    │
│                     6 layers, 8 heads                            │
│                       Output: (T × 512)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│      CTC HEAD           │     │   ATTENTION DECODER     │
│                         │     │                         │
│  Linear(512 → vocab)    │     │  Cross-Attention to     │
│  Log Softmax            │     │  encoder outputs        │
│                         │     │                         │
│  Loss: CTC Loss         │     │  Autoregressive         │
│  Inference: Fast        │     │  Loss: Cross-Entropy    │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       JOINT TRAINING                             │
│                                                                  │
│     Total Loss = λ_ctc × CTC_Loss + λ_ce × CrossEntropy_Loss    │
│                                                                  │
│     Default: λ_ctc = 0.3, λ_ce = 0.7                            │
└─────────────────────────────────────────────────────────────────┘
```

### Components

#### 1. Shared Encoder
- Same as current: CNN + Transformer
- Processes video frames into contextualized features
- Shared between CTC and Attention heads

#### 2. CTC Head (Recognition)
- Simple linear projection to vocabulary
- Used for fast inference at test time
- Provides alignment signal during training

#### 3. Attention Decoder (Translation/Generation)
- Transformer decoder with cross-attention
- Autoregressive: generates one token at a time
- Models output dependencies (gloss sequences)

---

## 3. Implementation Guide

### 3.1 Attention Decoder Module

```python
class GlossDecoder(nn.Module):
    """
    Autoregressive Transformer Decoder for gloss sequence generation.
    
    Uses cross-attention to encoder outputs and self-attention for
    modeling output dependencies.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 50
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Causal mask for autoregressive decoding
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len) * float('-inf'), diagonal=1)
        )
    
    def forward(
        self,
        encoder_output: torch.Tensor,
        target_ids: torch.Tensor,
        encoder_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, src_len, d_model) - from encoder
            target_ids: (batch, tgt_len) - target gloss IDs (shifted right)
            encoder_padding_mask: (batch, src_len) - padding mask for encoder
            
        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        batch_size, tgt_len = target_ids.shape
        
        # Embed target tokens
        positions = torch.arange(tgt_len, device=target_ids.device).unsqueeze(0)
        tgt_embed = self.embedding(target_ids) + self.pos_embedding(positions)
        
        # Causal mask for autoregressive decoding
        tgt_mask = self.causal_mask[:tgt_len, :tgt_len]
        
        # Decode
        decoder_output = self.decoder(
            tgt_embed,
            encoder_output,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=encoder_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_proj(decoder_output)
        
        return logits
    
    def generate(
        self,
        encoder_output: torch.Tensor,
        max_len: int = 50,
        start_token: int = 1,  # <sos>
        end_token: int = 2     # <eos>
    ) -> torch.Tensor:
        """
        Autoregressive generation at inference time.
        
        Args:
            encoder_output: (batch, src_len, d_model)
            max_len: Maximum output length
            start_token: Start-of-sequence token ID
            end_token: End-of-sequence token ID
            
        Returns:
            generated: (batch, seq_len) - generated token IDs
        """
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Start with <sos> token
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            # Get logits for next token
            logits = self.forward(encoder_output, generated)
            next_token_logits = logits[:, -1, :]  # (batch, vocab)
            
            # Greedy selection
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have produced <eos>
            if (next_token == end_token).all():
                break
        
        return generated
```

### 3.2 Hybrid Model

```python
class HybridSignLanguageModel(nn.Module):
    """
    Hybrid CTC + Attention model for Sign Language Recognition.
    
    Combines:
    - CTC head for alignment and fast inference
    - Attention decoder for output modeling and training stability
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        ctc_weight: float = 0.3
    ):
        super().__init__()
        
        self.ctc_weight = ctc_weight
        self.ce_weight = 1.0 - ctc_weight
        
        # Shared encoder (CNN + Transformer)
        self.cnn_backbone = CNNBackbone(output_dim=d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # CTC head
        self.ctc_proj = nn.Linear(d_model, vocab_size)
        
        # Attention decoder
        self.decoder = GlossDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Special tokens
        self.sos_token = vocab_size - 2  # <sos>
        self.eos_token = vocab_size - 1  # <eos>
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video frames to contextualized features."""
        # CNN features
        x = self.cnn_backbone(x)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.encoder(x)
        
        return x
    
    def forward(
        self,
        frames: torch.Tensor,
        target_ids: torch.Tensor = None,
        frame_lengths: torch.Tensor = None
    ):
        """
        Forward pass for training.
        
        Args:
            frames: (batch, seq_len, H, W, C) - input video
            target_ids: (batch, tgt_len) - target gloss IDs
            frame_lengths: (batch,) - actual frame lengths
            
        Returns:
            ctc_logits: For CTC loss
            decoder_logits: For cross-entropy loss
            encoder_output: For other uses
        """
        # Encode
        encoder_output = self.encode(frames)
        
        # CTC head
        ctc_logits = self.ctc_proj(encoder_output)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
        
        # Decoder (if targets provided)
        decoder_logits = None
        if target_ids is not None:
            # Shift right: add <sos> at start, remove last token
            decoder_input = self._shift_right(target_ids)
            decoder_logits = self.decoder(encoder_output, decoder_input)
        
        return ctc_log_probs, decoder_logits, encoder_output
    
    def _shift_right(self, target_ids: torch.Tensor) -> torch.Tensor:
        """Shift target IDs right and prepend <sos>."""
        batch_size = target_ids.size(0)
        sos = torch.full((batch_size, 1), self.sos_token, device=target_ids.device)
        shifted = torch.cat([sos, target_ids[:, :-1]], dim=1)
        return shifted
    
    def compute_loss(
        self,
        ctc_log_probs: torch.Tensor,
        decoder_logits: torch.Tensor,
        target_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        ctc_blank: int
    ):
        """
        Compute joint CTC + Cross-Entropy loss.
        """
        # CTC loss
        ctc_loss = F.ctc_loss(
            ctc_log_probs,
            target_ids,
            input_lengths,
            target_lengths,
            blank=ctc_blank,
            zero_infinity=True
        )
        
        # Cross-entropy loss for decoder
        # Flatten for cross-entropy
        ce_logits = decoder_logits.view(-1, decoder_logits.size(-1))
        ce_targets = target_ids.view(-1)
        ce_loss = F.cross_entropy(ce_logits, ce_targets, ignore_index=0)  # ignore padding
        
        # Combined loss
        total_loss = self.ctc_weight * ctc_loss + self.ce_weight * ce_loss
        
        return total_loss, ctc_loss, ce_loss
```

---

## 4. Training Strategy

### 4.1 Loss Weighting Schedule

```python
# Start with more CE weight, gradually shift to CTC
def get_loss_weights(epoch, total_epochs):
    """
    Dynamic loss weighting:
    - Early: More CE (decoder) to learn patterns
    - Later: More CTC for alignment
    """
    progress = epoch / total_epochs
    
    if progress < 0.3:
        # Early training: focus on decoder
        ctc_weight = 0.1
    elif progress < 0.7:
        # Mid training: balanced
        ctc_weight = 0.3
    else:
        # Late training: more CTC
        ctc_weight = 0.5
    
    return ctc_weight, 1.0 - ctc_weight
```

### 4.2 Training Loop Modifications

```python
def train_hybrid_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    model.train()
    
    ctc_weight, ce_weight = get_loss_weights(epoch, total_epochs)
    model.ctc_weight = ctc_weight
    model.ce_weight = ce_weight
    
    for batch in dataloader:
        frames = batch['frames'].to(device)
        target_ids = batch['gloss_ids'].to(device)
        frame_lengths = batch['frame_lengths']
        target_lengths = batch['gloss_lengths']
        
        optimizer.zero_grad()
        
        # Forward
        ctc_log_probs, decoder_logits, _ = model(frames, target_ids, frame_lengths)
        
        # Loss
        total_loss, ctc_loss, ce_loss = model.compute_loss(
            ctc_log_probs, decoder_logits, target_ids,
            frame_lengths, target_lengths, blank_idx
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
```

### 4.3 Inference Options

```python
def inference(model, frames, mode='ctc'):
    """
    Two inference modes:
    - 'ctc': Fast, uses CTC greedy decoding
    - 'attention': Slower, uses autoregressive decoder
    """
    model.eval()
    
    with torch.no_grad():
        encoder_output = model.encode(frames)
        
        if mode == 'ctc':
            # Fast CTC decoding
            ctc_logits = model.ctc_proj(encoder_output)
            predictions = ctc_greedy_decode(ctc_logits)
        else:
            # Autoregressive decoding
            predictions = model.decoder.generate(encoder_output)
    
    return predictions
```

---

## 5. Expected Results

### Performance Comparison

| Approach | Expected WER | Training Time | Inference Speed |
|----------|--------------|---------------|-----------------|
| Pure CTC (current) | 25-35% (if no collapse) | 1x | Fast |
| Pure Attention | 28-35% | 1.5x | Slow |
| **Hybrid** | **20-28%** | 1.3x | Fast (CTC mode) |

### Why Hybrid Avoids Collapse

1. **Decoder provides supervision**: Even if CTC collapses, decoder loss keeps training
2. **Shared encoder learns**: Encoder gets gradients from both heads
3. **Cross-entropy is stable**: No blank collapse in decoder
4. **Gradual transition**: Start with CE, then add CTC

---

## 6. Code Implementation

### 6.1 Required Changes to Vocabulary

```python
# Add special tokens to vocabulary
vocab = {
    '<pad>': 0,
    '<unk>': 1,
    '<blank>': 2,  # For CTC
    '<sos>': 3,    # Start of sequence (for decoder)
    '<eos>': 4,    # End of sequence (for decoder)
    # ... actual glosses ...
}
```

### 6.2 Dataset Modifications

```python
def __getitem__(self, idx):
    # ... existing code ...
    
    # Add <eos> to target for decoder training
    gloss_ids_with_eos = gloss_ids + [self.vocab['<eos>']]
    
    return {
        'frames': frames,
        'gloss_ids': gloss_ids,              # For CTC (no special tokens)
        'gloss_ids_decoder': gloss_ids_with_eos,  # For decoder (with <eos>)
        # ...
    }
```

### 6.3 Full Training Script Changes

```python
# In train.py

# 1. Update model creation
if args.model_type == 'hybrid':
    model = HybridSignLanguageModel(
        vocab_size=len(train_dataset.vocab) + 2,  # +2 for <sos>, <eos>
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=3,
        ctc_weight=0.3
    )

# 2. Update training loop
for epoch in range(args.epochs):
    # Dynamic loss weights
    ctc_w, ce_w = get_loss_weights(epoch, args.epochs)
    model.ctc_weight = ctc_w
    model.ce_weight = ce_w
    
    train_loss = train_hybrid_epoch(...)
    
    # Log both losses
    logger.info(f"Epoch {epoch}: Loss={train_loss:.4f}, CTC_w={ctc_w:.2f}, CE_w={ce_w:.2f}")
```

---

## 7. When to Use Hybrid

### Switch to Hybrid If:

| Condition | Threshold |
|-----------|-----------|
| Blank ratio after 5 epochs | > 95% |
| WER after 10 epochs | Still 100% |
| Loss not decreasing | Stuck for 3+ epochs |

### Keep Current CTC If:

| Condition | Threshold |
|-----------|-----------|
| Blank ratio decreasing | < 90% by epoch 5 |
| WER improving | < 90% by epoch 10 |
| Loss decreasing steadily | Consistent drop |

---

## 8. Implementation Checklist

If switching to hybrid approach:

- [ ] Add `<sos>` and `<eos>` to vocabulary
- [ ] Modify dataset to return decoder targets
- [ ] Implement `GlossDecoder` class
- [ ] Implement `HybridSignLanguageModel` class
- [ ] Update training loop for joint loss
- [ ] Add loss weight scheduling
- [ ] Update evaluation for both modes
- [ ] Test inference in both CTC and attention modes

---

## 9. References

1. **Sign Language Transformers** (Camgöz et al., 2020)
   - Joint CTC + Attention for CSLR
   - SOTA on PHOENIX at time of publication

2. **Listen, Attend and Spell** (Chan et al., 2016)
   - Original hybrid CTC + Attention for speech
   - Foundation for sequence transduction

3. **Hybrid CTC/Attention Architecture** (Watanabe et al., 2017)
   - Analysis of CTC vs Attention trade-offs
   - Joint training strategies

---

*Document created: December 16, 2025*
*Use if current CTC training continues to collapse after epoch 5-10*

