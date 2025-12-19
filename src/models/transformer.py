# -*- coding: utf-8 -*-
"""
Transformer Encoder for Sign Language Recognition

Architecture:
- CNN Feature Extractor (for raw frames)
- Positional Encoding
- Multi-Head Self-Attention
- Feed-Forward Networks
- CTC Output Layer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CNNBackbone(nn.Module):
    """
    CNN Feature Extractor for video frames.
    
    Takes raw frames and extracts spatial features.
    """
    
    def __init__(self, output_dim: int = 512, pretrained: bool = True):
        super().__init__()
        
        # Use ResNet18 as backbone (lightweight but effective)
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Remove final FC and avgpool
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Adaptive pooling + projection
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, output_dim)  # ResNet18 outputs 512 features
        
        # Freeze early layers for transfer learning
        for param in list(self.features.parameters())[:30]:  # Freeze first few layers
            param.requires_grad = False
        
        print(f"CNNBackbone initialized: ResNet18 -> {output_dim}D features")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, H, W, C) or (batch, seq_len, C, H, W)
        Returns:
            (batch, seq_len, output_dim)
        """
        batch_size, seq_len = x.shape[:2]
        
        # Handle channel dimension - convert to (B*T, C, H, W)
        if x.shape[-1] == 3:  # (B, T, H, W, C)
            x = x.permute(0, 1, 4, 2, 3)  # -> (B, T, C, H, W)
        
        # Reshape for CNN: (B*T, C, H, W)
        x = x.contiguous().view(batch_size * seq_len, *x.shape[2:])
        
        # Extract features
        features = self.features(x)
        features = self.pool(features)
        features = features.view(batch_size * seq_len, -1)
        features = self.proj(features)
        
        # Reshape back: (B, T, D)
        features = features.view(batch_size, seq_len, -1)
        
        return features


class SimpleCNNBackbone(nn.Module):
    """
    Lightweight CNN backbone for faster training.
    """
    
    def __init__(self, input_channels: int = 3, output_dim: int = 512):
        super().__init__()
        
        self.conv = nn.Sequential(
            # Conv1: 210x260 -> 105x130
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # -> 53x65
            
            # Conv2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> 27x33
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Conv3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # -> 14x17
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # -> 7x9
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.proj = nn.Linear(512, output_dim)
        
        print(f"SimpleCNNBackbone initialized: {input_channels}ch -> {output_dim}D features")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, H, W, C) or (batch, seq_len, C, H, W)
        Returns:
            (batch, seq_len, output_dim)
        """
        batch_size, seq_len = x.shape[:2]
        
        # Handle channel dimension
        if x.shape[-1] == 3:  # (B, T, H, W, C)
            x = x.permute(0, 1, 4, 2, 3)  # -> (B, T, C, H, W)
        
        # Reshape: (B*T, C, H, W)
        x = x.contiguous().view(batch_size * seq_len, *x.shape[2:])
        
        # Extract features
        features = self.conv(x)
        features = features.view(batch_size * seq_len, -1)
        features = self.proj(features)
        
        # Reshape: (B, T, D)
        features = features.view(batch_size, seq_len, -1)
        
        return features


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SignLanguageTransformer(nn.Module):
    """
    Transformer-based Sign Language Recognition Model.
    
    Input: 
        - Raw video frames (batch, seq_len, H, W, C) if use_cnn_backbone=True
        - Pre-extracted features (batch, seq_len, feature_dim) if use_cnn_backbone=False
    Output: Log probabilities for CTC (seq_len, batch, vocab_size)
    """
    
    def __init__(
        self,
        input_dim: int = 512,          # Input feature dimension (after CNN if used)
        d_model: int = 512,            # Transformer hidden dimension
        nhead: int = 8,                # Number of attention heads
        num_encoder_layers: int = 6,   # Number of transformer layers
        dim_feedforward: int = 2048,   # FFN hidden dimension
        dropout: float = 0.1,
        vocab_size: int = 1200,        # Number of glosses
        max_seq_len: int = 500,
        use_cnn_backbone: bool = True,  # Whether to use CNN for raw frames
        cnn_type: str = 'simple'        # 'simple' or 'resnet'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_cnn_backbone = use_cnn_backbone
        
        # CNN backbone for raw frames
        if use_cnn_backbone:
            if cnn_type == 'resnet':
                self.cnn_backbone = CNNBackbone(output_dim=input_dim, pretrained=True)
            else:
                self.cnn_backbone = SimpleCNNBackbone(input_channels=3, output_dim=input_dim)
        else:
            self.cnn_backbone = None
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output projection for CTC
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        print(f"SignLanguageTransformer initialized:")
        print(f"  CNN backbone: {cnn_type if use_cnn_backbone else 'None'}")
        print(f"  Input dim: {input_dim}")
        print(f"  Model dim: {d_model}")
        print(f"  Heads: {nhead}")
        print(f"  Layers: {num_encoder_layers}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        x: torch.Tensor, 
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Raw frames (batch, seq_len, H, W, C) or features (batch, seq_len, input_dim)
            src_mask: Attention mask (seq_len, seq_len)
            src_key_padding_mask: Padding mask (batch, seq_len)
            
        Returns:
            Log probabilities (seq_len, batch, vocab_size) for CTC
        """
        # CNN backbone for raw frames
        if self.cnn_backbone is not None:
            x = self.cnn_backbone(x)
        
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(
            x, 
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Output projection
        logits = self.output_proj(x)
        
        # Log softmax for CTC
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Transpose for CTC: (batch, seq, vocab) -> (seq, batch, vocab)
        log_probs = log_probs.transpose(0, 1)
        
        return log_probs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions without log softmax (for greedy decoding)."""
        if self.cnn_backbone is not None:
            x = self.cnn_backbone(x)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        logits = self.output_proj(x)
        return logits


class GlossDecoder(nn.Module):
    """
    Autoregressive Transformer Decoder for gloss sequence generation.
    
    Uses cross-attention to encoder outputs and self-attention for
    modeling output dependencies. Prevents CTC collapse by providing
    stable cross-entropy gradients.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.embed_scale = math.sqrt(d_model)
        
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
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        print(f"GlossDecoder initialized: {num_decoder_layers} layers, vocab={vocab_size}")
    
    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask
    
    def forward(
        self,
        encoder_output: torch.Tensor,
        target_ids: torch.Tensor,
        encoder_padding_mask: torch.Tensor = None,
        target_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, src_len, d_model) - from encoder
            target_ids: (batch, tgt_len) - target gloss IDs (shifted right, starts with <sos>)
            encoder_padding_mask: (batch, src_len) - True for padding positions
            target_padding_mask: (batch, tgt_len) - True for padding positions
            
        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        batch_size, tgt_len = target_ids.shape
        device = target_ids.device
        
        # Embed target tokens
        positions = torch.arange(tgt_len, device=device).unsqueeze(0).expand(batch_size, -1)
        tgt_embed = self.embedding(target_ids) * self.embed_scale + self.pos_embedding(positions)
        tgt_embed = self.dropout(tgt_embed)
        
        # Causal mask for autoregressive decoding
        tgt_mask = self._generate_square_subsequent_mask(tgt_len, device)
        
        # Decode
        decoder_output = self.decoder(
            tgt_embed,
            encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=encoder_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_proj(decoder_output)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        encoder_output: torch.Tensor,
        max_len: int = 50,
        sos_token: int = 3,
        eos_token: int = 4,
        encoder_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Autoregressive generation at inference time.
        
        Args:
            encoder_output: (batch, src_len, d_model)
            max_len: Maximum output length
            sos_token: Start-of-sequence token ID
            eos_token: End-of-sequence token ID
            
        Returns:
            generated: (batch, seq_len) - generated token IDs
        """
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Start with <sos> token
        generated = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len - 1):
            # Get logits for next token
            logits = self.forward(encoder_output, generated, encoder_padding_mask)
            next_token_logits = logits[:, -1, :]  # (batch, vocab)
            
            # Greedy selection
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            finished = finished | (next_token.squeeze(-1) == eos_token)
            if finished.all():
                break
        
        return generated


class HybridCTCAttentionModel(nn.Module):
    """
    Hybrid CTC + Attention model for Sign Language Recognition.
    
    Combines:
    - Shared CNN + Transformer Encoder
    - CTC head for alignment (recognition)
    - Attention Decoder for sequence modeling (prevents collapse)
    
    Joint Loss: L = λ_ctc × CTC_Loss + λ_ce × CrossEntropy_Loss
    
    Reference: "Sign Language Transformers" (Camgöz et al., 2020)
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
        max_seq_len: int = 500,
        ctc_weight: float = 0.3,
        use_resnet: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.ctc_weight = ctc_weight
        self.ce_weight = 1.0 - ctc_weight
        
        # CNN backbone (shared)
        if use_resnet:
            self.cnn_backbone = CNNBackbone(output_dim=d_model, pretrained=True)
        else:
            self.cnn_backbone = SimpleCNNBackbone(input_channels=3, output_dim=d_model)
        
        # Input projection
        self.input_proj = nn.Linear(d_model, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder (shared)
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
        
        # Initialize weights
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\nHybridCTCAttentionModel initialized:")
        print(f"  Encoder layers: {num_encoder_layers}")
        print(f"  Decoder layers: {num_decoder_layers}")
        print(f"  d_model: {d_model}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  CTC weight: {ctc_weight}")
        print(f"  CE weight: {1-ctc_weight}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def _init_weights(self):
        """Initialize non-pretrained weights."""
        for name, p in self.named_parameters():
            if 'cnn_backbone' not in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode video frames to contextualized features.
        
        Args:
            x: (batch, seq_len, H, W, C) - raw video frames
            src_key_padding_mask: (batch, seq_len) - True for padding
            
        Returns:
            encoder_output: (batch, seq_len, d_model)
        """
        # CNN features
        x = self.cnn_backbone(x)
        
        # Project and normalize
        x = self.input_proj(x)
        x = self.input_norm(x)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        return x
    
    def forward(
        self,
        frames: torch.Tensor,
        target_ids: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None
    ):
        """
        Forward pass for training.
        
        Args:
            frames: (batch, seq_len, H, W, C) - input video
            target_ids: (batch, tgt_len) - target gloss IDs (with <sos> prepended)
            src_key_padding_mask: (batch, seq_len) - source padding mask
            tgt_key_padding_mask: (batch, tgt_len) - target padding mask
            
        Returns:
            ctc_log_probs: (seq_len, batch, vocab) - for CTC loss
            decoder_logits: (batch, tgt_len, vocab) - for CE loss
            encoder_output: (batch, seq_len, d_model) - encoder features
        """
        # Encode
        encoder_output = self.encode(frames, src_key_padding_mask)
        
        # CTC head
        ctc_logits = self.ctc_proj(encoder_output)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (seq, batch, vocab)
        
        # Decoder (if targets provided)
        decoder_logits = None
        if target_ids is not None:
            decoder_logits = self.decoder(
                encoder_output, 
                target_ids,
                encoder_padding_mask=src_key_padding_mask,
                target_padding_mask=tgt_key_padding_mask
            )
        
        return ctc_log_probs, decoder_logits, encoder_output
    
    def compute_loss(
        self,
        ctc_log_probs: torch.Tensor,
        decoder_logits: torch.Tensor,
        ctc_targets: torch.Tensor,
        decoder_targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        ctc_blank: int,
        pad_idx: int = 0
    ):
        """
        Compute joint CTC + Cross-Entropy loss.
        
        Args:
            ctc_log_probs: (seq, batch, vocab) - CTC predictions
            decoder_logits: (batch, tgt_len, vocab) - Decoder predictions
            ctc_targets: (batch, max_target_len) - CTC targets (no special tokens)
            decoder_targets: (batch, tgt_len) - Decoder targets (ends with <eos>)
            input_lengths: (batch,) - input sequence lengths
            target_lengths: (batch,) - target sequence lengths
            ctc_blank: blank token index for CTC
            pad_idx: padding index to ignore in CE loss
            
        Returns:
            total_loss, ctc_loss, ce_loss
        """
        # CTC loss
        ctc_loss = F.ctc_loss(
            ctc_log_probs,
            ctc_targets,
            input_lengths,
            target_lengths,
            blank=ctc_blank,
            zero_infinity=True
        )
        
        # Cross-entropy loss for decoder
        # decoder_logits: (batch, tgt_len, vocab) -> (batch * tgt_len, vocab)
        # decoder_targets: (batch, tgt_len) -> (batch * tgt_len)
        ce_logits = decoder_logits.reshape(-1, decoder_logits.size(-1))
        ce_targets = decoder_targets.reshape(-1)
        ce_loss = F.cross_entropy(ce_logits, ce_targets, ignore_index=pad_idx)
        
        # Combined loss
        total_loss = self.ctc_weight * ctc_loss + self.ce_weight * ce_loss
        
        return total_loss, ctc_loss.item(), ce_loss.item()
    
    @torch.no_grad()
    def recognize(self, frames: torch.Tensor, mode: str = 'ctc') -> torch.Tensor:
        """
        Inference: Recognize glosses from video.
        
        Args:
            frames: (batch, seq_len, H, W, C) - input video
            mode: 'ctc' for fast greedy, 'attention' for autoregressive
            
        Returns:
            predictions: (batch, pred_len) - predicted token IDs
        """
        self.eval()
        encoder_output = self.encode(frames)
        
        if mode == 'ctc':
            # Fast CTC greedy decoding
            ctc_logits = self.ctc_proj(encoder_output)
            predictions = torch.argmax(ctc_logits, dim=-1)
        else:
            # Autoregressive decoding
            predictions = self.decoder.generate(encoder_output)
        
        return predictions
    
    @torch.no_grad()
    def greedy_decode(
        self,
        frames: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
        max_len: int = 100,
        sos_idx: int = 3,
        eos_idx: int = 4
    ) -> torch.Tensor:
        """
        Greedy decoding using attention decoder.
        
        Args:
            frames: (batch, seq_len, C, H, W) - input video frames
            src_key_padding_mask: (batch, seq_len) - source padding mask
            max_len: Maximum output length
            sos_idx: Start of sequence token index
            eos_idx: End of sequence token index
        
        Returns:
            predictions: (batch, seq_len) - decoded token IDs
        """
        self.eval()
        batch_size = frames.size(0)
        device = frames.device
        
        # Encode
        encoder_output = self.encode(frames, src_key_padding_mask)
        
        # Start with <sos>
        predictions = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            # Decode one step
            decoder_output = self.decoder(
                encoder_output,
                predictions,
                encoder_padding_mask=src_key_padding_mask
            )
            
            # Get next token (greedy)
            next_token = decoder_output[:, -1, :].argmax(dim=-1, keepdim=True)
            predictions = torch.cat([predictions, next_token], dim=1)
            
            # Stop if all sequences have produced <eos>
            if (next_token == eos_idx).all():
                break
        
        return predictions


class CNNTransformer(nn.Module):
    """
    CNN + Transformer hybrid model.
    
    Uses CNN for local feature extraction, Transformer for global context.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        vocab_size: int = 1200,
        kernel_size: int = 5,
        num_cnn_layers: int = 2
    ):
        super().__init__()
        
        # CNN layers for local feature extraction
        cnn_layers = []
        in_channels = input_dim
        for i in range(num_cnn_layers):
            cnn_layers.extend([
                nn.Conv1d(in_channels, d_model, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_channels = d_model
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        print(f"CNNTransformer initialized:")
        print(f"  CNN layers: {num_cnn_layers}")
        print(f"  Transformer layers: {num_encoder_layers}")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (seq_len, batch, vocab_size)
        """
        # CNN: (batch, seq, dim) -> (batch, dim, seq) -> CNN -> (batch, dim, seq) -> (batch, seq, dim)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        
        # Positional encoding + Transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Output
        logits = self.output_proj(x)
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs.transpose(0, 1)



