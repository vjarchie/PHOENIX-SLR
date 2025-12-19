"""
Hybrid CTC + Attention model for pre-extracted I3D features.

This model is simpler than the frame-based model since CNN feature extraction
is already done. It consists of:
1. Feature projection layer
2. Transformer encoder
3. CTC head
4. Attention decoder
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GlossDecoder(nn.Module):
    """Attention-based decoder for gloss sequence generation."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 500,
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        return self.output_proj(output)


class I3DHybridModel(nn.Module):
    """
    Hybrid CTC + Attention model for pre-extracted I3D features.
    
    Architecture:
        Input (I3D features) -> Projection -> Transformer Encoder -> CTC Head
                                                    |
                                                    v
                                            Attention Decoder -> CE Loss
    """
    
    def __init__(
        self,
        input_dim: int = 512,  # R3D-18 outputs 512, true I3D outputs 1024
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        vocab_size: int = 1200,
        max_seq_len: int = 500,
        pad_token_id: int = 0,
        blank_token_id: int = 2
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.blank_token_id = blank_token_id
        
        # Input projection (I3D features -> d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
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
        
        # CTC output layer
        self.ctc_output = nn.Linear(d_model, vocab_size)
        
        # Attention decoder
        self.decoder = GlossDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pad_token_id=pad_token_id
        )
        
        # Initialize weights
        self._init_weights()
        
        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nI3DHybridModel initialized:")
        print(f"  Input dim: {input_dim}")
        print(f"  d_model: {d_model}")
        print(f"  Encoder layers: {num_encoder_layers}")
        print(f"  Decoder layers: {num_decoder_layers}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        features: torch.Tensor,  # (batch, seq_len, feature_dim)
        tgt: torch.Tensor,  # (batch, tgt_seq_len)
        src_key_padding_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Pre-extracted I3D features (batch, seq_len, feature_dim)
            tgt: Target sequence for decoder (batch, tgt_seq_len)
            src_key_padding_mask: Mask for encoder (batch, seq_len)
            tgt_key_padding_mask: Mask for decoder (batch, tgt_seq_len)
        
        Returns:
            ctc_log_probs: CTC output (seq_len, batch, vocab_size)
            decoder_output: Decoder output (batch, tgt_seq_len, vocab_size)
        """
        # Project features
        x = self.input_proj(features)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Encode
        encoder_output = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # (batch, seq_len, d_model)
        
        # CTC output
        ctc_logits = self.ctc_output(encoder_output)
        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
        ctc_log_probs = ctc_log_probs.transpose(0, 1)  # (seq_len, batch, vocab)
        
        # Decoder output
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        decoder_output = self.decoder(
            tgt,
            encoder_output,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )  # (batch, tgt_seq_len, vocab_size)
        
        return ctc_log_probs, decoder_output
    
    @torch.no_grad()
    def greedy_decode(
        self,
        features: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
        max_len: int = 100,
        sos_idx: int = 3,
        eos_idx: int = 4
    ) -> torch.Tensor:
        """
        Greedy decoding for inference.
        
        Args:
            features: Pre-extracted I3D features (batch, seq_len, feature_dim)
            src_key_padding_mask: Mask for encoder
            max_len: Maximum output length
            sos_idx: Start of sequence token index
            eos_idx: End of sequence token index
        
        Returns:
            predictions: Decoded sequences (batch, max_len)
        """
        batch_size = features.size(0)
        device = features.device
        
        # Encode
        x = self.input_proj(features)
        x = self.pos_encoder(x)
        encoder_output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Start with <sos>
        predictions = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(predictions.size(1)).to(device)
            
            decoder_output = self.decoder(
                predictions,
                encoder_output,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            
            # Get next token
            next_token = decoder_output[:, -1, :].argmax(dim=-1, keepdim=True)
            predictions = torch.cat([predictions, next_token], dim=1)
            
            # Stop if all sequences have produced <eos>
            if (next_token == eos_idx).all():
                break
        
        return predictions


