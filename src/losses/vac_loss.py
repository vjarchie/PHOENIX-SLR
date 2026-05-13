# -*- coding: utf-8 -*-
"""
Visual Alignment Constraint (VAC) Loss for Sign Language Recognition

Reference: Min et al., "Visual Alignment Constraint for Continuous Sign Language Recognition" (ICCV 2021)

VAC addresses overfitting in CSLR by enforcing alignment between:
1. Visual features (from CNN backbone) 
2. Sequence features (from Transformer encoder)

Components:
- ConvCTC: CTC loss on visual features (before temporal modeling)
- SeqCTC: CTC loss on sequence features (after temporal modeling)  
- Distillation: KL divergence to align the two predictions

This prevents the model from overfitting to spurious patterns and improves
generalization by ensuring consistent predictions at different abstraction levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeqKD(nn.Module):
    """
    Sequence Knowledge Distillation Loss.
    
    Computes KL divergence between two sequence predictions to enforce alignment.
    Used to distill knowledge from the sequence model (teacher) to the visual model (student).
    
    Args:
        T: Temperature for softmax (higher = softer distributions)
    """
    
    def __init__(self, T: float = 8.0):
        super().__init__()
        self.T = T
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor, 
        use_blank: bool = False
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_logits: (seq_len, batch, vocab) - ConvCTC predictions (visual)
            teacher_logits: (seq_len, batch, vocab) - SeqCTC predictions (sequence)
            use_blank: Whether to include blank token in distillation
            
        Returns:
            Distillation loss (scalar)
        """
        start_idx = 0 if use_blank else 1
        
        # Get vocab size after potentially excluding blank
        vocab_size = teacher_logits.shape[-1] - start_idx
        
        # Apply temperature scaling and compute log-softmax for student
        student_log_probs = F.log_softmax(
            student_logits[:, :, start_idx:] / self.T, dim=-1
        ).view(-1, vocab_size)
        
        # Apply temperature scaling and compute softmax for teacher (detached)
        teacher_probs = F.softmax(
            teacher_logits[:, :, start_idx:].detach() / self.T, dim=-1
        ).view(-1, vocab_size)
        
        # KL divergence loss (scaled by T^2 as per distillation theory)
        loss = self.kl_loss(student_log_probs, teacher_probs) * (self.T ** 2)
        
        return loss


class VACLoss(nn.Module):
    """
    Visual Alignment Constraint Loss.
    
    Combines three losses:
    1. SeqCTC: CTC loss on final sequence features (main supervision)
    2. ConvCTC: CTC loss on visual features (auxiliary supervision)
    3. Distillation: KL divergence between the two (alignment constraint)
    
    Total Loss = w_seq * SeqCTC + w_conv * ConvCTC + w_dist * Distillation
    
    Default weights from VAC paper: SeqCTC=1.0, ConvCTC=1.0, Dist=25.0
    
    Args:
        seq_ctc_weight: Weight for sequence CTC loss (default: 1.0)
        conv_ctc_weight: Weight for visual CTC loss (default: 1.0) 
        dist_weight: Weight for distillation loss (default: 25.0)
        temperature: Temperature for distillation (default: 8.0)
        blank_idx: Index of blank token for CTC
    """
    
    def __init__(
        self,
        seq_ctc_weight: float = 1.0,
        conv_ctc_weight: float = 1.0,
        dist_weight: float = 25.0,
        temperature: float = 8.0,
        blank_idx: int = 2
    ):
        super().__init__()
        
        self.seq_ctc_weight = seq_ctc_weight
        self.conv_ctc_weight = conv_ctc_weight
        self.dist_weight = dist_weight
        self.blank_idx = blank_idx
        
        # CTC loss (shared for both heads)
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
        
        # Distillation loss
        self.distillation = SeqKD(T=temperature)
        
        print(f"VACLoss initialized:")
        print(f"  SeqCTC weight: {seq_ctc_weight}")
        print(f"  ConvCTC weight: {conv_ctc_weight}")
        print(f"  Distillation weight: {dist_weight}")
        print(f"  Temperature: {temperature}")
    
    def forward(
        self,
        seq_logits: torch.Tensor,
        conv_logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> dict:
        """
        Compute VAC loss.
        
        Args:
            seq_logits: (seq_len, batch, vocab) - Sequence CTC predictions
            conv_logits: (seq_len, batch, vocab) - Visual CTC predictions
            targets: (batch, max_target_len) - Target gloss IDs
            input_lengths: (batch,) - Input sequence lengths
            target_lengths: (batch,) - Target sequence lengths
            
        Returns:
            Dictionary with total loss and individual components
        """
        # Sequence CTC loss (main supervision)
        seq_log_probs = F.log_softmax(seq_logits, dim=-1)
        seq_ctc = self.ctc_loss(
            seq_log_probs,
            targets,
            input_lengths,
            target_lengths
        )
        
        # Visual CTC loss (auxiliary supervision)
        conv_log_probs = F.log_softmax(conv_logits, dim=-1)
        conv_ctc = self.ctc_loss(
            conv_log_probs,
            targets,
            input_lengths,
            target_lengths
        )
        
        # Distillation loss (alignment constraint)
        # Teacher: sequence logits (more refined)
        # Student: conv logits (needs to align with teacher)
        dist_loss = self.distillation(conv_logits, seq_logits, use_blank=False)
        
        # Combined loss
        total_loss = (
            self.seq_ctc_weight * seq_ctc +
            self.conv_ctc_weight * conv_ctc +
            self.dist_weight * dist_loss
        )
        
        return {
            'total': total_loss,
            'seq_ctc': seq_ctc.item(),
            'conv_ctc': conv_ctc.item(),
            'distillation': dist_loss.item()
        }


class VACHybridLoss(nn.Module):
    """
    VAC Loss adapted for Hybrid CTC+Attention architecture.
    
    Combines:
    1. SeqCTC: CTC loss on encoder output (alignment)
    2. ConvCTC: CTC loss on visual features (auxiliary)
    3. Distillation: KL divergence between CTC heads (alignment constraint)
    4. CrossEntropy: Decoder loss (sequence modeling)
    
    Total Loss = w_seq * SeqCTC + w_conv * ConvCTC + w_dist * Dist + w_ce * CE
    
    Args:
        seq_ctc_weight: Weight for sequence CTC (default: 0.3)
        conv_ctc_weight: Weight for visual CTC (default: 0.3)
        dist_weight: Weight for distillation (default: 10.0)
        ce_weight: Weight for cross-entropy decoder loss (default: 0.7)
        temperature: Distillation temperature (default: 8.0)
        blank_idx: Blank token index for CTC
        pad_idx: Padding token index for CE
        label_smoothing: Label smoothing for CE (default: 0.1)
    """
    
    def __init__(
        self,
        seq_ctc_weight: float = 0.3,
        conv_ctc_weight: float = 0.3,
        dist_weight: float = 10.0,
        ce_weight: float = 0.7,
        temperature: float = 8.0,
        blank_idx: int = 2,
        pad_idx: int = 0,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        
        self.seq_ctc_weight = seq_ctc_weight
        self.conv_ctc_weight = conv_ctc_weight
        self.dist_weight = dist_weight
        self.ce_weight = ce_weight
        self.blank_idx = blank_idx
        self.pad_idx = pad_idx
        
        # CTC loss
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
        
        # Cross-entropy loss with label smoothing
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=pad_idx, 
            label_smoothing=label_smoothing
        )
        
        # Distillation loss
        self.distillation = SeqKD(T=temperature)
        
        print(f"VACHybridLoss initialized:")
        print(f"  SeqCTC weight: {seq_ctc_weight}")
        print(f"  ConvCTC weight: {conv_ctc_weight}")
        print(f"  Distillation weight: {dist_weight}")
        print(f"  CE weight: {ce_weight}")
        print(f"  Label smoothing: {label_smoothing}")
    
    def forward(
        self,
        seq_logits: torch.Tensor,
        conv_logits: torch.Tensor,
        decoder_logits: torch.Tensor,
        ctc_targets: torch.Tensor,
        decoder_targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> dict:
        """
        Compute VAC + Hybrid loss.
        
        Args:
            seq_logits: (seq_len, batch, vocab) - Sequence CTC predictions
            conv_logits: (seq_len, batch, vocab) - Visual CTC predictions
            decoder_logits: (batch, tgt_len, vocab) - Decoder predictions
            ctc_targets: (batch, max_target_len) - CTC targets (no special tokens)
            decoder_targets: (batch, tgt_len) - Decoder targets (with <eos>)
            input_lengths: (batch,) - Input sequence lengths
            target_lengths: (batch,) - Target sequence lengths
            
        Returns:
            Dictionary with total loss and components
        """
        losses = {}
        total_loss = 0.0
        
        # Sequence CTC loss
        seq_log_probs = F.log_softmax(seq_logits, dim=-1)
        seq_ctc = self.ctc_loss(seq_log_probs, ctc_targets, input_lengths, target_lengths)
        losses['seq_ctc'] = seq_ctc.item()
        total_loss += self.seq_ctc_weight * seq_ctc
        
        # Visual CTC loss
        conv_log_probs = F.log_softmax(conv_logits, dim=-1)
        conv_ctc = self.ctc_loss(conv_log_probs, ctc_targets, input_lengths, target_lengths)
        losses['conv_ctc'] = conv_ctc.item()
        total_loss += self.conv_ctc_weight * conv_ctc
        
        # Distillation loss (align visual with sequence)
        dist_loss = self.distillation(conv_logits, seq_logits, use_blank=False)
        losses['distillation'] = dist_loss.item()
        total_loss += self.dist_weight * dist_loss
        
        # Cross-entropy loss for decoder
        if decoder_logits is not None:
            ce_logits = decoder_logits.reshape(-1, decoder_logits.size(-1))
            ce_targets = decoder_targets.reshape(-1)
            ce_loss = self.ce_loss(ce_logits, ce_targets)
            losses['ce'] = ce_loss.item()
            total_loss += self.ce_weight * ce_loss
        
        losses['total'] = total_loss
        
        return losses
