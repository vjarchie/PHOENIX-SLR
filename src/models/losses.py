# -*- coding: utf-8 -*-
"""
Loss functions for Sign Language Recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VACLoss(nn.Module):
    """
    Visual Alignment Constraint (VAC) Loss.
    
    Enforces monotonic alignment between the video features and the gloss sequence.
    This provides auxiliary supervision to the visual backbone, preventing it 
    from collapsing which is common when trained with just CTC.
    
    Reference: "VAC: Visual Alignment Constraint for CSLR" (ICCV 2021)
    """
    
    def __init__(self, blank_idx: int = 0):
        super().__init__()
        self.blank_idx = blank_idx
        # Utilize standard CTC loss for the visual predictions to enforce alignment
        self.ctc = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    def forward(self, 
                conv_ctc_log_probs: torch.Tensor, 
                targets: torch.Tensor, 
                input_lengths: torch.Tensor, 
                target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            conv_ctc_log_probs: (seq_len, batch, vocab_size) - Log probabilities from visual features
            targets: (batch, max_target_len) - Target gloss IDs
            input_lengths: (batch,) - Valid sequence lengths for input
            target_lengths: (batch,) - Valid sequence lengths for targets
            
        Returns:
            Scalar loss tensor
        """
        # The VAC loss simply relies on CTC supervision applied directly 
        # on the spatial/temporal features before the global transformer encoder
        # This keeps the representations aligned.
        loss = self.ctc(
            conv_ctc_log_probs,
            targets,
            input_lengths,
            target_lengths
        )
        return loss
