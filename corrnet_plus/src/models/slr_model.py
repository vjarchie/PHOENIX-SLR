"""
CorrNet+ Sign Language Recognition Model.
Combines: ResNet with Correlation + TemporalConv + BiLSTM + CTC
Based on: https://github.com/hulianyuyy/CorrNet_Plus
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
from modules import BiLSTMLayer, TemporalConv
from models.resnet import resnet18, resnet34


class Identity(nn.Module):
    """Identity module to replace unused layers."""
    def forward(self, x):
        return x


class NormLinear(nn.Module):
    """Linear layer with weight normalization."""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, F.normalize(self.weight, dim=0))


class SeqKD(nn.Module):
    """
    Sequence-level Knowledge Distillation.
    Used for VAC-style distillation between ConvCTC and SeqCTC heads.
    """
    
    def __init__(self, T: float = 8.0):
        super().__init__()
        self.T = T
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self,
        input_logits: torch.Tensor,
        target_logits: torch.Tensor,
        use_blank: bool = False
    ) -> torch.Tensor:
        """
        Args:
            input_logits: (T, B, V) student logits
            target_logits: (T, B, V) teacher logits (detached)
            use_blank: whether to include blank token
        
        Returns:
            KL divergence loss
        """
        if not use_blank:
            input_logits = input_logits[:, :, 1:]
            target_logits = target_logits[:, :, 1:]
        
        input_probs = F.log_softmax(input_logits / self.T, dim=-1)
        target_probs = F.softmax(target_logits / self.T, dim=-1)
        
        return self.kl_div(input_probs, target_probs) * (self.T ** 2)


class CorrNetPlusSLR(nn.Module):
    """
    Complete CorrNet+ model for Continuous Sign Language Recognition.
    
    Architecture:
    1. ResNet-18 with Spatial-Temporal Correlation (visual features)
    2. TemporalConv with LiftPooling (temporal downsampling + feature extraction)
    3. BiLSTM (sequential modeling)
    4. Dual CTC heads (ConvCTC + SeqCTC) with distillation
    
    Achieves 18.0% WER on PHOENIX-2014 (SOTA).
    """
    
    def __init__(
        self,
        num_classes: int,
        c2d_type: str = 'resnet18',
        conv_type: int = 2,
        hidden_size: int = 1024,
        use_bn: bool = False,
        weight_norm: bool = True,
        share_classifier: bool = True,
        loss_weights: dict = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.loss_weights = loss_weights or {
            'SeqCTC': 1.0,
            'ConvCTC': 1.0,
            'Dist': 25.0,
            'Cu': 0.2,
            'Cp': 0.5,
        }
        
        if c2d_type == 'resnet18':
            self.conv2d = resnet18(pretrained=True)
        elif c2d_type == 'resnet34':
            self.conv2d = resnet34(pretrained=True)
        else:
            raise ValueError(f"Unknown backbone: {c2d_type}")
        
        self.conv2d.fc = Identity()
        
        self.conv1d = TemporalConv(
            input_size=512,
            hidden_size=hidden_size,
            conv_type=conv_type,
            use_bn=use_bn,
            num_classes=num_classes
        )
        
        self.temporal_model = BiLSTMLayer(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True
        )
        
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        
        if share_classifier:
            self.conv1d.fc = self.classifier
        
        self._init_losses()
    
    def _init_losses(self):
        """Initialize loss functions."""
        self.ctc_loss = nn.CTCLoss(reduction='none', zero_infinity=True)
        self.distillation = SeqKD(T=8)
    
    def forward(
        self,
        x: torch.Tensor,
        len_x: torch.Tensor,
        label: torch.Tensor = None,
        label_lgt: torch.Tensor = None
    ) -> dict:
        """
        Args:
            x: (B, C, T, H, W) video tensor OR (B, C, T) pre-extracted features
            len_x: (B,) sequence lengths
            label: (B, max_label_len) gloss labels (for training)
            label_lgt: (B,) label lengths (for training)
        
        Returns:
            dict with logits, predictions, and auxiliary outputs
        """
        if len(x.shape) == 5:
            # Input should be (B, C, T, H, W) where C=3
            # But if C and T are swapped, fix it here
            if x.shape[1] != 3 and x.shape[2] == 3:
                # Input is (B, T, C, H, W), need to permute to (B, C, T, H, W)
                x = x.permute(0, 2, 1, 3, 4).contiguous()
            
            batch, channel, temp, height, width = x.shape
            framewise = self.conv2d(x)
            framewise = framewise.view(batch, temp, -1).permute(0, 2, 1)
        else:
            framewise = x
        
        conv1d_outputs = self.conv1d(framewise, len_x)
        
        visual_feat = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        
        tm_outputs = self.temporal_model(visual_feat, lgt)
        sequence_logits = self.classifier(tm_outputs['predictions'])
        
        return {
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": sequence_logits,
            "loss_liftpool_u": conv1d_outputs['loss_liftpool_u'],
            "loss_liftpool_p": conv1d_outputs['loss_liftpool_p'],
        }
    
    def compute_loss(
        self,
        outputs: dict,
        label: torch.Tensor,
        label_lgt: torch.Tensor
    ) -> tuple:
        """
        Compute CorrNet+ loss (SeqCTC + ConvCTC + Distillation + LiftPool).
        
        Args:
            outputs: forward() output dict
            label: (B, max_label_len) gloss labels
            label_lgt: (B,) label lengths
        
        Returns:
            tuple of (total_loss, loss_dict)
        """
        total_loss = 0
        loss_dict = {}
        feat_len = outputs['feat_len']
        
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                ctc_input = outputs['conv_logits'].log_softmax(-1)
                loss = self.ctc_loss(
                    ctc_input,
                    label.cpu().int(),
                    feat_len.cpu().int(),
                    label_lgt.cpu().int()
                ).mean()
                loss_dict['ConvCTC'] = weight * loss
                total_loss += loss_dict['ConvCTC']
            
            elif k == 'SeqCTC':
                ctc_input = outputs['sequence_logits'].log_softmax(-1)
                loss = self.ctc_loss(
                    ctc_input,
                    label.cpu().int(),
                    feat_len.cpu().int(),
                    label_lgt.cpu().int()
                ).mean()
                loss_dict['SeqCTC'] = weight * loss
                total_loss += loss_dict['SeqCTC']
            
            elif k == 'Dist':
                conv_logits = outputs['conv_logits']
                seq_logits = outputs['sequence_logits'].detach()
                
                # Align temporal dimensions if they differ
                min_t = min(conv_logits.size(0), seq_logits.size(0))
                conv_logits = conv_logits[:min_t]
                seq_logits = seq_logits[:min_t]
                
                loss = self.distillation(
                    conv_logits,
                    seq_logits,
                    use_blank=False
                )
                loss_dict['Dist'] = weight * loss
                total_loss += loss_dict['Dist']
            
            elif k == 'Cu':
                loss_dict['Cu'] = weight * outputs['loss_liftpool_u']
                total_loss += loss_dict['Cu']
            
            elif k == 'Cp':
                loss_dict['Cp'] = weight * outputs['loss_liftpool_p']
                total_loss += loss_dict['Cp']
        
        loss_dict['total'] = total_loss
        return total_loss, loss_dict
    
    @torch.no_grad()
    def decode(
        self,
        outputs: dict,
        decoder: object = None
    ) -> tuple:
        """
        Decode predictions using beam search.
        
        Args:
            outputs: forward() output dict
            decoder: CTC decoder object
        
        Returns:
            tuple of (seq_predictions, conv_predictions)
        """
        seq_pred = None
        conv_pred = None
        
        if decoder is not None:
            seq_pred = decoder.decode(
                outputs['sequence_logits'],
                outputs['feat_len'],
                batch_first=False,
                probs=False
            )
            conv_pred = decoder.decode(
                outputs['conv_logits'],
                outputs['feat_len'],
                batch_first=False,
                probs=False
            )
        
        return seq_pred, conv_pred


def build_model(
    num_classes: int,
    backbone: str = 'resnet18',
    hidden_size: int = 1024,
    **kwargs
) -> CorrNetPlusSLR:
    """
    Build CorrNet+ model with specified configuration.
    
    Args:
        num_classes: vocabulary size (including blank)
        backbone: 'resnet18' or 'resnet34'
        hidden_size: hidden dimension for temporal modeling
    
    Returns:
        CorrNetPlusSLR model
    """
    return CorrNetPlusSLR(
        num_classes=num_classes,
        c2d_type=backbone,
        hidden_size=hidden_size,
        **kwargs
    )
