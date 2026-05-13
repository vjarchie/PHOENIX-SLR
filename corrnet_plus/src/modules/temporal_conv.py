"""
Temporal Convolution modules for CorrNet+.
Includes Temporal LiftPool for learnable temporal downsampling.
Based on: https://github.com/hulianyuyy/CorrNet_Plus
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalWeighting(nn.Module):
    """Local weighting module for adaptive feature enhancement."""
    
    def __init__(self, input_size: int):
        super().__init__()
        self.conv = nn.Conv1d(input_size, input_size, kernel_size=5, stride=1, padding=2)
        self.insnorm = nn.InstanceNorm1d(input_size, affine=True)
        nn.init.zeros_(self.conv.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return x + x * (torch.sigmoid(self.insnorm(out)) - 0.5)


class TemporalLiftPool(nn.Module):
    """
    Temporal LiftPooling for learnable temporal downsampling.
    Uses lifting scheme with predict-update structure.
    """
    
    def __init__(self, input_size: int, kernel_size: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        
        self.predictor = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, stride=1, padding=1, groups=input_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_size, input_size, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )
        
        self.updater = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, stride=1, padding=1, groups=input_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_size, input_size, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )
        
        nn.init.zeros_(self.predictor[2].weight)
        nn.init.zeros_(self.updater[2].weight)
        
        self.weight1 = LocalWeighting(input_size)
        self.weight2 = LocalWeighting(input_size)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, C, T) input features
            
        Returns:
            tuple of (output, loss_u, loss_p)
        """
        B, C, T = x.size()
        
        # Handle odd sequence lengths by padding if necessary
        if T % self.kernel_size != 0:
            pad_len = self.kernel_size - (T % self.kernel_size)
            x = F.pad(x, (0, pad_len), mode='replicate')
            T = x.size(2)
        
        Xe = x[:, :, 0:T:self.kernel_size]
        Xo = x[:, :, 1:T:self.kernel_size]
        
        # Ensure Xe and Xo have same length
        min_len = min(Xe.size(2), Xo.size(2))
        Xe = Xe[:, :, :min_len]
        Xo = Xo[:, :, :min_len]
        
        d = Xo - self.predictor(Xe)
        s = Xe + self.updater(d)
        
        loss_u = torch.norm(s - Xo, p=2)
        loss_p = torch.norm(d, p=2)
        
        # Combine smooth and detail coefficients
        output = self.weight1(s) + self.weight2(d)
        
        return output, loss_u, loss_p


class TemporalConv(nn.Module):
    """
    Temporal convolution network with LiftPooling.
    Processes visual features along the temporal dimension.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        conv_type: int = 2,
        use_bn: bool = False,
        num_classes: int = -1
    ):
        super().__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type
        
        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', 'P2']
        elif self.conv_type == 2:
            self.kernel_size = ['K5', 'P2', 'K5', 'P2']
        
        self.temporal_conv = nn.ModuleList()
        
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            
            if ks[0] == 'P':
                self.temporal_conv.append(
                    TemporalLiftPool(input_size=input_sz, kernel_size=int(ks[1]))
                )
            elif ks[0] == 'K':
                kernel_sz = int(ks[1])
                padding = kernel_sz // 2  # Add padding to preserve length
                self.temporal_conv.append(
                    nn.Sequential(
                        nn.Conv1d(input_sz, self.hidden_size, kernel_size=kernel_sz, stride=1, padding=padding),
                        nn.BatchNorm1d(self.hidden_size),
                        nn.ReLU(inplace=True),
                    )
                )
        
        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)
    
    def update_lgt(self, feat_len: torch.Tensor) -> torch.Tensor:
        """Update feature length after temporal convolutions."""
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = feat_len // int(ks[1])
            # With padding, conv layers preserve length (no subtraction needed)
        return feat_len
    
    def forward(self, frame_feat: torch.Tensor, lgt: torch.Tensor) -> dict:
        """
        Args:
            frame_feat: (B, C, T) framewise features
            lgt: (B,) sequence lengths
            
        Returns:
            dict with visual_feat, conv_logits, feat_len, and lift losses
        """
        visual_feat = frame_feat
        loss_liftpool_u = 0
        loss_liftpool_p = 0
        
        for tempconv in self.temporal_conv:
            if isinstance(tempconv, TemporalLiftPool):
                visual_feat, loss_u, loss_d = tempconv(visual_feat)
                loss_liftpool_u += loss_u
                loss_liftpool_p += loss_d
            else:
                visual_feat = tempconv(visual_feat)
        
        lgt = self.update_lgt(lgt)
        
        logits = None
        if self.num_classes != -1:
            logits = self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),  # (T, B, C)
            "conv_logits": logits.permute(2, 0, 1) if logits is not None else None,  # (T, B, V)
            "feat_len": lgt,
            "loss_liftpool_u": loss_liftpool_u,
            "loss_liftpool_p": loss_liftpool_p,
        }
