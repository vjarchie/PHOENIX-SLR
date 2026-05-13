"""
Spatial-Temporal Correlation modules for CorrNet+.
Core innovation: captures cross-frame body trajectories without pose estimation.
Based on: https://github.com/hulianyuyy/CorrNet_Plus
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool2d(nn.Module):
    """
    Attention-based spatial pooling.
    Uses learnable queries to aggregate spatial information.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        output_dim: int = None,
        clusters: int = 1
    ):
        super().__init__()
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.clusters = clusters
        self.query = nn.Parameter(torch.rand(self.clusters, 1, embed_dim), requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, H, W) input tensor
            
        Returns:
            (N, C, T, clusters) attention-pooled features
        """
        N, C, T, H, W = x.shape
        x = x.flatten(start_dim=3).permute(3, 0, 2, 1).reshape(-1, N * T, C).contiguous()
        
        x, _ = F.multi_head_attention_forward(
            query=self.query.repeat(1, N * T, 1),
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        
        return x.view(self.clusters, N, T, C).contiguous().permute(1, 3, 2, 0)


class UnfoldTemporalWindows(nn.Module):
    """Unfold temporal windows for cross-frame correlation computation."""
    
    def __init__(self, window_size: int = 9, window_stride: int = 1, window_dilation: int = 1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation
        
        self.padding = (window_size + (window_size - 1) * (window_dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(
            kernel_size=(self.window_size, 1),
            dilation=(self.window_dilation, 1),
            stride=(self.window_stride, 1),
            padding=(self.padding, 0)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, H, W) input tensor
            
        Returns:
            (N, C, T, window_size, H, W) unfolded tensor
        """
        N, C, T, H, W = x.shape
        x = x.view(N, C, T, H * W)
        x = self.unfold(x)
        x = x.view(N, C, self.window_size, T, H, W).permute(0, 1, 3, 2, 4, 5).contiguous()
        return x


class TemporalWeighting(nn.Module):
    """
    Multi-scale temporal weighting module.
    Adaptively evaluates frame contributions.
    """
    
    def __init__(self, input_size: int):
        super().__init__()
        hidden_size = input_size // 16
        self.num = 3
        
        self.conv_transform = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.conv_back = nn.Conv1d(hidden_size, input_size, kernel_size=1)
        
        self.conv_enhance = nn.ModuleList([
            nn.Conv1d(
                hidden_size, hidden_size,
                kernel_size=3, stride=1, padding=i + 1,
                groups=hidden_size, dilation=i + 1
            )
            for i in range(self.num)
        ])
        
        self.weights = nn.Parameter(torch.ones(self.num) / self.num, requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, H, W) input tensor
            
        Returns:
            (N, C, T, H, W) weighted tensor
        """
        out = self.conv_transform(x.mean(-1).mean(-1))
        
        aggregated_out = 0
        for i in range(self.num):
            aggregated_out = aggregated_out + self.conv_enhance[i](out) * self.weights[i]
        
        out = self.conv_back(aggregated_out)
        
        return x * (torch.sigmoid(out.unsqueeze(-1).unsqueeze(-1)) - 0.5) * self.alpha


class GetCorrelation(nn.Module):
    """
    Core correlation module for CorrNet+.
    Computes cross-frame body trajectories via attention pooling and
    multi-scale spatial aggregation.
    """
    
    def __init__(self, channels: int, neighbors: int = 3):
        super().__init__()
        reduction_channel = channels // 16
        self.neighbors = neighbors
        self.clusters = 1
        
        self.down_conv2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.weights2 = nn.Parameter(
            torch.ones(self.neighbors * 2) / (self.neighbors * 2),
            requires_grad=True
        )
        
        self.unfold = UnfoldTemporalWindows(2 * self.neighbors + 1)
        self.weights3 = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.weights4 = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        
        self.attpool = AttentionPool2d(
            embed_dim=channels,
            num_heads=1,
            clusters=self.clusters
        )
        
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, reduction_channel, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(reduction_channel, channels, kernel_size=1),
        )
        
        self.down_conv = nn.Conv3d(channels, reduction_channel, kernel_size=1, bias=False)
        self.spatial_aggregation1 = nn.Conv3d(
            reduction_channel, reduction_channel,
            kernel_size=(9, 3, 3), padding=(4, 1, 1), groups=reduction_channel
        )
        self.spatial_aggregation2 = nn.Conv3d(
            reduction_channel, reduction_channel,
            kernel_size=(9, 3, 3), padding=(4, 2, 2), dilation=(1, 2, 2), groups=reduction_channel
        )
        self.spatial_aggregation3 = nn.Conv3d(
            reduction_channel, reduction_channel,
            kernel_size=(9, 3, 3), padding=(4, 3, 3), dilation=(1, 3, 3), groups=reduction_channel
        )
        
        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.conv_back = nn.Conv3d(reduction_channel, channels, kernel_size=1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, H, W) input tensor
            
        Returns:
            (N, C, T, 1, 1) correlation features
        """
        N, C, T, H, W = x.shape
        
        def clustering(query, key):
            affinities = torch.einsum('bctp,bctl->btpl', query, key)
            return torch.einsum('bctl,btpl->bctp', key, torch.sigmoid(affinities) - 0.5)
        
        x_mean = x.mean(3, keepdim=True).mean(4, keepdim=False)
        x_max = x.max(-1, keepdim=False)[0].max(-1, keepdim=True)[0]
        x_att = self.attpool(x)
        
        x2 = self.down_conv2(x)
        upfold = self.unfold(x2)
        upfold = (
            torch.cat([upfold[:, :, :, :self.neighbors], upfold[:, :, :, self.neighbors + 1:]], 3)
            * self.weights2.view(1, 1, 1, -1, 1, 1)
        ).view(N, C, T, -1)
        
        x_mean = x_mean * self.weights4[0] + x_max * self.weights4[1] + x_att * self.weights4[2]
        x_mean = clustering(x_mean, upfold)
        features = x_mean.view(N, C, T, self.clusters, 1)
        
        x_down = self.down_conv(x)
        aggregated_x = (
            self.spatial_aggregation1(x_down) * self.weights[0]
            + self.spatial_aggregation2(x_down) * self.weights[1]
            + self.spatial_aggregation3(x_down) * self.weights[2]
        )
        aggregated_x = self.conv_back(aggregated_x)
        
        features = features * (torch.sigmoid(aggregated_x) - 0.5)
        
        return features
