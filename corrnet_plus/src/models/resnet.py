"""
ResNet backbone with Spatial-Temporal Correlation for CorrNet+.
Integrates correlation modules at multiple feature scales.
Based on: https://github.com/hulianyuyy/CorrNet_Plus
"""
import gc
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import sys
sys.path.append('..')
from modules.correlation import GetCorrelation, TemporalWeighting


MODEL_URLS = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """3x3 spatial convolution (no temporal)."""
    return nn.Conv3d(
        in_planes, out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=False
    )


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
    expansion = 1
    
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet backbone with Spatial-Temporal Correlation modules.
    
    Key differences from standard ResNet:
    1. 3D convolutions with (1,k,k) kernels for spatial-only processing
    2. Correlation modules after layers 2, 3, 4 for cross-frame modeling
    3. Temporal weighting for adaptive frame importance
    """
    
    def __init__(self, block, layers: list, num_classes: int = 1000):
        super().__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv3d(
            3, 64,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.corr2 = GetCorrelation(self.inplanes, neighbors=1)
        self.temporal_weight2 = TemporalWeighting(self.inplanes)
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.corr3 = GetCorrelation(self.inplanes, neighbors=3)
        self.temporal_weight3 = TemporalWeighting(self.inplanes)
        
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.corr4 = GetCorrelation(self.inplanes, neighbors=5)
        self.temporal_weight4 = TemporalWeighting(self.inplanes)
        
        self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive for any input size
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=(1, stride, stride), bias=False
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )
        
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, H, W) video tensor
            
        Returns:
            (N*T, num_classes) or (N*T, 512) features
        """
        from torch.utils.checkpoint import checkpoint
        
        N, C, T, H, W = x.size()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Use gradient checkpointing for memory efficiency
        if self.training:
            x = checkpoint(self.layer1, x, use_reentrant=False)
            x = checkpoint(self.layer2, x, use_reentrant=False)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
        
        x = x + self.corr2(x) * self.alpha[0]
        x = x + self.temporal_weight2(x)
        
        if self.training:
            x = checkpoint(self.layer3, x, use_reentrant=False)
        else:
            x = self.layer3(x)
        
        x = x + self.corr3(x) * self.alpha[1]
        x = x + self.temporal_weight3(x)
        
        if self.training:
            x = checkpoint(self.layer4, x, use_reentrant=False)
        else:
            x = self.layer4(x)
        
        x = x + self.corr4(x) * self.alpha[2]
        x = x + self.temporal_weight4(x)
        
        x = x.transpose(1, 2).contiguous()
        x = x.view((-1,) + x.size()[2:])  # (N*T, C, H, W)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # (N*T, C)
        x = self.fc(x)
        
        return x


def resnet18(pretrained: bool = True, **kwargs) -> ResNet:
    """Constructs a ResNet-18 model with CorrNet+ correlation modules."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    if pretrained:
        checkpoint = model_zoo.load_url(MODEL_URLS['resnet18'], map_location='cpu')
        
        for ln in list(checkpoint.keys()):
            if 'conv' in ln or 'downsample.0.weight' in ln:
                checkpoint[ln] = checkpoint[ln].unsqueeze(2)
        
        model.load_state_dict(checkpoint, strict=False)
        del checkpoint
        gc.collect()
    
    return model


def resnet34(pretrained: bool = True, **kwargs) -> ResNet:
    """Constructs a ResNet-34 model with CorrNet+ correlation modules."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        checkpoint = model_zoo.load_url(MODEL_URLS['resnet34'], map_location='cpu')
        
        for ln in list(checkpoint.keys()):
            if 'conv' in ln or 'downsample.0.weight' in ln:
                checkpoint[ln] = checkpoint[ln].unsqueeze(2)
        
        model.load_state_dict(checkpoint, strict=False)
        del checkpoint
        gc.collect()
    
    return model
