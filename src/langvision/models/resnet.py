"""
ResNet implementation with LoRA support for efficient fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Type, Union
from .lora import LoRALinear, LoRAConfig


class LoRAConv2d(nn.Module):
    """Convolutional layer with LoRA adaptation."""
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int = 1, 
                 padding: int = 0,
                 r: int = 4,
                 alpha: float = 1.0,
                 dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        
        if r > 0:
            # LoRA decomposition for conv layers
            self.lora_A = nn.Parameter(torch.zeros(r, in_channels))
            self.lora_B = nn.Parameter(torch.zeros(out_channels, r))
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.reset_parameters()
    
    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.conv(x)
        
        if self.r > 0 and hasattr(self, 'lora_A'):
            # Apply LoRA adaptation for conv layers
            B, C, H, W = x.shape
            x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
            
            # LoRA forward pass
            lora_out = self.dropout(x_reshaped) @ self.lora_A.T @ self.lora_B.T
            lora_out = lora_out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            
            result = result + lora_out * self.scaling
        
        return result


class BasicBlock(nn.Module):
    """Basic ResNet block with LoRA support."""
    expansion = 1
    
    def __init__(self, 
                 inplanes: int, 
                 planes: int, 
                 stride: int = 1, 
                 downsample: Optional[nn.Module] = None,
                 lora_config: Optional[LoRAConfig] = None):
        super().__init__()
        
        if lora_config and lora_config.r > 0:
            self.conv1 = LoRAConv2d(inplanes, planes, 3, stride, 1, 
                                   r=lora_config.r, alpha=lora_config.alpha, dropout=lora_config.dropout)
            self.conv2 = LoRAConv2d(planes, planes, 3, 1, 1,
                                   r=lora_config.r, alpha=lora_config.alpha, dropout=lora_config.dropout)
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck ResNet block with LoRA support."""
    expansion = 4
    
    def __init__(self, 
                 inplanes: int, 
                 planes: int, 
                 stride: int = 1, 
                 downsample: Optional[nn.Module] = None,
                 lora_config: Optional[LoRAConfig] = None):
        super().__init__()
        
        if lora_config and lora_config.r > 0:
            self.conv1 = LoRAConv2d(inplanes, planes, 1, 1, 0,
                                   r=lora_config.r, alpha=lora_config.alpha, dropout=lora_config.dropout)
            self.conv2 = LoRAConv2d(planes, planes, 3, stride, 1,
                                   r=lora_config.r, alpha=lora_config.alpha, dropout=lora_config.dropout)
            self.conv3 = LoRAConv2d(planes, planes * self.expansion, 1, 1, 0,
                                   r=lora_config.r, alpha=lora_config.alpha, dropout=lora_config.dropout)
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
            self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """ResNet architecture with LoRA fine-tuning support."""
    
    def __init__(self, 
                 block: Type[Union[BasicBlock, Bottleneck]], 
                 layers: List[int], 
                 num_classes: int = 1000,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation: Optional[List[bool]] = None,
                 norm_layer: Optional[nn.Module] = None,
                 lora_config: Optional[LoRAConfig] = None):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.lora_config = lora_config
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                           "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                     dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                     dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                     dilate=replace_stride_with_dilation[2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head with LoRA support
        if lora_config and lora_config.r > 0:
            self.fc = LoRALinear(512 * block.expansion, num_classes,
                               r=lora_config.r, alpha=lora_config.alpha, dropout=lora_config.dropout)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, 
                   block: Type[Union[BasicBlock, Bottleneck]], 
                   planes: int, 
                   blocks: int,
                   stride: int = 1, 
                   dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                norm_layer(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 
                          groups=self.groups, base_width=self.base_width, 
                          dilation=previous_dilation, norm_layer=norm_layer,
                          lora_config=self.lora_config))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                              base_width=self.base_width, dilation=self.dilation,
                              norm_layer=norm_layer, lora_config=self.lora_config))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def resnet18(num_classes: int = 1000, lora_config: Optional[LoRAConfig] = None) -> ResNet:
    """ResNet-18 model with optional LoRA fine-tuning."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, lora_config=lora_config)


def resnet34(num_classes: int = 1000, lora_config: Optional[LoRAConfig] = None) -> ResNet:
    """ResNet-34 model with optional LoRA fine-tuning."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, lora_config=lora_config)


def resnet50(num_classes: int = 1000, lora_config: Optional[LoRAConfig] = None) -> ResNet:
    """ResNet-50 model with optional LoRA fine-tuning."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, lora_config=lora_config)


def resnet101(num_classes: int = 1000, lora_config: Optional[LoRAConfig] = None) -> ResNet:
    """ResNet-101 model with optional LoRA fine-tuning."""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, lora_config=lora_config)


def resnet152(num_classes: int = 1000, lora_config: Optional[LoRAConfig] = None) -> ResNet:
    """ResNet-152 model with optional LoRA fine-tuning."""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, lora_config=lora_config)
