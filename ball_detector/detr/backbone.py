# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Backbone modules."""
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from .utils import NestedTensor, is_main_process


@dataclass
class Parameters:
    hidden_dim: int = 256
    position_embedding: str = "sine"
    lr_backbone: float = 1e-5
    masks: bool = False
    backbone: str = "resnet50"


# pylint: disable=too-few-public-methods
class FrozenBatchNorm2d(torch.nn.Module):
    """BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any
    other models than torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load state dictionary.

        Custom handling for loading state dictionary.
        """
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


# pylint: disable=too-few-public-methods
class BackboneBase(nn.Module):
    """Base class for backbones."""

    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        """Initialize backbone base.

        Args:
            backbone: Backbone model
            train_backbone: Whether to train the backbone
            num_channels: Number of output channels
            return_interm_layers: Whether to return intermediate layers
        """
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        """Forward pass.

        Args:
            tensor_list: Input nested tensor

        Returns:
            Dictionary of output tensors
        """
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


# pylint: disable=too-few-public-methods
class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        """Initialize ResNet backbone.

        Args:
            name: Name of the backbone
            train_backbone: Whether to train the backbone
            return_interm_layers: Whether to return intermediate layers
            dilation: Whether to use dilation
        """
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=FrozenBatchNorm2d,
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


# pylint: disable=too-few-public-methods
class Joiner(nn.Sequential):
    """Joiner module that combines backbone with position encoding."""

    def __init__(self, backbone, position_embedding):
        """Initialize joiner.

        Args:
            backbone: Backbone model
            position_embedding: Position embedding module
        """
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        """Forward pass.

        Args:
            tensor_list: Input nested tensor

        Returns:
            Tuple of output tensors and position encodings
        """
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

