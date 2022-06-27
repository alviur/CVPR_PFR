from typing import OrderedDict
import torch
import lightly
from lightly.models.simsiam import _projection_mlp


def ss_resnet18_lightly(pretrained=False):
    # create a ResNet backbone and remove the classification head
    resnet = lightly.models.ResNetGenerator('resnet-18')
    return torch.nn.Sequential(*list(resnet.children())[:-1], torch.nn.AdaptiveAvgPool2d(1))


def resnet18_simsiam(pretrained=False):
    backbone = ss_resnet18_lightly(pretrained)
    return lightly.models.SimSiam(backbone, num_ftrs=512, num_mlp_layers=2)


def resnet18_and_proj(
    pretrained=False,
    num_ftrs: int = 2048,
    proj_hidden_dim: int = 2048,
    out_dim: int = 2048,
    num_mlp_layers: int = 3,
    num_classes=10
):
    # create a ResNet backbone and remove the classification head
    resnet = lightly.models.ResNetGenerator('resnet-18')
    # add projection layer
    proj = _projection_mlp(num_ftrs, proj_hidden_dim, out_dim, num_mlp_layers)
    model = torch.nn.Sequential(OrderedDict({
        'backbone': torch.nn.Sequential(*list(resnet.children())[:-1], torch.nn.AdaptiveAvgPool2d(1)),
        'proj': proj,
        'fc': torch.nn.Linear(out_dim, num_classes)
    }))
    model.head_var = 'fc'
    return model
