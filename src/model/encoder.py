from copy import deepcopy
import torch
from torch import nn
from torchvision import models as tv_models

from .tools import get_output_size, Identity, kaiming_weights_init


def get_resnet_model(name):
    if name is None:
        name = 'resnet18'
    return {
        'resnet18': tv_models.resnet18,
        'resnet34': tv_models.resnet34,
        'resnet50': tv_models.resnet50,
        'resnet101': tv_models.resnet101,
        'resnet152': tv_models.resnet152,
        'resnext50_32x4d': tv_models.resnext50_32x4d,
        'resnext101_32x8d': tv_models.resnext101_32x8d,
        'wide_resnet50_2': tv_models.wide_resnet50_2,
        'wide_resnet101_2': tv_models.wide_resnet101_2,
    }[name]


class TxtEncoder(nn.Module):
    color_channels = 3

    def __init__(self, img_size, **kwargs):
        super().__init__()
        kwargs = deepcopy(kwargs)
        self.with_pool = kwargs.pop('with_pool', True)
        name = kwargs.pop('name', 'resnet18')
        self.name = name
        pretrained = kwargs.pop('pretrained', False)
        freeze = kwargs.pop('freeze', False)
        n_features = kwargs.pop('n_features', 512)
        assert len(kwargs) == 0
        if name == 'identity':
            self.encoder = Identity()
        else:
            resnet = get_resnet_model(name)(pretrained=pretrained, progress=False)
            if freeze:
                for param in resnet.parameters():
                    param.requires_grad = False
            seq = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                   resnet.layer1, resnet.layer2, resnet.layer3]

            self.encoder = nn.Sequential(*seq)

        out_ch = get_output_size(self.color_channels, img_size, self.encoder, name=='vit')
        fc = nn.Sequential()
        if n_features is not None:
            proj = nn.Conv2d(out_ch, n_features, 1)
            out_ch = n_features
        self.out_ch = out_ch
        self.proj = proj

    def forward(self, x):
        return self.proj(self.encoder(x))


class Encoder(nn.Module):
    color_channels = 3

    def __init__(self, img_size, **kwargs):
        super().__init__()
        kwargs = deepcopy(kwargs)
        self.with_pool = kwargs.pop('with_pool', True)
        name = kwargs.pop('name', 'resnet18')
        self.name = name
        pretrained = kwargs.pop('pretrained', False)
        freeze = kwargs.pop('freeze', False)
        n_features = kwargs.pop('n_features', None)
        assert len(kwargs) == 0
        if name == 'identity':
            self.encoder = Identity()
        else:
            resnet = get_resnet_model(name)(pretrained=pretrained, progress=False)
            if freeze:
                for param in resnet.parameters():
                    param.requires_grad = False
            seq = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                   resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
            if self.with_pool:
                size = self.with_pool if isinstance(self.with_pool, (tuple, list)) else (1, 1)
                seq.append(torch.nn.AdaptiveAvgPool2d(output_size=size))
            self.encoder = nn.Sequential(*seq)

        out_ch = get_output_size(self.color_channels, img_size, self.encoder, name=='vit')
        fc = nn.Sequential()
        if n_features is not None:
            if out_ch != n_features:
                assert n_features < out_ch
                fc = nn.Linear(out_ch, n_features)
                _ = kaiming_weights_init(fc)
                out_ch = n_features
        self.out_ch = out_ch
        self.fc = fc

    def forward(self, x):
        if self.name == 'vit':
            y = self.encoder.get_intermediate_layers(x)[0]
            y=  y.flatten(1)
            y = self.fc(y)
            return y
        else:
            return self.fc(self.encoder(x).flatten(1))

