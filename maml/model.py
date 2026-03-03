from collections import OrderedDict  # ✅ CORRIGER CETTE LIGNE!

import torch.nn as nn
from torch.nn import functional as F
from torchmeta.modules import MetaConv2d, MetaLinear, MetaModule, MetaSequential

def conv_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
            track_running_stats=False)),
        ('relu', nn.ReLU()),
        ('pool', nn.MaxPool2d(2))
    ]))

class MetaConvModel(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64):
        super(MetaConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True))
        ]))
        
        # ✅ AJOUTER GlobalAvgPool
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = MetaLinear(hidden_size, out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        
        # ✅ MODIFIER: Au lieu de flatten
        features = self.global_avg_pool(features)  # (batch, hidden_size, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, hidden_size)
        
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits

class MetaMLPModel(MetaModule):
    """Multi-layer Perceptron architecture from [1]."""
    def __init__(self, in_features, out_features, hidden_sizes):
        super(MetaMLPModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes

        layer_sizes = [in_features] + hidden_sizes
        self.features = MetaSequential(OrderedDict([('layer{0}'.format(i + 1),
            MetaSequential(OrderedDict([
                ('linear', MetaLinear(hidden_size, layer_sizes[i + 1], bias=True)),
                ('relu', nn.ReLU())
            ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
        self.classifier = MetaLinear(hidden_sizes[-1], out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits

def ModelConvOmniglot(out_features, hidden_size=64):
    return MetaConvModel(1, out_features, hidden_size=hidden_size,
                         feature_size=hidden_size)

def ModelConvMiniImagenet(out_features, hidden_size=64):
    # ✅ MODIFIER: feature_size = hidden_size (pas 5*5*hidden_size)
    return MetaConvModel(3, out_features, hidden_size=hidden_size,
                         feature_size=hidden_size)

def ModelMLPSinusoid(hidden_sizes=[40, 40]):
    return MetaMLPModel(1, 1, hidden_sizes)

if __name__ == '__main__':
    model = ModelMLPSinusoid()