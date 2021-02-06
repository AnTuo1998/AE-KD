from .classifier import LinearClassifier
from .mobilenetv2 import mobile_half
from .resnet import (resnet8, resnet8x4, resnet14, resnet20, resnet32,
                     resnet32x4, resnet44, resnet56, resnet110)
from .resnet_feat_at import (resnet_feat_at_14, resnet_feat_at_20,
                             resnet_feat_at_110)
from .resnetv2 import ResNet18, ResNet34, ResNet50, ResNet101
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .vgg import vgg8_bn, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .wrn_cifar10 import (wrn_16_1_cifar10, wrn_16_2_cifar10, wrn_40_1_cifar10,
                          wrn_40_2_cifar10)

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'wrn_16_1_cifar10': wrn_16_1_cifar10,
    'wrn_16_2_cifar10': wrn_16_2_cifar10,
    'wrn_40_1_cifar10': wrn_40_1_cifar10,
    'wrn_40_2_cifar10': wrn_40_2_cifar10,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'resnet_feat_at_110': resnet_feat_at_110,
    'resnet_feat_at_14': resnet_feat_at_14,
    'resnet_feat_at_20': resnet_feat_at_20,
    'linear_classifier': LinearClassifier,
}
