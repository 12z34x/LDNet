from .resnet import resnet50, resnet101
from .xception import AlignedXception
from .mobilenetv2 import MobileNetV2
from .mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
from .ghostnet import ghostnet
from .vovnet import (vovnet27_slim, vovnet39, vovnet57)
from .res2net import (res2net101_v1b_26w_4s)


def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet50':
        return resnet50(output_stride, BatchNorm)

    elif backbone == 'resnet101':
        return resnet101(output_stride, BatchNorm)

    elif backbone == 'res2net101':
        return res2net101_v1b_26w_4s()

    elif backbone == 'vovnet27_slim':
        return vovnet27_slim()

    elif backbone == 'vovnet39':
        return vovnet39()

    elif backbone == 'vovnet57':
        return vovnet57(pretrained=True)

    elif backbone == 'xception':
        return AlignedXception(output_stride, BatchNorm)

    elif backbone == 'mobilenetv2':
        return MobileNetV2(output_stride, BatchNorm)

    elif backbone == 'mobilenetv3_s':
        return MobileNetV3_Small()

    elif backbone == 'mobilenetv3_l':
        return MobileNetV3_Large()

    elif backbone == "ghostnet":
        return ghostnet(output_stride=output_stride)

    else:
        raise NotImplementedError
