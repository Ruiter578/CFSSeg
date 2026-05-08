from typing import Union
from .utils import IntermediateLayerGetter
from ._deeplab import * 
from .backbone import resnet
from .backbone import mobilenetv2
from typing import Callable, Dict, Optional

class DeepLabModelFactory:
    """
    DeepLab 模型工厂类，用于创建不同配置的 DeepLabV3 和 DeepLabV3+ 模型。
    """

    def __init__(self):
        self.model_map: Dict[str, Callable] = self._get_model_map()

    def _get_model_map(self) -> Dict[str, Callable]:
        """
        获取模型名称到构建函数的映射字典。
        """
        return {
            'deeplabv3_resnet50': self.deeplabv3_resnet50,
            'deeplabv3plus_resnet50': self.deeplabv3plus_resnet50,
            'deeplabv3_resnet101': self.deeplabv3_resnet101,
            'deeplabv3plus_resnet101': self.deeplabv3plus_resnet101,
            'deeplabv3_mobilenet': self.deeplabv3_mobilenet,
            'deeplabv3plus_mobilenet': self.deeplabv3plus_mobilenet,
            'deeplabv3bga_resnet101': self.deeplabv3_resnet101_bga,
        }
    
    def _segm_resnet(self, name: str, backbone_name: str, num_classes: int, output_stride: int,
                    pretrained_backbone: bool, bn_freeze: bool) -> DeepLabV3:
        """
        创建基于 ResNet 的 DeepLab 模型。
        """
        if output_stride == 4:
            replace_stride_with_dilation = [True, True, True]
            aspp_dilate = [24, 48, 72]
        elif output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            replace_stride_with_dilation = [False, False, True]
            aspp_dilate = [6, 12, 18]

        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=replace_stride_with_dilation
        )

        inplanes = 2048
        low_level_planes = 256

        return_layers = {'layer4': 'out'}
        if name == 'deeplabv3':
            classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
        elif name == "deeplabv3_bga":
            classifier = DeepLabHeadBgA(inplanes, num_classes, aspp_dilate)
        else:
            raise ValueError(f"Unsupported model name '{name}' for ResNet backbone.")

        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        model = DeepLabV3(backbone, classifier, bn_freeze)
        return model
    
    def _segm_mobilenet(self, name: str, backbone_name: str, num_classes: int, output_stride: int,
                       pretrained_backbone: bool, bn_freeze: bool) -> DeepLabV3:
        """
        创建基于 MobileNetV2 的 DeepLab 模型。
        """
        if output_stride == 8:
            aspp_dilate = [12, 24, 36]
        else:
            aspp_dilate = [6, 12, 18]

        backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)

        # 重命名层
        backbone.low_level_features = backbone.features[0:4]
        backbone.high_level_features = backbone.features[4:-1]
        backbone.features = None
        backbone.classifier = None

        inplanes = 320
        low_level_planes = 24

        if name == 'deeplabv3plus':
            return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
            classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
        elif name == 'deeplabv3':
            return_layers = {'high_level_features': 'out'}
            classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
        else:
            raise ValueError(f"Unsupported model name '{name}' for MobileNetV2 backbone.")

        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        model = DeepLabV3(backbone, classifier, bn_freeze)
        return model

    def _load_model(self, arch_type: str, backbone: str, num_classes: int, output_stride: int,
                   pretrained_backbone: bool, bn_freeze: bool) -> DeepLabV3:
        """
        根据指定的 backbone 名称加载相应的模型。
        """
        if backbone == 'mobilenetv2':
            return self._segm_mobilenet(
                name=arch_type,
                backbone_name=backbone,
                num_classes=num_classes,
                output_stride=output_stride,
                pretrained_backbone=pretrained_backbone,
                bn_freeze=bn_freeze
            )
        elif backbone.startswith('resnet'):
            return self._segm_resnet(
                name=arch_type,
                backbone_name=backbone,
                num_classes=num_classes,
                output_stride=output_stride,
                pretrained_backbone=pretrained_backbone,
                bn_freeze=bn_freeze
            )
        else:
            raise NotImplementedError(f"Backbone '{backbone}' is not supported.")
    
    def deeplabv3_resnet50(self, num_classes: int = 21, output_stride: int = 8,
                           pretrained_backbone: bool = True, bn_freeze: bool = False) -> DeepLabV3:
        """构建一个带有 ResNet-50 主干的 DeepLabV3 模型。"""
        return self._load_model(
            arch_type='deeplabv3',
            backbone='resnet50',
            num_classes=num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
            bn_freeze=bn_freeze
        )

    def deeplabv3_resnet101(self, num_classes: int = 21, output_stride: int = 8,
                            pretrained_backbone: bool = True, bn_freeze: bool = False) -> DeepLabV3:
        """构建一个带有 ResNet-101 主干的 DeepLabV3 模型。"""
        return self._load_model(
            arch_type='deeplabv3',
            backbone='resnet101',
            num_classes=num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
            bn_freeze=bn_freeze
        )

    def deeplabv3_mobilenet(self, num_classes: int = 21, output_stride: int = 8,
                            pretrained_backbone: bool = True, bn_freeze: bool = False, **kwargs) -> DeepLabV3:
        """构建一个带有 MobileNetV2 主干的 DeepLabV3 模型。"""
        return self._load_model(
            arch_type='deeplabv3',
            backbone='mobilenetv2',
            num_classes=num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
            bn_freeze=bn_freeze
        )

    def deeplabv3_resnet101_bga(self, num_classes: int = 21, output_stride: int = 8,
                                 pretrained_backbone: bool = True, bn_freeze: bool = False) -> DeepLabV3:
        """构建一个带有 ResNet-101 主干的 DeepLabV3 BGA 模型。"""
        return self._load_model(
            arch_type='deeplabv3_bga',
            backbone='resnet101',
            num_classes=num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
            bn_freeze=bn_freeze
        )

    def deeplabv3plus_resnet50(self, num_classes: int = 21, output_stride: int = 8,
                                pretrained_backbone: bool = True, bn_freeze: bool = False) -> DeepLabV3:
        """构建一个带有 ResNet-50 主干的 DeepLabV3+ 模型。"""
        return self._load_model(
            arch_type='deeplabv3plus',
            backbone='resnet50',
            num_classes=num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
            bn_freeze=bn_freeze
        )

    def deeplabv3plus_resnet101(self, num_classes: int = 21, output_stride: int = 8,
                                 pretrained_backbone: bool = True, bn_freeze: bool = False) -> DeepLabV3:
        """构建一个带有 ResNet-101 主干的 DeepLabV3+ 模型。"""
        return self._load_model(
            arch_type='deeplabv3plus',
            backbone='resnet101',
            num_classes=num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
            bn_freeze=bn_freeze
        )

    def deeplabv3plus_mobilenet(self, num_classes: int = 21, output_stride: int = 8,
                                 pretrained_backbone: bool = True, bn_freeze: bool = False) -> DeepLabV3:
        """构建一个带有 MobileNetV2 主干的 DeepLabV3+ 模型。"""
        return self._load_model(
            arch_type='deeplabv3plus',
            backbone='mobilenetv2',
            num_classes=num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
            bn_freeze=bn_freeze
        )
