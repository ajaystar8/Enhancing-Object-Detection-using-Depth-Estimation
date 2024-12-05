import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models import ResNet18_Weights, resnet18

from utils.utils import extract_layers

"""
Tuple: (filters, kernel_size, stride) 
List: ["B", number of repeats]
"U": upsampling the feature map and concatenating with a previous layer
"S": scale prediction block and computing the YOLO loss
Note: Every convolution is a "same" convolution 
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    # first route from the end of the previous block
    (512, 3, 2),
    ["B", 8],
    # second route from the end of the previous block
    (1024, 3, 2),
    ["B", 4],
    # To this point is Darknet-53
    (512, 1, 1),
    # resnet concat -> out-shape = in-shape -> No changes required
    "C",
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class ResnetBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet_backbone = torch.nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.resnet_backbone(x)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super(CNNBlock, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kwargs.get("kernel_size")

        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if bn_act else None
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers.append(
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1)
                )
            )
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ScalePrediction, self).__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1)
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class ResYOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super(ResYOLOv3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    """
    Loads weights for the backbone of yolov3. Backbone is till (not including) the conv layer: Conv2d(512, 1, 1)
    """

    def freeze_backbone_weights(self):
        module_list = extract_layers(self)
        for idx, module_item in enumerate(module_list):
            if idx == 62:
                break
            _, module = module_item[0], module_item[1]
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x, hha):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            elif isinstance(layer, ResnetBlock):
                x = x + layer(hha)
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(CNNBlock(in_channels,
                                       out_channels,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=1 if kernel_size == 3 else 0))
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
                    in_channels = in_channels * 3

                elif module == "C":
                    layers.append(ResnetBlock())

        return layers


def test():
    num_classes = 19
    model = ResYOLOv3(num_classes=19)
    img_size = 416
    x = torch.randn((2, 3, img_size, img_size))
    hha = torch.randn((2, 3, img_size, img_size))
    out = model(x, hha)
    assert out[0].shape == (2, 3, img_size // 32, img_size // 32, 5 + num_classes)
    assert out[1].shape == (2, 3, img_size // 16, img_size // 16, 5 + num_classes)
    assert out[2].shape == (2, 3, img_size // 8, img_size // 8, 5 + num_classes)
    summary(model, [(3, 608, 608), (3, 608, 608)])
    print("Success!")


if __name__ == '__main__':
    test()
