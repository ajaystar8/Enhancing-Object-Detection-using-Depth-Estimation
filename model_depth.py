import torch
import torch.nn as nn
from torchsummary import summary


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

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20, depth_in_channels=3):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes

        # Pre-Merge Configurations
        self.rgb_config = [
            (32, 3, 1),
            (64, 3, 2),
        ]
        self.depth_config = [
            (32, 3, 1),
            (64, 3, 2),
        ]
        
        # Create Layers
        self.rgb_pre_merge_layers = self._create_conv_layers(self.rgb_config, in_channels=in_channels)
        self.depth_pre_merge_layers = self._create_conv_layers(self.depth_config, in_channels=depth_in_channels)

        self.merged_channels = self._calculate_merged_channels()

        # Post-Merge Configurations
        self.post_merge_config = [
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

        # Create post-merge layers
        self.post_merge_layers = self._create_conv_layers(self.post_merge_config, in_channels=self.merged_channels)

    def forward(self, rgb, depth):
        # Process RGB through pre-merge layers
        x_rgb = rgb
        for layer in self.rgb_pre_merge_layers:
            x_rgb = layer(x_rgb)

        # Process Depth through pre-merge layers
        x_depth = depth
        for layer in self.rgb_pre_merge_layers:
            x_depth = layer(x_depth)

        print(f"RGB output shape: {x_rgb.shape}")
        print(f"Depth output shape: {x_depth.shape}")

        # Merge RGB and Depth features
        x = torch.cat([x_rgb, x_depth], dim=1)

        print(f"Merged tensor shape: {x.shape}")
        print(f"Merged channels: {self.merged_channels}")

        # Process merged features through post-merge layers
        outputs = []
        route_connections = []
        for layer in self.post_merge_layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs
    
    def _calculate_merged_channels(self):
        """Calculate the number of channels after merging RGB and Depth features."""
        rgb_out_channels = self._get_output_channels(self.rgb_config, in_channels=3)
        depth_out_channels = self._get_output_channels(self.depth_config, in_channels=3)
        return rgb_out_channels + depth_out_channels

    def _get_output_channels(self, config, in_channels):
        """Helper to calculate the output channels of a config sequence."""
        for module in config:
            if isinstance(module, tuple):
                out_channels, _, _ = module
                in_channels = out_channels
            elif isinstance(module, list):
                pass  # Residual blocks do not change the number of channels
        return in_channels

    def _create_conv_layers(self, config, in_channels=3):
        layers = nn.ModuleList()
        print(f"Beginning with {in_channels} channels")
        for module in config:
            print(f"Creating Conv Layer: in_channels={in_channels}")
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                # print(f"Creating Conv Layer: in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size}")
                layers.append(
                    CNNBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1 if kernel_size == 3 else 0)
                )
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
                    layers.append(nn.Upsample(scale_factor=2, mode="bilinear"))
                    in_channels = in_channels * 3
        return layers


# Testing the model
def test():
    num_classes = 80
    model = YOLOv3(num_classes=num_classes).to("cuda")

    # RGB input: 3 channels, Depth input: 3 channel
    rgb = torch.randn((2, 3, 608, 608)).to("cuda")
    depth = torch.randn((2, 3, 608, 608)).to("cuda")

    outputs = model(rgb, depth)

    assert outputs[0].shape == (2, 3, 608 // 32, 608 // 32, 5 + num_classes)
    summary(model, [(3, 608, 608), (3, 608, 608)])
    print("Success!")


if __name__ == "__main__":
    test()