from torch import nn
import torch


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(kernel_size=3,
                               in_channels=in_channels,
                               out_channels=out_channels,
                               stride=stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(kernel_size=3,
                               in_channels=out_channels,
                               out_channels=out_channels,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResidualLayer(nn.Sequential):

    def __init__(self, layers):
        super().__init__(*layers)

    @classmethod
    def make_layer(cls, in_channels, out_channels, num_conv):
        layers = []
        if in_channels == out_channels:
            layers.append(ResidualBlock(in_channels=in_channels, out_channels=out_channels, stride=1, downsample=None))
        else:
            layers.append(ResidualBlock(in_channels=in_channels, out_channels=out_channels, stride=2,
                                        downsample=nn.Sequential(
                                            nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=2),
                                            nn.BatchNorm2d(num_features=out_channels))))
        for _ in range(1, num_conv):
            layers.append(ResidualBlock(in_channels=out_channels, out_channels=out_channels, stride=1, downsample=None))
        return ResidualLayer(layers)