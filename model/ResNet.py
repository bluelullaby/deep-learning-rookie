from torch import nn
from layer.ResNetLayer import *
import torch


class ResNet18(nn.Sequential):

    def __init__(self):
        layers = [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualLayer.make_layer(64, 64, 2),
            ResidualLayer.make_layer(64, 128, 2),
            ResidualLayer.make_layer(128, 256, 2),
            ResidualLayer.make_layer(256, 512, 2),
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        ]
        super(ResNet18, self).__init__(*layers)
