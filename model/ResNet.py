from torch import nn
import torch

if __name__ == '__main__':
    net = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(num_features=64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    X = torch.rand((256, 3, 224, 224))
    print(net(X).shape)