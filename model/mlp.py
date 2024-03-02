import torch
import torch.nn as nn


def MLP():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


def test():
    net = MLP()
    y = net(torch.randn(16, 1, 28, 28))
    print(y.size())


if __name__ == "main":
    print("main")
