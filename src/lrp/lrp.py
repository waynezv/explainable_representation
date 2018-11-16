# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    def __init__(self, linear):
        super().__init__(
            in_features=linear.in_features,
            out_features=linear.out_features
        )

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias

    def relprop(self, R):
        V = torch.clamp(self.weight, min=0)
        Z = torch.mm(self.X, torch.transpose(V, 0, 1)) + 1e-9
        S = R / Z
        C = torch.mm(S, V)
        R = self.X * C
        return R


class Conv2d(nn.Conv2d):
    def __init__(self, conv2d):
        super().__init__(
            conv2d.in_channels,
            conv2d.out_channels,
            conv2d.kernel_size,
            stride=conv2d.stride,
            padding=conv2d.padding,
            dilation=conv2d.dilation,
            groups=conv2d.groups,
            bias=True
        )

        self.weight = conv2d.weight
        self.bias = conv2d.bias

    def gradprop(self, DY):
        output_padding = self.X.size()[2] - \
            ((self.Y.size()[2] - 1) * self.stride[0] -
             2 * self.padding[0] + self.kernel_size[0])

        return F.conv_transpose2d(DY, self.weight, stride=self.stride,
                                  padding=self.padding,
                                  output_padding=output_padding)

    def relprop(self, R):
        Z = self.Y + 1e-9
        S = R / Z
        C = self.gradprop(S)
        R = self.X * C
        return R


class MaxPool2d(nn.MaxPool2d):
    def __init__(self, maxpool2d):
        super().__init__(
            kernel_size=maxpool2d.kernel_size,
            stride=maxpool2d.stride
        )

        self.kernel_size = maxpool2d.kernel_size
        self.stride = maxpool2d.stride
        self.padding = maxpool2d.padding
        self.dilation = maxpool2d.dilation
        self.return_indices = maxpool2d.return_indices
        self.ceil_mode = maxpool2d.ceil_mode

    def gradprop(self, DY):
        DX = self.X * 0

        _, indices = F.max_pool2d(self.X, self.kernel_size, self.stride,
                                  self.padding, self.dilation,
                                  self.ceil_mode, True)

        out_size = ((DY.size(2) - 1) * self.stride -
                    2 * self.padding + self.kernel_size)
        # BUG: size mismatch
        if out_size == 55:
            out_size = 55 + 1
        output_size = torch.Size([DY.size(0), DY.size(1), out_size, out_size])

        DX = F.max_unpool2d(DY, indices, self.kernel_size, self.stride,
                            self.padding,
                            output_size=output_size)
        return DX

    def relprop(self, R):
        Z = self.Y + 1e-9
        S = R / Z
        C = self.gradprop(S)
        R = self.X * C
        return R


class ReLU(nn.ReLU):
    def relprop(self, R):
        return R


class Reshape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(-1, 256 * 6 * 6)

    def relprop(self, R):
        return R.view(-1, 256, 6, 6)


class AlexNetLRP(nn.Module):
    def __init__(self, alex):
        super().__init__()
        self.layers = nn.Sequential(
            Conv2d(alex.features[0]),
            ReLU(),
            MaxPool2d(alex.features[2]),
            Conv2d(alex.features[3]),
            ReLU(),
            MaxPool2d(alex.features[5]),
            Conv2d(alex.features[6]),
            ReLU(),
            Conv2d(alex.features[8]),
            ReLU(),
            Conv2d(alex.features[10]),
            ReLU(),
            MaxPool2d(alex.features[12]),
            Reshape(),
            Linear(alex.classifier[1]),
            ReLU(),
            Linear(alex.classifier[4]),
            ReLU(),
            Linear(alex.classifier[6])
        )

    def forward(self, x):
        x = self.layers(x)
        return x

    def relprop(self, R):
        for l in range(len(self.layers), 0, -1):
            R = self.layers[l - 1].relprop(R)
        return R
