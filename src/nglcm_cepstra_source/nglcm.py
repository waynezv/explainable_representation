# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import pdb


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.projector = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024 + 1024, num_classes),
        )

        # self.classifier = nn.Sequential(
            # nn.Linear(1024, num_classes),
        # )

    def _project(self, Fa, Fg):
        Fl = torch.matmul(
            (1 - torch.matmul(
                torch.matmul(Fg, torch.inverse(torch.matmul(Fg.t(), Fg))),
                Fg.t())
             ),
            Fa)

        return Fl

    def forward(self, x, gx):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        hx = self.projector(x)

        hgx = torch.cat((hx, gx), dim=1)

        Fa = self.classifier(hgx)

        hx_null = torch.zeros(*hx.size()).to(x.device)
        Fg = self.classifier(torch.cat((hx_null, gx), dim=1))

        Fl = self._project(Fa, Fg)

        # Fl = self.classifier(hx)

        return Fl

    def predict(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        hx = self.projector(x)

        gx_null = torch.zeros(*hx.size()).to(x.device)

        hgx = torch.cat((hx, gx_null), dim=1)

        Fp = self.classifier(hgx)

        # Fp = self.classifier(hx)

        p_yx = nn.functional.softmax(Fp, dim=1)
        _, yh = torch.max(p_yx, 1)
        return yh


class FilterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(11, 3), stride=(4, 1), padding=(2, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=(5, 3), stride=(2, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.projector = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        gx = self.projector(x)

        return gx
