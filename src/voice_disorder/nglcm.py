# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import pdb


class Predictor(nn.Module):
    def __init__(self, input_dim=None, num_classes=4):
        super().__init__()
        assert input_dim is not None, 'Input_dim unspecified!'
        self.input_dim = input_dim

        self.features = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 300),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # self.classifier = nn.Sequential(
            # nn.Linear(300 + 300, num_classes),
        # )

        self.classifier = nn.Sequential(
            nn.Linear(300, num_classes),
        )

    def _project(self, Fa, Fg):
        tol = torch.eye(Fa.size(1)).to(Fa.device) * 0.
        Fl = torch.matmul(
            (1 - torch.matmul(
                torch.matmul(Fg,
                             torch.inverse(torch.matmul(Fg.t(), Fg) + tol)),
                Fg.t())
             ),
            Fa)
        return Fl

    def forward(self, x, gx):
        x = x.view(x.size(0), self.input_dim)
        hx = self.features(x)

        # hgx = torch.cat((hx, gx), dim=1)
        # Fa = self.classifier(hgx)

        # hx_null = torch.zeros(*hx.size()).to(x.device)
        # Fg = self.classifier(torch.cat((hx_null, gx), dim=1))

        # Fl = self._project(Fa, Fg)

        Fl = self.classifier(hx)

        return Fl

    def predict(self, x):
        x = x.view(x.size(0), self.input_dim)
        hx = self.features(x)

        # gx_null = torch.zeros(*hx.size()).to(x.device)

        # hgx = torch.cat((hx, gx_null), dim=1)

        # Fp = self.classifier(hgx)

        Fp = self.classifier(hx)

        p_yx = nn.functional.softmax(Fp, dim=1)
        _, yh = torch.max(p_yx, 1)
        return yh


class FilterNet(nn.Module):
    def __init__(self, input_dim=None):
        super().__init__()
        assert input_dim is not None, 'Input_dim unspecified!'
        self.input_dim = input_dim

        self.features = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        gx = self.features(x)
        return gx
