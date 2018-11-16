# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import pdb


def _Dmat(N):
    '''
    Generate D matrix of dimension N-by-N.
    Along direction 0.
    '''
    D = torch.eye(N, dtype=torch.float32)
    D = D + torch.diag(-torch.ones(N - 1, dtype=torch.float32),
                       diagonal=1)
    return D


class GLCM(nn.Module):
    '''
    Neural gray-level co-occurrence matrix model.
    '''
    def __init__(self, in_dim=227 * 227, num_pixel_vals=256):
        super().__init__()
        self.in_dim = in_dim
        self.num_pixel_vals = num_pixel_vals

        self.D = _Dmat(self.in_dim)
        self.phi_a = nn.Parameter(torch.FloatTensor(self.num_pixel_vals).
                                  uniform_(0, num_pixel_vals))
        self.phi_b = nn.Parameter(torch.FloatTensor(self.num_pixel_vals).
                                  uniform_(0, num_pixel_vals))

        self.weight = nn.Parameter(torch.FloatTensor(
            self.num_pixel_vals * num_pixel_vals, 32).normal_())
        self.bias = nn.Parameter(torch.FloatTensor(32).normal_())

    def _b(self, a):
        '''
        Compute b.

        Parameters
        ----------
        a: torch.FloatTensor, shape (B, C * M * M)

        Returns
        -------
        b: torch.FloatTensor, shape (B, C * M * M)
        '''
        # Memory efficient version
        # B, m = a.size()
        # b = torch.zeros(B, m).to(a.device)

        # for i in range(m - 1):
            # b[:, i] = a[:, i] - a[:, i + 1]

        # b[:, -1] = a[:, -1]

        # Computation efficient version
        D = self.D.to(a.device)
        b = torch.einsum('ij,jk->ik', (D, a.t())).t()

        return b

    def _s_fun(self, a, b):
        '''
        S thresholding function.

        Parameters
        ----------
        a, b: torch.FloatTensor, shape (B, C * M * M)
            B batchsize, C channel, M image size.

        Returns
        -------
        sa, sb: torch.FloatTensor, shape (B, N, C * M * M)
            N num_pixel_vals.
        '''
        B, m = a.size()
        N = self.phi_a.size(0)

        a = a.reshape((B, 1, m))
        b = b.reshape((B, 1, m))
        phi_a = self.phi_a.reshape((N, 1))
        phi_b = self.phi_b.reshape((N, 1))

        a = torch.clamp(a - phi_a, min=0., max=1.)
        b = torch.clamp(b - phi_b, min=0., max=1.)

        return a, b

    def forward(self, x):
        '''
        Parameters
        ----------
        x: torch.FloatTensor, shape (B, C, M, M)
            C channel is 1, gray.

        Returns
        -------
        out: torch.FloatTensor, shape (B, out_size)
        '''
        B, _, _, _ = x.size()

        # Flatten
        x = x.view(B, -1)

        a = x
        b = self._b(a)

        sa, sb = self._s_fun(a, b)

        # out = torch.matmul(sa, sb.permute(0, 2, 1)).view(B, -1)
        out = torch.einsum('bij,bjk->bik', (sa, sb.permute(0, 2, 1))). \
            view(B, -1)

        # out = torch.matmul(out, self.weight) + self.bias
        # out = torch.sigmoid(torch.matmul(out, self.weight) + self.bias)
        # out = torch.tanh(torch.matmul(out, self.weight) + self.bias)
        out = torch.relu(torch.matmul(out, self.weight) + self.bias)
        return out


class AlexNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
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
                nn.Linear(256 * 1 * 1, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(32, 32),
                nn.ReLU(inplace=True),
            )

            self.classifier = nn.Sequential(
                nn.Linear(32 + 32, num_classes),
            )

            # self.classifier = nn.Sequential(
                # nn.Linear(32, num_classes),
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
            x = x.view(x.size(0), 256 * 1 * 1)
            hx = self.projector(x)

            hgx = torch.cat((hx, gx), dim=1)

            # Normalize
            # hgx_min = hgx.min(dim=0, keepdim=True)[0]
            # hgx_max = hgx.max(dim=0, keepdim=True)[0]
            # hgx = (hgx - hgx_min) / (hgx_max - hgx_min)

            Fa = self.classifier(hgx)

            hx_null = torch.zeros(*hx.size()).to(x.device)
            Fg = self.classifier(torch.cat((hx_null, gx), dim=1))

            Fl = self._project(Fa, Fg)

            # Fl = self.classifier(hx)

            return Fl

        def predict(self, x):
            B, C, _, _ = x.size()

            gx_null = torch.zeros(B, 32).to(x.device)

            x = self.features(x)
            x = x.view(x.size(0), 256 * 1 * 1)
            hx = self.projector(x)

            hgx = torch.cat((hx, gx_null), dim=1)

            Fp = self.classifier(hgx)

            # Fp = self.classifier(hx)

            p_yx = nn.functional.softmax(Fp, dim=1)
            _, yh = torch.max(p_yx, 1)
            return yh
