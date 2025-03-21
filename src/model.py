"""
Discriminator and Generator implementation from DCGAN paper

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-01: Initial coding
* 2022-12-20: Small revision of code, checked that it works with latest PyTorch version
"""
import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary


class Discriminator(nn.Module):
    def __init__(self, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            # input: N x img_size[0] x W x H
            nn.Conv2d(self.img_size[0], 32 * self.img_size[0], kernel_size=3, stride=1, padding=1), # img: W x H
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(32 * self.img_size[0], 64 * self.img_size[0], 3, 2, 1), # img: W/2 x H/2
            self._block(64 * self.img_size[0], 128 * self.img_size[0], 3, 2, 1), # img: W/4 x H/4
            self._block(128 * self.img_size[0], 128 * self.img_size[0], 3, 2, 1), # img: W/8 x H/8
            # nn.Conv2d(128, 1, kernel_size=3, stride=2, padding=0), #img: 1x1
        )
        self.adv = nn.Sequential(
            nn.Linear(2 * np.prod(self.img_size), 1)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        x = self.disc(x)
        x = x.reshape(x.shape[0], -1)
        return self.adv(x)


class Generator(nn.Module):
    def __init__(self, img_size, latent_dim):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.layer1 = nn.Sequential(
            nn.Linear(latent_dim, 2 * np.prod(img_size)),
            nn.LeakyReLU(0.2),
        )
        self.net = nn.Sequential(
            # Input: N x 128C x W/8 x H/8
            self._block(128 * self.img_size[0], 128 * self.img_size[0], 3, 1, 1),  # img: W/8 x H/8
            self._block(128 * self.img_size[0], 64 * self.img_size[0], 4, 2, 1),  # img: W/4 x H/4
            self._block(64 * self.img_size[0], 32 * self.img_size[0], 4, 2, 1),  # img: W/2 x H/2
            # self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                32 * self.img_size[0], img_size[0], kernel_size=4, stride=2, padding=1 # img: W x H
            ),
            # Output: N x img_size[0] x W x H
            # nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x.reshape(x.shape[0], 128 * self.img_size[0], self.img_size[1]//8, self.img_size[2]//8)
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    # N, in_channels, H, W = 8, 3, 16, 16
    img_size = (16, 32, 256)
    latent_dim = 100
    x = torch.randn((1, 16, 32, 256))
    z = torch.randn((1, latent_dim))
    disc = Discriminator(img_size)
    gen = Generator(img_size, latent_dim)
    summary(disc,(1, 16, 32, 256))
    summary(gen,(1, 100))
    # disc_out = disc(x)
    # gen_out = gen(z)
    # assert disc(x).shape == (1, 1), "Discriminator test succeeds"
    # assert gen(z).shape == (1, 16, 32, 256), "Generator test succeeds"


# test()
