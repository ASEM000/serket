from itertools import product

import numpy.testing as npt
import torch

import serket as sk


def test_conv1d():
    padding_choice = [0, 1, 2, 3]
    kernel_size = [1, 2, 3]
    stride = [1, 2, 3]
    dilation = [1, 2, 3]
    groups = [1, 3]

    for ki, si, di, gi, pi in product(kernel_size, stride, dilation, groups, padding_choice):  # fmt: skip
        torch_conv = torch.nn.Conv1d(3, 3, ki, stride=si, padding=pi, dilation=di, groups=gi)  # fmt: skip
        serket_conv = sk.nn.Conv1D(3, 3, ki, strides=si, padding=pi, kernel_dilation=di, groups=gi)  # fmt: skip

        # set weights of torch to serket
        serket_conv = serket_conv.at["weight"].set(torch_conv.weight.detach().numpy())
        serket_conv = serket_conv.at["bias"].set(torch_conv.bias.detach().numpy().reshape(-1, 1))  # fmt: skip

        # test forward pass
        x = torch.randn(1, 3, 32)
        y = torch_conv(x)
        z = serket_conv(x.detach().numpy()[0])

        npt.assert_allclose(y.detach().numpy()[0], z, atol=1e-5)


def test_conv2d():
    padding_choice = [0, 1, 2, 3]
    kernel_size = [1, 2, 3]
    stride = [1, 2, 3]
    dilation = [1, 2, 3]
    groups = [1, 3]

    for ki, si, di, gi, pi in product(kernel_size, stride, dilation, groups, padding_choice):  # fmt: skip
        torch_conv = torch.nn.Conv2d(3, 3, ki, stride=si, padding=pi, dilation=di, groups=gi)  # fmt: skip
        serket_conv = sk.nn.Conv2D(3, 3, ki, strides=si, padding=pi, kernel_dilation=di, groups=gi)  # fmt: skip

        # set weights of torch to serket
        serket_conv = serket_conv.at["weight"].set(torch_conv.weight.detach().numpy())
        serket_conv = serket_conv.at["bias"].set(torch_conv.bias.detach().numpy().reshape(-1, 1, 1))  # fmt: skip

        # test forward pass
        x = torch.randn(1, 3, 32, 32)
        y = torch_conv(x)
        z = serket_conv(x.detach().numpy()[0])

        npt.assert_allclose(y.detach().numpy()[0], z, atol=1e-5)


def test_conv3d():
    padding_choice = [0, 1, 2, 3]
    kernel_size = [1, 2, 3]
    stride = [1, 2, 3]
    dilation = [1, 2, 3]
    groups = [1, 3]

    for ki, si, di, gi, pi in product(kernel_size, stride, dilation, groups, padding_choice):  # fmt: skip
        torch_conv = torch.nn.Conv3d(3, 3, ki, stride=si, padding=pi, dilation=di, groups=gi)  # fmt: skip
        serket_conv = sk.nn.Conv3D(3, 3, ki, strides=si, padding=pi, kernel_dilation=di, groups=gi)  # fmt: skip

        # set weights of torch to serket
        serket_conv = serket_conv.at["weight"].set(torch_conv.weight.detach().numpy())
        serket_conv = serket_conv.at["bias"].set(torch_conv.bias.detach().numpy().reshape(-1, 1, 1, 1))  # fmt: skip

        # test forward pass
        x = torch.randn(1, 3, 32, 32, 32)
        y = torch_conv(x)
        z = serket_conv(x.detach().numpy()[0])

        npt.assert_allclose(y.detach().numpy()[0], z, atol=1e-5)
