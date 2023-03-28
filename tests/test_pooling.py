from __future__ import annotations

from itertools import product

import jax.numpy as jnp
import numpy.testing as npt

from serket.nn.pooling import (
    AdaptiveAvgPool1D,
    AdaptiveAvgPool2D,
    AdaptiveAvgPool3D,
    AdaptiveMaxPool1D,
    AdaptiveMaxPool2D,
    AdaptiveMaxPool3D,
    AvgPool1D,
    AvgPool2D,
    AvgPool3D,
    GlobalAvgPool1D,
    GlobalAvgPool2D,
    GlobalAvgPool3D,
    GlobalMaxPool1D,
    GlobalMaxPool2D,
    GlobalMaxPool3D,
    LPPool1D,
    LPPool2D,
    MaxPool1D,
    MaxPool2D,
    MaxPool3D,
)


def test_MaxPool1D():
    layer = MaxPool1D(kernel_size=2, padding="SAME", strides=1)
    x = jnp.arange(1, 11).reshape(1, 10).astype(jnp.float32)
    npt.assert_allclose(layer(x), jnp.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 10]]))

    layer = MaxPool1D(kernel_size=2, padding="SAME", strides=2)
    npt.assert_allclose(layer(x), jnp.array([[2, 4, 6, 8, 10]]))

    layer = MaxPool1D(kernel_size=2, padding="VALID", strides=1)
    npt.assert_allclose(layer(x), jnp.array([[2, 3, 4, 5, 6, 7, 8, 9, 10]]))

    layer = MaxPool1D(kernel_size=2, padding="VALID", strides=2)
    npt.assert_allclose(layer(x), jnp.array([[2, 4, 6, 8, 10]]))


def test_MaxPool2D():
    layer = MaxPool2D(kernel_size=2, padding="SAME", strides=1)
    x = jnp.arange(1, 10).reshape(1, 3, 3).astype(jnp.float32)
    npt.assert_allclose(layer(x), jnp.array([[[5, 6, 6], [8, 9, 9], [8, 9, 9]]]))


def test_MaxPool3D():
    layer = MaxPool3D(kernel_size=(1, 2, 2), padding="SAME", strides=1)
    x = jnp.arange(1, 10).reshape(1, 1, 3, 3).astype(jnp.float32)
    npt.assert_allclose(layer(x), jnp.array([[[[5, 6, 6], [8, 9, 9], [8, 9, 9]]]]))


def test_AvgPool1D():
    layer = AvgPool1D(kernel_size=2, padding="SAME", strides=1)
    x = jnp.arange(1, 11).reshape(1, 10).astype(jnp.float32)
    npt.assert_allclose(
        layer(x), jnp.array([[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 5]])
    )


def test_AvgPool2D():
    layer = AvgPool2D(kernel_size=2, padding="SAME", strides=1)
    x = jnp.arange(1, 10).reshape(1, 3, 3).astype(jnp.float32)
    npt.assert_allclose(
        layer(x), jnp.array([[[3, 4, 2.25], [6, 7, 3.75], [3.75, 4.25, 2.25]]])
    )


def test_AvgPool3D():
    layer = AvgPool3D(kernel_size=(1, 2, 2), padding="SAME", strides=1)
    x = jnp.arange(1, 10).reshape(1, 1, 3, 3).astype(jnp.float32)
    npt.assert_allclose(
        layer(x), jnp.array([[[[3, 4, 2.25], [6, 7, 3.75], [3.75, 4.25, 2.25]]]])
    )


def test_GlobalMaxPool1D():
    layer = GlobalMaxPool1D()
    x = jnp.arange(1, 11).reshape(1, 10).astype(jnp.float32)
    npt.assert_allclose(layer(x), jnp.array(10).reshape(1, 1))

    layer = GlobalMaxPool1D(keepdims=False)
    x = jnp.arange(1, 11).reshape(1, 10).astype(jnp.float32)
    npt.assert_allclose(layer(x), jnp.array(10))


def test_GlobalMaxPool2D():
    layer = GlobalMaxPool2D()
    x = jnp.arange(1, 10).reshape(1, 3, 3).astype(jnp.float32)
    npt.assert_allclose(layer(x), jnp.array(9).reshape(1, 1, 1))

    layer = GlobalMaxPool2D(keepdims=False)
    x = jnp.arange(1, 10).reshape(1, 3, 3).astype(jnp.float32)
    npt.assert_allclose(layer(x), jnp.array(9))


def test_GlobalMaxPool3D():
    layer = GlobalMaxPool3D()
    x = jnp.arange(1, 10).reshape(1, 1, 3, 3).astype(jnp.float32)
    npt.assert_allclose(layer(x), jnp.array(9).reshape(1, 1, 1, 1))

    layer = GlobalMaxPool3D(keepdims=False)
    x = jnp.arange(1, 10).reshape(1, 1, 3, 3).astype(jnp.float32)
    npt.assert_allclose(layer(x), jnp.array(9))


def test_GlobalAvgPool1D():
    layer = GlobalAvgPool1D()
    x = jnp.arange(1, 11).reshape(1, 10).astype(jnp.float32)
    npt.assert_allclose(layer(x), jnp.array(5.5).reshape(1, 1))

    layer = GlobalAvgPool1D(keepdims=False)
    x = jnp.arange(1, 11).reshape(1, 10).astype(jnp.float32)
    npt.assert_allclose(layer(x), jnp.array(5.5))


def test_GlobalAvgPool2D():
    layer = GlobalAvgPool2D()
    x = jnp.arange(1, 10).reshape(1, 3, 3).astype(jnp.float32)
    npt.assert_allclose(layer(x), jnp.array(5).reshape(1, 1, 1))

    layer = GlobalAvgPool2D(keepdims=False)
    x = jnp.arange(1, 10).reshape(1, 3, 3).astype(jnp.float32)
    npt.assert_allclose(layer(x), jnp.array(5))


def test_GlobalAvgPool3D():
    layer = GlobalAvgPool3D()
    x = jnp.arange(1, 10).reshape(1, 1, 3, 3).astype(jnp.float32)
    npt.assert_allclose(layer(x), jnp.array(5).reshape(1, 1, 1, 1))

    layer = GlobalAvgPool3D(keepdims=False)
    x = jnp.arange(1, 10).reshape(1, 1, 3, 3).astype(jnp.float32)
    npt.assert_allclose(layer(x), jnp.array(5))


def test_llpool1d():
    layer = LPPool1D(6.0, (2,))
    x = jnp.array(
        [
            [1.2993, 0.4735, -1.1191, 0.4357],
            [0.7256, -1.4759, 0.2187, 1.3955],
            [1.6344, 0.2285, 1.9801, 0.2249],
        ]
    )

    y = jnp.array(
        [[1.2998067, 1.1197486], [1.4793532, 1.3955034], [1.634402, 1.9801008]]
    )

    npt.assert_allclose(layer(x), y, atol=1e-4)


def test_llpool2d():
    layer = LPPool2D(6.0, (2, 2))
    x = jnp.array(
        (
            [
                [
                    [0.1881, 0.9284, -0.7165, -1.3147],
                    [1.1024, 2.5158, -0.2471, -0.7024],
                    [-0.8504, 0.7049, 0.3620, 0.3718],
                    [-1.5475, 0.3283, -1.9352, 0.5100],
                ],
                [
                    [0.6891, -0.2228, -0.3587, -0.9167],
                    [-0.8310, 0.7738, -0.3007, -1.7645],
                    [1.8144, 0.6532, -2.2035, 0.0986],
                    [0.1442, 1.3068, 1.8348, -0.0607],
                ],
                [
                    [0.0585, -0.5723, 0.2643, 0.6170],
                    [0.9788, 2.2352, 0.8158, -0.6996],
                    [-0.6741, -1.2954, -1.0559, 1.1246],
                    [1.4273, 1.0977, 0.6395, 1.5917],
                ],
            ]
        )
    )

    y = jnp.array(
        [
            [[2.5198, 1.3253], [1.5568, 1.9354]],
            [[0.9310, 1.7703], [1.8550, 2.3118]],
            [[2.2379, 0.8810], [1.5708, 1.6439]],
        ]
    )

    npt.assert_allclose(layer(x), y, atol=1e-4)


def test_adaptive_pool1d():
    layer_avg = AdaptiveAvgPool1D(2)
    layer_max = AdaptiveMaxPool1D(2)
    for input_shape in [2, 3, 4, 5, 10, 13, 14]:
        x = jnp.ones([1, input_shape])
        assert layer_avg(x).shape == (1, 2)
        assert layer_max(x).shape == (1, 2)


def test_adaptive_pool2d():
    layer_avg = AdaptiveAvgPool2D((2, 3))
    layer_max = AdaptiveMaxPool2D((2, 3))
    for input_size in product([2, 3, 4, 5], [3, 4]):
        x = jnp.ones([1, *input_size])
        assert layer_avg(x).shape == (1, 2, 3)
        assert layer_max(x).shape == (1, 2, 3)

    layer_avg = AdaptiveAvgPool2D((4, 7))
    layer_max = AdaptiveMaxPool2D((4, 7))
    for input_size in product([4, 5], [7, 8, 9, 16, 18]):
        x = jnp.ones([1, *input_size])
        assert layer_avg(x).shape == (1, 4, 7)
        assert layer_max(x).shape == (1, 4, 7)


def test_adaptive_pool3d():
    layer_avg = AdaptiveAvgPool3D((2, 3, 4))
    layer_max = AdaptiveMaxPool3D((2, 3, 4))
    for input_size in product([2, 3, 4, 5], [3, 4, 5], [4, 5]):
        x = jnp.ones([1, *input_size])
        assert layer_avg(x).shape == (1, 2, 3, 4)
        assert layer_max(x).shape == (1, 2, 3, 4)

    layer_avg = AdaptiveAvgPool3D((4, 7, 8))
    layer_max = AdaptiveMaxPool3D((4, 7, 8))
    for input_size in product([4, 5], [7, 8, 9, 16, 18], [8, 9]):
        x = jnp.ones([1, *input_size])
        assert layer_avg(x).shape == (1, 4, 7, 8)
        assert layer_max(x).shape == (1, 4, 7, 8)
