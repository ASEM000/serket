from __future__ import annotations

import jax.numpy as jnp
import numpy.testing as npt

from serket.nn import ConvScan1D, ConvScan2D, ConvScan3D


def test_convolution_scan_1d():
    layer = ConvScan1D(
        in_features=1, out_features=1, kernel_size=3, strides=1, padding=1
    )
    layer = layer.at["weight"].set(jnp.array([[[1, 2, 3]]]))
    x = jnp.array([[1, 2, 3, 4, 5, 6]])
    npt.assert_allclose(
        jnp.array([[8.0, 21.0, 39.0, 62.0, 90.0, 102.0]]), layer(x), atol=1e-5
    )


def test_convolution_scan_2d():
    layer = ConvScan2D(
        in_features=1, out_features=1, kernel_size=3, strides=1, padding=1
    )
    layer = layer.at["weight"].set(jnp.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]))
    x = jnp.ones([1, 3, 3])
    npt.assert_allclose(
        jnp.array(
            [
                [
                    [28.0, 147.0, 608.0],
                    [525.0, 4281.0, 18507.0],
                    [13904.0, 120235.0, 522240.0],
                ]
            ]
        ),
        layer(x),
        atol=1e-5,
    )


def test_convolution_scan_3d():
    layer = ConvScan3D(
        in_features=1, out_features=1, kernel_size=(3, 3, 3), strides=1, padding=1
    )
    layer = layer.at["weight"].set(jnp.arange(1, 28).reshape(1, 1, 3, 3, 3))
    x = jnp.ones([1, 3, 3, 3])
    # layer(x)

    npt.assert_allclose(
        layer(x),
        jnp.array(
            [
                [
                    [
                        [1.6400000e02, 2.3590000e03, 3.0810000e04],
                        [3.0317000e04, 7.9171700e05, 1.0655003e07],
                        [9.8342080e06, 2.6471693e08, 3.5664422e09],
                    ],
                    [
                        [7.3831270e06, 1.9861917e08, 2.6729948e09],
                        [4.9306798e09, 1.3278608e11, 1.7880492e12],
                        [1.6493103e12, 4.4430371e13, 5.9861013e14],
                    ],
                    [
                        [1.2357488e12, 3.3271042e13, 4.4777161e14],
                        [8.2673550e14, 2.2265059e16, 2.9981332e17],
                        [2.7655004e17, 7.4499197e18, 1.0037273e20],
                    ],
                ]
            ]
        ),
        atol=1e-5,
    )
