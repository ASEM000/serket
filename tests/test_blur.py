from __future__ import annotations

import jax.numpy as jnp
import numpy.testing as npt
import pytest

from serket.nn import AvgBlur2D, FFTFilter2D, Filter2D, GaussianBlur2D


def test_AvgBlur2D():
    x = AvgBlur2D(1, 3)(jnp.arange(1, 26).reshape([1, 5, 5]).astype(jnp.float32))

    y = [
        [
            [1.7777778, 3.0, 3.6666667, 4.3333335, 3.1111112],
            [4.3333335, 7.0, 8.0, 9.0, 6.3333335],
            [7.6666665, 12.0, 13.0, 13.999999, 9.666667],
            [11.0, 17.0, 17.999998, 19.0, 13.0],
            [8.444445, 13.0, 13.666667, 14.333334, 9.777778],
        ]
    ]

    npt.assert_allclose(x, y, atol=1e-5)

    with pytest.raises(ValueError):
        AvgBlur2D(1, 0)

    with pytest.raises(ValueError):
        AvgBlur2D(0, 1)


def test_GaussBlur2D():
    layer = GaussianBlur2D(in_features=1, kernel_size=3, sigma=1.0)
    x = jnp.ones([1, 5, 5])

    npt.assert_allclose(
        jnp.array(
            [
                [
                    [0.5269764, 0.7259314, 0.7259314, 0.7259314, 0.5269764],
                    [0.7259314, 1.0, 1.0, 1.0, 0.7259314],
                    [0.7259314, 1.0, 1.0, 1.0, 0.7259314],
                    [0.7259314, 1.0, 1.0, 1.0, 0.7259314],
                    [0.5269764, 0.7259314, 0.7259314, 0.7259314, 0.5269764],
                ]
            ]
        ),
        layer(x),
        atol=1e-5,
    )

    with pytest.raises(ValueError):
        GaussianBlur2D(1, 0, sigma=1.0)

    with pytest.raises(ValueError):
        GaussianBlur2D(0, 1, sigma=1.0)


# def test_lazy_blur():
#     layer = GaussianBlur2D(in_features=None, kernel_size=3, sigma=1.0)
#     assert layer(jnp.ones([10, 5, 5])).shape == (10, 5, 5)

#     layer = AvgBlur2D(None, 3)
#     assert layer(jnp.ones([10, 5, 5])).shape == (10, 5, 5)

#     layer = Filter2D(None, jnp.ones([3, 3]))
#     assert layer(jnp.ones([10, 5, 5])).shape == (10, 5, 5)

#     with pytest.raises(ConcretizationTypeError):
#         jax.jit(GaussianBlur2D(in_features=None, kernel_size=3, sigma=1.0))(jnp.ones([10, 5, 5]))  # fmt: skip

#     with pytest.raises(ConcretizationTypeError):
#         jax.jit(AvgBlur2D(in_features=None, kernel_size=3))(jnp.ones([10, 5, 5]))

#     with pytest.raises(ConcretizationTypeError):
#         jax.jit(Filter2D(in_features=None, kernel=jnp.ones([4, 4])))(jnp.ones([10, 5, 5]))  # fmt: skip


def test_filter2d():
    layer = Filter2D(in_features=1, kernel=jnp.ones([3, 3]) / 9.0)
    x = jnp.ones([1, 5, 5])

    npt.assert_allclose(AvgBlur2D(1, 3)(x), layer(x), atol=1e-4)

    layer2 = FFTFilter2D(in_features=1, kernel=jnp.ones([3, 3]) / 9.0)

    npt.assert_allclose(layer(x), layer2(x), atol=1e-4)
