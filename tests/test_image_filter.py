# Copyright 2023 Serket authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import jax.numpy as jnp
import numpy.testing as npt
import pytest

import serket as sk
from serket.nn.image import (
    AvgBlur2D,
    FFTFilter2D,
    Filter2D,
    GaussianBlur2D,
    HorizontalShear2D,
    HorizontalTranslate2D,
    JigSaw2D,
    Pixelate2D,
    RandomHorizontalShear2D,
    RandomHorizontalTranslate2D,
    RandomRotate2D,
    RandomVerticalShear2D,
    RandomVerticalTranslate2D,
    Rotate2D,
    Solarize2D,
    VerticalShear2D,
    VerticalTranslate2D,
)


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

    # with pytest.raises(ValueError):
    # AvgBlur2D(1, 0)

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


def test_solarize2d():
    x = jnp.arange(1, 26).reshape(1, 5, 5)
    layer = Solarize2D(threshold=10, max_val=25)
    npt.assert_allclose(
        layer(x),
        jnp.array(
            [
                [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 15],
                    [14, 13, 12, 11, 10],
                    [9, 8, 7, 6, 5],
                    [4, 3, 2, 1, 0],
                ]
            ]
        ),
    )


def test_horizontal_translate():
    x = jnp.arange(1, 26).reshape(1, 5, 5)
    layer = HorizontalTranslate2D(2)
    npt.assert_allclose(
        layer(x),
        jnp.array(
            [
                [
                    [0, 0, 1, 2, 3],
                    [0, 0, 6, 7, 8],
                    [0, 0, 11, 12, 13],
                    [0, 0, 16, 17, 18],
                    [0, 0, 21, 22, 23],
                ]
            ]
        ),
    )

    layer = HorizontalTranslate2D(-2)
    npt.assert_allclose(
        layer(x),
        jnp.array(
            [
                [
                    [3, 4, 5, 0, 0],
                    [8, 9, 10, 0, 0],
                    [13, 14, 15, 0, 0],
                    [18, 19, 20, 0, 0],
                    [23, 24, 25, 0, 0],
                ]
            ]
        ),
    )

    layer = HorizontalTranslate2D(0)

    npt.assert_allclose(layer(x), x)

    layer = RandomHorizontalTranslate2D((0, 0))

    npt.assert_allclose(layer(x), x)


def test_vertical_translate():
    x = jnp.arange(1, 26).reshape(1, 5, 5)
    layer = VerticalTranslate2D(2)
    npt.assert_allclose(
        layer(x),
        jnp.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                ]
            ]
        ),
    )

    layer = VerticalTranslate2D(-2)
    npt.assert_allclose(
        layer(x),
        jnp.array(
            [
                [
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ]
        ),
    )

    layer = VerticalTranslate2D(0)

    npt.assert_allclose(layer(x), x)

    layer = RandomVerticalTranslate2D((0, 0))

    npt.assert_allclose(layer(x), x)


def test_jigsaw():
    x = jnp.arange(1, 17).reshape(1, 4, 4)
    layer = JigSaw2D(2)
    npt.assert_allclose(
        layer(x),
        jnp.array([[[9, 10, 3, 4], [13, 14, 7, 8], [11, 12, 1, 2], [15, 16, 5, 6]]]),
    )


def test_rotate():
    layer = Rotate2D(90)

    x = jnp.arange(1, 26).reshape(1, 5, 5)
    # ccw rotation

    rot = jnp.array(
        [
            [
                [5, 10, 15, 20, 25],
                [4, 9, 14, 19, 24],
                [3, 8, 13, 18, 23],
                [2, 7, 12, 17, 22],
                [1, 6, 11, 16, 21],
            ]
        ]
    )

    npt.assert_allclose(layer(x), rot)

    # random roate

    layer = RandomRotate2D((90, 90))

    npt.assert_allclose(layer(x), rot)
    npt.assert_allclose(sk.tree_eval(layer)(x), x)


def test_horizontal_shear():
    x = jnp.arange(1, 26).reshape(1, 5, 5)
    layer = HorizontalShear2D(45)
    shear = jnp.array(
        [
            [
                [0, 0, 1, 2, 3],
                [0, 6, 7, 8, 9],
                [11, 12, 13, 14, 15],
                [17, 18, 19, 20, 0],
                [23, 24, 25, 0, 0],
            ]
        ]
    )

    npt.assert_allclose(layer(x), shear)

    layer = RandomHorizontalShear2D((45, 45))
    npt.assert_allclose(layer(x), shear)

    npt.assert_allclose(sk.tree_eval(layer)(x), x)


def test_vertical_shear():
    x = jnp.arange(1, 26).reshape(1, 5, 5)
    layer = VerticalShear2D(45)
    shear = jnp.array(
        [
            [
                [0, 0, 3, 9, 15],
                [0, 2, 8, 14, 20],
                [1, 7, 13, 19, 25],
                [6, 12, 18, 24, 0],
                [11, 17, 23, 0, 0],
            ]
        ]
    )

    npt.assert_allclose(layer(x), shear)

    layer = RandomVerticalShear2D((45, 45))
    npt.assert_allclose(layer(x), shear)

    npt.assert_allclose(sk.tree_eval(layer)(x), x)


def test_posterize():
    x = jnp.arange(1, 26).reshape(1, 5, 5)
    layer = sk.nn.Posterize2D(4)
    posterized = jnp.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [16, 16, 16, 16, 16],
                [16, 16, 16, 16, 16],
            ]
        ]
    )

    npt.assert_allclose(layer(x), posterized)


def test_pixelate():
    x = jnp.arange(1, 26).reshape(1, 5, 5)
    layer = Pixelate2D(1)
    npt.assert_allclose(layer(x), x)
