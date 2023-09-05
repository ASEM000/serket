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

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

import serket as sk
from serket.image.augment import (
    AdjustContrast2D,
    JigSaw2D,
    PixelShuffle2D,
    RandomContrast2D,
)
from serket.image.filter import (
    AvgBlur2D,
    FFTAvgBlur2D,
    FFTFilter2D,
    FFTGaussianBlur2D,
    Filter2D,
    GaussianBlur2D,
)
from serket.image.geometric import (
    HorizontalFlip2D,
    HorizontalShear2D,
    HorizontalTranslate2D,
    Pixelate2D,
    RandomHorizontalShear2D,
    RandomRotate2D,
    RandomVerticalShear2D,
    Rotate2D,
    Solarize2D,
    VerticalFlip2D,
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

    # test with
    z = FFTAvgBlur2D(1, 3)(jnp.arange(1, 26).reshape([1, 5, 5]).astype(jnp.float32))
    npt.assert_allclose(y, z, atol=1e-5)


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

    z = FFTGaussianBlur2D(1, 3, sigma=1.0)(jnp.ones([1, 5, 5])).astype(jnp.float32)

    npt.assert_allclose(layer(x), z, atol=1e-5)


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
    layer = sk.image.Posterize2D(4)
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


def test_adjust_contrast_2d():
    x = jnp.array(
        [
            [
                [0.19165385, 0.4459561, 0.03873193],
                [0.58923364, 0.0923605, 0.2597469],
                [0.83097064, 0.4854728, 0.03308535],
            ],
            [
                [0.10485303, 0.10068893, 0.408355],
                [0.40298176, 0.6227188, 0.8612417],
                [0.52223504, 0.3363577, 0.1300546],
            ],
        ]
    )
    y = jnp.array(
        [
            [
                [0.26067203, 0.38782316, 0.18421106],
                [0.45946193, 0.21102534, 0.29471856],
                [0.5803304, 0.4075815, 0.18138777],
            ],
            [
                [0.24628687, 0.24420482, 0.39803785],
                [0.39535123, 0.50521976, 0.6244812],
                [0.45497787, 0.3620392, 0.25888765],
            ],
        ]
    )

    npt.assert_allclose(AdjustContrast2D(contrast_factor=0.5)(x), y, atol=1e-5)


def test_random_contrast_2d():
    x = jnp.array(
        [
            [
                [0.19165385, 0.4459561, 0.03873193],
                [0.58923364, 0.0923605, 0.2597469],
                [0.83097064, 0.4854728, 0.03308535],
            ],
            [
                [0.10485303, 0.10068893, 0.408355],
                [0.40298176, 0.6227188, 0.8612417],
                [0.52223504, 0.3363577, 0.1300546],
            ],
        ]
    )

    y = jnp.array(
        [
            [
                [0.23179087, 0.4121493, 0.1233343],
                [0.5137658, 0.1613692, 0.28008443],
                [0.68521255, 0.44017565, 0.11932957],
            ],
            [
                [0.18710288, 0.1841496, 0.40235513],
                [0.39854428, 0.55438805, 0.7235553],
                [0.4831221, 0.3512926, 0.20497654],
            ],
        ]
    )

    npt.assert_allclose(
        RandomContrast2D(contrast_range=(0.5, 1))(x, key=jax.random.PRNGKey(0)),
        y,
        atol=1e-5,
    )


def test_flip_left_right_2d():
    flip = HorizontalFlip2D()
    x = jnp.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    y = flip(x)
    npt.assert_allclose(y, jnp.array([[[3, 2, 1], [6, 5, 4], [9, 8, 7]]]))


def test_flip_up_down_2d():
    flip = VerticalFlip2D()
    x = jnp.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    y = flip(x)
    npt.assert_allclose(y, jnp.array([[[7, 8, 9], [4, 5, 6], [1, 2, 3]]]))


def test_pixel_shuffle():
    x = jnp.array(
        [
            [[0.08482574, 1.9097648], [0.29561743, 1.120948]],
            [[0.33432344, -0.82606775], [0.6481277, 1.0434873]],
            [[-0.7824839, -0.4539462], [0.6297971, 0.81524646]],
            [[-0.32787678, -1.1234448], [-1.6607416, 0.27290547]],
        ]
    )

    ps = PixelShuffle2D(2)
    y = jnp.array([0.08482574, 0.33432344, 1.9097648, -0.82606775])

    npt.assert_allclose(ps(x)[0, 0], y, atol=1e-5)

    with pytest.raises(ValueError):
        PixelShuffle2D(3)(jnp.ones([6, 4, 4]))

    with pytest.raises(ValueError):
        PixelShuffle2D(-3)(jnp.ones([9, 6, 4]))
