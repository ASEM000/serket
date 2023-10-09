# Copyright 2023 serket authors
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
import jax.random as jr
import numpy.testing as npt
import pytest

import serket as sk


def test_AvgBlur2D():
    x = sk.image.AvgBlur2D(3)(jnp.arange(1, 26).reshape([1, 5, 5]).astype(jnp.float32))

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

    # test with
    z = sk.image.FFTAvgBlur2D(3)(
        jnp.arange(1, 26).reshape([1, 5, 5]).astype(jnp.float32)
    )
    npt.assert_allclose(y, z, atol=1e-5)

    layer = sk.tree_mask(sk.image.FFTAvgBlur2D((3, 5)))
    grads = jax.grad(lambda node: jnp.sum(node(x)))(layer)
    npt.assert_allclose(grads.kernel_x, jnp.zeros_like(grads.kernel_x))
    npt.assert_allclose(grads.kernel_y, jnp.zeros_like(grads.kernel_y))


def test_GaussBlur2D():
    layer = sk.image.GaussianBlur2D(kernel_size=3, sigma=1.0)
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

    z = sk.image.FFTGaussianBlur2D(3, sigma=1.0)(jnp.ones([1, 5, 5])).astype(
        jnp.float32
    )

    npt.assert_allclose(layer(x), z, atol=1e-5)


def test_filter2d():
    layer = sk.image.Filter2D(kernel=jnp.ones([3, 3]) / 9.0)
    x = jnp.ones([1, 5, 5])

    npt.assert_allclose(sk.image.AvgBlur2D(3)(x), layer(x), atol=1e-4)

    layer2 = sk.image.FFTFilter2D(kernel=jnp.ones([3, 3]) / 9.0)

    npt.assert_allclose(layer(x), layer2(x), atol=1e-4)

    layer = sk.tree_mask(sk.image.AvgBlur2D((3, 5)))
    grads = jax.grad(lambda node: jnp.sum(node(x)))(layer)
    npt.assert_allclose(grads.kernel_x, jnp.zeros_like(grads.kernel_x))
    npt.assert_allclose(grads.kernel_y, jnp.zeros_like(grads.kernel_y))


def test_solarize2d():
    x = jnp.arange(1, 26).reshape(1, 5, 5)
    layer = sk.image.Solarize2D(threshold=10, max_val=25)
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
    layer = sk.image.HorizontalTranslate2D(2)
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

    layer = sk.image.HorizontalTranslate2D(-2)
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

    layer = sk.image.HorizontalTranslate2D(0)

    npt.assert_allclose(layer(x), x)


def test_vertical_translate():
    x = jnp.arange(1, 26).reshape(1, 5, 5)
    layer = sk.image.VerticalTranslate2D(2)
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

    layer = sk.image.VerticalTranslate2D(-2)
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

    layer = sk.image.VerticalTranslate2D(0)

    npt.assert_allclose(layer(x), x)


def test_jigsaw():
    x = jnp.arange(1, 17).reshape(1, 4, 4)
    layer = sk.image.RandomJigSaw2D(2)
    npt.assert_allclose(
        layer(x, key=jax.random.PRNGKey(0)),
        jnp.array([[[9, 10, 3, 4], [13, 14, 7, 8], [11, 12, 1, 2], [15, 16, 5, 6]]]),
    )


def test_rotate():
    layer = sk.image.Rotate2D(90)

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

    layer = sk.image.RandomRotate2D((90, 90))

    npt.assert_allclose(layer(x, key=jax.random.PRNGKey(0)), rot)
    npt.assert_allclose(sk.tree_eval(layer)(x), x)

    with pytest.raises(ValueError):
        sk.image.RandomRotate2D((90, 0, 9))(x, key=jax.random.PRNGKey(0))


def test_horizontal_shear():
    x = jnp.arange(1, 26).reshape(1, 5, 5)
    layer = sk.image.HorizontalShear2D(45)
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

    layer = sk.image.RandomHorizontalShear2D((45, 45))
    npt.assert_allclose(layer(x, key=jax.random.PRNGKey(0)), shear)

    npt.assert_allclose(sk.tree_eval(layer)(x), x)

    with pytest.raises(ValueError):
        sk.image.RandomHorizontalShear2D((45, 0, 9))(x, key=jax.random.PRNGKey(0))


def test_vertical_shear():
    x = jnp.arange(1, 26).reshape(1, 5, 5)
    layer = sk.image.VerticalShear2D(45)
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

    layer = sk.image.RandomVerticalShear2D((45, 45))
    npt.assert_allclose(layer(x, key=jax.random.PRNGKey(0)), shear)

    npt.assert_allclose(sk.tree_eval(layer)(x), x)

    with pytest.raises(ValueError):
        sk.image.RandomVerticalShear2D((45, 0, 9))(x, key=jax.random.PRNGKey(0))


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
    layer = sk.image.Pixelate2D(1)
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

    npt.assert_allclose(sk.image.AdjustContrast2D(factor=0.5)(x), y, atol=1e-5)


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
        sk.image.RandomContrast2D(range=(0.5, 1))(x, key=jax.random.PRNGKey(0)),
        y,
        atol=1e-5,
    )

    with pytest.raises(ValueError):
        sk.image.RandomContrast2D(range=(1, 0))(x, key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError):
        sk.image.RandomContrast2D(range=(0, 0, 0))(x, key=jax.random.PRNGKey(0))


def test_flip_left_right_2d():
    flip = sk.image.HorizontalFlip2D()
    x = jnp.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    y = flip(x)
    npt.assert_allclose(y, jnp.array([[[3, 2, 1], [6, 5, 4], [9, 8, 7]]]))


def test_random_flip_left_right_2d():
    flip = sk.image.RandomHorizontalFlip2D(rate=1.0)
    x = jnp.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    y = flip(x, key=jax.random.PRNGKey(0))
    npt.assert_allclose(y, jnp.array([[[3, 2, 1], [6, 5, 4], [9, 8, 7]]]))


def test_flip_up_down_2d():
    flip = sk.image.VerticalFlip2D()
    x = jnp.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    y = flip(x)
    npt.assert_allclose(y, jnp.array([[[7, 8, 9], [4, 5, 6], [1, 2, 3]]]))


def test_random_flip_up_down_2d():
    flip = sk.image.RandomVerticalFlip2D(rate=1.0)
    x = jnp.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    y = flip(x, key=jax.random.PRNGKey(0))
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

    ps = sk.image.PixelShuffle2D(2)
    y = jnp.array([0.08482574, 0.33432344, 1.9097648, -0.82606775])

    npt.assert_allclose(ps(x)[0, 0], y, atol=1e-5)

    with pytest.raises(ValueError):
        sk.image.PixelShuffle2D(3)(jnp.ones([6, 4, 4]))

    with pytest.raises(TypeError):
        sk.image.PixelShuffle2D(-3)(jnp.ones([9, 6, 4]))


def test_unsharp_mask():
    x = jax.random.uniform(jax.random.PRNGKey(0), (2, 10, 10))

    guassian_x = sk.image.GaussianBlur2D(3, sigma=1.0)(x)

    npt.assert_allclose(
        sk.image.UnsharpMask2D(3, sigma=1.0)(x),
        x + (x - guassian_x),
        atol=1e-5,
    )

    npt.assert_allclose(
        sk.image.FFTUnsharpMask2D(3, sigma=1.0)(x),
        x + (x - guassian_x),
        atol=1e-5,
    )

    layer = sk.tree_mask(sk.image.UnsharpMask2D((3, 5)))
    grads = jax.grad(lambda node: jnp.sum(node(x)))(layer)
    npt.assert_allclose(grads.kernel_x, jnp.zeros_like(grads.kernel_x))
    npt.assert_allclose(grads.kernel_y, jnp.zeros_like(grads.kernel_y))


def test_box_blur():
    x = jnp.arange(1, 17).reshape(1, 4, 4).astype(jnp.float32)
    y = jnp.array(
        [
            [
                [1.6000001, 2.4, 2.4, 2.0],
                [3.6000001, 5.2000003, 5.2000003, 4.2],
                [6.0000005, 8.400001, 8.400001, 6.6],
                [4.8, 6.666667, 6.666667, 5.2000003],
            ]
        ]
    )

    npt.assert_allclose(sk.image.BoxBlur2D((3, 5))(x), y, atol=1e-6)
    npt.assert_allclose(sk.image.FFTBoxBlur2D((3, 5))(x), y, atol=1e-6)

    layer = sk.tree_mask(sk.image.BoxBlur2D((3, 5)))
    grads = jax.grad(lambda node: jnp.sum(node(x)))(layer)
    npt.assert_allclose(grads.kernel_x, jnp.zeros_like(grads.kernel_x))
    npt.assert_allclose(grads.kernel_y, jnp.zeros_like(grads.kernel_y))


def test_laplacian():
    x = jax.random.uniform(jax.random.PRNGKey(0), (2, 10, 10))

    kernel = jnp.array(([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]]))

    npt.assert_allclose(
        sk.image.Laplacian2D(3)(x),
        sk.image.Filter2D(kernel)(x),
        atol=1e-5,
    )

    npt.assert_allclose(
        sk.image.FFTLaplacian2D(3)(x),
        sk.image.Filter2D(kernel)(x),
        atol=1e-5,
    )

    layer = sk.tree_mask(sk.image.Laplacian2D((3, 5)))
    grads = jax.grad(lambda node: jnp.sum(node(x)))(layer)
    npt.assert_allclose(grads.kernel, jnp.zeros_like(grads.kernel))


def test_motion_blur():
    x = jnp.arange(1, 17).reshape(1, 4, 4) + 0.0
    y = sk.image.MotionBlur2D(3, angle=30, direction=0.5)(x)
    ytrue = jnp.array(
        [
            [
                [0.7827108, 2.4696379, 3.3715053, 3.8119273],
                [2.8356633, 6.3387947, 7.3387947, 7.1810846],
                [5.117592, 10.338796, 11.338796, 10.550241],
                [6.472714, 10.020969, 10.770187, 9.100007],
            ]
        ]
    )
    npt.assert_allclose(y, ytrue, atol=1e-6)
    y = sk.image.FFTMotionBlur2D(3, angle=30, direction=0.5)(x)
    npt.assert_allclose(y, ytrue, atol=1e-6)

    layer = sk.tree_mask(sk.image.MotionBlur2D(3))
    grads = jax.grad(lambda node: jnp.sum(node(x)))(layer)
    npt.assert_allclose(grads.kernel, jnp.zeros_like(grads.kernel))


def test_median_blur():
    # against kornia
    x = jnp.arange(1, 26).reshape(1, 5, 5) + 0.0
    y = sk.image.MedianBlur2D(3)(x)
    z = jnp.array(
        [
            [
                [0.0, 2.0, 3.0, 4.0, 0.0],
                [2.0, 7.0, 8.0, 9.0, 5.0],
                [7.0, 12.0, 13.0, 14.0, 10.0],
                [12.0, 17.0, 18.0, 19.0, 15.0],
                [0.0, 17.0, 18.0, 19.0, 0.0],
            ]
        ]
    )
    npt.assert_allclose(y, z, atol=1e-6)


def test_wave_transform():
    x = jnp.ones((1, 5, 5))
    y = sk.image.WaveTransform2D(length=1, amplitude=0)(x)
    npt.assert_allclose(x, y, atol=1e-6)

    y = sk.image.RandomWaveTransform2D(length_range=[1, 1], amplitude_range=[0, 0])(
        x, key=jax.random.PRNGKey(0)
    )
    npt.assert_allclose(x, y, atol=1e-6)


def test_adjust_log_2d():
    x = jnp.arange(1, 17).reshape(1, 4, 4) / 16.0
    y = sk.image.AdjustLog2D()(x)
    z = jnp.array(
        [
            [
                [0.08746284, 0.16992499, 0.24792752, 0.32192808],
                [0.3923174, 0.45943162, 0.52356195, 0.5849625],
                [0.64385617, 0.7004397, 0.75488746, 0.8073549],
                [0.857981, 0.9068906, 0.9541963, 1.0],
            ]
        ],
    )
    npt.assert_allclose(y, z, atol=1e-6)


def test_adjust_sigmoid_2d():
    x = jnp.arange(1, 17).reshape(1, 4, 4) / 16.0
    y = sk.image.AdjustSigmoid2D()(x)
    z = jnp.array(
        [
            [
                [0.01243165, 0.02297737, 0.04208773, 0.07585818],
                [0.13296424, 0.22270013, 0.34864512, 0.5],
                [0.6513549, 0.7772999, 0.86703575, 0.9241418],
                [0.95791227, 0.97702265, 0.9875683, 0.9933072],
            ]
        ]
    )
    npt.assert_allclose(y, z, atol=1e-6)


def test_rgb_to_grayscale():
    gray = jnp.ones((1, 5, 5))
    assert sk.image.GrayscaleToRGB2D()(gray).shape == (3, 5, 5)
    rgb = jnp.ones((3, 5, 5))
    assert sk.image.RGBToGrayscale2D()(rgb).shape == (1, 5, 5)


def test_rgb_to_hsv_to_rgb():
    array = jnp.arange(1, 28).reshape(3, 3, 3) / 27.0
    hsv = sk.image.RGBToHSV2D()(array)
    target = jnp.array(
        [
            [
                [3.665191888809204, 3.665191411972046, 3.665191888809204],
                [3.665191411972046, 3.665191888809204, 3.665191888809204],
                [3.665191411972046, 3.665191888809204, 3.665191411972046],
            ],
            [
                [0.947368443012238, 0.899999976158142, 0.857142865657806],
                [0.818181753158569, 0.782608687877655, 0.750000000000000],
                [0.719999969005585, 0.692307710647583, 0.666666626930237],
            ],
            [
                [0.703703701496124, 0.740740716457367, 0.777777791023254],
                [0.814814805984497, 0.851851880550385, 0.888888895511627],
                [0.925925910472870, 0.962962985038757, 1.000000000000000],
            ],
        ]
    )
    # with kornia
    npt.assert_allclose(target, hsv, atol=1e-6)

    # identity
    npt.assert_allclose(sk.image.HSVToRGB2D()(hsv), array, atol=1e-6)


def test_adjust_hue():
    array = jnp.arange(1, 28).reshape(3, 3, 3) / 27.0
    layer = sk.image.AdjustHue2D(2.0)
    target = jnp.array(
        [
            [
                [0.703703701496124, 0.740740716457367, 0.777777791023254],
                [0.814814805984497, 0.851851880550385, 0.888888895511627],
                [0.925925910472870, 0.962962985038757, 1.000000000000000],
            ],
            [
                [0.037037022411823, 0.074074089527130, 0.111111104488373],
                [0.148148193955421, 0.185185194015503, 0.222222223877907],
                [0.259259283542633, 0.296296298503876, 0.333333373069763],
            ],
            [
                [0.430464237928391, 0.467501282691956, 0.504538357257843],
                [0.541575372219086, 0.578612446784973, 0.615649461746216],
                [0.652686476707458, 0.689723551273346, 0.726760566234589],
            ],
        ]
    )

    npt.assert_allclose(layer(array), target, atol=1e-6)
    layer = sk.image.RandomHue2D((0.0, 0.0))
    npt.assert_allclose((layer)(array, key=jr.PRNGKey(0)), array, atol=1e-6)


def test_adjust_saturation():
    array = jnp.arange(1, 28).reshape(3, 3, 3) / 27.0
    layer = sk.image.AdjustSaturation2D(0.6)
    target = jnp.array(
        [
            [
                [0.303703695535660, 0.340740710496902, 0.377777755260468],
                [0.414814800024033, 0.451851814985275, 0.488888859748840],
                [0.525925874710083, 0.562962949275970, 0.600000023841858],
            ],
            [
                [0.503703594207764, 0.540740728378296, 0.577777683734894],
                [0.614814817905426, 0.651851773262024, 0.688888788223267],
                [0.725925922393799, 0.762962877750397, 0.800000011920929],
            ],
            [
                [0.703703701496124, 0.740740716457367, 0.777777791023254],
                [0.814814805984497, 0.851851880550385, 0.888888895511627],
                [0.925925910472870, 0.962962985038757, 1.000000000000000],
            ],
        ]
    )

    npt.assert_allclose(layer(array), target, atol=1e-6)
    layer = sk.image.RandomSaturation2D((1.0, 1.0))
    npt.assert_allclose(layer(array, key=jr.PRNGKey(0)), array, atol=1e-6)


def test_random_horizontal_translate_2d():
    layer = sk.image.RandomHorizontalTranslate2D()
    assert layer(jnp.ones([3, 10, 10]), key=jr.PRNGKey(0)).shape == (3, 10, 10)


def test_random_vertical_translate_2d():
    layer = sk.image.RandomVerticalTranslate2D()
    assert layer(jnp.ones([3, 10, 10])).shape == (3, 10, 10)


def test_random_perspective_2d():
    layer = sk.image.RandomPerspective2D(scale=0)
    npt.assert_allclose(
        layer(jnp.ones([3, 10, 10]), key=jr.PRNGKey(0)),
        jnp.ones([3, 10, 10]),
        atol=1e-6,
    )


def test_sobel_2d():
    x = jnp.arange(1, 26).reshape(1, 5, 5).astype(jnp.float32)
    layer = sk.image.Sobel2D()
    target = jnp.array(
        [
            [
                [21.954498, 28.635643, 32.55764, 36.496574, 33.61547],
                [41.036568, 40.792156, 40.792156, 40.792156, 46.8615],
                [56.603886, 40.792156, 40.792156, 40.792156, 63.529522],
                [74.323616, 40.792156, 40.792156, 40.792156, 81.706795],
                [78.24321, 68.26419, 72.249565, 76.23647, 89.27486],
            ]
        ]
    )

    npt.assert_allclose(layer(x), target, atol=1e-6)

    layer = sk.image.FFTSobel2D()
    npt.assert_allclose(layer(x), target, atol=1e-5)


def test_adjust_brightness():
    x = jnp.arange(1, 10).reshape(1, 3, 3) / 10
    npt.assert_allclose(
        sk.image.AdjustBrightness2D(0.5)(x),
        jnp.array([[[0.6, 0.7, 0.8], [0.9, 1.0, 1.0], [1.0, 1.0, 1.0]]]),
        atol=1e-6,
    )


def test_random_brightness():
    x = jnp.arange(1, 10).reshape(1, 3, 3) / 10
    npt.assert_allclose(
        sk.image.RandomBrightness2D((0.5, 0.5))(x, key=jr.PRNGKey(0)),
        jnp.array([[[0.6, 0.7, 0.8], [0.9, 1.0, 1.0], [1.0, 1.0, 1.0]]]),
        atol=1e-6,
    )


def test_elastic_transform_2d():
    layer = sk.image.ElasticTransform2D(kernel_size=5, sigma=1.0, alpha=1.0)
    key = jr.PRNGKey(0)
    image = jnp.arange(1, 26).reshape(1, 5, 5).astype(jnp.float32)
    y = layer(image, key=key)
    npt.assert_allclose(
        jnp.array(
            [
                [
                    [1.016196, 2.1166031, 3.101904, 3.9978292, 4.950251],
                    [6.4011364, 7.65492, 8.359732, 8.447475, 9.246953],
                    [12.352943, 13.375501, 13.5076475, 13.214482, 13.972327],
                    [17.171738, 17.772211, 17.501146, 17.446032, 18.450916],
                    [20.999998, 21.80103, 22.054277, 22.589563, 23.693525],
                ]
            ]
        ),
        y,
        atol=1e-6,
    )

    layer = sk.image.FFTElasticTransform2D(kernel_size=5, sigma=1.0, alpha=1.0)
    y_ = layer(image, key=key)
    npt.assert_allclose(y, y_, atol=1e-6)
