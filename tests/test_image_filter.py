# Copyright 2024 serket authors
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


def test_laplacian():
    x = jax.random.uniform(jax.random.PRNGKey(0), (2, 10, 10))

    kernel = jnp.array(([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]]))
    y = jax.vmap(sk.image.filter_2d, in_axes=(0, None))(x, kernel)
    npt.assert_allclose(sk.image.Laplacian2D(3)(x), y, atol=1e-5)

    npt.assert_allclose(sk.image.FFTLaplacian2D(3)(x), y, atol=1e-5)


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


def test_random_horizontal_translate_2d():
    layer = sk.image.RandomHorizontalTranslate2D()
    assert layer(jnp.ones([3, 10, 10]), key=jr.PRNGKey(0)).shape == (3, 10, 10)


def test_random_vertical_translate_2d():
    layer = sk.image.RandomVerticalTranslate2D()
    assert layer(jnp.ones([3, 10, 10]), key=jax.random.PRNGKey(0)).shape == (3, 10, 10)


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


def test_bilateral_blur_2d():
    # against kornia
    x = jnp.array(
        [
            [
                [0.69434124, 0.86752045, 0.81006658, 0.20813388, 0.69910944],
                [0.54937655, 0.79141474, 0.52549887, 0.20865983, 0.17488086],
                [0.57635713, 0.30588913, 0.46992487, 0.06162852, 0.13264960],
                [0.73067701, 0.44296265, 0.23169714, 0.31364781, 0.70484269],
                [0.85146296, 0.50539654, 0.68432248, 0.66344428, 0.49181402],
            ]
        ]
    )

    layer = sk.image.BilateralBlur2D((3, 5), sigma_space=(1.2, 1.3), sigma_color=1.5)

    y = jnp.array(
        [
            [
                [0.35344556, 0.45698392, 0.43362546, 0.29663152, 0.2049025],
                [0.42099416, 0.5300394, 0.48385006, 0.33084756, 0.19578457],
                [0.36466658, 0.4202085, 0.386836, 0.27474657, 0.18085244],
                [0.3913928, 0.44740027, 0.437398, 0.3687499, 0.29071397],
                [0.3237696, 0.37171572, 0.39108697, 0.34648296, 0.26146644],
            ]
        ]
    )

    npt.assert_allclose(layer(x), y, atol=1e-6)


def test_joint_bilateral_blur_2d():
    x = jnp.array(
        [
            [
                [0.70148504, 0.5235718, 0.796878, 0.13653928, 0.19069368],
                [0.13707995, 0.7230044, 0.43493855, 0.03692275, 0.02392286],
                [0.90786785, 0.11590564, 0.97039336, 0.5512939, 0.628494],
                [0.54870003, 0.16357982, 0.4420898, 0.47006166, 0.6418716],
                [0.89536285, 0.07819122, 0.08694249, 0.8098613, 0.97447044],
            ]
        ]
    )

    y = jnp.array(
        [
            [
                [
                    3.01010489e-01,
                    4.04902875e-01,
                    9.10854340e-02,
                    5.20312428e-01,
                    5.13539314e-02,
                ],
                [
                    1.99799776e-01,
                    3.98275018e-01,
                    7.71528661e-01,
                    9.39785659e-01,
                    5.81741035e-01,
                ],
                [
                    1.27980113e-01,
                    7.86668301e-01,
                    6.20619059e-01,
                    5.28359771e-01,
                    7.00435698e-01,
                ],
                [
                    9.26082611e-01,
                    2.72210836e-02,
                    1.38149798e-01,
                    1.54755056e-01,
                    9.10691023e-02,
                ],
                [
                    8.69154930e-04,
                    1.15489244e-01,
                    6.57627940e-01,
                    5.47029793e-01,
                    8.61521661e-01,
                ],
            ]
        ]
    )

    z = jnp.array(
        [
            [
                [0.2569141, 0.3378913, 0.30954438, 0.19656985, 0.08877079],
                [0.35686886, 0.4755023, 0.4675542, 0.34694836, 0.2059895],
                [0.32238984, 0.43757084, 0.48053238, 0.4226889, 0.30991194],
                [0.32943073, 0.39682606, 0.47729316, 0.49428663, 0.41849864],
                [0.20187145, 0.23701364, 0.30271348, 0.34831038, 0.34285933],
            ]
        ]
    )

    layer = sk.image.JointBilateralBlur2D(
        (3, 5),
        sigma_color=1.5,
        sigma_space=(1.2, 1.3),
    )
    npt.assert_allclose(layer(x, y), z, atol=1e-6)


def test_blur_pool_2d():
    # test against kornia
    x = jnp.array(
        [
            [
                [0.4139353, 0.9534693, 0.75953954, 0.58387464, 0.96926004],
                [0.1243794, 0.95703536, 0.40123367, 0.50165236, 0.14891201],
                [0.15226442, 0.57921094, 0.28390247, 0.69786924, 0.00421089],
                [0.69519806, 0.14696085, 0.37243962, 0.7077443, 0.32187456],
                [0.99285686, 0.884988, 0.8843235, 0.5459984, 0.03278035],
            ]
        ]
    )
    y = jnp.array(
        [
            [
                [0.29802963, 0.5233751, 0.36526662],
                [0.28191438, 0.47190684, 0.222722],
                [0.45492253, 0.49992818, 0.16091321],
            ]
        ]
    )

    layer = sk.image.BlurPool2D((3, 3), strides=2)

    npt.assert_allclose(layer(x), y, atol=1e-6)

    layer = sk.image.FFTBlurPool2D((3, 3), strides=2)

    npt.assert_allclose(layer(x), y, atol=1e-6)
