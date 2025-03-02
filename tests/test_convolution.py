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
import os

os.environ["KERAS_BACKEND"] = "jax"

from itertools import product

import jax
import jax.numpy as jnp
import keras
import numpy.testing as npt
import pytest

import serket as sk


def test_depthwise_fft_conv():
    x = jnp.ones([10, 1])
    npt.assert_allclose(
        sk.nn.DepthwiseFFTConv1D(10, 3, key=jax.random.key(0))(x),
        sk.nn.DepthwiseConv1D(10, 3, key=jax.random.key(0))(x),
        atol=1e-4,
    )

    x = jnp.ones([10, 1, 1])
    npt.assert_allclose(
        sk.nn.DepthwiseFFTConv2D(10, 3, key=jax.random.key(0))(x),
        sk.nn.DepthwiseConv2D(10, 3, key=jax.random.key(0))(x),
        atol=1e-4,
    )

    x = jnp.ones([10, 1, 1, 1])
    npt.assert_allclose(
        sk.nn.DepthwiseFFTConv3D(10, 3, key=jax.random.key(0))(x),
        sk.nn.DepthwiseConv3D(10, 3, key=jax.random.key(0))(x),
        atol=1e-4,
    )


def test_conv_transpose():
    x = jnp.ones([10, 4])
    npt.assert_allclose(
        sk.nn.Conv1DTranspose(10, 4, 3, key=jax.random.key(0))(x),
        sk.nn.FFTConv1DTranspose(10, 4, 3, key=jax.random.key(0))(x),
        atol=1e-4,
    )

    x = jnp.ones([10, 4])
    npt.assert_allclose(
        sk.nn.Conv1DTranspose(10, 4, 3, dilation=2, key=jax.random.key(0))(x),
        sk.nn.FFTConv1DTranspose(10, 4, 3, dilation=2, key=jax.random.key(0))(x),
        atol=1e-5,
    )

    x = jnp.ones([10, 4, 4])
    npt.assert_allclose(
        sk.nn.Conv2DTranspose(10, 4, 3, key=jax.random.key(0))(x),
        sk.nn.FFTConv2DTranspose(10, 4, 3, key=jax.random.key(0))(x),
        atol=1e-4,
    )

    x = jnp.ones([10, 4, 4, 4])
    npt.assert_allclose(
        sk.nn.Conv3DTranspose(10, 4, 3, dilation=2, key=jax.random.key(0))(x),
        sk.nn.FFTConv3DTranspose(10, 4, 3, dilation=2, key=jax.random.key(0))(x),
        atol=1e-5,
    )


def test_separable_conv():
    x = jnp.ones([10, 4])
    npt.assert_allclose(
        sk.nn.SeparableConv1D(10, 4, 3, key=jax.random.key(0))(x),
        sk.nn.SeparableFFTConv1D(10, 4, 3, key=jax.random.key(0))(x),
        atol=1e-4,
    )

    x = jnp.ones([10, 4, 4])
    npt.assert_allclose(
        sk.nn.SeparableConv2D(10, 4, 3, key=jax.random.key(0))(x),
        sk.nn.SeparableFFTConv2D(10, 4, 3, key=jax.random.key(0))(x),
        atol=1e-4,
    )

    x = jnp.ones([10, 4, 4, 4])
    npt.assert_allclose(
        sk.nn.SeparableConv3D(10, 4, 3, key=jax.random.key(0))(x),
        sk.nn.SeparableFFTConv3D(10, 4, 3, key=jax.random.key(0))(x),
        atol=1e-4,
    )


def test_conv1D():
    layer = sk.nn.Conv1D(
        in_features=1,
        out_features=1,
        kernel_size=2,
        padding="same",
        strides=1,
        key=jax.random.key(0),
    )

    layer = layer.at["weight"].set(jnp.ones([1, 1, 2], dtype=jnp.float32))  # OIW
    x = jnp.arange(1, 11).reshape([1, 10]).astype(jnp.float32)
    npt.assert_allclose(layer(x), jnp.array([[3, 5, 7, 9, 11, 13, 15, 17, 19, 10]]))

    layer = sk.nn.Conv1D(
        in_features=1,
        out_features=1,
        kernel_size=2,
        padding="same",
        strides=2,
        key=jax.random.key(0),
    )
    layer = layer.at["weight"].set(jnp.ones([1, 1, 2], dtype=jnp.float32))
    x = jnp.arange(1, 11).reshape([1, 10]).astype(jnp.float32)

    npt.assert_allclose(layer(x), jnp.array([[3, 7, 11, 15, 19]]))

    layer = sk.nn.Conv1D(
        in_features=1,
        out_features=1,
        kernel_size=2,
        padding="VALID",
        strides=1,
        key=jax.random.key(0),
    )
    layer = layer.at["weight"].set(jnp.ones([1, 1, 2], dtype=jnp.float32))
    x = jnp.arange(1, 11).reshape([1, 10]).astype(jnp.float32)

    npt.assert_allclose(layer(x), jnp.array([[3, 5, 7, 9, 11, 13, 15, 17, 19]]))

    x = jnp.array(
        [
            [
                -0.10502207,
                -0.56205004,
                -0.56485987,
                -1.7063935,
                0.56626016,
                -0.42215332,
                1.0077653,
                0.9922631,
                -0.61236995,
                -1.8450408,
            ]
        ]
    )
    w = jnp.array(
        [
            [[0.12903854, -0.69336414, 0.41909721]],
            [[0.68273675, 0.16725819, -0.08097019]],
        ]
    )

    b = jnp.array([[0.0], [0.0]])

    y = jnp.array(
        [
            [
                -0.16391265,
                -0.3254394,
                0.61541975,
                0.9337027,
                -0.04316157,
                0.48837015,
                -0.8823215,
                -1.5157251,
                0.55463594,
                1.4073253,
            ],
            [
                0.02817101,
                0.04415954,
                -0.21203005,
                -0.6349587,
                -0.37253788,
                -1.31597,
                0.60474735,
                0.02713783,
                0.5856145,
                0.36885628,
            ],
        ]
    )

    layer = sk.nn.Conv1D(
        1, 2, 3, padding=2, strides=1, dilation=2, key=jax.random.key(0)
    )
    layer = layer.at["weight"].set(w)
    layer = layer.at["bias"].set(b)

    npt.assert_allclose(layer(x), y)

    layer = sk.nn.Conv1D(
        1,
        2,
        3,
        padding=2,
        strides=1,
        dilation=2,
        bias_init=None,
        key=jax.random.key(0),
    )
    layer = layer.at["weight"].set(w)
    npt.assert_allclose(layer(x), y)


def test_conv2D():
    layer = sk.nn.Conv2D(
        in_features=1, out_features=1, kernel_size=2, key=jax.random.key(0)
    )
    layer = layer.at["weight"].set(jnp.ones([1, 1, 2, 2], dtype=jnp.float32))  # OIHW
    x = jnp.arange(1, 17).reshape([1, 4, 4]).astype(jnp.float32)

    npt.assert_allclose(
        layer(x)[0],
        jnp.array(
            [[14, 18, 22, 12], [30, 34, 38, 20], [46, 50, 54, 28], [27, 29, 31, 16]]
        ),
    )

    layer = sk.nn.Conv2D(
        in_features=1,
        out_features=1,
        kernel_size=2,
        padding="VALID",
        key=jax.random.key(0),
    )
    layer = layer.at["weight"].set(jnp.ones([1, 1, 2, 2], dtype=jnp.float32))
    x = jnp.arange(1, 17).reshape([1, 4, 4]).astype(jnp.float32)

    npt.assert_allclose(
        layer(x)[0],
        jnp.array(
            [
                [14, 18, 22],
                [30, 34, 38],
                [46, 50, 54],
            ]
        ),
    )

    layer = sk.nn.Conv2D(1, 2, 2, padding="same", strides=2, key=jax.random.key(0))
    layer = layer.at["weight"].set(jnp.ones([2, 1, 2, 2], dtype=jnp.float32))
    x = jnp.arange(1, 17).reshape([1, 4, 4]).astype(jnp.float32)

    npt.assert_allclose(
        layer(x)[0],
        jnp.array(
            [
                [14, 22],
                [46, 54],
            ]
        ),
    )

    layer = sk.nn.Conv2D(1, 2, 2, padding="same", strides=1, key=jax.random.key(0))
    layer = layer.at["weight"].set(jnp.ones([2, 1, 2, 2], dtype=jnp.float32))
    x = jnp.arange(1, 17).reshape([1, 4, 4]).astype(jnp.float32)

    npt.assert_allclose(
        layer(x)[0],
        jnp.array(
            [[14, 18, 22, 12], [30, 34, 38, 20], [46, 50, 54, 28], [27, 29, 31, 16]]
        ),
    )

    npt.assert_allclose(
        layer(x)[1],
        jnp.array(
            [[14, 18, 22, 12], [30, 34, 38, 20], [46, 50, 54, 28], [27, 29, 31, 16]]
        ),
    )

    layer = sk.nn.Conv2D(
        1, 2, 2, padding="same", strides=1, bias_init=None, key=jax.random.key(0)
    )
    layer = layer.at["weight"].set(jnp.ones([2, 1, 2, 2], dtype=jnp.float32))
    x = jnp.arange(1, 17).reshape([1, 4, 4]).astype(jnp.float32)

    npt.assert_allclose(
        layer(x)[0],
        jnp.array(
            [[14, 18, 22, 12], [30, 34, 38, 20], [46, 50, 54, 28], [27, 29, 31, 16]]
        ),
    )

    npt.assert_allclose(
        layer(x)[1],
        jnp.array(
            [[14, 18, 22, 12], [30, 34, 38, 20], [46, 50, 54, 28], [27, 29, 31, 16]]
        ),
    )


def test_conv3D():
    layer = sk.nn.Conv3D(1, 3, 3, key=jax.random.key(0))
    layer = layer.at["weight"].set(jnp.ones([3, 1, 3, 3, 3]))
    layer = layer.at["bias"].set(jnp.zeros([3, 1, 1, 1]))
    npt.assert_allclose(
        layer(jnp.ones([1, 1, 3, 3])),
        jnp.tile(jnp.array([[4, 6, 4], [6, 9, 6], [4, 6, 4]]), [3, 1, 1, 1]),
    )


def test_conv1dtranspose():
    x = jnp.array(
        [
            [0.88300896, 0.34756768, 0.41512465],
            [0.13573384, 0.75572586, 0.1611284],
            [0.6713624, 0.6773077, 0.24859095],
            [0.72523665, 0.01929283, 0.60781384],
        ]
    )

    w = jnp.array(
        [
            [
                [0.30227104, -0.05764396, 0.2769581],
                [0.18765458, 0.11693919, 0.34051096],
                [-0.4997116, 0.26043424, 0.10502717],
                [-0.03766551, 0.01165446, -0.12174296],
            ]
        ]
    )

    b = jnp.array([[[0.0]]])

    layer = sk.nn.Conv1DTranspose(
        4, 1, 3, padding=2, strides=1, dilation=2, key=jax.random.key(0)
    )
    layer = layer.at["weight"].set(w)
    layer = layer.at["bias"].set(b)
    y = jnp.array([[0.27022034, 0.24495776, -0.00368674]])
    npt.assert_allclose(layer(x), y, atol=1e-5)

    layer = sk.nn.Conv1DTranspose(
        4,
        1,
        3,
        padding=2,
        strides=1,
        dilation=2,
        bias_init=None,
        key=jax.random.key(0),
    )
    layer = layer.at["weight"].set(w)
    y = jnp.array([[0.27022034, 0.24495776, -0.00368674]])
    npt.assert_allclose(layer(x), y, atol=1e-5)


def test_conv2dtranspose():
    x = jnp.array(
        [
            [
                [0.07875574, 0.6884998, 0.584759, 0.96977115],
                [0.8446715, 0.07353628, 0.3616743, 0.455294],
                [0.59947085, 0.92952085, 0.07017219, 0.7658876],
                [0.52333033, 0.20134509, 0.08446538, 0.6004138],
            ],
            [
                [0.8891939, 0.53262544, 0.04431581, 0.48456],
                [0.93436563, 0.2775594, 0.6189116, 0.9289819],
                [0.43007636, 0.85937333, 0.9426601, 0.7450541],
                [0.19154763, 0.57595146, 0.21673548, 0.18218327],
            ],
            [
                [0.13518727, 0.7918651, 0.167938, 0.54408634],
                [0.02775729, 0.24681842, 0.2679938, 0.49933732],
                [0.525504, 0.17591465, 0.5766901, 0.9605187],
                [0.56262374, 0.28900325, 0.7377145, 0.57581556],
            ],
        ],
    )

    w = jnp.array(
        [
            [
                [
                    [0.42048344, -0.12954444, 0.12225959],
                    [-0.3275112, 0.23747145, 0.2635427],
                    [-0.02253131, -0.09689994, -0.05774301],
                ],
                [
                    [-0.06914992, 0.05235992, 0.28782916],
                    [-0.19883825, 0.41525128, -0.02345275],
                    [-0.31591812, 0.18510762, 0.26573473],
                ],
                [
                    [0.16306466, -0.00594486, 0.20800874],
                    [-0.04522677, -0.12785907, 0.06354138],
                    [-0.09737353, 0.08018816, -0.03065838],
                ],
            ]
        ]
    )

    b = jnp.array([[[0.0]]])

    layer = sk.nn.Conv2DTranspose(
        3, 1, 3, padding=2, strides=1, dilation=2, key=jax.random.key(0)
    )

    layer = layer.at["weight"].set(w)
    layer = layer.at["bias"].set(b)

    y = jnp.array(
        [
            [
                [0.826823, 0.76963556, -0.05952819, -0.17411183],
                [0.7428247, 0.33745894, -0.19105533, 0.14668494],
                [0.44151804, 1.1056999, -0.05200444, 0.1634948],
                [0.4133723, 0.87103486, 0.05985051, -0.01025786],
            ]
        ]
    )

    npt.assert_allclose(layer(x), y, atol=1e-5)

    layer = sk.nn.Conv2DTranspose(
        3,
        1,
        3,
        padding=2,
        strides=1,
        dilation=2,
        bias_init=None,
        key=jax.random.key(0),
    )

    layer = layer.at["weight"].set(w)

    y = jnp.array(
        [
            [
                [0.826823, 0.76963556, -0.05952819, -0.17411183],
                [0.7428247, 0.33745894, -0.19105533, 0.14668494],
                [0.44151804, 1.1056999, -0.05200444, 0.1634948],
                [0.4133723, 0.87103486, 0.05985051, -0.01025786],
            ]
        ]
    )

    npt.assert_allclose(layer(x), y, atol=1e-5)


def test_conv3dtranspose():
    x = jnp.array(
        [
            [
                [
                    [0.9922036, 0.8135023, 0.4357065],
                    [0.5499866, 0.16954863, 0.7430643],
                    [0.43268728, 0.39654946, 0.4743309],
                ],
                [
                    [0.8118012, 0.95373034, 0.2214334],
                    [0.36190224, 0.49149978, 0.51192653],
                    [0.33870387, 0.16532409, 0.34585416],
                ],
                [
                    [0.4589044, 0.00961053, 0.76880145],
                    [0.9786881, 0.7666111, 0.05819798],
                    [0.15056634, 0.2265861, 0.7614187],
                ],
            ],
            [
                [
                    [0.3882742, 0.53167224, 0.43066823],
                    [0.07477725, 0.03106737, 0.679559],
                    [0.48226452, 0.663074, 0.5331527],
                ],
                [
                    [0.5678816, 0.9485518, 0.9766922],
                    [0.650491, 0.5300664, 0.52210283],
                    [0.8029467, 0.2828722, 0.40568137],
                ],
                [
                    [0.03113937, 0.5991132, 0.56445587],
                    [0.17777038, 0.63313735, 0.9898604],
                    [0.67996955, 0.849555, 0.01604772],
                ],
            ],
            [
                [
                    [0.4631437, 0.9124212, 0.45101917],
                    [0.7450532, 0.54239345, 0.45996523],
                    [0.14280403, 0.01203346, 0.6348034],
                ],
                [
                    [0.672565, 0.21366453, 0.43834722],
                    [0.20767748, 0.31813693, 0.88000405],
                    [0.16432023, 0.25118446, 0.98615324],
                ],
                [
                    [0.1503222, 0.9133204, 0.50562966],
                    [0.609205, 0.96970797, 0.7225517],
                    [0.15559411, 0.33902645, 0.20180452],
                ],
            ],
            [
                [
                    [0.71946573, 0.8797904, 0.91639566],
                    [0.3203174, 0.34817457, 0.6113894],
                    [0.94209945, 0.62192047, 0.47244883],
                ],
                [
                    [0.72698116, 0.37481964, 0.9780731],
                    [0.5172117, 0.6892407, 0.59901524],
                    [0.04232013, 0.02548707, 0.05988038],
                ],
                [
                    [0.78978825, 0.7785101, 0.6678246],
                    [0.2033236, 0.27756643, 0.16570795],
                    [0.40070426, 0.9758557, 0.5060563],
                ],
            ],
        ]
    )

    w = jnp.array(
        [
            [
                [
                    [
                        [0.12643094, 0.21056306, 0.15895613],
                        [0.05221418, 0.09544074, 0.01061383],
                        [-0.1907051, 0.08492875, 0.02936444],
                    ],
                    [
                        [-0.05912174, -0.16569473, 0.14907388],
                        [0.06837426, 0.20294279, -0.01972006],
                        [-0.03040608, -0.0954188, 0.0557409],
                    ],
                    [
                        [0.09990136, -0.08285338, -0.19272368],
                        [-0.06107553, -0.04925374, -0.14729367],
                        [0.21084067, 0.16285245, -0.04423034],
                    ],
                ],
                [
                    [
                        [0.0213276, -0.04020498, -0.0695809],
                        [-0.07439377, -0.12844822, -0.07155278],
                        [-0.15585335, 0.0193371, 0.02764635],
                    ],
                    [
                        [0.07196217, -0.08014142, -0.21173581],
                        [-0.08674188, 0.0465454, 0.04854336],
                        [-0.07645767, -0.03202511, -0.13292158],
                    ],
                    [
                        [0.15233557, -0.09984318, -0.18849371],
                        [0.08223559, -0.08179485, -0.13408582],
                        [-0.12540144, -0.02884547, 0.05229353],
                    ],
                ],
                [
                    [
                        [-0.15763171, -0.02407911, -0.00681908],
                        [0.03250497, 0.1550481, 0.10114194],
                        [-0.09490093, 0.00461772, -0.11466124],
                    ],
                    [
                        [-0.02422531, 0.0127817, 0.18543416],
                        [0.06330952, -0.00709089, -0.20145036],
                        [-0.04776909, 0.14656785, -0.02975747],
                    ],
                    [
                        [0.0158229, -0.10876098, 0.05956696],
                        [-0.01243071, 0.12164979, -0.13510725],
                        [-0.00212572, 0.13608131, 0.05084516],
                    ],
                ],
                [
                    [
                        [0.00098468, 0.08638827, -0.07876199],
                        [-0.03128693, -0.08964928, 0.04160309],
                        [0.11917851, 0.08862061, -0.02827655],
                    ],
                    [
                        [0.17352532, -0.06490915, -0.07816768],
                        [0.00646151, 0.05263481, -0.15061238],
                        [-0.0843135, 0.05057508, -0.0828398],
                    ],
                    [
                        [0.03397794, 0.03386745, -0.14823097],
                        [0.04261533, 0.14061622, 0.05397691],
                        [0.09352212, 0.01038896, 0.12974001],
                    ],
                ],
            ]
        ]
    )

    b = jnp.array([[[[0.0]]]])

    layer = sk.nn.Conv3DTranspose(
        4, 1, 3, padding=2, strides=1, dilation=2, key=jax.random.key(0)
    )
    layer = layer.at["weight"].set(w)
    layer = layer.at["bias"].set(b)

    y = jnp.array(
        [
            [
                [
                    [-0.09683423, 0.44358936, 0.3575339],
                    [-0.22975557, 0.11778405, 0.27501053],
                    [-0.730394, -0.11462384, 0.11293405],
                ],
                [
                    [-0.06865323, 0.26918745, 0.21076179],
                    [-0.12277332, 0.15844066, 0.13829224],
                    [-0.47520664, -0.20936649, -0.02070317],
                ],
                [
                    [0.12662104, 0.28816995, 0.16732623],
                    [0.2505706, 0.25785616, 0.19034886],
                    [0.03599086, 0.14876032, 0.24214049],
                ],
            ]
        ]
    )

    npt.assert_allclose(y, layer(x), atol=1e-5)

    layer = sk.nn.Conv3DTranspose(
        4,
        1,
        3,
        padding=2,
        strides=1,
        dilation=2,
        bias_init=None,
        key=jax.random.key(0),
    )
    layer = layer.at["weight"].set(w)

    y = jnp.array(
        [
            [
                [
                    [-0.09683423, 0.44358936, 0.3575339],
                    [-0.22975557, 0.11778405, 0.27501053],
                    [-0.730394, -0.11462384, 0.11293405],
                ],
                [
                    [-0.06865323, 0.26918745, 0.21076179],
                    [-0.12277332, 0.15844066, 0.13829224],
                    [-0.47520664, -0.20936649, -0.02070317],
                ],
                [
                    [0.12662104, 0.28816995, 0.16732623],
                    [0.2505706, 0.25785616, 0.19034886],
                    [0.03599086, 0.14876032, 0.24214049],
                ],
            ]
        ]
    )

    npt.assert_allclose(y, layer(x), atol=1e-5)


def test_depthwise_conv1d():
    x = jnp.array(
        [
            [0.47723162, 0.9154197],
            [0.32835495, 0.83435106],
            [0.8260038, 0.8944353],
            [0.8727443, 0.47783256],
            [0.13558674, 0.70415115],
        ]
    )

    w = jnp.array(
        [
            [[0.0599848, -0.2638429, 0.52350515]],
            [[-0.28128156, -0.08704478, 0.2306788]],
            [[-0.22776055, 0.27533752, 0.2010997]],
            [[-0.03780305, 0.0926038, 0.11419511]],
            [[0.27788645, -0.17414662, 0.49673325]],
            [[-0.34119204, 0.01004243, 0.10704458]],
            [[0.04834682, 0.13226765, 0.08824182]],
            [[0.2252047, -0.4809752, 0.3147189]],
            [[-0.3383687, -0.4130007, -0.29718262]],
            [[-0.2222034, 0.4465317, -0.4457184]],
        ]
    )

    y = jnp.array(
        [
            [0.35331273, -0.21290036],
            [0.1696274, -0.21391895],
            [0.25819618, 0.15494184],
            [0.12568572, 0.06485126],
            [0.30044997, 0.07377239],
            [0.10403953, -0.27284363],
            [0.15760066, 0.1053962],
            [-0.26938546, -0.03327949],
            [-0.2652589, -0.33669323],
            [-0.25330934, 0.28429797],
        ]
    )

    layer = sk.nn.DepthwiseConv1D(
        in_features=5, kernel_size=3, depth_multiplier=2, key=jax.random.key(0)
    )
    layer = layer.at["weight"].set(w)

    npt.assert_allclose(y, layer(x), atol=1e-5)


def test_depthwise_conv2d():
    w = jnp.array(
        [
            [
                [
                    [-0.2693168, -0.01502147, 0.15012494],
                    [-0.00952473, 0.2306507, -0.18514359],
                    [0.42974612, 0.01279813, -0.3835524],
                ]
            ],
            [
                [
                    [0.22893754, -0.10614857, 0.3973861],
                    [0.2680097, 0.02779007, -0.40917027],
                    [0.22730389, -0.15551737, 0.1432856],
                ]
            ],
        ]
    )

    x = jnp.array(
        [
            [
                [0.10212982, 0.91221213, 0.20904398, 0.68790114, 0.70880795],
                [0.22495127, 0.9364072, 0.40070045, 0.212129, 0.04311025],
                [0.9548882, 0.140378, 0.14465642, 0.8334881, 0.00511444],
                [0.21266508, 0.0991888, 0.1724143, 0.09609783, 0.28935385],
                [0.5082896, 0.07669151, 0.21158075, 0.34280586, 0.6379497],
            ],
            [
                [0.9899399, 0.24278629, 0.6639457, 0.8549551, 0.5985389],
                [0.9747602, 0.10861945, 0.37414157, 0.0194155, 0.49901915],
                [0.80062544, 0.42624974, 0.7938998, 0.07342649, 0.1975286],
                [0.12891269, 0.5729337, 0.12007415, 0.7519882, 0.9888351],
                [0.31168306, 0.35112584, 0.5317881, 0.7633387, 0.0442704],
            ],
        ],
    )

    y = jnp.array(
        [
            [
                [-0.5016162, 0.12569302, 0.23835005, 0.1838218, 0.24864832],
                [-0.02769451, 0.48650065, -0.35882273, 0.14777793, 0.170266],
                [0.29613215, 0.00853828, -0.34067634, 0.04963186, -0.01953573],
                [0.01450753, -0.11036392, 0.01036078, -0.23313387, -0.00324076],
                [0.11473458, -0.05920575, -0.03027398, -0.04549751, 0.11365113],
            ],
            [
                [-0.20785895, 0.2586774, -0.29701594, 0.11032546, 0.17257676],
                [-0.08939171, 0.80533236, 0.3403615, 0.3930706, 0.13723959],
                [-0.15041986, 0.21929488, 0.31843108, 0.4679199, -0.00620808],
                [-0.1446053, 0.5473113, -0.00182522, -0.09056022, 0.39148763],
                [0.07898343, -0.10788733, 0.21379803, 0.48623982, 0.27300733],
            ],
        ]
    )

    layer = sk.nn.DepthwiseConv2D(2, 3, key=jax.random.key(0))
    layer = layer.at["weight"].set(w)

    npt.assert_allclose(y, layer(x), atol=1e-5)


def test_seperable_conv1d():
    x = jnp.array(
        [
            [0.60440326, 0.44108868, 0.89211035, 0.86819136, 0.99565816],
            [0.9063431, 0.64626694, 0.97458243, 0.163221, 0.24878585],
        ]
    )

    w1 = jnp.array(
        [
            [[-0.680353, -0.6851049, 0.60482687]],
            [[-0.02965522, -0.5761329, -0.11722302]],
            [[-0.12148273, 0.23472422, -0.00963646]],
            [[0.15432, 0.4803856, 0.00417626]],
        ]
    )

    w2 = jnp.array([[[0.4585532], [-0.3802476], [0.29934883], [0.80850005]]])

    y = jnp.array([[0.5005436, 0.44051802, 0.5662357, 0.13085097, -0.22720146]])

    layer = sk.nn.SeparableConv1D(
        in_features=2,
        out_features=1,
        kernel_size=3,
        depth_multiplier=2,
        key=jax.random.key(0),
    )

    layer = layer.at["depthwise_weight"].set(w1)
    layer = layer.at["pointwise_weight"].set(w2)

    npt.assert_allclose(y, layer(x), atol=1e-5)


def test_seperable_conv2d():
    x = jnp.array(
        [
            [
                [0.648191, 0.691234, 0.21900558, 0.94197, 0.3107084],
                [0.48518872, 0.8150476, 0.01268113, 0.7480334, 0.356174],
                [0.43563485, 0.86477315, 0.17455602, 0.21244049, 0.5058414],
                [0.3497224, 0.05337083, 0.13146913, 0.5132158, 0.308689],
                [0.86405003, 0.6903384, 0.08874571, 0.10805643, 0.61204815],
            ],
            [
                [0.08067358, 0.4133495, 0.40587842, 0.9662497, 0.07789862],
                [0.33498573, 0.73515475, 0.903263, 0.7960495, 0.09835494],
                [0.39089763, 0.09410667, 0.7823404, 0.1541977, 0.02740514],
                [0.2192421, 0.36280763, 0.46072757, 0.4836924, 0.87127614],
                [0.34191585, 0.82141125, 0.36559772, 0.25070393, 0.29197383],
            ],
        ]
    )

    w1 = jnp.array(
        [
            [
                [
                    [-0.14634436, -0.36800927, -0.13477623],
                    [-0.30637866, -0.03396568, 0.39051646],
                    [-0.25704288, 0.0559361, 0.00828293],
                ]
            ],
            [
                [
                    [-0.24296674, 0.20260447, 0.03023961],
                    [-0.14025584, 0.19635099, 0.3088829],
                    [-0.37352383, -0.27563155, -0.07043174],
                ]
            ],
            [
                [
                    [0.1397506, -0.20479576, -0.12788275],
                    [0.02316645, -0.34143952, -0.38499755],
                    [-0.34102032, -0.02208391, -0.14697403],
                ]
            ],
            [
                [
                    [-0.30659765, -0.05266225, 0.25238943],
                    [-0.3700197, 0.01858535, 0.36415017],
                    [0.06069544, 0.17667967, -0.40375128],
                ]
            ],
        ]
    )

    w2 = jnp.array([[[[-0.8940864]], [[0.04388905]], [[-0.3335355]], [[0.3573779]]]])

    y = jnp.array(
        [
            [
                [-0.1752224, 0.33442956, 0.37111858, 0.1924801, 0.3907704],
                [0.32045367, 0.85511434, 0.75993437, 0.41820422, 0.22250973],
                [0.13431981, 0.7713539, 0.5504457, 0.01695335, 0.36642462],
                [0.28530243, 0.8071514, 0.464118, 0.25935113, 0.40009925],
                [0.23932455, 0.49517187, 0.28972542, 0.17167322, 0.18646029],
            ]
        ]
    )

    layer_jax = sk.nn.SeparableConv2D(
        in_features=2,
        out_features=1,
        kernel_size=3,
        depth_multiplier=2,
        key=jax.random.key(0),
    )

    layer_jax = layer_jax.at["depthwise_weight"].set(w1)
    layer_jax = layer_jax.at["pointwise_weight"].set(w2)

    npt.assert_allclose(y, layer_jax(x), atol=1e-5)

    layer_jax = sk.nn.SeparableFFTConv2D(
        in_features=2,
        out_features=1,
        kernel_size=3,
        depth_multiplier=2,
        key=jax.random.key(0),
    )
    layer_jax = layer_jax.at["depthwise_weight"].set(w1)
    layer_jax = layer_jax.at["pointwise_weight"].set(w2)
    npt.assert_allclose(y, layer_jax(x), atol=1e-5)


def test_conv1d_local():
    x = jnp.ones((2, 28))

    w = jnp.array(
        [
            [
                [
                    -0.15961593,
                    -0.24700482,
                    -0.03241882,
                    0.16830233,
                    0.17315423,
                    -0.15926096,
                    0.11997852,
                    0.19070503,
                    0.2514011,
                    0.01214904,
                    0.12885782,
                    -0.13484786,
                    -0.19441944,
                ],
                [
                    0.04949,
                    -0.25033048,
                    0.16624266,
                    0.16315845,
                    -0.07447895,
                    0.24705955,
                    -0.2545433,
                    -0.04126936,
                    0.11806422,
                    0.0186832,
                    0.05966297,
                    0.1445893,
                    -0.22556928,
                ],
                [
                    -0.23063937,
                    -0.03540739,
                    -0.08050132,
                    -0.07008512,
                    0.00241017,
                    0.05862182,
                    0.02273929,
                    -0.03910652,
                    0.2082575,
                    0.02895498,
                    -0.22927788,
                    0.17321557,
                    0.00570095,
                ],
                [
                    0.08137614,
                    0.25040284,
                    -0.24458233,
                    0.12267029,
                    -0.12790537,
                    0.17364258,
                    0.05551764,
                    0.03889585,
                    0.24334595,
                    0.16311577,
                    -0.05199888,
                    0.1495251,
                    -0.12573873,
                ],
                [
                    -0.04205574,
                    -0.05446893,
                    0.16779593,
                    0.15733883,
                    -0.11077882,
                    0.11935773,
                    0.06591952,
                    0.13060933,
                    -0.19740026,
                    -0.13954279,
                    0.00263405,
                    0.00692567,
                    -0.10878837,
                ],
                [
                    0.01163918,
                    0.06446105,
                    -0.01387599,
                    -0.11335008,
                    -0.18015893,
                    0.03948161,
                    0.2264004,
                    0.21437737,
                    -0.25310597,
                    0.14172179,
                    -0.14102633,
                    0.18787548,
                    -0.2130653,
                ],
            ]
        ]
    )

    y = jnp.array(
        [
            [
                -0.2898057,
                -0.27234778,
                -0.03733987,
                0.42803472,
                -0.31775767,
                0.47890234,
                0.23601207,
                0.4942117,
                0.37056252,
                0.22508198,
                -0.23114826,
                0.52728325,
                -0.8618802,
            ]
        ]
    )

    layer = sk.nn.Conv1DLocal(
        in_features=2,
        out_features=1,
        kernel_size=3,
        strides=2,
        in_size=(28,),
        padding="valid",
        key=jax.random.key(0),
    )

    layer = layer.at["weight"].set(w)

    npt.assert_allclose(y, layer(x), atol=1e-5)


def test_conv2d_local():
    w = jnp.array(
        [
            [
                [[-0.21994266, -0.2716022]],
                [[0.2665612, -0.15465173]],
                [[0.08909398, -0.36371586]],
                [[0.1832841, -0.07327542]],
                [[0.21907657, 0.00321126]],
                [[-0.28950286, 0.33936942]],
                [[0.26961517, 0.28448611]],
                [[-0.44536602, 0.2771259]],
                [[-0.02320498, 0.08787525]],
                [[-0.47078812, 0.39727128]],
                [[0.00807443, 0.41162848]],
                [[0.33439147, 0.00823227]],
            ]
        ]
    )

    y = jnp.array([[[-0.07870772, 0.9459547]]])

    x = jnp.ones((2, 4, 4))

    layer = sk.nn.Conv2DLocal(
        2,
        1,
        (3, 2),
        in_size=(4, 4),
        padding="valid",
        strides=2,
        key=jax.random.key(0),
    )
    layer = layer.at["weight"].set(w)

    npt.assert_allclose(y, layer(x), atol=1e-5)


def test_in_feature_error():
    with pytest.raises(ValueError):
        sk.nn.Conv1D(0, 1, 2, key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv2D(0, 1, 2, key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv3D(0, 1, 2, key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv1DLocal(0, 1, 2, in_size=(2,), key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv2DLocal(0, 1, 2, in_size=(2, 2), key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv1DTranspose(0, 1, 3, key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv2DTranspose(0, 1, 3, key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv3DTranspose(0, 1, 3, key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.DepthwiseConv1D(0, 1, key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.DepthwiseConv2D(0, 1, key=jax.random.key(0))


def test_out_feature_error():
    with pytest.raises(ValueError):
        sk.nn.Conv1D(1, 0, 2, key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv2D(1, 0, 2, key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv3D(1, 0, 2, key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv1DLocal(1, 0, 2, in_size=(2,), key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv2DLocal(1, 0, 2, in_size=(2, 2), key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv1DTranspose(1, 0, 3, key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv2DTranspose(1, 0, 3, key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv3DTranspose(1, 0, 3, key=jax.random.key(0))


def test_groups_error():
    with pytest.raises(ValueError):
        sk.nn.Conv1D(1, 1, 2, groups=0, key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv2D(1, 1, 2, groups=0, key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv3D(1, 1, 2, groups=0, key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv1DTranspose(1, 1, 3, groups=0, key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv2DTranspose(1, 1, 3, groups=0, key=jax.random.key(0))

    with pytest.raises(ValueError):
        sk.nn.Conv3DTranspose(1, 1, 3, groups=0, key=jax.random.key(0))


@pytest.mark.parametrize(
    "layer,array,expected_shape",
    [
        [sk.nn.Conv1D, jnp.ones([10, 3]), (1, 3)],
        [sk.nn.Conv2D, jnp.ones([10, 3, 3]), (1, 3, 3)],
        [sk.nn.Conv3D, jnp.ones([10, 3, 3, 3]), (1, 3, 3, 3)],
        [sk.nn.Conv1DTranspose, jnp.ones([10, 3]), (1, 3)],
        [sk.nn.Conv2DTranspose, jnp.ones([10, 3, 3]), (1, 3, 3)],
        [sk.nn.Conv3DTranspose, jnp.ones([10, 3, 3, 3]), (1, 3, 3, 3)],
    ],
)
def test_lazy_conv(layer, array, expected_shape):
    lazy = layer(None, 1, 3, key=jax.random.key(0))
    value, material = sk.value_and_tree(lambda layer: layer(array))(lazy)

    assert value.shape == expected_shape
    assert material.in_features == 10


def test_lazy_conv_local():
    layer = sk.nn.Conv1DLocal(None, 1, 3, in_size=(3,), key=jax.random.key(0))
    _, layer = sk.value_and_tree(lambda layer: layer(jnp.ones([10, 3])))(layer)
    assert layer.in_features == 10
    layer = sk.nn.Conv1DLocal(2, 1, 2, in_size=None, key=jax.random.key(0))

    with pytest.raises(ValueError):
        # should raise error because in_features is specified = 2 and
        # input in_features is 10
        _, layer = sk.value_and_tree(lambda layer: layer(jnp.ones([10, 3])))(layer)

    _, layer = sk.value_and_tree(lambda layer: layer(jnp.ones([2, 3])))(layer)
    assert layer.in_features == 2


@pytest.mark.parametrize(
    (
        "sk_layer",
        "keras_layer",
        "kernel_size",
        "strides",
        "padding",
        "dilation",
        "ndim",
    ),
    [
        *product(
            [sk.nn.Conv1D, sk.nn.FFTConv1D],
            [keras.layers.Conv1D],
            [3, 5],
            [1, 2],
            ["same", "valid"],
            [1],
            [1],
        ),
        *product(
            [sk.nn.Conv2D, sk.nn.FFTConv2D],
            [keras.layers.Conv2D],
            [(3, 3), (3, 5), 2],
            [(2, 1), (1, 2), 1],
            ["same", "valid"],
            [1],
            [2],
        ),
        *product(
            [sk.nn.Conv3D, sk.nn.FFTConv3D],
            [keras.layers.Conv3D],
            [(3, 3, 3), (3, 5, 3), 2],
            [(2, 1, 1), (1, 2, 1), 1],
            ["same", "valid"],
            [1],
            [3],
        ),
    ],
)
def test_conv_keras(
    sk_layer,
    keras_layer,
    kernel_size,
    strides,
    padding,
    dilation,
    ndim,
):
    shape = [4] + [10] * ndim
    x = jax.random.uniform(jax.random.key(0), shape)
    layer_keras = keras_layer(
        filters=3,
        kernel_size=kernel_size,
        padding=padding,
        dilation_rate=dilation,
        strides=strides,
        data_format="channels_first",
    )
    layer_sk = sk_layer(
        4,
        3,
        kernel_size=kernel_size,
        padding=padding,
        dilation=dilation,
        strides=strides,
        key=jax.random.key(0),
    )
    layer_keras.build((1, *shape))

    weight = layer_keras.get_weights()[0]
    bias = layer_keras.get_weights()[1]

    # *KIO -> OI*K
    axes = list(range(weight.ndim))
    axes = axes[-2:][::-1] + axes[:-2]
    weight_ = jnp.transpose(weight, axes)
    bias_ = jnp.expand_dims(bias, list(range(1, ndim + 1)))

    layer_sk = layer_sk.at["weight"].set(weight_).at["bias"].set(bias_)
    npt.assert_allclose(layer_sk(x), layer_keras(x[None])[0], atol=5e-6)


# @pytest.mark.parametrize(
#     "sk_layer,keras_layer,kernel_size,strides,padding,ndim",
#     [
#         *product(
#             [sk.nn.DepthwiseConv1D, sk.nn.DepthwiseFFTConv1D],
#             [keras.layers.DepthwiseConv1D],
#             [3, 5],
#             [1, 2],
#             ["same"],
#             [1],
#         ),
#         # *product(
#         #     [sk.nn.DepthwiseConv2D, sk.nn.DepthwiseFFTConv2D],
#         #     [keras.layers.DepthwiseConv2D],
#         #     [(3, 3), (3, 5), 2],
#         #     [(2, 1), (1, 2), 1],
#         #     ["same", "valid"],
#         #     [2],
#         # ),
#         # *product(
#         #     [sk.nn.DepthwiseConv3D, sk.nn.DepthwiseFFTConv3D],
#         #     [keras.layers.DepthwiseConv3D],
#         #     [(3, 3, 3), (3, 5, 3), 2],
#         #     [(2, 1, 1), (1, 2, 1), 1],
#         #     ["same", "valid"],
#         #     [1],
#         #     [3],
#         # ),
#     ],
# )
# def test_depthwiseconv_keras(
#     sk_layer,
#     keras_layer,
#     kernel_size,
#     strides,
#     padding,
#     ndim,
# ):
#     shape = [4] + [10] * ndim
#     x = jax.random.uniform(jax.random.key(0), shape)
#     layer_keras = keras_layer(
#         kernel_size=kernel_size,
#         depth_multiplier=4,
#         padding=padding,
#         strides=strides,
#         data_format="channels_first",
#     )
#     layer_sk = sk_layer(
#         4,
#         kernel_size=kernel_size,
#         padding=padding,
#         strides=strides,
#         depth_multiplier=4,
#         key=jax.random.key(0),
#     )
#     layer_keras.build((1, *shape))

#     weight = layer_keras.get_weights()[0]
#     bias = layer_keras.get_weights()[1]

#     # *KIO -> OI*K
#     axes = list(range(weight.ndim))
#     axes = axes[-2:][::-1] + axes[:-2]
#     weight_ = jnp.transpose(weight, axes)
#     bias_ = jnp.expand_dims(bias, list(range(1, ndim + 1)))

#     layer_sk = layer_sk.at["weight"].set(weight_).at["bias"].set(bias_)
#     npt.assert_allclose(layer_sk(x), layer_keras(x[None])[0], atol=5e-6)


def test_spectral_conv_1d():
    layer = sk.nn.SpectralConv1D(1, 2, modes=10, key=jax.random.key(0))
    layer = (
        layer.at["weight_r"]
        .set(
            jnp.array(
                [
                    [
                        [
                            [
                                0.2481,
                                0.0442,
                                0.1537,
                                0.2450,
                                0.2278,
                                0.1744,
                                0.0112,
                                0.1469,
                                0.3488,
                                0.0805,
                            ]
                        ],
                        [
                            [
                                0.3408,
                                0.1985,
                                0.2097,
                                0.4764,
                                0.0926,
                                0.1526,
                                0.0880,
                                0.0753,
                                0.1041,
                                0.3616,
                            ]
                        ],
                    ]
                ]
            )
        )
        .at["weight_i"]
        .set(
            jnp.array(
                [
                    [
                        [
                            [
                                0.3841,
                                0.0660,
                                0.3170,
                                0.4482,
                                0.3162,
                                0.2009,
                                0.0844,
                                0.2593,
                                0.4000,
                                0.1411,
                            ]
                        ],
                        [
                            [
                                0.4576,
                                0.4371,
                                0.2765,
                                0.0181,
                                0.1867,
                                0.4660,
                                0.1349,
                                0.0159,
                                0.4649,
                                0.3712,
                            ]
                        ],
                    ]
                ]
            )
        )
    )
    x = jnp.array(
        [
            [
                -0.2159,
                -0.7425,
                0.5627,
                0.2596,
                0.5943,
                1.5419,
                0.5073,
                -0.5910,
                -0.5692,
                0.9200,
                1.1108,
                1.2899,
                -1.4959,
                -0.1938,
                0.4455,
                1.3253,
                -1.6293,
                -0.5497,
                -0.4798,
                -0.4997,
            ]
        ]
    )
    npt.assert_allclose(
        layer(x),
        jnp.array(
            [
                [
                    0.0328842,
                    -0.09070581,
                    0.37291762,
                    0.1076019,
                    0.22965696,
                    0.09572654,
                    -0.3921412,
                    -0.3544573,
                    0.11821049,
                    0.4223338,
                    0.4160973,
                    -0.32546958,
                    -0.37600443,
                    0.1276577,
                    0.38620248,
                    0.06922199,
                    -0.46361017,
                    0.02187163,
                    -0.12691171,
                    0.12352063,
                ],
                [
                    -0.05426618,
                    0.19038501,
                    0.6024397,
                    0.03671208,
                    0.67035574,
                    0.25951746,
                    0.091704,
                    -0.49809638,
                    -0.01539469,
                    0.38438407,
                    0.2571618,
                    -0.30642772,
                    -0.38458398,
                    0.10909297,
                    -0.13115075,
                    -0.24731942,
                    -0.6856997,
                    0.403228,
                    -0.07954475,
                    -0.06045485,
                ],
            ]
        ),
        atol=1e-6,
    )


def test_spectral_conv_2d():
    layer_ = sk.nn.SpectralConv2D(1, 2, modes=(4, 3), key=jax.random.key(0))

    w_r = jnp.stack(
        [
            jnp.array(
                [
                    [
                        [
                            [0.3396, 0.3814, 0.4588],
                            [0.0463, 0.0990, 0.1236],
                            [0.0493, 0.3005, 0.1481],
                            [0.0028, 0.3075, 0.0591],
                        ]
                    ],
                    [
                        [
                            [0.1275, 0.0414, 0.0860],
                            [0.2221, 0.1159, 0.2975],
                            [0.1586, 0.1410, 0.0511],
                            [0.4905, 0.2091, 0.2593],
                        ]
                    ],
                ]
            ),
            jnp.array(
                [
                    [
                        [
                            [0.3153, 0.4131, 0.2464],
                            [0.4483, 0.3570, 0.0767],
                            [0.0404, 0.2813, 0.0969],
                            [0.2975, 0.3396, 0.0586],
                        ]
                    ],
                    [
                        [
                            [0.3214, 0.3991, 0.0238],
                            [0.2986, 0.0493, 0.4539],
                            [0.3181, 0.1280, 0.3648],
                            [0.0987, 0.3623, 0.1733],
                        ]
                    ],
                ]
            ),
        ]
    )

    w_i = jnp.stack(
        [
            jnp.array(
                [
                    [
                        [
                            [0.0507, 0.2707, 0.0512],
                            [0.1630, 0.0753, 0.0865],
                            [0.2492, 0.1557, 0.2747],
                            [0.0589, 0.2555, 0.3134],
                        ]
                    ],
                    [
                        [
                            [0.4104, 0.4585, 0.1077],
                            [0.0841, 0.1328, 0.2571],
                            [0.3806, 0.1017, 0.2582],
                            [0.4697, 0.4653, 0.3250],
                        ]
                    ],
                ]
            ),
            jnp.array(
                [
                    [
                        [
                            [0.3802, 0.2843, 0.3507],
                            [0.0795, 0.2979, 0.0433],
                            [0.0788, 0.4163, 0.3746],
                            [0.1387, 0.0899, 0.2833],
                        ]
                    ],
                    [
                        [
                            [0.4518, 0.0037, 0.0324],
                            [0.2218, 0.4991, 0.1319],
                            [0.0193, 0.4955, 0.0179],
                            [0.0938, 0.1494, 0.1309],
                        ]
                    ],
                ]
            ),
        ]
    )

    layer_ = layer_.at["weight_r"].set(w_r).at["weight_i"].set(w_i)
    x = jnp.array(
        [
            [
                [-0.1066, 0.4910, -0.1327, -0.1839, 0.4081],
                [-0.4769, 0.4919, 0.2303, -0.6603, 0.3051],
                [1.4438, -0.4929, 0.2225, 1.3679, -0.1168],
                [-0.0901, 0.0051, 0.5716, -1.0714, -0.7623],
                [-1.2768, -0.9546, -0.8639, -0.2049, -0.2692],
            ]
        ]
    )

    npt.assert_allclose(
        layer_(x),
        jnp.array(
            [
                [
                    [0.12254247, 0.13477372, -0.22648641, 0.23759897, 0.03508948],
                    [0.08804332, 0.3369788, -0.11117554, -0.11362755, 0.00313328],
                    [0.09802226, -0.36852267, 0.45606878, 0.18551637, 0.12071826],
                    [0.09001034, 0.01557234, -0.07713533, -0.56744605, -0.13950717],
                    [-0.54261154, -0.24243608, -0.02123806, -0.08897388, -0.14689755],
                ],
                [
                    [0.24471198, 0.03794191, -0.00319719, 0.11428356, 0.22155367],
                    [-0.05815738, 0.23682189, -0.0759994, -0.31811824, 0.25815266],
                    [0.14537422, -0.12251858, 0.28575662, 0.4039829, -0.19226913],
                    [0.03137114, -0.05654814, -0.26489, -0.59775954, 0.07911236],
                    [-0.37372565, -0.07363869, -0.0180975, 0.2213602, -0.3965687],
                ],
            ]
        ),
        atol=1e-6,
    )


def test_spectral_conv_3d():
    w_r = jnp.stack(
        [
            jnp.array(
                [
                    [
                        [
                            [[0.4422, 0.2378, 0.2697], [0.3186, 0.0630, 0.4937]],
                            [[0.4556, 0.2324, 0.3278], [0.1899, 0.4577, 0.2965]],
                            [[0.1820, 0.2199, 0.3645], [0.3871, 0.2518, 0.3968]],
                        ]
                    ],
                    [
                        [
                            [[0.2688, 0.4664, 0.4470], [0.3320, 0.0973, 0.4505]],
                            [[0.2332, 0.3554, 0.4052], [0.2642, 0.1790, 0.4521]],
                            [[0.2201, 0.3265, 0.1480], [0.3731, 0.3868, 0.4348]],
                        ]
                    ],
                ]
            ),
            jnp.array(
                [
                    [
                        [
                            [[0.1021, 0.0450, 0.2536], [0.3417, 0.2413, 0.0771]],
                            [[0.4366, 0.4174, 0.2173], [0.1230, 0.2131, 0.0276]],
                            [[0.4427, 0.0157, 0.4741], [0.3420, 0.2471, 0.1925]],
                        ]
                    ],
                    [
                        [
                            [[0.1400, 0.4713, 0.3775], [0.2534, 0.1483, 0.1195]],
                            [[0.0349, 0.3495, 0.0479], [0.2350, 0.0278, 0.2931]],
                            [[0.1592, 0.0883, 0.4120], [0.1571, 0.4263, 0.2472]],
                        ]
                    ],
                ]
            ),
            jnp.array(
                [
                    [
                        [
                            [[0.1397, 0.1473, 0.2587], [0.1917, 0.2872, 0.0730]],
                            [[0.2375, 0.0188, 0.1536], [0.2670, 0.1198, 0.2248]],
                            [[0.4007, 0.4197, 0.4730], [0.3005, 0.0074, 0.4410]],
                        ]
                    ],
                    [
                        [
                            [[0.0063, 0.0102, 0.4848], [0.1350, 0.1516, 0.2781]],
                            [[0.3461, 0.2521, 0.1409], [0.2032, 0.1563, 0.2196]],
                            [[0.2589, 0.1135, 0.4836], [0.3718, 0.3525, 0.2745]],
                        ]
                    ],
                ]
            ),
            jnp.array(
                [
                    [
                        [
                            [[0.1911, 0.1833, 0.3703], [0.4744, 0.2607, 0.4507]],
                            [[0.3662, 0.2172, 0.4775], [0.2457, 0.0858, 0.1412]],
                            [[0.2551, 0.4672, 0.3425], [0.3431, 0.0920, 0.1829]],
                        ]
                    ],
                    [
                        [
                            [[0.0839, 0.4867, 0.3175], [0.0408, 0.4319, 0.4521]],
                            [[0.4711, 0.0948, 0.2310], [0.3798, 0.1946, 0.3274]],
                            [[0.0108, 0.1521, 0.4714], [0.3973, 0.4319, 0.3664]],
                        ]
                    ],
                ]
            ),
        ]
    )

    w_i = jnp.stack(
        [
            jnp.array(
                [
                    [
                        [
                            [[0.1228, 0.2616, 0.1616], [0.0463, 0.3155, 0.0100]],
                            [[0.4512, 0.4287, 0.1517], [0.4981, 0.3791, 0.1393]],
                            [[0.4824, 0.3823, 0.1861], [0.0159, 0.2970, 0.3337]],
                        ]
                    ],
                    [
                        [
                            [[0.0635, 0.2905, 0.2268], [0.4156, 0.1227, 0.2814]],
                            [[0.0172, 0.0739, 0.2992], [0.1306, 0.1702, 0.3850]],
                            [[0.1325, 0.2009, 0.1637], [0.3154, 0.4834, 0.3377]],
                        ]
                    ],
                ]
            ),
            jnp.array(
                [
                    [
                        [
                            [[0.2815, 0.0128, 0.3531], [0.3258, 0.3700, 0.1727]],
                            [[0.3728, 0.0030, 0.3264], [0.0184, 0.2255, 0.4447]],
                            [[0.4829, 0.0999, 0.1491], [0.1728, 0.1023, 0.1339]],
                        ]
                    ],
                    [
                        [
                            [[0.1623, 0.1927, 0.0045], [0.0780, 0.0841, 0.4370]],
                            [[0.1134, 0.1457, 0.1222], [0.0190, 0.4638, 0.4037]],
                            [[0.3939, 0.2813, 0.1856], [0.1781, 0.2129, 0.3442]],
                        ]
                    ],
                ]
            ),
            jnp.array(
                [
                    [
                        [
                            [[0.4053, 0.4150, 0.4490], [0.3855, 0.3994, 0.2125]],
                            [[0.1191, 0.1685, 0.2662], [0.4737, 0.4967, 0.0080]],
                            [[0.0063, 0.3181, 0.2767], [0.4962, 0.3980, 0.4969]],
                        ]
                    ],
                    [
                        [
                            [[0.4781, 0.4388, 0.1791], [0.0499, 0.0445, 0.0923]],
                            [[0.1128, 0.3076, 0.2786], [0.4791, 0.0022, 0.2538]],
                            [[0.3828, 0.1285, 0.1179], [0.1898, 0.0898, 0.3909]],
                        ]
                    ],
                ]
            ),
            jnp.array(
                [
                    [
                        [
                            [[0.0403, 0.1285, 0.0995], [0.0785, 0.2148, 0.0999]],
                            [[0.0990, 0.4321, 0.2214], [0.3379, 0.4558, 0.3288]],
                            [[0.4520, 0.3815, 0.3552], [0.2973, 0.1899, 0.0380]],
                        ]
                    ],
                    [
                        [
                            [[0.1185, 0.0394, 0.3835], [0.4336, 0.3033, 0.4651]],
                            [[0.3038, 0.0242, 0.1272], [0.0196, 0.0155, 0.2290]],
                            [[0.4361, 0.4002, 0.2696], [0.3674, 0.1206, 0.2150]],
                        ]
                    ],
                ]
            ),
        ]
    )

    layer_ = sk.nn.SpectralConv3D(1, 2, modes=(3, 2, 3), key=jax.random.key(0))
    layer_ = layer_.at["weight_r"].set(w_r).at["weight_i"].set(w_i)
    x_ = jnp.array(
        [
            [
                [
                    [-0.8077, -0.0889, -1.1965, 0.4884, -0.2976],
                    [-0.0915, 0.0849, -0.8136, -0.7932, -0.8501],
                    [0.0728, -0.4103, 0.3818, 0.8184, 0.4787],
                    [-2.0097, -0.0694, 2.5465, -0.0400, -0.6161],
                    [0.4981, -0.7493, 0.5346, -1.9640, -1.9855],
                ],
                [
                    [-0.3440, 0.9739, 0.2299, -0.7442, -1.4439],
                    [-1.3525, 0.8112, -0.7390, -0.5703, -1.9879],
                    [-0.7590, -2.1056, -0.8409, 0.6213, -0.1782],
                    [-1.1078, 1.3205, -0.1416, 1.2144, 0.3062],
                    [1.0100, 2.6431, 1.1126, 0.0221, -1.1145],
                ],
                [
                    [2.9416, -0.2999, 0.0600, -0.2478, 0.5127],
                    [-0.4685, -0.5397, -1.9055, -0.3839, 0.5221],
                    [0.3877, -0.1948, -1.8084, 0.1575, -1.7568],
                    [1.9861, 0.8096, -0.7128, -0.1727, -2.0262],
                    [-0.6110, 0.8884, 0.8381, 2.3556, -0.2357],
                ],
                [
                    [1.0616, -0.5036, -0.0585, -0.5807, 0.6485],
                    [-1.5226, 1.0455, 0.5958, -1.3031, 0.3210],
                    [-0.1032, 0.1299, -0.2774, -0.8132, 1.0651],
                    [0.0047, 0.0368, 0.0468, 0.8399, 0.7140],
                    [1.2273, -0.2270, -1.3760, 0.0756, 0.4333],
                ],
                [
                    [0.5033, -0.0075, -1.1936, 0.6505, 0.8711],
                    [-1.8358, -0.0492, 1.6722, -0.0853, 0.6354],
                    [0.3888, 0.9005, -1.1166, 0.4884, 1.0565],
                    [-0.3597, -1.7342, 0.4268, -1.0826, -0.7833],
                    [0.5278, 0.1570, -0.1778, 0.6771, -0.0876],
                ],
            ]
        ]
    )

    import numpy.testing as npt

    npt.assert_allclose(
        layer_(x_),
        jnp.array(
            [
                [
                    [
                        [
                            -0.00952937,
                            -0.09997971,
                            -0.15907927,
                            -0.04657828,
                            -0.3884752,
                        ],
                        [-0.20385663, -0.29563156, -0.3580177, -0.21104312, 0.14476638],
                        [0.05787412, 0.07032249, 0.1654335, -0.22869341, -0.22280629],
                        [-0.2619114, 0.47952688, 0.4127133, -0.46945438, -0.22931936],
                        [-0.00530472, 0.01531786, 0.40660092, -0.82888436, -0.46972078],
                    ],
                    [
                        [0.73715407, 0.15441893, -0.3581801, -0.26002598, -0.27185518],
                        [0.08890544, 0.18254262, -0.6515506, -0.4800409, -0.60789806],
                        [-0.7648498, -0.12228681, -0.05160045, 0.1934209, -0.7153443],
                        [-0.02933906, 0.0705153, -0.35353458, 0.52394825, -0.1143944],
                        [0.7213779, 0.70545316, -0.09953299, -0.12526603, -0.09469531],
                    ],
                    [
                        [0.51321805, -0.55638933, 0.11096933, 0.2885401, 0.6772172],
                        [0.23915023, -0.2182792, -0.1611944, 0.19546825, -0.213074],
                        [-0.1407576, -0.5288355, -0.50846803, 0.02196375, -0.07204127],
                        [1.0944732, 0.00640141, -0.62735915, -0.4159209, -0.39573458],
                        [0.3980013, 0.32021376, 0.07743512, 0.21066219, -0.20130579],
                    ],
                    [
                        [0.0795495, -0.3690523, -0.03239918, 0.08990704, 0.5434028],
                        [-0.39513332, 0.50895506, 0.01622826, -0.10175109, 0.29015338],
                        [-0.29087728, 0.45998833, -0.17015672, -0.01697872, 0.14216062],
                        [-0.0174965, 0.25106546, -0.21963704, 0.3009086, -0.02599236],
                        [-0.23155355, -0.6227041, -0.06031143, 0.30637437, 0.33302101],
                    ],
                    [
                        [-0.0447557, 0.1414328, 0.07026405, 0.26351675, 0.08906306],
                        [-0.51435304, 0.25030228, 0.5550475, -0.04011752, 0.03652146],
                        [-0.22774845, -0.12961094, 0.20103218, 0.27164754, 0.06311806],
                        [
                            -0.13335203,
                            -0.20071849,
                            -0.14559206,
                            -0.13275158,
                            -0.00467712,
                        ],
                        [0.02101387, -0.50302416, 0.14187033, 0.11888778, -0.11930912],
                    ],
                ],
                [
                    [
                        [0.06833271, -0.23742741, -0.41210794, 0.19206612, -0.41307318],
                        [-0.14734407, -0.35206315, -0.19915298, 0.0115882, -0.12853135],
                        [-0.10532138, -0.03484996, 0.45841172, 0.26824796, -0.2892223],
                        [-0.10966449, 0.4651556, 0.394094, -0.17670897, 0.28249568],
                        [0.22479233, 0.01744341, 0.2180783, -0.69224876, -0.27506685],
                    ],
                    [
                        [0.3394785, 0.53365016, 0.1283809, -0.2581764, -0.46553305],
                        [
                            -0.01016378,
                            -0.04189893,
                            -0.33788094,
                            -0.26559076,
                            -0.38592568,
                        ],
                        [-0.4349737, -0.26480386, -0.0483421, 0.08696102, -0.44293502],
                        [-0.05198468, 0.10266227, -0.18406327, 0.4219227, -0.46086884],
                        [0.33701414, 0.9826842, -0.02537738, -0.11621875, -0.5175969],
                    ],
                    [
                        [0.06611431, -0.38909042, -0.05015415, 0.20969553, 0.27406588],
                        [0.4408156, 0.14405787, -0.56176203, -0.18683532, -0.2516399],
                        [0.29772887, -0.1703223, -0.5146025, -0.08789919, -0.42210284],
                        [1.0012482, -0.09730292, -0.29393196, 0.04346997, -0.60627633],
                        [0.09716587, 0.14595854, 0.2803105, 0.39357546, -0.53534424],
                    ],
                    [
                        [0.02507782, -0.32720768, 0.03543025, -0.28854078, 0.28156114],
                        [-0.49916995, 0.54552394, -0.06522982, -0.4807392, 0.22325806],
                        [-0.3486275, 0.2818212, -0.10975402, 0.07556005, 0.16782479],
                        [0.5435274, -0.01019724, -0.27392554, 0.2969026, 0.28230852],
                        [0.40475816, -0.46541727, 0.14079623, -0.11414856, 0.11832531],
                    ],
                    [
                        [-0.1994129, 0.11780342, -0.553713, 0.24488054, 0.34777692],
                        [-0.500532, 0.42688718, 0.03705249, -0.05564727, 0.20532164],
                        [0.04618343, 0.23428321, -0.07118867, 0.34591073, 0.19023784],
                        [0.00423366, 0.08053151, 0.04566277, -0.18225242, 0.20350815],
                        [-0.0545201, -0.7047, 0.08382111, 0.15045369, 0.08529431],
                    ],
                ],
            ]
        ),
        atol=1e-6,
    )
