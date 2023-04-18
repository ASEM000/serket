import jax.numpy as jnp
import numpy.testing as npt
import pytest

from serket.nn.convolution import (  # Conv3DLocal,; Conv1DSemiLocal,; Conv2DSemiLocal,; Conv3DSemiLocal,
    Conv1D,
    Conv1DLocal,
    Conv1DTranspose,
    Conv2D,
    Conv2DLocal,
    Conv2DTranspose,
    Conv3D,
    Conv3DTranspose,
    DepthwiseConv1D,
    DepthwiseConv2D,
    SeparableConv1D,
    SeparableConv2D,
)


def test_conv1D():
    layer = Conv1D(
        in_features=1,
        out_features=1,
        kernel_size=2,
        padding="SAME",
        strides=1,
    )

    layer = layer.at["weight"].set(jnp.ones([1, 1, 2], dtype=jnp.float32))  # OIW
    x = jnp.arange(1, 11).reshape([1, 10]).astype(jnp.float32)
    npt.assert_allclose(layer(x), jnp.array([[3, 5, 7, 9, 11, 13, 15, 17, 19, 10]]))

    layer = Conv1D(
        in_features=1, out_features=1, kernel_size=2, padding="SAME", strides=2
    )
    layer = layer.at["weight"].set(jnp.ones([1, 1, 2], dtype=jnp.float32))
    x = jnp.arange(1, 11).reshape([1, 10]).astype(jnp.float32)

    npt.assert_allclose(layer(x), jnp.array([[3, 7, 11, 15, 19]]))

    layer = Conv1D(
        in_features=1, out_features=1, kernel_size=2, padding="VALID", strides=1
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

    layer = Conv1D(1, 2, 3, padding=2, strides=1, kernel_dilation=2)
    layer = layer.at["weight"].set(w)
    layer = layer.at["bias"].set(b)

    npt.assert_allclose(layer(x), y)

    layer = Conv1D(
        1, 2, 3, padding=2, strides=1, kernel_dilation=2, bias_init_func=None
    )
    layer = layer.at["weight"].set(w)
    npt.assert_allclose(layer(x), y)


def test_conv2D():
    layer = Conv2D(in_features=1, out_features=1, kernel_size=2)
    layer = layer.at["weight"].set(jnp.ones([1, 1, 2, 2], dtype=jnp.float32))  # OIHW
    x = jnp.arange(1, 17).reshape([1, 4, 4]).astype(jnp.float32)

    npt.assert_allclose(
        layer(x)[0],
        jnp.array(
            [[14, 18, 22, 12], [30, 34, 38, 20], [46, 50, 54, 28], [27, 29, 31, 16]]
        ),
    )

    layer = Conv2D(in_features=1, out_features=1, kernel_size=2, padding="VALID")
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

    layer = Conv2D(1, 2, 2, padding="SAME", strides=2)
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

    layer = Conv2D(1, 2, 2, padding="SAME", strides=1)
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

    layer = Conv2D(1, 2, 2, padding="SAME", strides=1, bias_init_func=None)
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
    layer = Conv3D(1, 3, 3)
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

    layer = Conv1DTranspose(4, 1, 3, padding=2, strides=1, kernel_dilation=2)
    layer = layer.at["weight"].set(w)
    layer = layer.at["bias"].set(b)
    y = jnp.array([[0.27022034, 0.24495776, -0.00368674]])
    npt.assert_allclose(layer(x), y, atol=1e-5)

    layer = Conv1DTranspose(
        4, 1, 3, padding=2, strides=1, kernel_dilation=2, bias_init_func=None
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

    layer = Conv2DTranspose(3, 1, 3, padding=2, strides=1, kernel_dilation=2)

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

    layer = Conv2DTranspose(
        3, 1, 3, padding=2, strides=1, kernel_dilation=2, bias_init_func=None
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

    layer = Conv3DTranspose(4, 1, 3, padding=2, strides=1, kernel_dilation=2)
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

    layer = Conv3DTranspose(
        4, 1, 3, padding=2, strides=1, kernel_dilation=2, bias_init_func=None
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

    layer = DepthwiseConv1D(in_features=5, kernel_size=3, depth_multiplier=2)
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

    layer = DepthwiseConv2D(2, 3)
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

    layer = SeparableConv1D(
        in_features=2, out_features=1, kernel_size=3, depth_multiplier=2
    )

    layer = layer.at["depthwise_conv"].at["weight"].set(w1)
    layer = layer.at["pointwise_conv"].at["weight"].set(w2)

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

    layer_jax = SeparableConv2D(
        in_features=2, out_features=1, kernel_size=3, depth_multiplier=2
    )

    layer_jax = layer_jax.at["depthwise_conv"].at["weight"].set(w1)
    layer_jax = layer_jax.at["pointwise_conv"].at["weight"].set(w2)

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

    layer = Conv1DLocal(
        in_features=2,
        out_features=1,
        kernel_size=3,
        strides=2,
        in_size=(28,),
        padding="valid",
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

    layer = Conv2DLocal(2, 1, (3, 2), in_size=(4, 4), padding="valid", strides=2)
    layer = layer.at["weight"].set(w)

    npt.assert_allclose(y, layer(x), atol=1e-5)


def test_in_feature_error():
    with pytest.raises(ValueError):
        Conv1D(0, 1, 2)

    with pytest.raises(ValueError):
        Conv2D(0, 1, 2)

    with pytest.raises(ValueError):
        Conv3D(0, 1, 2)

    with pytest.raises(ValueError):
        Conv1DLocal(0, 1, 2, in_size=(2,))

    with pytest.raises(ValueError):
        Conv2DLocal(0, 1, 2, in_size=(2, 2))

    with pytest.raises(ValueError):
        Conv1DTranspose(0, 1, 3)

    with pytest.raises(ValueError):
        Conv2DTranspose(0, 1, 3)

    with pytest.raises(ValueError):
        Conv3DTranspose(0, 1, 3)

    with pytest.raises(ValueError):
        DepthwiseConv1D(0, 1)

    with pytest.raises(ValueError):
        DepthwiseConv2D(0, 1)


def test_out_feature_error():
    with pytest.raises(ValueError):
        Conv1D(1, 0, 2)

    with pytest.raises(ValueError):
        Conv2D(1, 0, 2)

    with pytest.raises(ValueError):
        Conv3D(1, 0, 2)

    with pytest.raises(ValueError):
        Conv1DLocal(1, 0, 2, in_size=(2,))

    with pytest.raises(ValueError):
        Conv2DLocal(1, 0, 2, in_size=(2, 2))

    with pytest.raises(ValueError):
        Conv1DTranspose(1, 0, 3)

    with pytest.raises(ValueError):
        Conv2DTranspose(1, 0, 3)

    with pytest.raises(ValueError):
        Conv3DTranspose(1, 0, 3)


def test_groups_error():
    with pytest.raises(ValueError):
        Conv1D(1, 1, 2, groups=0)

    with pytest.raises(ValueError):
        Conv2D(1, 1, 2, groups=0)

    with pytest.raises(ValueError):
        Conv3D(1, 1, 2, groups=0)

    with pytest.raises(ValueError):
        Conv1DTranspose(1, 1, 3, groups=0)

    with pytest.raises(ValueError):
        Conv2DTranspose(1, 1, 3, groups=0)

    with pytest.raises(ValueError):
        Conv3DTranspose(1, 1, 3, groups=0)


# def test_lazy_conv():
#     layer = Conv1D(None, 1, 3)
#     assert layer(jnp.ones([10, 3])).shape == (1, 3)

#     layer = Conv2D(None, 1, 3)
#     assert layer(jnp.ones([10, 3, 3])).shape == (1, 3, 3)

#     layer = Conv3D(None, 1, 3)
#     assert layer(jnp.ones([10, 3, 3, 3])).shape == (1, 3, 3, 3)

#     layer = Conv1DTranspose(None, 1, 3)
#     assert layer(jnp.ones([10, 3])).shape == (1, 3)

#     layer = Conv2DTranspose(None, 1, 3)
#     assert layer(jnp.ones([10, 3, 3])).shape == (1, 3, 3)

#     layer = Conv3DTranspose(None, 1, 3)
#     assert layer(jnp.ones([10, 3, 3, 3])).shape == (1, 3, 3, 3)

#     layer = DepthwiseConv1D(None, 3)
#     assert layer(jnp.ones([10, 3])).shape == (10, 3)

#     layer = DepthwiseConv2D(None, 3)
#     assert layer(jnp.ones([10, 3, 3])).shape == (10, 3, 3)

#     layer = Conv1DLocal(None, 1, 3, in_size=(3,))
#     assert layer(jnp.ones([10, 3])).shape == (1, 3)

#     layer = Conv2DLocal(None, 1, 3, in_size=(3, 3))
#     assert layer(jnp.ones([10, 3, 3])).shape == (1, 3, 3)

#     layer = SeparableConv1D(None, 1, 3)
#     assert layer(jnp.ones([10, 3])).shape == (1, 3)

#     layer = SeparableConv2D(None, 1, 3)
#     assert layer(jnp.ones([10, 3, 3])).shape == (1, 3, 3)

#     with pytest.raises(ConcretizationTypeError):
#         jax.jit(Conv1D(None, 1, 3))(jnp.ones([10, 3]))

#     with pytest.raises(ConcretizationTypeError):
#         jax.jit(Conv2D(None, 1, 3))(jnp.ones([10, 3, 3]))

#     with pytest.raises(ConcretizationTypeError):
#         jax.jit(Conv3D(None, 1, 3))(jnp.ones([10, 3, 3, 3]))

#     with pytest.raises(ConcretizationTypeError):
#         jax.jit(Conv1DTranspose(None, 1, 3))(jnp.ones([10, 3]))

#     with pytest.raises(ConcretizationTypeError):
#         jax.jit(Conv2DTranspose(None, 1, 3))(jnp.ones([10, 3, 3]))

#     with pytest.raises(ConcretizationTypeError):
#         jax.jit(Conv3DTranspose(None, 1, 3))(jnp.ones([10, 3, 3, 3]))

#     with pytest.raises(ConcretizationTypeError):
#         jax.jit(DepthwiseConv1D(None, 3))(jnp.ones([10, 3]))

#     with pytest.raises(ConcretizationTypeError):
#         jax.jit(DepthwiseConv2D(None, 3))(jnp.ones([10, 3, 3]))

#     with pytest.raises(ConcretizationTypeError):
#         jax.jit(Conv1DLocal(None, 1, 3, in_size=(3,)))(jnp.ones([10, 3]))

#     with pytest.raises(ConcretizationTypeError):
#         jax.jit(Conv2DLocal(None, 1, 3, in_size=(3, 3)))(jnp.ones([10, 3, 3]))

#     with pytest.raises(ConcretizationTypeError):
#         jax.jit(SeparableConv1D(None, 1, 3))(jnp.ones([10, 3]))

#     with pytest.raises(ConcretizationTypeError):
#         jax.jit(SeparableConv2D(None, 1, 3))(jnp.ones([10, 3, 3]))
