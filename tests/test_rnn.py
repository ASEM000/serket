# testing against keras
# import tensorflow.keras as tfk
# import tensorflow as tf
# import numpy as np
# from serket.nn.recurrent import  LSTMCell, ScanRNN

# batch_size = 1
# time_steps = 2
# in_features = 3
# hidden_features=2

# inputs = np.ones([batch_size,time_steps, in_features]).astype(np.float32)
# inp = tf.keras.Input(shape=(time_steps, in_features))
# rnn = (tf.keras.layers.LSTM(hidden_features, return_sequences=True, return_state=False))(inp)
# rnn = tf.keras.Model(inputs=inp, outputs=rnn)
# # rnn(inputs)
# w_in_to_hidden = jnp.array(rnn.weights[0].numpy())
# w_hidden_to_hidden = jnp.array(rnn.weights[1].numpy())
# b_hidden_to_hidden = jnp.array(rnn.weights[2].numpy())
# x = jnp.ones([time_steps, in_features])
# cell = LSTMCell(in_features, hidden_features, recurrent_weight_init_func="glorot_uniform", bias_init_func="zeros",
#  weight_init_func="glorot_uniform")
# cell = cell.at["in_to_hidden.weight"].set(w_in_to_hidden)
# cell = cell.at["hidden_to_hidden.weight"].set(w_hidden_to_hidden)
# cell = cell.at["hidden_to_hidden.bias"].set(b_hidden_to_hidden)
# ScanRNN(cell, return_sequences=True)(x) ,rnn(inputs)

# testing with keras
# inputs = np.ones([batch_size,time_steps, in_features]).astype(np.float32)
# inp = tf.keras.Input(shape=(time_steps, in_features))
# rnn = tfk.layers.Bidirectional(tf.keras.layers.LSTM(hidden_features, return_sequences=False))(inp)
# rnn = tf.keras.Model(inputs=inp, outputs=rnn)
# # rnn(inputs)
# w_in_to_hidden = jnp.array(rnn.weights[0].numpy())
# w_hidden_to_hidden = jnp.array(rnn.weights[1].numpy())
# b_hidden_to_hidden = jnp.array(rnn.weights[2].numpy())
# x = jnp.ones([time_steps, in_features])
# cell = LSTMCell(in_features, hidden_features)
# cell = cell.at["in_to_hidden.weight"].set(w_in_to_hidden)
# cell = cell.at["hidden_to_hidden.weight"].set(w_hidden_to_hidden)
# cell = cell.at["hidden_to_hidden.bias"].set(b_hidden_to_hidden)

# w_in_to_hidden_reverse = jnp.array(rnn.weights[3].numpy())
# w_hidden_to_hidden_reverse = jnp.array(rnn.weights[4].numpy())
# b_hidden_to_hidden_reverse = jnp.array(rnn.weights[5].numpy())
# reverse_cell = LSTMCell(in_features, hidden_features)

# reverse_cell = reverse_cell.at["in_to_hidden.weight"].set(w_in_to_hidden_reverse)
# reverse_cell = reverse_cell.at["hidden_to_hidden.weight"].set(w_hidden_to_hidden_reverse)
# reverse_cell = reverse_cell.at["hidden_to_hidden.bias"].set(b_hidden_to_hidden_reverse)


import jax.numpy as jnp
import numpy.testing as npt
import pytest

from serket.nn.recurrent import (  # ConvGRU1DCell,; ConvGRU2DCell,; ConvGRU3DCell,; ConvLSTM2DCell,; ConvLSTM3DCell,
    ConvLSTM1DCell,
    GRUCell,
    LSTMCell,
    ScanRNN,
    SimpleRNNCell,
)

# import pytest


def test_vanilla_rnn():
    in_features = 2
    hidden_features = 3
    # batch_size = 1
    time_steps = 10

    # test against keras
    # copy weights from keras to serket and compare outputs
    # inputs = np.ones([batch_size,time_steps, in_features]).astype(np.float32)
    # inp = tf.keras.Input(shape=(time_steps, in_features))
    # rnn = (tf.keras.layers.SimpleRNN(hidden_features, return_sequences=False, return_state=False))(inp)
    # rnn = tf.keras.Model(inputs=inp, outputs=rnn)

    x = jnp.ones([time_steps, in_features]).astype(jnp.float32)

    w_in_to_hidden = jnp.array(
        [[0.6252413, -0.34832734, 0.6286191], [0.84620893, 0.52448165, 0.13104844]]
    )

    w_hidden_to_hidden = jnp.array(
        [
            [-0.24631214, -0.86077654, -0.44541454],
            [-0.96763766, 0.24441445, 0.06276101],
            [-0.05484254, -0.4464587, 0.893122],
        ]
    )

    cell = SimpleRNNCell(
        in_features=in_features,
        hidden_features=hidden_features,
        recurrent_weight_init_func="glorot_uniform",
    )

    w_combined = jnp.concatenate([w_in_to_hidden, w_hidden_to_hidden], axis=0)
    cell = cell.at["in_and_hidden_to_hidden"].at["weight"].set(w_combined)
    sk_layer = ScanRNN(cell)
    y = jnp.array([0.9637042, -0.8282256, 0.7314449])
    npt.assert_allclose(sk_layer(x), y)


def test_lstm():
    # tensorflow
    in_features = 2
    hidden_features = 3
    # batch_size = 1
    time_steps = 10

    # inputs = np.ones([batch_size,time_steps, in_features]).astype(np.float32)
    # inp = tf.keras.Input(shape=(time_steps, in_features))
    # rnn = (tf.keras.layers.LSTM(hidden_features, return_sequences=False, return_state=False))(inp)
    # rnn = tf.keras.Model(inputs=inp, outputs=rnn)

    # w_in_to_hidden = jnp.array(rnn.weights[0].numpy())
    # w_hidden_to_hidden = jnp.array(rnn.weights[1].numpy())
    # b_hidden_to_hidden = jnp.array(rnn.weights[2].numpy())

    x = jnp.ones([time_steps, in_features]).astype(jnp.float32)

    w_in_to_hidden = jnp.array(
        [
            [
                -0.1619612,
                -0.17861447,
                -0.374527,
                0.21063584,
                0.1806348,
                0.0344786,
                0.44189203,
                -0.55044144,
                0.28518462,
                -0.09390897,
                0.56036115,
                0.19108337,
            ],
            [
                0.03269911,
                -0.21127799,
                0.55661833,
                -0.6470987,
                -0.27472985,
                -0.21884575,
                0.2479667,
                -0.34201348,
                0.00261247,
                -0.6468279,
                0.5003185,
                0.6460693,
            ],
        ]
    )

    w_hidden_to_hidden = jnp.array(
        [
            [
                0.3196982,
                0.25284654,
                -0.18152222,
                0.44958767,
                -0.44068673,
                -0.19395973,
                -0.00905689,
                -0.17610262,
                0.21773854,
                -0.47118214,
                -0.07700437,
                0.24598895,
            ],
            [
                -0.23678103,
                -0.01854092,
                -0.15681103,
                -0.20309119,
                -0.51169145,
                0.33006623,
                0.35155487,
                0.1802753,
                -0.08975402,
                -0.30867696,
                0.37548447,
                -0.3264465,
            ],
            [
                -0.14270899,
                0.26242012,
                -0.31327525,
                0.206014,
                0.5501963,
                0.14983827,
                -0.15515868,
                0.2578809,
                -0.14565073,
                -0.33286166,
                0.4204296,
                0.21370588,
            ],
        ]
    )

    b_hidden_to_hidden = jnp.array(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    cell = LSTMCell(
        in_features=in_features,
        hidden_features=hidden_features,
        recurrent_weight_init_func="glorot_uniform",
    )
    w_combined = jnp.concatenate([w_in_to_hidden, w_hidden_to_hidden], axis=0)
    cell = cell.at["in_and_hidden_to_hidden"].at["weight"].set(w_combined)
    cell = cell.at["in_and_hidden_to_hidden"].at["bias"].set(b_hidden_to_hidden)

    sk_layer = ScanRNN(cell, return_sequences=False)
    y = jnp.array([0.18658024, -0.6338659, 0.3445018])
    npt.assert_allclose(y, sk_layer(x), atol=1e-5)

    w_in_to_hidden = jnp.array(
        [
            [
                0.11943924,
                -0.609248,
                -0.45503575,
                -0.3439762,
                -0.33675978,
                0.05291432,
                -0.12904513,
                -0.22977036,
                0.32492596,
                0.06835997,
                0.0484916,
                0.07520777,
            ],
            [
                0.39872873,
                -0.08020723,
                -0.4879259,
                -0.61926323,
                -0.45951623,
                -0.44556192,
                -0.05298251,
                0.54848397,
                0.19754452,
                0.6012858,
                -0.06859863,
                0.16502213,
            ],
        ]
    )

    w_hidden_to_hidden = jnp.array(
        [
            [
                0.18880641,
                0.21262297,
                -0.2961502,
                0.33976135,
                -0.09891935,
                -0.00502901,
                0.34378093,
                0.4202192,
                0.36584634,
                0.08396737,
                -0.4975226,
                0.15165171,
            ],
            [
                0.30486387,
                -0.46795598,
                -0.07052832,
                0.51685417,
                -0.23734125,
                0.1711132,
                0.16389124,
                -0.08915165,
                -0.02928232,
                -0.2173849,
                0.19655496,
                -0.45694238,
            ],
            [
                -0.1722902,
                -0.23029403,
                0.05032581,
                0.21182823,
                0.5298174,
                -0.50670344,
                -0.18930247,
                0.30799994,
                -0.18611868,
                -0.08317372,
                -0.26286182,
                -0.30177474,
            ],
        ]
    )

    b_hidden_to_hidden = jnp.array(
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    cell = LSTMCell(
        in_features=in_features,
        hidden_features=hidden_features,
        recurrent_weight_init_func="glorot_uniform",
    )

    w_combined = jnp.concatenate([w_in_to_hidden, w_hidden_to_hidden], axis=0)

    cell = cell.at["in_and_hidden_to_hidden"].at["weight"].set(w_combined)
    cell = cell.at["in_and_hidden_to_hidden"].at["bias"].set(b_hidden_to_hidden)

    sk_layer = ScanRNN(cell, return_sequences=True)

    y = jnp.array(
        [
            [-0.07431775, 0.05081949, 0.07480226],
            [-0.12263095, 0.07622699, 0.1146026],
            [-0.15380122, 0.0886446, 0.13589925],
            [-0.17376699, 0.0944333, 0.14736715],
            [-0.18647897, 0.09689739, 0.1535385],
            [-0.19453025, 0.09775667, 0.15683244],
            [-0.19960524, 0.09789205, 0.1585632],
            [-0.20278986, 0.0977404, 0.15945096],
            [-0.20477988, 0.09750732, 0.15989034],
            [-0.20601842, 0.09728104, 0.16009602],
        ]
    )

    npt.assert_allclose(y, sk_layer(x), atol=1e-5)

    cell = LSTMCell(
        in_features=in_features,
        hidden_features=hidden_features,
        recurrent_weight_init_func="glorot_uniform",
    )

    sk_layer = ScanRNN(cell, return_sequences=True)
    assert sk_layer(x).shape == (10, 3)


def test_gru():
    w1 = jnp.array(
        [
            [
                -0.04667467,
                0.25340378,
                0.26873875,
                0.15961742,
                0.56519365,
                0.46263158,
                -0.0030899,
                0.31380886,
                0.44481528,
            ]
        ]
    )

    w2 = jnp.array(
        [
            [
                0.23404205,
                0.10193896,
                0.27892762,
                -0.488236,
                -0.4173184,
                -0.0588184,
                0.41350085,
                0.36151117,
                -0.45407838,
            ],
            [
                -0.560196,
                -0.22648495,
                -0.12656957,
                0.31881046,
                0.47110367,
                0.30805635,
                0.41259462,
                0.40002275,
                -0.0368616,
            ],
            [
                0.5745573,
                0.4343021,
                0.42046744,
                -0.09401041,
                0.5539224,
                -0.13675115,
                -0.5197817,
                -0.21241805,
                -0.16732433,
            ],
        ]
    )

    cell = GRUCell(1, 3, bias_init_func=None)
    cell = cell.at["in_to_hidden"].at["weight"].set(w1)
    cell = cell.at["hidden_to_hidden"].at["weight"].set(w2)
    y = jnp.array([[-0.00142191, 0.11011646, 0.1613554]])
    ypred = ScanRNN(cell, return_sequences=True)(jnp.ones([1, 1]))
    npt.assert_allclose(y, ypred, atol=1e-4)


def test_conv_lstm1d():
    w_in_to_hidden = jnp.array(
        [
            [
                [0.3159187, -0.37110862, 0.23497376],
                [0.06916022, 0.16520068, -0.1498835],
            ],
            [
                [0.13892826, -0.2475906, 0.11548725],
                [-0.14935534, 0.0077568, 0.31523505],
            ],
            [
                [0.20523027, 0.333159, -0.26372582],
                [0.21769527, -0.28275424, 0.07145688],
            ],
            [
                [-0.32436138, 0.17985162, -0.05102682],
                [-0.33781663, 0.07652837, 0.14034107],
            ],
            [
                [-0.2476197, 0.27073297, -0.15494357],
                [-0.17142114, 0.0436784, -0.2635818],
            ],
            [
                [-0.1563589, -0.30193892, -0.3076105],
                [0.30359367, -0.37472126, 0.08727607],
            ],
            [
                [0.02532503, -0.33569914, -0.16816947],
                [-0.28197324, -0.20834318, -0.31490648],
            ],
            [
                [0.37559494, -0.10307714, -0.28350165],
                [0.16282192, 0.25434867, 0.14521858],
            ],
            [
                [-0.3619054, -0.05932748, 0.13838741],
                [0.317831, -0.01710135, 0.01839554],
            ],
            [
                [-0.33236656, -0.15234765, 0.23833898],
                [-0.0525074, -0.1169591, 0.22625437],
            ],
            [
                [0.3350378, 0.3527101, -0.08017969],
                [-0.25890553, 0.24611798, 0.30005935],
            ],
            [
                [-0.07834777, -0.02483597, -0.28757787],
                [-0.15855587, 0.14020738, -0.3187018],
            ],
        ]
    )

    w_hidden_to_hidden = jnp.array(
        [
            [
                [0.44095814, 0.12996325, 0.1313585],
                [0.18582591, 0.07248487, -0.7859758],
                [-0.17839126, 0.15680492, -0.08622836],
            ],
            [
                [-0.11601712, 0.00761805, 0.43996823],
                [0.27362385, 0.0799137, 0.2580722],
                [-0.563254, 0.19736156, 0.26167846],
            ],
            [
                [-0.28901652, -0.25223732, -0.10025343],
                [0.56027263, -0.28712046, -0.18524358],
                [0.37074035, 0.3996833, 0.1725195],
            ],
            [
                [0.07441625, 0.20128009, 0.30421543],
                [-0.06981394, -0.17527759, 0.22605616],
                [0.11372325, 0.63972735, -0.19949353],
            ],
            [
                [0.08129799, -0.06646754, -0.44094074],
                [-0.09799376, 0.16513337, 0.1980969],
                [-0.01823295, 0.33500522, 0.19564764],
            ],
            [
                [-0.4375121, -0.07695349, 0.27423194],
                [0.25537497, 0.64107186, -0.09421141],
                [0.21401826, -0.15687335, -0.07473418],
            ],
            [
                [-0.37147775, 0.06210529, -0.04531584],
                [-0.38045418, 0.26204777, -0.17553791],
                [-0.16380772, 0.39306286, -0.444068],
            ],
            [
                [-0.08250815, 0.5762788, 0.3014125],
                [0.08091379, -0.20550683, 0.06467859],
                [0.02479128, -0.16484486, 0.09149422],
            ],
            [
                [-0.1793791, 0.23342696, -0.33710676],
                [0.4355502, -0.23507121, 0.11481185],
                [-0.21538775, -0.16292992, -0.6203824],
            ],
            [
                [-0.1719443, 0.04258863, -0.35778967],
                [0.12353352, 0.0826712, -0.10358769],
                [-0.55321497, 0.07205058, 0.29797262],
            ],
            [
                [-0.52755165, 0.27079415, -0.04477403],
                [-0.3376618, -0.32239383, -0.3393156],
                [0.04485175, -0.04528336, 0.30485243],
            ],
            [
                [-0.14193594, -0.634814, 0.28351584],
                [-0.16348608, -0.4000306, -0.08978741],
                [-0.26926947, -0.12314601, -0.19621553],
            ],
        ]
    )

    b_hidden_to_hidden = jnp.array(
        [
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [1.0],
            [1.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
        ]
    )

    in_features = 2
    hidden_features = 3
    time_steps = 1
    spatial_dim = (3,)

    # inputs = np.ones([batch_size,time_steps, in_features,*spatial_dim]).astype(np.float32)
    # inp = tf.keras.Input(shape=(time_steps, in_features,*spatial_dim))
    # rnn = (tf.keras.layers.ConvLSTM1D(hidden_features,recurrent_activation="sigmoid", kernel_size=3, padding='same',
    # return_sequences=False,data_format='channels_first'))(inp)
    # rnn = tf.keras.Model(inputs=inp, outputs=rnn)

    cell = ConvLSTM1DCell(
        in_features=in_features,
        hidden_features=hidden_features,
        recurrent_act_func="sigmoid",
        kernel_size=3,
        padding="same",
        weight_init_func="glorot_uniform",
        recurrent_weight_init_func="glorot_uniform",
        bias_init_func="zeros",
    )

    cell = cell.at["in_to_hidden"].at["weight"].set(w_in_to_hidden)
    cell = cell.at["hidden_to_hidden"].at["weight"].set(w_hidden_to_hidden)
    cell = cell.at["hidden_to_hidden"].at["bias"].set(b_hidden_to_hidden)

    x = jnp.ones([time_steps, in_features, *spatial_dim])

    res_sk = ScanRNN(cell, return_sequences=False)(x)

    y = jnp.array(
        [
            [-0.19088623, -0.20386685, -0.11864982],
            [0.00493522, 0.18935747, 0.16954307],
            [0.01413723, 0.00672858, -0.03464129],
        ]
    )

    assert jnp.allclose(res_sk, y, atol=1e-5)

    cell = ConvLSTM1DCell(
        in_features=in_features,
        hidden_features=hidden_features,
        recurrent_act_func="sigmoid",
        kernel_size=3,
        padding="same",
        weight_init_func="glorot_uniform",
        recurrent_weight_init_func="glorot_uniform",
        bias_init_func="zeros",
    )

    res_sk = ScanRNN(cell, return_sequences=False)(x)
    assert res_sk.shape == (3, 3)


def test_bilstm():
    # batch_size = 1
    time_steps = 2
    in_features = 3
    hidden_features = 2

    x = jnp.ones([time_steps, in_features])
    cell = LSTMCell(in_features, hidden_features)
    reverse_cell = LSTMCell(in_features, hidden_features)

    w_in_to_hidden = jnp.array(
        [
            [
                -0.6061297,
                0.6038931,
                0.0219295,
                -0.53232527,
                0.63680524,
                -0.1877076,
                0.5494583,
                0.5319734,
            ],
            [
                -0.11174804,
                0.1967476,
                -0.01281184,
                0.6291546,
                -0.10848027,
                -0.32045278,
                0.07772851,
                -0.07741755,
            ],
            [
                0.69948727,
                -0.48679155,
                0.39291233,
                -0.0054667,
                0.5324392,
                0.62987834,
                -0.2530458,
                -0.5623743,
            ],
        ]
    )

    w_hidden_to_hidden = jnp.array(
        [
            [
                -0.07784259,
                0.5912869,
                -0.08792564,
                -0.07326522,
                -0.07806911,
                -0.75162244,
                0.01986005,
                0.24453232,
            ],
            [
                0.23444527,
                -0.5768899,
                0.24225983,
                -0.23526284,
                -0.2299888,
                -0.444415,
                0.4977502,
                0.00633401,
            ],
        ]
    )

    b_hidden_to_hidden = jnp.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    w_in_to_hidden_reverse = jnp.array(
        [
            [
                0.28273338,
                -0.1472258,
                0.3937468,
                0.34040576,
                -0.299861,
                -0.38785607,
                0.00533426,
                0.06143087,
            ],
            [
                -0.40093276,
                0.39314228,
                -0.43308863,
                0.532469,
                -0.71949875,
                0.16529655,
                -0.07926816,
                -0.5383911,
            ],
            [
                -0.0023067,
                -0.5820745,
                0.31508905,
                0.29104167,
                -0.35113502,
                -0.6884494,
                0.14833266,
                -0.46562153,
            ],
        ]
    )

    w_hidden_to_hidden_reverse = jnp.array(
        [
            [
                3.12127233e-01,
                7.36315727e-01,
                -1.91057637e-01,
                1.89247921e-01,
                4.54114564e-02,
                6.95739524e-04,
                5.34631252e-01,
                1.43038025e-02,
            ],
            [
                3.68674904e-01,
                -1.35606900e-01,
                -3.05835426e-01,
                -1.86572984e-01,
                -7.80997992e-01,
                2.84251571e-02,
                -1.41527206e-02,
                3.26157391e-01,
            ],
        ]
    )

    b_hidden_to_hidden_reverse = jnp.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    combined_w = jnp.concatenate([w_in_to_hidden, w_hidden_to_hidden], axis=0)
    cell = cell.at["in_and_hidden_to_hidden"].at["weight"].set(combined_w)
    cell = cell.at["in_and_hidden_to_hidden"].at["bias"].set(b_hidden_to_hidden)

    combined_w_reverse = jnp.concatenate(
        [w_in_to_hidden_reverse, w_hidden_to_hidden_reverse], axis=0
    )
    reverse_cell = (
        reverse_cell.at["in_and_hidden_to_hidden"].at["weight"].set(combined_w_reverse)
    )
    reverse_cell = (
        reverse_cell.at["in_and_hidden_to_hidden"]
        .at["bias"]
        .set(b_hidden_to_hidden_reverse)
    )

    res = ScanRNN(cell, backward_cell=reverse_cell, return_sequences=False)(x)

    y = jnp.array([0.35901642, 0.00826644, -0.3015435, -0.13661332])

    npt.assert_allclose(res, y, atol=1e-5)


def test_rnn_error():
    with pytest.raises(TypeError):
        ScanRNN(None)

    with pytest.raises(TypeError):
        ScanRNN(SimpleRNNCell(3, 3), 1)

    layer = ScanRNN(SimpleRNNCell(3, 3), SimpleRNNCell(3, 3))
    with pytest.raises(TypeError):
        layer(jnp.ones([10, 3]), 1.0)

    with pytest.raises(ValueError):
        layer(jnp.ones([10, 3, 3]))
