# from itertools import product

# import jax.numpy as jnp
# import numpy.testing as npt
# import pytest
# import tensorflow as tf
# import tensorflow.keras as tfk

# import serket as sk


# @pytest.mark.parametrize(
#     "tf_layer,sk_layer,ndim",
#     [
#         (tfk.layers.Conv1D, sk.nn.Conv1D, 1),
#         (tfk.layers.Conv2D, sk.nn.Conv2D, 2),
#         (tfk.layers.Conv3D, sk.nn.Conv3D, 3),
#     ],
# )
# def test_convnd_tf(tf_layer, sk_layer, ndim):
#     in_features = [5]
#     out_features = [2]
#     padding_choice = ["same", "valid"]
#     kernel_size = [1, 2, 3]
#     stride = [1, 2, 3]
#     dilation = [1]
#     groups = [1, 3]
#     spatial_dim = (10,) * ndim
#     batch_size = 1

#     for di, do, ki, si, ddi, gi, pi in product(
#         in_features,
#         out_features,
#         kernel_size,
#         stride,
#         dilation,
#         groups,
#         padding_choice,
#     ):
#         tf_conv = tf_layer(
#             do * gi,
#             kernel_size=ki,
#             strides=si,
#             padding=pi,
#             dilation_rate=ddi,
#             groups=gi,
#         )
#         tf_conv.build((batch_size, *spatial_dim, di * gi))

#         sk_conv = sk_layer(
#             di * gi,
#             do * gi,
#             kernel_size=ki,
#             strides=si,
#             padding=pi,
#             kernel_dilation=ddi,
#             groups=gi,
#         )

#         tf_bias = tf_conv.bias.numpy().reshape(-1, *(1,) * ndim)
#         tf_weight = tf_conv.weights[0].numpy()

#         # transform the weights
#         # (kernel_sizes, in_dim, out_dim) -> (out_dim, in_dim, kernel_sizes)
#         tf_weight = jnp.transpose(tf_weight, (-1, -2, *range(ndim)))

#         # set the weights to the serket layer
#         sk_conv = sk_conv.at["weight"].set(tf_weight)
#         sk_conv = sk_conv.at["bias"].set(tf_bias)

#         # create a random input
#         x_tf = tf.random.normal((batch_size, *spatial_dim, di * gi))

#         # convert the output to numpy channels first
#         y_tf = tf_conv(x_tf)[0].numpy()
#         y_tf = jnp.transpose(y_tf, (-1, *range(ndim)))

#         # remove the batch dimension
#         x_sk = x_tf.numpy()[0]
#         # convert the input to channels first input
#         x_sk = jnp.transpose(x_sk, (-1, *range(ndim)))
#         y_sk = sk_conv(x_sk)

#         npt.assert_allclose(y_tf, y_sk, atol=5e-6)


# @pytest.mark.parametrize(
#     "tf_layer,sk_layer,ndim",
#     [
#         (tfk.layers.Conv1DTranspose, sk.nn.Conv1DTranspose, 1),
#         (tfk.layers.Conv2DTranspose, sk.nn.Conv2DTranspose, 2),
#         (tfk.layers.Conv3DTranspose, sk.nn.Conv3DTranspose, 3),
#     ],
# )
# def test_convnd_transpose_tf(tf_layer, sk_layer, ndim):
#     in_features = [1, 2, 3]
#     out_features = [2, 3, 4]
#     padding_choice = ["valid"]
#     kernel_size = [1, 2, 3]
#     stride = [1]
#     dilation = [1]
#     groups = [1]
#     spatial_dim = (4,) * ndim
#     batch_size = 1

#     for di, do, ki, si, ddi, gi, pi in product(
#         in_features,
#         out_features,
#         kernel_size,
#         stride,
#         dilation,
#         groups,
#         padding_choice,
#     ):
#         tf_conv = tf_layer(
#             do * gi,
#             kernel_size=ki,
#             strides=si,
#             padding=pi,
#             dilation_rate=ddi,
#             groups=gi,
#             use_bias=False,
#         )
#         tf_conv.build((batch_size, *spatial_dim, di * gi))

#         sk_conv = sk_layer(
#             di * gi,
#             do * gi,
#             kernel_size=ki,
#             strides=si,
#             padding=pi,
#             kernel_dilation=ddi,
#             groups=gi,
#             bias_init_func=None,
#         )

#         tf_weight = tf_conv.weights[0].numpy()

#         # (*K, O, I) -> (O, I, *K)
#         tf_weight = jnp.transpose(tf_weight, (-2, -1, *range(ndim)))

#         # set the weights to the serket layer
#         sk_conv = sk_conv.at["weight"].set(tf_weight)

#         # create a random input
#         x_tf = tf.ones((*spatial_dim, di * gi))
#         x_tf = tf.expand_dims(x_tf, axis=0)

#         # convert the output to numpy channels first
#         y_tf = tf_conv(x_tf)[0].numpy()
#         y_tf = jnp.moveaxis(y_tf, -1, 0)
#         # flip along the spatial dimensions
#         y_tf = jnp.flip(y_tf, axis=range(1, ndim + 1))

#         x_sk = jnp.ones((di * gi, *spatial_dim))
#         # convert the input to channels first input
#         y_sk = sk_conv(x_sk)
#         npt.assert_allclose(y_tf, y_sk, atol=5e-6)


# @pytest.mark.parametrize(
#     "tf_layer,sk_layer,ndim",
#     [
#         (tfk.layers.DepthwiseConv1D, sk.nn.DepthwiseConv1D, 1),
#         (tfk.layers.DepthwiseConv2D, sk.nn.DepthwiseConv2D, 2),
#     ],
# )
# def test_depthwise_convnd_tf(tf_layer, sk_layer, ndim):
#     in_features = [5]
#     padding_choice = ["same", "valid"]
#     kernel_size = [1, 2, 3]
#     stride = [1, 2, 3]
#     spatial_dim = (10,) * ndim
#     batch_size = 1

#     for di, ki, si, pi in product(
#         in_features,
#         kernel_size,
#         stride,
#         padding_choice,
#     ):
#         tf_conv = tf_layer(
#             kernel_size=ki,
#             strides=si,
#             padding=pi,
#         )
#         tf_conv.build((batch_size, *spatial_dim, di))

#         sk_conv = sk_layer(
#             di,
#             kernel_size=ki,
#             strides=si,
#             padding=pi,
#         )

#         tf_bias = tf_conv.bias.numpy().reshape(-1, *(1,) * ndim)
#         tf_weight = tf_conv.weights[0].numpy()

#         # transform the weights
#         # (kernel_sizes, in_dim, out_dim) -> (out_dim, in_dim, kernel_sizes)
#         tf_weight = jnp.transpose(tf_weight, (-2, -1, *range(ndim)))

#         # set the weights to the serket layer
#         sk_conv = sk_conv.at["weight"].set(tf_weight)
#         sk_conv = sk_conv.at["bias"].set(tf_bias)

#         # create a random input
#         x_tf = tf.random.normal((batch_size, *spatial_dim, di))

#         # convert the output to numpy channels first
#         y_tf = tf_conv(x_tf)[0].numpy()
#         y_tf = jnp.transpose(y_tf, (-1, *range(ndim)))

#         # remove the batch dimension
#         x_sk = x_tf.numpy()[0]
#         # convert the input to channels first input
#         x_sk = jnp.transpose(x_sk, (-1, *range(ndim)))
#         y_sk = sk_conv(x_sk)

#         npt.assert_allclose(y_tf, y_sk, atol=5e-6)
