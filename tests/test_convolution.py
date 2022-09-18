import jax.numpy as jnp
import numpy.testing as npt
import pytest

from serket.nn.convolution import (
    Conv1D,
    Conv2D,
    Conv3D,
    _check_and_return_kernel,
    _check_and_return_padding,
    _check_and_return_rate,
    _check_and_return_stride,
)


def test_check_and_return():

    assert _check_and_return_kernel(3, 2) == (3, 3)
    assert _check_and_return_kernel((3, 3), 2) == (3, 3)
    assert _check_and_return_kernel((3, 3, 3), 3) == (3, 3, 3)

    with pytest.raises(AssertionError):
        _check_and_return_kernel((3, 3), 3)

    with pytest.raises(AssertionError):
        _check_and_return_kernel((3, 3, 3), 2)

    with pytest.raises(AssertionError):
        _check_and_return_kernel((3, 3, 3), 1)

    assert _check_and_return_rate(3, 2) == (3, 3)
    assert _check_and_return_rate((3, 3), 2) == (3, 3)
    assert _check_and_return_rate((3, 3, 3), 3) == (3, 3, 3)

    assert _check_and_return_stride(3, 2) == (3, 3)
    assert _check_and_return_stride((3, 3), 2) == (3, 3)
    assert _check_and_return_stride((3, 3, 3), 3) == (3, 3, 3)

    assert _check_and_return_padding(((1, 1), (1, 1)), 2) == ((1, 1), (1, 1))
    assert _check_and_return_padding(((1, 1), (1, 1), (1, 1)), 3) == (
        (1, 1),
        (1, 1),
        (1, 1),
    )


def test_conv1D():

    layer = Conv1D(
        in_features=1, out_features=1, kernel_size=2, padding="SAME", strides=1
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

    layer = Conv2D(
        in_features=1, out_features=2, kernel_size=2, padding="SAME", strides=2
    )
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

    layer = Conv2D(
        in_features=1, out_features=2, kernel_size=2, padding="SAME", strides=1
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

    layer = Conv3D(1, 3, 3)
    layer = layer.at["weight"].set(jnp.ones([3, 1, 3, 3, 3]))
    layer = layer.at["bias"].set(jnp.zeros([3, 1, 1, 1]))
    npt.assert_allclose(
        layer(jnp.ones([1, 1, 3, 3])),
        jnp.tile(jnp.array([[4, 6, 4], [6, 9, 6], [4, 6, 4]]), [3, 1, 1, 1]),
    )
