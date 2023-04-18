from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from serket.nn.callbacks import init_func_cb, isinstance_factory, positive_int_cb
from serket.nn.utils import InitFuncType


@ft.lru_cache(maxsize=None)
def _multilinear_einsum_string(degree: int) -> str:
    # Generate einsum string for a linear layer of degree n
    # Example:
    #     >>> _multilinear_einsum_string(1)
    #     '...a,ab->....b'
    #     >>> _multilinear_einsum_string(2)
    #     '...a,...b,abc->....c'

    alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    if not (1 <= degree <= len(alpha) - 1):
        msg = f"degree must be between 1 and {len(alpha)-1}, got {degree}"
        raise ValueError(msg)

    xs_string = [f"...{i}" for i in alpha[:degree]]
    output_string = ",".join(xs_string)
    output_string += f",{alpha[:degree+1]}->...{alpha[degree]}"
    return output_string


@ft.lru_cache(maxsize=None)
def _general_linear_einsum_string(*axes: tuple[int, ...]) -> str:
    # Return the einsum string for a general linear layer.
    # Example:
    #     # apply linear layer to last axis
    #     >>> _general_linear_einsum_string(-1)
    #     '...0,01->...1'

    #     # apply linear layer to last two axes
    #     >>> _general_linear_einsum_string(-1,-2)
    #     '...01,012->...2'

    #     # apply linear layer to second last axis
    #     >>> _general_linear_einsum_string(-2)
    #     '...01,02->...12'

    #     # apply linear layer to last and third last axis
    #     >>> _general_linear_einsum_string(-1,-3)
    #     '...012,023->...13'

    if not all([i < 0 for i in axes]):
        raise ValueError("axes should be negative")

    axes = sorted(axes)
    total_axis = abs(min(axes))  # get the total number of axes
    alpha = "".join(map(str, range(total_axis + 1)))
    input_string = "..." + alpha[:total_axis]
    weight_string = "".join([input_string[axis] for axis in axes]) + alpha[total_axis]
    result_string = "".join([ai for ai in input_string if ai not in weight_string])
    result_string += alpha[total_axis]
    return f"{input_string},{weight_string}->{result_string}"


class Multilinear(pytc.TreeClass):
    weight: jax.Array
    bias: jax.Array

    in_features: tuple[int, ...] | None = pytc.field(callbacks=[isinstance_factory((int, tuple))])  # fmt: skip
    out_features: int

    def __init__(
        self,
        in_features: int | tuple[int, ...] | None,
        out_features: int,
        *,
        weight_init_func: InitFuncType = "he_normal",
        bias_init_func: InitFuncType = "ones",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """Linear layer with arbitrary number of inputs applied to last axis of each input

        Args:
            in_features: number of input features for each input
            out_features: number of output features
            weight_init_func: function to initialize the weights
            bias_init_func: function to initialize the bias
            key: key for the random number generator

        Example:
            >>> # Bilinear layer
            >>> layer = Multilinear((5,6), 7)
            >>> layer(jnp.ones((1,5)), jnp.ones((1,6))).shape
            (1, 7)

            >>> # Trilinear layer
            >>> layer = Multilinear((5,6,7), 8)
            >>> layer(jnp.ones((1,5)), jnp.ones((1,6)), jnp.ones((1,7))).shape
            (1, 8)
        """
        if not isinstance(in_features, (tuple, int)):
            msg = f"Expected tuple or int for in_features, got {type(in_features)}"
            raise ValueError(msg)

        self.in_features = in_features
        self.out_features = out_features

        self.weight_init_func = init_func_cb(weight_init_func)
        self.bias_init_func = init_func_cb(bias_init_func)

        weight_shape = (*self.in_features, out_features)
        self.weight = self.weight_init_func(key, weight_shape)

        if self.bias_init_func is None:
            self.bias = None
        else:
            self.bias = self.bias_init_func(key, (out_features,))

    def __call__(self, *x, **k) -> jax.Array:
        einsum_string = _multilinear_einsum_string(len(self.in_features))
        x = jnp.einsum(einsum_string, *x, self.weight)

        if self.bias is None:
            return x
        return x + self.bias


class Linear(Multilinear):
    """Linear layer with 1 input applied to last axis of input

    Args:
        in_features: number of input features
        out_features: number of output features
        weight_init_func: function to initialize the weights
        bias_init_func: function to initialize the bias
        key: key for the random number generator

    Example:
        >>> layer = Linear(5, 6)
        >>> layer(jnp.ones((1,5))).shape
        (1, 6)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        weight_init_func: InitFuncType = "he_normal",
        bias_init_func: InitFuncType = "ones",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        super().__init__(
            (in_features,),
            out_features,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=key,
        )


class Bilinear(Multilinear):
    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        *,
        weight_init_func: InitFuncType = "he_normal",
        bias_init_func: InitFuncType = "ones",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """Bilinear layer

        Args:
            in1_features: number of input features for the first input
            in2_features: number of input features for the second input
            out_features: number of output features
            weight_init_func: function to initialize the weights
            bias_init_func: function to initialize the bias
            key: key for the random number generator

        Example:
            >>> layer = Bilinear(5, 6, 7)
            >>> layer(jnp.ones((1,5)), jnp.ones((1,6))).shape
            (1, 7)
        """
        super().__init__(
            (in1_features, in2_features),
            out_features,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
            key=key,
        )


class GeneralLinear(pytc.TreeClass):
    weight: jax.Array
    bias: jax.Array

    in_features: tuple[int, ...] = pytc.field(callbacks=[isinstance_factory(tuple)])
    out_features: tuple[int, ...]
    in_axes: tuple[int, ...] = pytc.field(callbacks=[isinstance_factory(tuple)])

    def __init__(
        self,
        in_features: tuple[int, ...],
        out_features: int,
        *,
        in_axes: tuple[int, ...],
        weight_init_func: InitFuncType = "he_normal",
        bias_init_func: InitFuncType = "ones",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """Apply a Linear Layer to input at in_axes

        Args:
            in_features: number of input features corresponding to in_axes
            out_features: number of output features
            in_axes: axes to apply the linear layer to
            weight_init_func: weight initialization function
            bias_init_func: bias initialization function
            key: random key

        Example:
            >>> x = jnp.ones([1, 2, 3, 4])
            >>> layer = GeneralLinear(in_features=(1, 2), in_axes=(0, 1), out_features=5)
            >>> assert layer(x).shape == (3, 4, 5)

        Note:
            This layer is similar to to flax linen's DenseGeneral, the difference is that
            this layer uses einsum to apply the linear layer to the specified axes.
        """

        self.in_features = in_features
        self.out_features = out_features
        self.in_axes = in_axes

        if len(in_axes) != len(in_features):
            msg = "Expected in_axes and in_features to have the same length,"
            msg += f"got {len(in_axes)} and {len(in_features)}"
            raise ValueError(msg)

        self.weight_init_func = init_func_cb(weight_init_func)
        self.bias_init_func = init_func_cb(bias_init_func)
        self.weight = self.weight_init_func(key, (*self.in_features, self.out_features))

        if self.bias_init_func is None:
            self.bias = None
        else:
            self.bias = self.bias_init_func(key, (self.out_features,))

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        # ensure negative axes
        axes = map(lambda i: i if i < 0 else i - x.ndim, self.in_axes)
        einsum_string = _general_linear_einsum_string(*axes)
        x = jnp.einsum(einsum_string, x, self.weight)
        return x


class Identity(pytc.TreeClass):
    """Identity layer"""

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return x


class Embedding(pytc.TreeClass):
    in_features: int = pytc.field(callbacks=[positive_int_cb])
    out_features: int = pytc.field(callbacks=[positive_int_cb])
    weight: jax.Array

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """Defines an embedding layer.

        Args:
            in_features: vocabulary size.
            out_features: embedding size.
            key: random key to initialize the weights.

        Example:
            >>> import serket as sk
            >>> # 10 words in the vocabulary, each word is represented by a 3 dimensional vector
            >>> table = sk.nn.Embedding(10,3)
            >>> # take the last word in the vocab
            >>> table(jnp.array([9]))
            Array([[0.43810904, 0.35078037, 0.13254273]], dtype=float32)
        """
        self.in_features = in_features
        self.out_features = out_features
        self.weight = jr.uniform(key, (self.in_features, self.out_features))

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        """Embeds the input.

        Args:
            x: integer index array of subdtype integer.

        Returns:
            Embedding of the input.

        """
        if not jnp.issubdtype(x.dtype, jnp.integer):
            raise TypeError("Input must be an integer array.")

        return jnp.take(self.weight, x, axis=0)


class MergeLinear(pytc.TreeClass):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, *layers: tuple[Linear, ...]):
        """Merge multiple linear layers with the same `out_features`.

        Args:
            layers: linear layers to merge

        Example:
            >>> import serket as sk
            >>> import numpy.testing as npt
            >>> layer1 = sk.nn.Linear(5, 6)  # 5 input features, 6 output features
            >>> layer2 = sk.nn.Linear(7, 6)  # 7 input features, 6 output features
            >>> merged_layer = sk.nn.MergeLinear(layer1, layer2)  # 12 input features, 6 output features
            >>> x1 = jnp.ones([1, 5])  # 1 sample, 5 features
            >>> x2 = jnp.ones([1, 7])  # 1 sample, 7 features
            >>> y = merged_layer(x1, x2)  # one matrix multiplication
            >>> z = layer1(x1) + layer2(x2)  # two matrix multiplications
            >>> npt.assert_allclose(y, z, atol=1e-6)

        Note:
            Use this layer to reduce the matrix multiplication operations in the forward pass.
        """
        out_dim0 = layers[0].out_features

        for layer in layers[1:]:
            if layer.out_features != out_dim0:
                msg = "All layers must have the same output dimension."
                msg += f" Got {out_dim0} and {layer.out_features}"
                raise ValueError(msg)

        self.weight = jnp.concatenate([L.weight for L in layers], axis=0)
        self.bias = sum([L.bias for L in layers if L.bias_init_func])

    def __call__(self, *xs: tuple[jax.Array, ...], **k) -> jax.Array:
        xs = jnp.concatenate(xs, axis=-1)
        return xs @ self.weight + self.bias
