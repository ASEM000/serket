from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

from serket.nn.linear import Linear
from serket.nn.utils import ActivationType, InitFuncType, resolve_activation


class FNN(pytc.TreeClass):
    def __init__(
        self,
        layers: Sequence[int],
        *,
        act_func: ActivationType = jax.nn.relu,
        weight_init_func: InitFuncType = "he_normal",
        bias_init_func: InitFuncType = "ones",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        """Fully connected neural network
        Args:
            layers: Sequence of layer sizes
            act_func: Activation function to use. Defaults to jax.nn.relu.
            weight_init_func: Weight initializer function. Defaults to jax.nn.initializers.he_normal().
            bias_init_func: Bias initializer function. Defaults to lambda key, shape: jnp.ones(shape).
            key: Random key for weight and bias initialization. Defaults to jr.PRNGKey(0).

        Example:
            >>> fnn = FNN([10, 5, 2])
            >>> fnn(jnp.ones((3, 10))).shape
            (3, 2)
        """

        keys = jr.split(key, len(layers) - 1)
        self.act_func = resolve_activation(act_func)

        self.layers = tuple(
            Linear(
                in_features=in_dim,
                out_features=out_dim,
                key=ki,
                weight_init_func=weight_init_func,
                bias_init_func=bias_init_func,
            )
            for (ki, in_dim, out_dim) in (zip(keys, layers[:-1], layers[1:]))
        )

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act_func(x)
        return self.layers[-1](x)


class PFNN(pytc.TreeClass):
    """Parallel fully connected neural network with subnetworks for each output.

    Example:
        >>> nn = sk.nn.PFNN([1, 2, [4, 5], 2])
        >>> #         |---> 4 -> 1
        >>> # 1 -> 2 -|
        >>> #         |---> 5 -> 1

    Note:
        https://github.com/lululxvi/deepxde/blob/master/deepxde/nn/pytorch/fnn.py
    """

    def __init__(
        self,
        layers: Sequence[int],
        act_func: ActivationType = "relu",
        weight_init_func: InitFuncType = "glorot_uniform",
        bias_init_func: InitFuncType = "zeros",
        key: jr.KeyArray = jr.PRNGKey(0),
    ):
        if not isinstance(layers, (tuple, list)):
            raise TypeError(f"layers must be a tuple or list, got {type(layers)}")

        if not isinstance(layers[-1], int) or not isinstance(layers[0], int):
            msg = "First and last layers must be integers, specifying input and output dimensions."
            raise TypeError(msg)

        split_index = None

        # first pass to check if layers are valid
        for i, layer in enumerate(layers):
            if isinstance(layer, int):
                if i < len(layers) - 1 and split_index is not None:
                    # prevent joining of paths except at the last layer
                    # ex: [1,[2,3],"4",5] is not allowed because 4 is not the last layer
                    msg = f"Cannot join paths at layer {i} after splitting at layer {i-1}."
                    msg += "Joining paths is only allowed at the last layer."
                    raise ValueError(msg)

            elif isinstance(layer, (tuple, list)):
                if not all(isinstance(item, int) for item in layer):
                    msg = f"All layers in {layer} must be integers."
                    raise TypeError(msg)

                if len(layer) != layers[-1]:
                    # ex: [1, "[2,3]", 3] is not allowed becuase the split path must have
                    # the same output size
                    msg = f"Length of {layer} must match output size {layers[-1]}."
                    raise ValueError(msg)

                if split_index is None:
                    # found the first split index
                    split_index = i - 1

            else:
                msg = f"Layer {i} must be an integer or a list/tuple of integers."
                raise TypeError(msg)

        if split_index is None:
            msg = "No split index found. Cannot create unshared layers."
            msg += "Use FNN instead to create a fully connected shared network."
            raise ValueError(msg)

        # key for shared layers and a key
        # for each path in the unshared layers if any
        shared_key, *unshared_keys = jr.split(key, layers[-1] + 1)

        if split_index > 0:
            self.shared_layers = FNN(
                layers[: split_index + 1],
                act_func=act_func,
                weight_init_func=weight_init_func,
                bias_init_func=bias_init_func,
                key=shared_key,
            )
        else:
            # ex: [1, [2,3], 2]
            # if split_index == 0, then there are no shared layers
            self.shared_layers = None

        in_dim = layers[split_index]
        unshared_spec = list(zip(*layers[split_index + 1 : -1]))
        self.unshared_layers = []

        for i in range(layers[-1]):
            # for each output add a subnetwork of fully connected layers
            # ex: [1, [2,3], 2] will create two subnetworks
            # one with layers [2,2] and another with layers [3,2]
            self.unshared_layers += [
                FNN(
                    [in_dim, *unshared_spec[i], 1],
                    act_func=act_func,
                    weight_init_func=weight_init_func,
                    bias_init_func=bias_init_func,
                    key=unshared_keys[i],
                )
            ]
        self.unshared_layers = tuple(self.unshared_layers)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        if self.shared_layers:
            x = self.shared_layers(x)
        x = [layer(x) for layer in self.unshared_layers]
        return jnp.concatenate(x, axis=-1)
