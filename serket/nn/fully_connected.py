from __future__ import annotations

import functools as ft
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytreeclass as pytc

from serket.nn.linear import Linear


@pytc.treeclass
class FNN:
    layers: Sequence[Linear]
    act_func: Callable

    def __init__(
        self,
        layers: Sequence[int, ...],
        *,
        act_func: Callable = jax.nn.relu,
        weight_init_func: str | Callable = "he_normal",
        bias_init_func: str | Callable = "ones",
        key=jr.PRNGKey(0),
    ):
        """Fully connected neural network
        Args:
            layers: Sequence of layer sizes
            act_func: Activation function to use. Defaults to jax.nn.relu.
            weight_init_func: Weight initializer function. Defaults to jax.nn.initializers.he_normal().
            bias_init_func: Bias initializer function. Defaults to lambda key, shape: jnp.ones(shape).
            key: Random key for weight and bias initialization. Defaults to jr.PRNGKey(0).

        Examples:
            >>> fnn = FNN([10, 5, 2])
            >>> fnn(jnp.ones((3, 10))).shape
            (3, 2)
        """

        keys = jr.split(key, len(layers))
        self.act_func = (
            jtu.Partial(act_func) if not pytc.is_treeclass(act_func) else act_func
        )
        self.layers = [
            Linear(
                in_features=in_dim,
                out_features=out_dim,
                key=ki,
                weight_init_func=weight_init_func,
                bias_init_func=bias_init_func,
            )
            for (ki, in_dim, out_dim) in (zip(keys, layers[:-1], layers[1:]))
        ]

    def __call__(self, x, **kwargs):
        *layers, last = self.layers
        for layer in layers:
            x = layer(x)
            x = self.act_func(x)
        return last(x)


@pytc.treeclass
class PFNN:
    """Parallel fully connected neural network

    Each output node is represented by a seperate subnetwork.
    see https://github.com/lululxvi/deepxde/blob/master/deepxde/nn/pytorch/fnn.py

    Args:
        layers :
            tuple item `i` maps to the number of neurons in the layer `i`
            (1,2,2,3) -> 1 input, 2 hidden layers with 2 neurons each, 3 output

        act_func :
            Activation function for hidden layers. Defaults to jax.nn.relu.

        weight_init_func :
            Weight initializer function . Defaults to jax.nn.initializers.he_normal().

        bias_init_func :
            Bias initializer function . Defaults to ones.

        key :
            Random key for weight and bias initialization. Defaults to jr.PRNGKey(0).

    Example:
        >>> print(PFNN([3,[5,4], 2]).tree_diagram())
        PFNN
            ├── layers=<class 'list'>
            │   ├── layers_0=<class 'list'>
            │   │   ├── layers_0_0=Linear
            │   │   │   ├── weight=f32[3,5]
            │   │   │   ├── bias=f32[5]
            │   │   │   ├*─ in_features=3
            │   │   │   ├*─ out_features=5
            │   │   │   ├*─ weight_init_func=init(key,shape,dtype)
            │   │   │   └*─ bias_init_func=Lambda(key,shape)
            │   │   └── layers_0_1=Linear
            │   │       ├── weight=f32[5,1]
            │   │       ├── bias=f32[1]
            │   │       ├*─ in_features=5
            │   │       ├*─ out_features=1
            │   │       ├*─ weight_init_func=init(key,shape,dtype)
            │   │       └*─ bias_init_func=Lambda(key,shape)
            │   └── layers_1=<class 'list'>
            │       ├── layers_1_0=Linear
            │       │   ├── weight=f32[3,4]
            │       │   ├── bias=f32[4]
            │       │   ├*─ in_features=3
            │       │   ├*─ out_features=4
            │       │   ├*─ weight_init_func=init(key,shape,dtype)
            │       │   └*─ bias_init_func=Lambda(key,shape)
            │       └── layers_1_1=Linear
            │           ├── weight=f32[4,1]
            │           ├── bias=f32[1]
            │           ├*─ in_features=4
            │           ├*─ out_features=1
            │           ├*─ weight_init_func=init(key,shape,dtype)
            │           └*─ bias_init_func=Lambda(key,shape)
            └*─ act_func=relu(*args,**kwargs)
    """

    layers: Sequence[Sequence[Linear, ...], ...]
    act_func: Callable

    def __init__(
        self,
        layers: Sequence[int, ...],
        *,
        act_func: Callable = jax.nn.relu,
        weight_init_func: str | Callable = "he_normal",
        bias_init_func: str | Callable = "zeros",
        key=jr.PRNGKey(0),
    ):

        self.act_func = (
            jtu.Partial(act_func) if not pytc.is_treeclass(act_func) else act_func
        )

        # check input/output node
        assert isinstance(layers[0], int), "Input node must be an integer"
        assert isinstance(layers[-1], int), "Output node must be an integer"
        Layer = ft.partial(
            Linear,
            weight_init_func=weight_init_func,
            bias_init_func=bias_init_func,
        )
        keys = jr.split(key, len(layers) - 1)

        self.layers = []

        in_dim = layers[0]

        for i, (key, out_dim) in enumerate(zip(keys, layers[1:])):
            sub_keys = jr.split(key, layers[-1])

            if isinstance(in_dim, (list, tuple)) and isinstance(out_dim, (list, tuple)):
                assert (
                    len(out_dim) == layers[-1]
                ), f"Size of sequence must match output node. Found len({out_dim})= {len(out_dim)} != {layers[-1]}"

                self.layers.append(
                    [
                        Layer(in_dim[j], out_dim[j], key=ki)
                        for j, ki in enumerate(sub_keys)
                    ]
                )

            elif isinstance(in_dim, int) and isinstance(out_dim, int):
                out_dim = 1 if i == (len(layers) - 2) else out_dim
                self.layers.append(
                    [Layer(in_dim, out_dim, key=ki) for _, ki in enumerate(sub_keys)]
                )

            elif isinstance(in_dim, int) and isinstance(out_dim, (list, tuple)):
                assert (
                    len(out_dim) == layers[-1]
                ), f"Size of sequence must match output node. Found len({out_dim})= {len(out_dim)} != {layers[-1]}"

                self.layers.append(
                    [Layer(in_dim, out_dim[j], key=ki) for j, ki in enumerate(sub_keys)]
                )

            elif isinstance(in_dim, (list, tuple)) and isinstance(out_dim, int):
                if i != (len(layers) - 2):
                    raise ValueError(
                        "Subnetworks can only joined at the end of the network."
                    )

                self.layers.append(
                    [Layer(in_dim[j], 1) for j, ki in enumerate(sub_keys)]
                )

            else:
                raise TypeError("Layers definition be an integer or a sequence.")

            in_dim = out_dim

        self.layers = list(map(list, zip(*self.layers)))

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        X = [x] * len(self.layers)
        for i, sub_network in enumerate(self.layers):
            *layers, last = sub_network
            for layer in layers:
                X[i] = layer(X[i])
                X[i] = self.act_func(X[i])
            X[i] = last(X[i])

        return jnp.concatenate(X, axis=-1)
