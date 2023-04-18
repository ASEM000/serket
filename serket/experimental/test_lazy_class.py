import functools as ft

import jax
import jax.numpy as jnp
import pytest
import pytreeclass as pytc

from serket.experimental import lazy_class


def test_lazy_class():
    @ft.partial(
        lazy_class,
        lazy_keywords=["in_features"],  # -> `in_features` is lazy evaluated
        infer_func=lambda self, x: (x.shape[-1],),
        infer_method_name="__call__",  # -> `infer_func` is applied to `__call__` method
        lazy_marker=None,  # -> `None` is used to indicate a lazy argument
    )
    class LazyLinear(pytc.TreeClass):
        weight: jax.Array
        bias: jax.Array

        def __init__(self, in_features: int, out_features: int):
            self.in_features = in_features
            self.out_features = out_features
            self.weight = jax.random.normal(
                jax.random.PRNGKey(0), (in_features, out_features)
            )
            self.bias = jax.random.normal(jax.random.PRNGKey(0), (out_features,))

        def __call__(self, x):
            return x @ self.weight + self.bias

    layer = LazyLinear(None, 20)
    x = jnp.ones([10, 1])

    assert layer(x).shape == (10, 20)

    layer = LazyLinear(None, 20)

    with pytest.raises(ValueError):
        jax.vmap(layer)(jnp.ones([10, 1, 1]))
