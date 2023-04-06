import functools as ft

import jax
import jax.numpy as jnp


def diff(func, *args, **kwargs):
    """grad of sum(func)"""

    @ft.partial(jax.grad, *args, **kwargs)
    @ft.wraps(func)
    def sum_func(*ar, **kw):
        return jnp.sum(func(*ar, **kw))

    return sum_func


def value_and_diff(func, *args, **kwargs):
    """value and grad of sum(func)"""

    @ft.partial(jax.value_and_grad, *args, **kwargs)
    @ft.wraps(func)
    def sum_func(*ar, **kw):
        return jnp.sum(func(*ar, **kw))

    return sum_func
