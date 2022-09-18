import jax
import jax.numpy as jnp


def diff(func, *args, **kwargs):
    """grad of summed func"""
    return jax.grad(lambda *ar, **kw: jnp.sum(func(*ar, **kw)), **kwargs)


def diff_and_grad(func, **kwargs):
    """value and grad of summed func"""
    return jax.value_and_grad(lambda *ar, **kw: jnp.sum(func(*ar, **kw)), **kwargs)
