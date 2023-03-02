from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


def range_cb(min_val: float = -float("inf"), max_val: float = float("inf")):
    """Return a function that checks if the input is in the range [min_val, max_val]."""

    def range_check(value: float):
        if jnp.min(value) <= value <= jnp.max(max_val):
            return value
        raise ValueError(f"Expected value between {min_val} and {max_val}, got {value}")

    return range_check


def instance_cb(expected_type: type | tuple[type]):
    """Return a function that checks if the input is an instance of expected_type."""

    def instance_check(value: Any):
        if isinstance(value, expected_type):
            return value
        raise ValueError(f"Expected value of type {expected_type}, got {type(value)}")

    return instance_check


def scalar_like_cb(value: Any):
    """Return a function that checks if the input is a trainable scalar"""
    if isinstance(value, (float, complex)):
        return value
    if (
        isinstance(value, (jnp.ndarray, np.ndarray))
        and np.issubdtype(value.dtype, np.inexact)
        and value.shape == ()
    ):
        return value

    msg = f"Expected value to be a float, complex, or array-like object, got {type(value)}"
    raise ValueError(msg)


def canonicalize_cb(value, ndim):
    if isinstance(value, int):
        return (value,) * ndim
    elif isinstance(value, jnp.ndarray):
        return jnp.repeat(value, ndim)
    elif isinstance(value, tuple):
        if len(value) != ndim:
            msg = f"Expected tuple of length {ndim}, got {len(value)}: {value}"
            raise ValueError(msg)
        return tuple(value)
    raise ValueError(f"Expected int or tuple for , got {value}.")


def merge_cb(callbacks):
    def merged_cb(value):
        for cb in callbacks:
            value = cb(value)
        return value

    return merged_cb


non_negative_scalar_cb = merge_cb([range_cb(0), scalar_like_cb])
