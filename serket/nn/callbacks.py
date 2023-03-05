from __future__ import annotations

from types import FunctionType
from typing import Any, Callable

import jax.nn.initializers as ji
import jax.numpy as jnp
import jax.tree_util as jtu
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


def and_cb(*callbacks):
    """Return a function that checks if the input matches all of the callbacks."""

    def wrapper(value):
        for cb in callbacks:
            value = cb(value)
        return value

    return wrapper


def or_cb(*callbacks):
    """Return a function that checks if the input matches one of the callbacks."""

    def wrapper(value):
        for cb in callbacks:
            try:
                return cb(value)
            except Exception:
                pass
        raise ValueError(f"Expected value to match one of {callbacks}, got {value}")

    return wrapper


def positive_int_cb(value):
    """Return if value is a positive integer, otherwise raise an error."""
    if value is None:
        # prolly lazy
        return value
    if not isinstance(value, int):
        raise ValueError(f"value must be an integer, got {type(value)}")
    if value <= 0:
        raise ValueError(f"value must be positive, got {value}")
    return value


non_negative_scalar_cb = and_cb(range_cb(0), scalar_like_cb)

maybe_positive_int_cb = or_cb(positive_int_cb, instance_cb(type(None)))


def _rename_func(func: Callable, name: str) -> Callable:
    """Rename a function."""
    func.__name__ = name
    return func


_init_func_dict = {
    "he_normal": _rename_func(ji.he_normal(), "he_normal_init"),
    "he_uniform": _rename_func(ji.he_uniform(), "he_uniform_init"),
    "glorot_normal": _rename_func(ji.glorot_normal(), "glorot_normal_init"),
    "glorot_uniform": _rename_func(ji.glorot_uniform(), "glorot_uniform_init"),
    "lecun_normal": _rename_func(ji.lecun_normal(), "lecun_normal_init"),
    "lecun_uniform": _rename_func(ji.lecun_uniform(), "lecun_uniform_init"),
    "normal": _rename_func(ji.normal(), "normal_init"),
    "uniform": _rename_func(ji.uniform(), "uniform_init"),
    "ones": _rename_func(ji.ones, "ones_init"),
    "zeros": _rename_func(ji.zeros, "zeros_init"),
    "xavier_normal": _rename_func(ji.xavier_normal(), "xavier_normal_init"),
    "xavier_uniform": _rename_func(ji.xavier_uniform(), "xavier_uniform_init"),
    "orthogonal": _rename_func(ji.orthogonal(), "orthogonal_init"),
}


def init_func_cb(init_func: str | Callable) -> Callable:
    if isinstance(init_func, FunctionType):
        return jtu.Partial(init_func)

    if isinstance(init_func, str):
        if init_func in _init_func_dict:
            return jtu.Partial(_init_func_dict[init_func])
        raise ValueError(f"value must be one of {list(_init_func_dict.keys())}")

    if init_func is None:
        return None

    raise ValueError("Value must be a string or a function.")
