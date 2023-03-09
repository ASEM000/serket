from __future__ import annotations

import functools as ft
from types import FunctionType
from typing import Any, Callable, Sequence

import jax.nn.initializers as ji
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytreeclass as pytc


def _rename_func(func: Callable, name: str) -> Callable:
    """Rename a function."""
    func.__name__ = name
    return func


_INIT_FUNC_MAP = {
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


def range_cb(min_val: float = -float("inf"), max_val: float = float("inf")):
    """Return a function that checks if the input is in the range [min_val, max_val]."""

    def range_check(value: float):
        if jnp.min(value) <= value <= jnp.max(max_val):
            return value
        raise ValueError(f"Expected value between {min_val} and {max_val}, got {value}")

    return range_check


def instance_cb(expected_type: type | Sequence[type]):
    """Return a function that checks if the input is an instance of expected_type."""

    def instance_check(value: Any):
        if isinstance(value, expected_type):
            return value

        msg = f"Expected value of type {expected_type.__name__}, "
        msg += f"got {type(value).__name__}"
        raise ValueError(msg)

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


def canonicalize_cb(value, ndim, name: str | None = None):
    # in essence this is a type check that allows for int, tuple, and jnp.ndarray
    # canonicalization is done by converting to a tuple of length ndim
    if isinstance(value, int):
        return (value,) * ndim
    if isinstance(value, jnp.ndarray):
        return jnp.repeat(value, ndim)
    if isinstance(value, tuple):
        if len(value) != ndim:
            msg = f"Expected tuple of length {ndim}, got {len(value)}: {value}"
            msg += f" for {name}" if name is not None else ""
            raise ValueError(msg)
        return tuple(value)
    msg = f"Expected int or tuple for , got {value}."
    msg += f" for {name}" if name is not None else ""
    raise ValueError(msg)


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


def init_func_cb(init_func: str | Callable) -> Callable:
    if isinstance(init_func, FunctionType):
        return jtu.Partial(init_func)

    if isinstance(init_func, str):
        if init_func in _INIT_FUNC_MAP:
            return jtu.Partial(_INIT_FUNC_MAP[init_func])
        raise ValueError(f"value must be one of {list(_INIT_FUNC_MAP.keys())}")

    if init_func is None:
        return None

    raise ValueError("Value must be a string or a function.")


@ft.lru_cache(maxsize=None)
def canonicalize_padding_cb(
    padding: tuple[int | tuple[int, int] | str, ...] | int | str,
    kernel_size: tuple[int, ...],
    name: str | None = None,
):
    """
    Resolve padding to a tuple of tuples of ints.

    Args:
        padding: padding to resolve
        kernel_size: kernel size to use for resolving padding
        name: name of the argument being resolved

    Examples:
        >>> padding= (1, (2, 3), "same")
        >>> kernel_size = (3, 3, 3)
        >>> _canonicalize_padding(padding, kernel_size)
        ((1, 1), (2, 3), (1, 1))
    """

    def resolve_tuple_padding(padding, kernel_size):
        if len(padding) != len(kernel_size):
            msg = f"Expected padding to be of length {len(kernel_size)}, got {len(padding)}"
            msg += f" for {name}" if name is not None else ""
            raise ValueError(msg)

        resolved_padding = [[]] * len(kernel_size)

        for i, item in enumerate(padding):
            if isinstance(item, int):
                # ex: padding = (1, 2, 3)
                resolved_padding[i] = (item, item)

            elif isinstance(item, tuple):
                # ex: padding = ((1, 2), (3, 4), (5, 6))
                if len(item) != 2:
                    msg = f"Expected tuple of length 2, got {len(item)}"
                    msg += f" for {name}" if name is not None else ""
                    raise ValueError(msg)

                resolved_padding[i] = item

            elif isinstance(item, str):
                # ex: padding = ("same", "valid", "same")
                if item.lower() == "same":
                    lhs, rhs = ((kernel_size[i] - 1) // 2), (kernel_size[i] // 2)
                    resolved_padding[i] = (lhs, rhs)

                elif item.lower() == "valid":
                    resolved_padding[i] = (0, 0)

                else:
                    msg = f'String argument must be in ["same","valid"].Found {item}'
                    msg += f" for {name}" if name is not None else ""
                    raise ValueError(msg)
        return tuple(resolved_padding)

    def resolve_int_padding(padding, kernel_size):
        return ((padding, padding),) * len(kernel_size)

    def resolve_string_padding(padding, kernel_size):
        if padding.lower() == "same":
            return tuple(((wi - 1) // 2, wi // 2) for wi in kernel_size)

        elif padding.lower() == "valid":
            return ((0, 0),) * len(kernel_size)

        raise ValueError(f'string argument must be in ["same","valid"].Found {padding}')

    if isinstance(padding, int):
        return resolve_int_padding(padding, kernel_size)

    if isinstance(padding, str):
        return resolve_string_padding(padding, kernel_size)

    if isinstance(padding, tuple):
        return resolve_tuple_padding(padding, kernel_size)

    msg = f"Expected padding to be of type int, str or tuple, got {type(padding)}"
    msg += f" for {name}" if name is not None else ""
    raise ValueError(msg)


def validate_spatial_in_shape(call_wrapper, *, attribute_name: str):
    """Decorator to validate spatial input shape."""

    def check_spatial_in_shape(x, spatial_ndim: int) -> None:
        spatial_tuple = ("rows", "cols", "depths")
        if x.ndim != spatial_ndim + 1:
            msg = f"Input must be a {spatial_ndim+1}D tensor in shape of "
            msg += f"(in_features, {', '.join(spatial_tuple[:spatial_ndim])}), "
            msg += f"but got {x.shape}."
            raise ValueError(msg)
        return x

    @ft.wraps(call_wrapper)
    def wrapper(self, array, *a, **k):
        array = check_spatial_in_shape(array, getattr(self, attribute_name))
        return call_wrapper(self, array, *a, **k)

    return wrapper


def validate_in_features(call_wrapper, *, attribute_name: str, axis: int = 0):
    """Decorator to validate input features."""

    def check_in_features(x, in_features: int, axis: int) -> None:
        if x.shape[axis] != in_features:
            msg = f"Specified input_features={in_features} ,"
            msg += f"but got input with input_features={x.shape[axis]}."
            raise ValueError(msg)
        return x

    @ft.wraps(call_wrapper)
    def wrapper(self, array, *a, **k):
        check_in_features(array, getattr(self, attribute_name), axis)
        return call_wrapper(self, array, *a, **k)

    return wrapper


non_negative_scalar_cb = and_cb(range_cb(0), scalar_like_cb)
frozen_positive_int_cb = and_cb(positive_int_cb, pytc.freeze)
