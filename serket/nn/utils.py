from __future__ import annotations

import dataclasses
import functools as ft
from types import FunctionType
from typing import Any, Callable, Sequence, Tuple, Union

import jax
import jax.nn.initializers as ji
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu


def _calculate_transpose_padding(padding, kernel_size, input_dilation, extra_padding):
    """
    Transpose padding to get the padding for the transpose convolution.

    Args:
        padding: padding to transpose
        kernel_size: kernel size to use for transposing padding
        input_dilation: input dilation to use for transposing padding
        extra_padding: extra padding to use for transposing padding
    """
    return tuple(
        ((ki - 1) * di - pl, (ki - 1) * di - pr + ep)
        for (pl, pr), ki, ep, di in zip(
            padding, kernel_size, extra_padding, input_dilation
        )
    )


def _rename_func(func: Callable, name: str) -> Callable:
    """Rename a function."""
    func.__name__ = name
    return func


_ACT_FUNC_MAP = {
    "tanh": jax.nn.tanh,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "hard_sigmoid": jax.nn.hard_sigmoid,
    None: lambda x: x,
}


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


def _canonicalize_init_func(init_func: str | Callable, name: str) -> Callable | None:
    if isinstance(init_func, FunctionType):
        return jtu.Partial(init_func)

    elif isinstance(init_func, str):
        if init_func in _INIT_FUNC_MAP:
            return jtu.Partial(_INIT_FUNC_MAP[init_func])
        raise ValueError(f"{name} must be one of {list(_INIT_FUNC_MAP.keys())}")

    elif init_func is None:
        return None

    raise ValueError(f"`{name}` must be a string or a function.")


def _calculate_convolution_output_shape(shape, kernel_size, padding, strides):
    """Compute the shape of the output of a convolutional layer."""
    return tuple(
        (xi + (li + ri) - ki) // si + 1
        for xi, ki, si, (li, ri) in zip(shape, kernel_size, strides, padding)
    )


@ft.lru_cache(maxsize=None)
def _canonicalize_padding(
    padding: tuple[int | tuple[int, int] | str, ...] | int | str,
    kernel_size: tuple[int, ...],
):
    """
    Resolve padding to a tuple of tuples of ints.

    Args:
        padding: padding to resolve
        kernel_size: kernel size to use for resolving padding

    Examples:
        >>> padding= (1, (2, 3), "same")
        >>> kernel_size = (3, 3, 3)
        >>> _canonicalize_padding(padding, kernel_size)
        ((1, 1), (2, 3), (1, 1))
    """

    def _resolve_tuple_padding(padding, kernel_size):
        msg = f"Expected padding to be of length {len(kernel_size)}, got {len(padding)}"
        assert len(padding) == len(kernel_size), msg

        resolved_padding = [[]] * len(kernel_size)

        for i, item in enumerate(padding):
            if isinstance(item, int):
                # ex: padding = (1, 2, 3)
                resolved_padding[i] = (item, item)

            elif isinstance(item, tuple):
                # ex: padding = ((1, 2), (3, 4), (5, 6))
                assert len(item) == 2, f"Expected tuple of length 2, got {len(item)}"
                resolved_padding[i] = item

            elif isinstance(item, str):
                # ex: padding = ("same", "valid", "same")
                if item.lower() == "same":
                    resolved_padding[i] = ((kernel_size[i] - 1) // 2), (
                        kernel_size[i] // 2
                    )

                elif item.lower() == "valid":
                    resolved_padding[i] = (0, 0)

                else:
                    msg = f'string argument must be in ["same","valid"].Found {item}'
                    raise ValueError(msg)

        return tuple(resolved_padding)

    def _resolve_int_padding(padding, kernel_size):
        return ((padding, padding),) * len(kernel_size)

    def _resolve_string_padding(padding, kernel_size):
        if padding.lower() == "same":
            return tuple(((wi - 1) // 2, wi // 2) for wi in kernel_size)

        elif padding.lower() == "valid":
            return ((0, 0),) * len(kernel_size)

        raise ValueError(f'string argument must be in ["same","valid"].Found {padding}')

    if isinstance(padding, int):
        return _resolve_int_padding(padding, kernel_size)

    elif isinstance(padding, str):
        return _resolve_string_padding(padding, kernel_size)

    elif isinstance(padding, tuple):
        return _resolve_tuple_padding(padding, kernel_size)

    msg = f"Expected padding to be of type int, str or tuple, got {type(padding)}"
    raise ValueError(msg)


def _canonicalize(value, ndim, name):
    if isinstance(value, int):
        return (value,) * ndim
    elif isinstance(value, jax.Array):
        return jnp.repeat(value, ndim)
    elif isinstance(value, tuple):
        assert len(value) == ndim, f"{name} must be a tuple of length {ndim}"
        return tuple(value)
    raise ValueError(f"Expected int or tuple for {name}, got {value}.")


def _canonicalize_positive_int(value, name):
    """Return if value is a positive integer, otherwise raise an error."""
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {type(value)}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _check_spatial_in_shape(x, spatial_ndim: int) -> None:
    spatial_tuple = ("rows", "cols", "depths")
    if x.ndim != spatial_ndim + 1:
        msg = f"Input must be a {spatial_ndim+1}D tensor in shape of "
        msg += f"(in_features, {', '.join(spatial_tuple[:spatial_ndim])}), "
        msg += f"but got {x.shape}."
        raise ValueError(msg)
    return x


def _check_in_features(x, in_features: int, axis: int = 0) -> None:
    if x.shape[axis] != in_features:
        msg = f"Specified input_features={in_features} ,"
        msg += f"but got input with input_features={x.shape[axis]}."
        raise ValueError(msg)
    return x


_canonicalize_kernel = ft.partial(_canonicalize, name="kernel_size")
_canonicalize_strides = ft.partial(_canonicalize, name="strides")
_canonicalize_input_dilation = ft.partial(_canonicalize, name="input_dilation")
_canonicalize_kernel_dilation = ft.partial(_canonicalize, name="kernel_dilation")  # fmt: skip
_canonicalize_input_size = ft.partial(_canonicalize, name="input_size")  # fmt: skip


def _create_fields_from_container(items: Sequence[Any]) -> dict:
    return_map = {}
    for i, item in enumerate(items):
        field_item = dataclasses.field(repr=True)
        field_name = f"{(item.__class__.__name__)}_{i}"
        object.__setattr__(field_item, "name", field_name)
        object.__setattr__(field_item, "type", type(item))
        return_map[field_name] = field_item
    return return_map


def _create_fields_from_mapping(items: dict[str, Any]) -> dict:
    return_map = {}
    for field_name, item in items.items():
        field_item = dataclasses.field(repr=True)
        object.__setattr__(field_item, "name", field_name)
        object.__setattr__(field_item, "type", type(item))
        return_map[field_name] = field_item
    return return_map


def _check_non_tracer(*x, name: str = "Class"):
    if any(isinstance(xi, jax.core.Tracer) for xi in x):
        _TRACER_ERROR_MSG = (
            f"Using Tracers as input to a lazy layer is not supported. "
            "Use non-Tracer input to initialize the layer.\n"
            "This error can occur if jax transformations are applied to a layer before "
            "calling it with a non Tracer input.\n"
            "Example: \n"
            "# This will fail\n"
            ">>> x = jax.numpy.ones(...)\n"
            f">>> layer = {name}(None, ...)\n"
            ">>> layer = jax.jit(layer)\n"
            ">>> layer(x) \n"
            "# Instead, first initialize the layer with a non Tracer input\n"
            "# and then apply jax transformations\n"
            f">>> layer = {name}(None, ...)\n"
            ">>> layer(x) # dry run to initialize the layer\n"
            ">>> layer = jax.jit(layer)\n"
        )

        raise ValueError(_TRACER_ERROR_MSG)


def _range_cb(min_val: float = -float("inf"), max_val: float = float("inf")):
    """Return a function that checks if the input is in the range [min_val, max_val]."""

    def range_check(value: float):
        if jnp.min(value) <= value <= jnp.max(max_val):
            return value
        raise ValueError(f"Expected value between {min_val} and {max_val}, got {value}")

    return range_check


def _instance_cb(expected_type: type | tuple[type]):
    """Return a function that checks if the input is an instance of expected_type."""

    def instance_check(value: Any):
        if isinstance(value, expected_type):
            return value
        raise ValueError(f"Expected value of type {expected_type}, got {type(value)}")

    return instance_check


KernelSizeType = Union[int, Sequence[int]]
StridesType = Union[int, Sequence[int]]
PaddingType = Union[str, int, Sequence[int], Sequence[Tuple[int, int]]]
DilationType = Union[int, Sequence[int]]
InitFuncType = Union[str, Callable[[jr.PRNGKey, Sequence[int]], jax.Array]]


def canonicalize(value, ndim, name: str | None = None):
    if isinstance(value, int):
        return (value,) * ndim
    if isinstance(value, jax.Array):
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


@ft.lru_cache(maxsize=None)
def canonicalize_padding(
    padding: tuple[int | tuple[int, int] | str, ...] | int | str,
    kernel_size: tuple[int, ...],
    name: str | None = None,
):
    """
    Resolve padding to a tuple of tuples of ints.

    Args:
    ----
        padding: padding to resolve
        kernel_size: kernel size to use for resolving padding
        name: name of the argument being resolved

    Examples:
    --------
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
