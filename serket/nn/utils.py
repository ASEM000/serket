from __future__ import annotations

import dataclasses
import functools as ft
from types import FunctionType
from typing import Any, Callable, Sequence

import jax
import jax.nn.initializers as ji
import jax.numpy as jnp
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


_act_func_map = {
    "tanh": jax.nn.tanh,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "hard_sigmoid": jax.nn.hard_sigmoid,
    None: lambda x: x,
}

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


def _check_and_return_init_func(
    init_func: str | Callable, name: str
) -> Callable | None:
    if isinstance(init_func, FunctionType):
        return jtu.Partial(init_func)

    elif isinstance(init_func, str):
        if init_func in _init_func_dict:
            return jtu.Partial(_init_func_dict[init_func])
        raise ValueError(f"{name} must be one of {list(_init_func_dict.keys())}")

    elif init_func is None:
        return None

    raise ValueError(f"`{name}` must be a string or a function.")


def _calculate_convolution_output_shape(shape, kernel_size, padding, strides):
    """Compute the shape of the output of a convolutional layer."""
    return tuple(
        (xi + (li + ri) - ki) // si + 1
        for xi, ki, si, (li, ri) in zip(shape, kernel_size, strides, padding)
    )


def _check_and_return_padding(
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
        >>> _check_and_return_padding(padding, kernel_size)
        ((1, 1), (2, 3), (1, 1))
    """

    def _resolve_tuple_padding(padding, kernel_size):

        assert len(padding) == len(
            kernel_size
        ), f"Expected padding to be of length {len(kernel_size)}, got {len(padding)}"

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
                    raise ValueError(
                        f'string argument must be in ["same","valid"].Found {item}'
                    )
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

    raise ValueError(
        f"Expected padding to be of type int, str or tuple, got {type(padding)}"
    )


def _check_and_return(value, ndim, name):
    if isinstance(value, int):
        return (value,) * ndim
    elif isinstance(value, jnp.ndarray):
        return jnp.repeat(value, ndim)
    elif isinstance(value, tuple):
        assert len(value) == ndim, f"{name} must be a tuple of length {ndim}"
        return tuple(value)
    raise ValueError(f"Expected int or tuple for {name}, got {value}.")


def _check_and_return_positive_int(value, name):
    """Check if value is a positive integer."""
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {type(value)}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _check_spatial_in_shape(func):
    """Decorator to check the input shape of a spatial layer"""
    spatial_tuple = ("rows", "cols", "depths")

    @ft.wraps(func)
    def wrapper(self, x, *args, **kwargs):
        if x.ndim != self.ndim + 1:
            msg = f"Input must be a {self.ndim+1}D tensor in shape of "
            msg += f"(in_features, {', '.join(spatial_tuple[:self.ndim])}), "
            msg += f"but got {x.shape}."
            raise ValueError(msg)
        return func(self, x, *args, **kwargs)

    return wrapper


def _check_in_features(func):
    """Check if the input feature dimension is the same as the layer's in_features"""

    @ft.wraps(func)
    def wrapper(self, x, *args, **kwargs):
        if x.shape[0] != self.in_features:
            msg = f"Specified input_features={self.in_features} ,"
            msg += f"but got input with input_features={x.shape[0]}."
            raise ValueError(msg)

        return func(self, x, *args, **kwargs)

    return wrapper


_check_and_return_kernel = ft.partial(_check_and_return, name="kernel_size")
_check_and_return_strides = ft.partial(_check_and_return, name="strides")
_check_and_return_input_dilation = ft.partial(_check_and_return, name="input_dilation")
_check_and_return_kernel_dilation = ft.partial(_check_and_return, name="kernel_dilation")  # fmt: skip
_check_and_return_input_size = ft.partial(_check_and_return, name="input_size")  # fmt: skip


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


_TRACER_ERROR_MSG = lambda cls_name: (
    "Using Tracers as input to a lazy layer is not supported. "
    "Use non-Tracer input to initialize the layer.\n"
    "This error can occur if jax transformations are applied to a layer before "
    "calling it with a non Tracer input.\n"
    "Example: \n"
    "# This will fail\n"
    ">>> x = jax.numpy.ones(...)\n"
    f">>> layer = {cls_name}(None, ...)\n"
    ">>> layer = jax.jit(layer)\n"
    ">>> layer(x) \n"
    "# Instead, first initialize the layer with a non Tracer input\n"
    "# and then apply jax transformations\n"
    f">>> layer = {cls_name}(None, ...)\n"
    ">>> layer(x) # dry run to initialize the layer\n"
    ">>> layer = jax.jit(layer)\n"
)


@ft.lru_cache(maxsize=128)
def _multilinear_einsum_string(degree: int) -> str:
    """Generate einsum string for a linear layer of degree n
    Example:
        >>> _multilinear_einsum_string(1)
        '...a,ab->....b'
        >>> _multilinear_einsum_string(2)
        '...a,...b,abc->....c'
    """
    alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert 1 <= degree <= len(alpha) - 1, f"degree must be between 1 and {len(alpha)-1}"

    xs_string = [f"...{i}" for i in alpha[:degree]]
    output_string = ",".join(xs_string)
    output_string += f",{alpha[:degree+1]}->...{alpha[degree]}"
    return output_string


@ft.lru_cache(maxsize=128)
def _general_linear_einsum_string(*axes: tuple[int, ...]) -> str:
    """Return the einsum string for a general linear layer.
    Example:
        # apply linear layer to last axis
        >>> _general_linear_einsum_string(-1)
        '...a,ab->...b'

        # apply linear layer to last two axes
        >>> _general_linear_einsum_string(-1,-2)
        '...ab,abc->...c'

        # apply linear layer to second last axis
        >>> _general_linear_einsum_string(-2)
        '...ab,ac->...bc'

        # apply linear layer to last and third last axis
        >>> _general_linear_einsum_string(-1,-3)
        '...abc,acd->...bd'
    """
    assert all([i < 0 for i in axes]), "axes should be negative"
    axes = sorted(axes)
    total_axis = abs(min(axes))  # get the total number of axes
    alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    input_string = "..." + alpha[:total_axis]
    weight_string = "".join([input_string[axis] for axis in axes]) + alpha[total_axis]
    result_string = "".join([ai for ai in input_string if ai not in weight_string])
    result_string += alpha[total_axis]
    return f"{input_string},{weight_string}->{result_string}"


def _getattr(item, path):
    return (
        _getattr(getattr(item, path[0]), path[1:])
        if len(path) > 1
        else getattr(item, path[0])
    )


def _hasattr(item, path):
    return (
        _hasattr(getattr(item, path[0]), path[1:])
        if len(path) > 1
        else hasattr(item, path[0])
    )


def _lazy_call(infer_func: Callable[[Any], dict[str, Any]], partial_kw: str):
    """Decorator to lazily initialize a layer. The layer is initialized when the first call is made.
    Args:
        infer_func: a function that takes the layer as input and returns a dictionary of inferred values.
        partial_kw: the keyword argument that stores the partial function.
    """

    partial_kw = partial_kw.split(".")

    def func_wrapper(func):
        @ft.wraps(func)
        def wrapper(self, *args, **kwargs):
            if _hasattr(self, partial_kw):
                if any(isinstance(x, jax.core.Tracer) for x in args):
                    raise ValueError(_TRACER_ERROR_MSG(self.__class__.__name__))
                _getattr(self, partial_kw)(**infer_func(self, *args, **kwargs))

            return func(self, *args, **kwargs)

        return wrapper

    return func_wrapper


def _lazy_general_linear(func):
    def infer_func(self, *a, **k):
        return {"in_features": tuple(a[0].shape[i] for i in self.in_axes)}

    return _lazy_call(infer_func, "_partial_init")(func)


def _lazy_multi_linear(func):
    def infer_func(self, *a, **k):
        return {"in_features": tuple(ai.shape[-1] for ai in a)}

    return _lazy_call(infer_func, "_partial_init")(func)


def _lazy_conv(func):
    def infer_func(self, *a, **k):
        return {"in_features": a[0].shape[0]}

    return _lazy_call(infer_func, "_partial_init")(func)


_lazy_rnn_cell = _lazy_blur = _lazy_norm = _lazy_conv


def _lazy_local_conv(func):
    def infer_func(self, *a, **k):
        return {"in_features": a[0].shape[0], "in_size": a[0].shape[1:]}

    return _lazy_call(infer_func, "_partial_init")(func)


def _lazy_fwd_rnn(func):
    def infer_func(self, *a, **k):
        return {"in_features": a[0].shape[1]}

    return _lazy_call(infer_func, "cell._partial_init")(func)


def _lazy_bwd_rnn(func):
    def infer_func(self, *a, **k):
        return {"in_features": a[0].shape[1]}

    return _lazy_call(infer_func, "backward_cell._partial_init")(func)
