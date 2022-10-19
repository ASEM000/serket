from __future__ import annotations

import dataclasses
import functools as ft
import inspect
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


_init_func_dict = {
    "he_normal": _rename_func(ji.he_normal(), "he_normal"),
    "he_uniform": _rename_func(ji.he_uniform(), "he_uniform"),
    "glorot_normal": _rename_func(ji.glorot_normal(), "glorot_normal"),
    "glorot_uniform": _rename_func(ji.glorot_uniform(), "glorot_uniform"),
    "lecun_normal": _rename_func(ji.lecun_normal(), "lecun_normal"),
    "lecun_uniform": _rename_func(ji.lecun_uniform(), "lecun_uniform"),
    "normal": _rename_func(ji.normal(), "normal"),
    "uniform": _rename_func(ji.uniform(), "uniform"),
    "ones": ji.ones,
    "zeros": ji.zeros,
    "xavier_normal": _rename_func(ji.xavier_normal(), "xavier_normal"),
    "xavier_uniform": _rename_func(ji.xavier_uniform(), "xavier_uniform"),
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


def _lazy_class(lazy_map: dict[str, Callable]):
    """transform a class into a lazy class that initializes its attributes lazily after the first call
    Args:
        lazy_map:
            a mapping from field name to a callable that returns the field value from the input at call time
        Example:
            >>> class Foo:
            ....    def __init__(self, in_features):
            ....        self.in_features = in_features
            ....    def __call__(self, x):
            ....        return self.in_features + x

            # infer `in_features` from call input
            # lets say in_featues is last dimension of input
            lazy_foo = _lazy_class({"in_features": lambda x: x.shape[-1]})
    """

    def cls_wrapper(cls):
        def _init_wrapper(func):
            @ft.wraps(func)
            def wrapper(self, *args, **kwargs):
                # get all parameters from the input
                params = inspect.signature(func).parameters
                params = {k: v.default for k, v in params.items()}
                del params["self"]  # remove self
                params.update(kwargs)  # merge user kwargs
                params.update(dict(zip(params.keys(), args)))  # merge user args

                # check if any of the params is marked as lazy (i.e. = None)
                if any(params[lazy_argname] is None for lazy_argname in lazy_map):
                    # set all fields to None to give the user a hint that they are lazy
                    # and to avoid errors when calling the layer before initializing it
                    for field_item in dataclasses.fields(self):
                        setattr(self, field_item.name, None)

                    # remove lazy arg so that it is not part of the partial function
                    for lazy_argname in lazy_map:
                        del params[lazy_argname]

                    setattr(self, "_partial_init", ft.partial(func, self, **params))
                    return
                # execute original init
                return func(self, *args, **kwargs)

            return wrapper

        def _call_wrapper(func):
            @ft.wraps(func)
            def wrapper(self, *args, **kwargs):
                if hasattr(self, "_partial_init"):
                    # check if jax transformations are applied to the layer
                    # by checking if any of the input is a Tracer
                    if any(isinstance(a, jax.core.Tracer) for a in args):
                        raise ValueError(_TRACER_ERROR_MSG(self.__class__.__name__))
                    inferred = {k: v(*args, **kwargs) for k, v in lazy_map.items()}
                    # initialize the layer with the inferred values
                    getattr(self, "_partial_init")(**inferred)
                    # remove the partial function so that it is not called again
                    object.__delattr__(self, "_partial_init")

                # execute original call
                return func(self, *args, **kwargs)

            return wrapper

        cls.__init__ = _init_wrapper(cls.__init__)
        cls.__call__ = _call_wrapper(cls.__call__)
        return cls

    return cls_wrapper
