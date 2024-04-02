# Copyright 2023 serket authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools as ft
import inspect
import operator as op
from types import MethodType
from typing import Any, Callable, Sequence, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.util import safe_zip
from typing_extensions import Annotated, Literal, ParamSpec

import serket as sk

KernelSizeType = Union[int, Sequence[int]]
StridesType = Union[int, Sequence[int]]
PaddingType = Union[str, int, Sequence[int], Sequence[Tuple[int, int]]]
DilationType = Union[int, Sequence[int]]
P = ParamSpec("P")
T = TypeVar("T")
HWArray = Annotated[jax.Array, "HW"]
CHWArray = Annotated[jax.Array, "CHW"]
PaddingLiteral = Literal[
    "constant",
    "edge",
    "linear_ramp",
    "maximum",
    "mean",
    "median",
    "minimum",
    "reflect",
    "symmetric",
    "wrap",
]
PaddingMode = Union[PaddingLiteral, Union[int, float], Callable]


@ft.lru_cache(maxsize=None)
def generate_conv_dim_numbers(spatial_ndim) -> jax.lax.ConvDimensionNumbers:
    return jax.lax.ConvDimensionNumbers(*((tuple(range(spatial_ndim + 2)),) * 3))


@ft.lru_cache(maxsize=128)
def calculate_transpose_padding(
    padding,
    kernel_size,
    input_dilation,
    extra_padding,
):
    """Transpose padding to get the padding for the transpose convolution.

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


def calculate_convolution_output_shape(
    shape: tuple[int, ...],
    kernel_size: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
    strides: tuple[int, ...],
):
    """Compute the shape of the output of a convolutional layer."""
    return tuple(
        (xi + (li + ri) - ki) // si + 1
        for xi, ki, si, (li, ri) in zip(shape, kernel_size, strides, padding)
    )


def same_padding_along_dim(
    in_dim: int,
    kernel_size: int,
    stride: int,
) -> tuple[int, int]:
    # https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2
    # di: input dimension
    # ki: kernel size
    # si: stride
    if in_dim % stride == 0:
        pad = max(kernel_size - stride, 0)
    else:
        pad = max(kernel_size - (in_dim % stride), 0)

    return (pad // 2, pad - pad // 2)


def resolve_tuple_padding(
    in_dim: tuple[int, ...],
    padding: PaddingType,
    kernel_size: KernelSizeType,
    strides: StridesType,
) -> tuple[tuple[int, int], ...]:
    del in_dim, strides
    if len(padding) != len(kernel_size):
        raise ValueError(f"Length mismatch {len(kernel_size)=}!={len(padding)=}.")

    resolved_padding = [[]] * len(kernel_size)

    for i, item in enumerate(padding):
        if isinstance(item, int):
            resolved_padding[i] = (item, item)  # ex: padding = (1, 2, 3)

        elif isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError(f"Expected tuple of length 2, got {len(item)=}")
            resolved_padding[i] = item

    return tuple(resolved_padding)


def resolve_int_padding(
    in_dim: tuple[int, ...],
    padding: PaddingType,
    kernel_size: KernelSizeType,
    strides: StridesType,
):
    del in_dim, strides
    return ((padding, padding),) * len(kernel_size)


def resolve_string_padding(in_dim, padding, kernel_size, strides):
    if padding.lower() == "same":
        return tuple(
            same_padding_along_dim(di, ki, si)
            for di, ki, si in zip(in_dim, kernel_size, strides)
        )

    if padding.lower() == "valid":
        return ((0, 0),) * len(kernel_size)

    raise ValueError(f'string argument must be in ["same","valid"].Found {padding}')


@ft.lru_cache(maxsize=128)
def delayed_canonicalize_padding(
    in_dim: tuple[int, ...],
    padding: PaddingType,
    kernel_size: KernelSizeType,
    strides: StridesType,
):
    # in case of `str` padding, we need to know the input dimension
    # to calculate the padding thus we need to delay the canonicalization
    # until the call

    if isinstance(padding, int):
        return resolve_int_padding(in_dim, padding, kernel_size, strides)

    if isinstance(padding, str):
        return resolve_string_padding(in_dim, padding, kernel_size, strides)

    if isinstance(padding, tuple):
        return resolve_tuple_padding(in_dim, padding, kernel_size, strides)

    raise ValueError(
        "Expected padding to be of:\n"
        "* int, for same padding along all dimensions\n"
        "* str, for `same` or `valid` padding along all dimensions\n"
        "* tuple of int, for individual padding along each dimension\n"
        "* tuple of tuple of int, for padding before and after each dimension\n"
        f"Got {padding=}."
    )


def canonicalize(value, ndim, name: str | None = None):
    if isinstance(value, (int, float)):
        return (value,) * ndim
    if isinstance(value, jax.Array):
        return jnp.repeat(value, ndim)
    if isinstance(value, tuple):
        if len(value) != ndim:
            raise ValueError(f"{len(value)=} != {ndim=} for {name=} and {value=}.")
        return tuple(value)

    raise ValueError(f"Expected int or tuple , got {value=}.")


@sk.autoinit
class Range(sk.TreeClass):
    min_val: float = -float("inf")
    max_val: float = float("inf")
    min_inclusive: bool = True
    max_inclusive: bool = True

    def __call__(self, value: Any):
        lop, ls = (op.ge, "[") if self.min_inclusive else (op.gt, "(")
        rop, rs = (op.le, "]") if self.max_inclusive else (op.lt, ")")

        if lop(value, self.min_val) and rop(value, self.max_val):
            return value

        raise ValueError(f"Not in {ls}{self.min_val}, {self.max_val}{rs} got {value=}.")


@sk.autoinit
class IsInstance(sk.TreeClass):
    klass: type | Sequence[type]

    def __call__(self, value: Any):
        if isinstance(value, self.klass):
            return value
        raise TypeError(f"Expected {self.klass}, got {type(value).__name__}")


class ScalarLike(sk.TreeClass):
    """Check if the input is a scalar"""

    def __call__(self, value: Any):
        if isinstance(value, (float, complex)):
            return value
        if (
            isinstance(value, (jax.Array, np.ndarray))
            and np.issubdtype(value.dtype, np.inexact)
            and value.shape == ()
        ):
            return value
        raise ValueError(f"Expected inexact type got {value=}")


def positive_int_cb(value):
    """Return if value is a positive integer, otherwise raise an error."""
    if not isinstance(value, int):
        raise ValueError(f"value must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{value=} must be positive.")
    return value


def recursive_getattr(obj, attr: Sequence[str]):
    return (
        getattr(obj, attr[0])
        if len(attr) == 1
        else recursive_getattr(getattr(obj, attr[0]), attr[1:])
    )


def validate_spatial_ndim(func: Callable[P, T], argnum: int = 0) -> Callable[P, T]:
    """Decorator to validate spatial input shape."""

    @ft.wraps(func)
    def wrapper(self, *args, **kwargs):
        input = args[argnum]
        spatial_ndim = self.spatial_ndim

        if input.ndim != spatial_ndim + 1:
            spatial = ", ".join(("rows", "cols", "depths")[:spatial_ndim])
            name = type(self).__name__
            raise ValueError(
                f"Dimesion mismatch error in inputs of {name}\n"
                f"Input should satisfy:\n"
                f"  - {(spatial_ndim + 1) = } dimension, but got {input.ndim = }.\n"
                f"  - shape of (in_features, {spatial}), but got {input.shape = }.\n"
                + (
                    # maybe the user apply the layer on a batched input
                    "The input should be unbatched (no batch dimension).\n"
                    "To apply on batched input, use `jax.vmap(...)(input)`."
                    if input.ndim == spatial_ndim + 2
                    else ""
                )
            )
        return func(self, *args, **kwargs)

    return wrapper


def validate_in_features_shape(func: Callable[P, T], axis: int) -> Callable[P, T]:
    """Decorator to validate input features."""

    def check_axis_shape(input, in_features: int, axis: int) -> None:
        if input.shape[axis] != in_features:
            raise ValueError(f"Specified {in_features=}, got {input.shape[axis]=}.")
        return input

    @ft.wraps(func)
    def wrapper(self, array, *a, **k):
        check_axis_shape(array, self.in_features, axis)
        return func(self, array, *a, **k)

    return wrapper


@ft.lru_cache(maxsize=128)
def get_params(func: MethodType) -> tuple[inspect.Parameter, ...]:
    """Get the arguments of func."""
    return tuple(inspect.signature(func).parameters.values())


def tuplify(value: T) -> T | tuple[T]:
    return value if isinstance(value, tuple) else (value,)


def kernel_map(
    func: dict,
    shape: tuple[int, ...],
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
    padding_mode: PaddingMode = "constant",
) -> Callable:
    """Minimal implementation of kmap from kernex"""
    # Copyright 2023 Kernex authors
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     https://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    # copied here to avoid requiring kernex as a dependency
    # does not support most of the kernex features
    if isinstance(padding_mode, (int, float)):
        pad_kwargs = dict(mode="constant", constant_values=padding_mode)
    elif isinstance(padding_mode, (str, Callable)):
        pad_kwargs = dict(mode=padding_mode)

    gather_kwargs = dict(
        mode="promise_in_bounds",
        indices_are_sorted=True,
        unique_indices=True,
    )

    def calculate_kernel_map_output_shape(
        shape: tuple[int, ...],
        kernel_size: tuple[int, ...],
        strides: tuple[int, ...],
        border: tuple[tuple[int, int], ...],
    ) -> tuple[int, ...]:
        return tuple(
            (xi + (li + ri) - ki) // si + 1
            for xi, ki, si, (li, ri) in safe_zip(shape, kernel_size, strides, border)
        )

    @ft.partial(jax.profiler.annotate_function, name="general_arange")
    def general_arange(di: int, ki: int, si: int, x0: int, xf: int) -> jax.Array:
        # this function is used to calculate the windows indices for a given dimension
        start, end = -x0 + ((ki - 1) // 2), di + xf - (ki // 2)
        size = end - start

        res = (
            jax.lax.broadcasted_iota(dtype=jnp.int32, shape=(size, ki), dimension=0)
            + jax.lax.broadcasted_iota(dtype=jnp.int32, shape=(ki, size), dimension=0).T
            + (start)
            - ((ki - 1) // 2)
        )

        # res[::si] is slightly slower.
        return (res) if si == 1 else (res)[::si]

    @ft.lru_cache(maxsize=128)
    def recursive_vmap(*, ndim: int):
        def nvmap(n):
            in_axes = [None] * ndim
            in_axes[-n] = 0
            return (
                jax.vmap(lambda *x: x, in_axes=in_axes)
                if n == 1
                else jax.vmap(nvmap(n - 1), in_axes=in_axes)
            )

        return nvmap(ndim)

    @ft.partial(jax.profiler.annotate_function, name="general_product")
    def general_product(*args: jax.Array):
        return recursive_vmap(ndim=len(args))(*args)

    def generate_views(
        shape: tuple[int, ...],
        kernel_size: tuple[int, ...],
        strides: tuple[int, ...],
        border: tuple[tuple[int, int], ...],
    ) -> tuple[jax.Array, ...]:
        dim_range = tuple(
            general_arange(di, ki, si, x0, xf)
            for (di, ki, si, (x0, xf)) in zip(shape, kernel_size, strides, border)
        )
        matrix = general_product(*dim_range)
        return tuple(map(lambda xi, wi: xi.reshape(-1, wi), matrix, kernel_size))

    def absolute_wrapper(*a, **k):
        def map_func(view: tuple[jax.Array, ...], array: jax.Array):
            patch = array.at[ix_(*view)].get(**gather_kwargs)
            return func(patch, *a, **k)

        return map_func

    def ix_(*args):
        """modified version of jnp.ix_"""
        n = len(args)
        output = []
        for i, a in enumerate(args):
            shape = [1] * n
            shape[i] = a.shape[0]
            output.append(jax.lax.broadcast_in_dim(a, shape, (i,)))
        return tuple(output)

    pad_width = tuple([0, max(0, pi[0]) + max(0, pi[1])] for pi in padding)
    args = (shape, kernel_size, strides, padding)
    views = generate_views(*args)
    output_shape = calculate_kernel_map_output_shape(*args)

    def single_call_wrapper(array: jax.Array, *a, **k):
        padded_array = jnp.pad(array, pad_width, **pad_kwargs)
        reduced_func = absolute_wrapper(*a, **k)

        def map_func(view):
            return reduced_func(view, padded_array)

        result = jax.vmap(map_func)(views)
        return result.reshape(*output_shape, *result.shape[1:])

    return single_call_wrapper


def single_dispatch(argnum: int = 0):
    """Single dispatch with argnum"""

    def decorator(func):
        dispatcher = ft.singledispatch(func)

        @ft.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                klass = type(args[argnum])
            except IndexError:
                argname = get_params(func)[argnum].name
                klass = type(kwargs[argname])
            return dispatcher.dispatch(klass)(*args, **kwargs)

        wrapper.def_type = dispatcher.register
        wrapper.registry = dispatcher.registry
        ft.update_wrapper(wrapper, func)
        return wrapper

    return decorator


"""This module provides decorators to handle lazy layers in a functional way.


Creating a _lazy_ ``Linear`` layer example:

In this example, we create a _lazy_ ``Linear`` that initializes the weights
and biases based on the input. The ``__init__`` method is decorated with 
``maybe_lazy_init`` with the condition that if ``in_features`` is ``None`` 
then the layer is lazy. and the ``__call__`` method is decorated with 
``maybe_lazy_call`` with the condition that if the instance ``in_features`` 
is ``None`` then the layer is lazy. addditionally, we define an update function 
that infers the ``in_features`` from the input shape and updates the 
``in_features`` attribute to then 
re-initialize the layer with the inferred ``in_features``.

One benefit of this approach is that we can use the layer as a lazy layer
or a materialized layer without changing the code. This is useful to 
translate code from both explicit and implicit shaped layer found in 
libraries like ``pytorch`` and ``tensorflow``.

As quick sketch how this work is in the following example:

>>> import jax
>>> class Lazy:
...     def __init__(self, dim_size: int | None):
...         # let dim size be the array size
...         # and if we dont have the array size
...         # we can set it to None to be inferred later
...         self.dim_size = dim_size
...     def __call__(self, x):
...         return x * self.dim_size
>>> def maybe_lazy_init(func):
...     def wrapper(self, dim_size):
...         if input is not None:
...             return func(self, dim_size)
...         # we do not execute the init function
...         # because its lazy
...         return None
...     return wrapper
>>> def maybe_lazy_call(func):
...     def wrapper(self, x):
...         if self.dim_size is not None:
...             return func(self, x)
...         # the input is lazy , so we do infer the dim size
...         # here. because `TreeClass` is immutable we need to
...         # return a new instance of the class with the updated
...         # dim size, but here we are just updating the dim size
...         # of the current instance that is not immutable
...         self.dim_size = x.size
...         return func(self, x)
...     return wrapper
>>> # now lets decorate our lazy class
>>> Lazy.__init__ = maybe_lazy_init(Lazy.__init__)
>>> Lazy.__call__ = maybe_lazy_call(Lazy.__call__)
>>> print(Lazy(2)(jax.numpy.ones([2])))
>>> print(Lazy(None)(jax.numpy.ones([2])))


Now lets create a lazy ``Linear`` layer using ``serket``:

>>> import functools as ft
>>> import serket as sk
>>> import jax.numpy as jnp
>>> from serket._src.utils import maybe_lazy_call, maybe_lazy_init
>>> def is_lazy_init(self, in_features, out_features):
...    # we need to define how to tell if the layer is lazy
...    # based on the inputs
...    return in_features is None  # or anything else really
>>> def is_lazy_call(self, x):
...    # we need to define how to tell if the layer is lazy
...    # at the call time
...    # replicating the lazy init condition
...    return getattr(self, "in_features", False) is None
>>> def infer_in_features(self, x):
...    # we need to define how to infer the in_features
...    # based on the inputs at call time
...    # for linear layers, we can infer the in_features as the last dimension
...    return x.shape[-1]
>>> # lastly we need to assign this function to a dictionary that has the name
>>> # of the feature we want to infer
>>> updates = dict(in_features=infer_in_features)
>>> class SimpleLinear(sk.TreeClass):
...    @ft.partial(maybe_lazy_init, is_lazy=is_lazy_init)
...    def __init__(self, in_features, out_features):
...        self.in_features = in_features
...        self.out_features = out_features
...        self.weight = jnp.ones((in_features, out_features))  # dummy weight
...        self.bias = jnp.zeros((out_features,))  # dummy bias
>>>    @ft.partial(maybe_lazy_call, is_lazy=is_lazy_call, updates=updates)
...    def __call__(self, x):
...        return x
>>> simple_lazy = SimpleLinear(None, 1)
>>> x = jnp.ones([10, 2])  # last dimension is the in_features of the layer
>>> print(repr(simple_lazy))
SimpleLinear(in_features=None, out_features=1)
>>> _, material = sk.value_and_tree(lambda layer: layer(x))(simple_lazy)
>>> print(repr(material))
SimpleLinear(
  in_features=2, 
  out_features=1, 
  weight=f32[2,1](μ=1.00, σ=0.00, ∈[1.00,1.00]), 
  bias=f32[1](μ=0.00, σ=0.00, ∈[0.00,0.00])
)
"""


# Maybe expose this as a public API
# Handling lazy layers


def maybe_lazy_init(
    func: Callable[P, T],
    is_lazy: Callable[..., bool],
) -> Callable[P, T]:
    """Sets input arguments to instance attribute if lazy initialization is ``True``.

    The key idea is to store the input arguments to the instance attribute to
    be used later when the instance is re-initialized using ``maybe_lazy_call``
    decorator. ``maybe_lazy_call`` assumes that the input arguments are stored
    in the instance attribute and can be retrieved using ``vars(instance)``.
    Meaning that upon re-initialization, ``obj.__init__(**vars(obj))`` will
    re-initialize the instance with the same input arguments.

    Args:
        func: The ``__init__`` method of a class.
        is_lazy: A function that returns ``True`` if lazy initialization is ``True``.
            the function accepts the same arguments as ``func``.

    Returns:
        The decorated ``__init__`` method.
    """

    def inner(instance, *a, **k):
        if not is_lazy(instance, *a, **k):
            # continue with the original initialization
            return func(instance, *a, **k)

        # store the input arguments to the instance
        # until the instance is re-initialized (materialized)
        # then use the stored arguments to re-initialize the instance
        kwargs: dict[str, Any] = dict()

        for index, param in enumerate(get_params(func)[1:]):
            # skip the self argument
            if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                # fetch from the positional arguments
                # or the keyword arguments or the default value if exists
                if len(a) > index:
                    # fetch from the positional arguments if available
                    kwargs[param.name] = a[index]
                elif param.name in k:
                    # fetch from the keyword arguments
                    kwargs[param.name] = k[param.name]
                elif param.default is not inspect.Parameter.empty:
                    # fetch from the default value if exists
                    kwargs[param.name] = param.default

            elif param.kind is inspect.Parameter.KEYWORD_ONLY:
                # fetch from the keyword arguments
                # or the default value
                if param.name in k:
                    # fetch from the keyword arguments if exists
                    kwargs[param.name] = k[param.name]
                elif param.default is not inspect.Parameter.empty:
                    # fetch from the default value if exists
                    kwargs[param.name] = param.default
            else:
                # dont support positional only arguments, etc.
                # not to complicate things
                raise NotImplementedError(f"{param.kind=}")

        for key, value in kwargs.items():
            # set the attribute to the instance
            # these will be reused to re-initialize the instance
            # after the first call
            setattr(instance, key, value)

        # halt the initialization of the instance
        # and move to the next call
        return None

    return ft.wraps(func)(inner)


LAZY_CALL_ERROR = """\
Cannot call ``{func_name}`` directly on a lazy layer.
use ``value_and_tree(lambda layer: layer{func_name}(...))(layer)`` instead to return a tuple of:
    - Layer output.
    - Materialized layer.

Example:
    >>> layer = {class_name}(...)
    >>> layer(input)  # this will raise an error
    ...
    
    Instead use the following pattern:

    >>> output, material = value_and_tree(lambda layer: layer{func_name}(input))(layer)
    >>> material(input)
    ...
"""


def maybe_lazy_call(
    func: Callable[P, T],
    is_lazy: Callable[..., bool],
    updates: dict[str, Callable[..., Any]],
) -> Callable[P, T]:
    """Reinitialize the instance if it is lazy.

    Accompanying decorator for ``maybe_lazy_init``.

    Args:
        func: The method to decorate that accepts the arguments needed to re-initialize
            the instance.
        is_lazy: A function that returns ``True`` if lazy initialization is ``True``.
            the function accepts the same arguments as ``func``.
        updates: A dictionary of updates to the instance attributes. this dictionary
            maps the attribute name to a function that accepts the attribute value
            and returns the updated value. the function accepts the same arguments
            as ``func``.
    """

    @ft.wraps(func)
    def inner(instance, *a, **k):
        if not is_lazy(instance, *a, **k):
            return func(instance, *a, **k)

        # the instance variables are the input arguments
        # to the ``__init__`` method
        kwargs = dict(vars(instance))

        for key, update in updates.items():
            kwargs[key] = update(instance, *a, **k)

        try:
            for key in kwargs:
                # clear the instance information (i.e. the initial input arguments)
                # use ``delattr`` to raise an error if the instance is immutable
                # which is marking the instance as lazy and immutable
                delattr(instance, key)
        except AttributeError:
            # the instance is lazy and immutable
            func_name = func.__name__
            func_name = "" if func_name == "__call__" else f".{func_name}"
            class_name = type(instance).__name__
            kwargs = dict(func_name=func_name, class_name=class_name)
            raise RuntimeError(LAZY_CALL_ERROR.format(**kwargs))

        # re-initialize the instance with the resolved arguments
        # this will only works under `value_and_tree` that allows
        # the instance to be mutable with it's context after being copied first
        type(instance).__init__(instance, **kwargs)
        # call the decorated function
        return func(instance, *a, **k)

    return inner
