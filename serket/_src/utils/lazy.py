# Copyright 2024 serket authors
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

"""This module provides decorators to handle lazy layers in a functional way."""

from __future__ import annotations

import functools as ft
import inspect
from typing import Any, Callable, TypeVar

from typing_extensions import ParamSpec

from serket._src.utils.inspect import get_params

P = ParamSpec("P")
T = TypeVar("T")


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
