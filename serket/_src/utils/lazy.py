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

"""This module provides decorators to handle lazy layers in a functional way.

For instance:
>>> net = Linear(None, 10, key=...)
is a lazy layer because the input features are passed as ``None``. The layer
is not materialized yet. To materialize the layer, you can use ``value_and_tree``
to materialize the layer and get the output along with the materialized layer:
>>> output, material = value_and_tree(lambda layer: layer(input))(net)

- Contrary to other framework, `serket` is eager-first lazy-second. This means
  that the lazy is opt-in and not the default behavior.

- The key idea is to store the input arguments to the instance attribute to
  be used later when the instance is re-initialized using ``maybe_lazy_call``
  decorator. ``maybe_lazy_call`` assumes that the input arguments are stored
  in the instance attribute and can be retrieved using ``vars(instance)``.
  Meaning that upon re-initialization, ``obj.__init__(**vars(obj))`` will
  re-initialize the instance with the same input arguments.

- Because the instance is immutable, the process of re-initialization is
  performed under ``value_and_tree`` that allows the instance to be mutable
  with it's context after being copied first.
"""

from __future__ import annotations

import functools as ft
import inspect
from typing import Any, Callable, TypeVar

from typing_extensions import ParamSpec

from serket._src.utils.inspect import get_params

P = ParamSpec("P")
T = TypeVar("T")


def handle_pos_or_kw(param: inspect.Parameter, index: int, args, kwargs):
    if len(args) > index:
        return args[index]
    if param.name in kwargs:
        return kwargs[param.name]
    if param.default is not inspect.Parameter.empty:
        return param.default
    raise TypeError(f"{param.name} is required")


def handle_kw_only(param: inspect.Parameter, index, args, kwargs):
    del index, args
    if param.name in kwargs:
        return kwargs[param.name]
    if param.default is not inspect.Parameter.empty:
        return param.default
    raise TypeError(f"{param.name} is required")


ParamHandler = Callable[[inspect.Parameter, int, tuple[Any, ...], dict[str, Any]], Any]
rules: dict[Any, ParamHandler] = {}
rules[inspect.Parameter.POSITIONAL_OR_KEYWORD] = handle_pos_or_kw
rules[inspect.Parameter.KEYWORD_ONLY] = handle_kw_only


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

    @ft.wraps(func)
    def inner(instance, *a, **k):
        if not is_lazy(instance, *a, **k):
            return func(instance, *a, **k)

        # store the input arguments to the instance
        # until the instance is re-initialized (materialized)
        # then use the stored arguments to re-initialize the instance
        for i, p in enumerate(get_params(func)[1:]):
            setattr(instance, p.name, rules[p.kind](p, i, a, k))

        return None

    return inner


LAZY_ERROR = """\
Cannot call ``{fname}`` directly on a lazy layer.
use ``value_and_tree(lambda layer: layer{fname}(...))(layer)`` instead to return a tuple of:
    - Layer output.
    - Materialized layer.

Example:
    >>> layer = {cname}(...)
    >>> layer(input)  # this will raise an error
    ...
    
    Instead use the following pattern:

    >>> output, material = value_and_tree(lambda layer: layer{fname}(input))(layer)
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
        partial_mapping = dict(vars(instance))

        for key, update in updates.items():
            partial_mapping[key] = update(instance, *a, **k)

        try:
            for key in partial_mapping:
                # clear the instance information (i.e. the initial input arguments)
                # use ``delattr`` to raise an error if the instance is immutable
                # which is marking the instance as lazy and immutable
                delattr(instance, key)
        except AttributeError:
            # the instance is lazy and immutable
            fname = "" if (fname := func.__name__) == "__call__" else f".{fname}"
            cname = type(instance).__name__
            raise RuntimeError(LAZY_ERROR.format(fname=fname, cname=cname))

        # re-initialize the instance with the resolved arguments
        # this will only works under `value_and_tree` that allows
        # the instance to be mutable with it's context after being copied first
        init = getattr(type(instance), "__init__")
        init(instance, **partial_mapping)
        # call the decorated function
        return func(instance, *a, **k)

    return inner
