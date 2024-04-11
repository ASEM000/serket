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

from __future__ import annotations

import functools as ft

from serket._src.utils.inspect import get_params


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
                try:
                    klass = type(kwargs[argname])
                except KeyError:
                    raise TypeError(f"{func.__name__} missing required {argname=}")
            return dispatcher.dispatch(klass)(*args, **kwargs)

        wrapper.def_type = dispatcher.register
        wrapper.registry = dispatcher.registry
        ft.update_wrapper(wrapper, func)
        return wrapper

    return decorator
