from __future__ import annotations

import functools as ft
import inspect
from typing import Any, Callable, Sequence, TypeVar

import jax

T = TypeVar("T")


_lazy_placeholder = object()
LAZY_KW = "__lazy_init__"
_uninitialized = type("Uninitialized", (), {"__repr__": lambda _: "Uninitialized"})()


class LazyPartial(ft.partial):
    def __call__(self, *args, **keywords) -> Callable:
        # https://stackoverflow.com/a/7811270
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is _lazy_placeholder else arg for arg in self.args)  # type: ignore
        return self.func(*args, *iargs, **keywords)


def lazy_class(
    klass: type[T],
    *,
    lazy_keywords: Sequence[str],
    infer_func: Callable,
    infer_method_name: str = "__call__",
    lazy_marker: Any = None,
) -> type[T]:
    """a decorator that allows for lazy initialization of a class by wrapping the init method
    to allow for partialization of the lazy arguments, and wrapping the call method to allow for
    inference of the lazy arguments at call time.

    Args:
        klass: the class to wrap
        lazy_keywords: the keywords that are lazy evaluated
        infer_func:
            the function that is applied to the input of `infer_method` to infer the lazy arguments
            should return a tuple of the same length as `lazy_keywords`
        infer_method_name: the method name in which `infer_func` is applied to infer the lazy arguments
        lazy_marker: the marker that is used to indicate a lazy argument. e.x. `None`

    Example:
        >>> import functools as ft
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import pytreeclass as pytc
        >>> @ft.partial(
        ...    lazy_class,
        ...    lazy_keywords=["in_features"],  # -> `in_features` is lazy evaluated
        ...    infer_func=lambda self, x: (x.shape[-1],),  # -> `in_features` is inferred from `x` last dim at
        ...    infer_method_name="__call__",  # -> `infer_func` is applied to `__call__` method
        ...    lazy_marker=None,  # -> `None` is used to indicate a lazy argument
        ... )
        ... @pytc.treeclass
        ... class LazyLinear:
        ...    weight: jax.Array
        ...    bias: jax.Array
        ...    def __init__(self, in_features: int, out_features: int):
        ...        self.in_features = in_features
        ...        self.out_features = out_features
        ...        self.weight = jax.random.normal(
        ...            jax.random.PRNGKey(0), (in_features, out_features)
        ...        )
        ...        self.bias = jax.random.normal(jax.random.PRNGKey(0), (out_features,))
        ...    def __call__(self, x):
        ...        return x @ self.weight + self.bias

        >>> layer = LazyLinear(None, 20)  # -> `in_features` is lazy marked as `None` and inferred at call time
        >>> x = jnp.ones([10, 1])  # `in_features` is inferred from `x` last dim=1 at call time
        >>> assert layer(x).shape == (10, 20)
    """

    init_sig = inspect.signature(klass.__init__)
    non_self_params = list(init_sig.parameters.values())[1:]

    def is_marked(item: Any) -> bool:
        if isinstance(item, (tuple, list)):
            return any(i is lazy_marker for i in item)
        return item is lazy_marker

    def check_tracer_input(args, *, name: str = None):
        msg = (
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

        for arg in args:
            if isinstance(arg, jax.core.Tracer):
                raise ValueError(msg)
        return args

    def lazy_init(init_func: Callable) -> Callable:
        @ft.wraps(init_func)
        def wrapper(self, *args, **kwargs):
            is_lazy_names = []
            masked_args = []
            masked_kwargs = dict()

            # first pass to check if any of the lazy keywords are None
            for i, param in enumerate(non_self_params):
                # fetch the value and check if it's None
                if param.kind == param.POSITIONAL_ONLY:
                    # value must be in args or in defaults
                    value = args[i] if len(args) > i else param.default
                    is_lazy_names += [param.name in lazy_keywords and is_marked(value)]
                    masked_args += [_lazy_placeholder] if is_lazy_names[-1] else [value]
                elif param.kind == param.POSITIONAL_OR_KEYWORD and len(args) > i:
                    # value might be in args or kwargs if in `POSITIONAL_OR_KEYWORD`
                    # so we double check the length of args to make sure we don't fetch from kwargs
                    value = args[i] if len(args) > i else kwargs.get(param.name, param.default)  # fmt: skip
                    is_lazy_names += [param.name in lazy_keywords and is_marked(value)]
                    masked_args += [_lazy_placeholder] if is_lazy_names[-1] else [value]
                else:
                    # value must be in kwargs or in defaults
                    value = kwargs[param.name] if param.name in kwargs else param.default  # fmt: skip
                    is_lazy_names += [param.name in lazy_keywords and is_marked(value)]
                    masked_kwargs[param.name] = (_lazy_placeholder if is_lazy_names[-1] else value)  # fmt: skip

            # partialize the init function
            partial_init = LazyPartial(init_func, self, *masked_args, **masked_kwargs)

            if True in is_lazy_names:
                # store the partialized init function in the instance
                vars(self)[LAZY_KW] = partial_init
                # return to the call method
                return
            else:
                # not lazy, so delete the lazy func
                del partial_init

            if hasattr(self, LAZY_KW):
                # we came from the call method, so we can delete the lazy
                # partialized init function
                check_tracer_input(args, name=init_func.__qualname__)

                del vars(self)[LAZY_KW]
            # call the original init function
            return init_func(self, *args, **kwargs)

        return wrapper

    def lazy_call(call_func):
        @ft.wraps(call_func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, LAZY_KW):
                return call_func(self, *args, **kwargs)

            # we are lazy, so we can call the original call function
            # we are forwarded from the init method, so we need
            # to infer the lazy arguments and call the init method
            # the self input here is the self path
            partial_func = getattr(self, LAZY_KW)

            # check if the input is a Tracer
            # in essence since lazy modifies the init method then we need to
            # to raise an error if the input is a Tracer. as this mutation is not
            # compatible with jax transformations
            args = check_tracer_input(args, name=type(self).__name__)

            # get the inferred arguments
            output = infer_func(self, *args, **kwargs)

            # in essence, we need to decide how to merge the output with the masked args and kwargs
            fargs, fkwargs = partial_func.args, partial_func.keywords
            lazy_args = [None for arg in fargs if arg is _lazy_placeholder]
            lazy_kwargs = {k: None for k in fkwargs if fkwargs[k] is _lazy_placeholder}

            keys = list(lazy_kwargs.keys())

            for i, item in enumerate(output):
                # merge the output with the masked args and kwargs
                if i < len(lazy_args):
                    # assume positional args are returned first
                    # by the `infer_func``
                    lazy_args[i] = item
                else:
                    # assume kwargs are returned last
                    index = i - len(lazy_args)
                    if index < len(keys):
                        lazy_kwargs[keys[index]] = item
            partial_func(*lazy_args, **lazy_kwargs)
            return call_func(self, *args, **kwargs)

        return wrapper

    for name, wrapper in (("__init__", lazy_init), (infer_method_name, lazy_call)):
        setattr(klass, name, wrapper(getattr(klass, name)))

    return klass
