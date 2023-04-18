from __future__ import annotations

import functools as ft
import inspect
from typing import Any, Callable, TypeVar

import jax.tree_util as jtu
from jax.core import concrete_or_error
from pytreeclass._src.tree_indexer import _mutable_context

T = TypeVar("T")


class Lazy:
    def __repr__(self):
        return "Lazy"


LAZY = Lazy()

LAZY_KW = "_lazy_init"
_lazy_init_registry: dict[int, Callable] = {}


class LazyPartial(ft.partial):
    def __call__(self, *args, **keywords) -> Callable:
        # https://stackoverflow.com/a/7811270
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is LAZY else arg for arg in self.args)  # type: ignore
        return self.func(*args, *iargs, **keywords)


def is_lazy(x: Any) -> bool:
    return id(x) in _lazy_init_registry


def get_lazy_init_entry(id: Any) -> Callable:
    return _lazy_init_registry.get(id, None)


def lazy_class(
    klass: type[T],
    *,
    is_lazy: Callable[[str, Any], bool],
    infer_func: Callable,
    hook_name: str = "__call__",
) -> type[T]:
    """a decorator that allows for lazy initialization of a class by wrapping the init method
    to allow for partialization of the lazy arguments, and wrapping the call method to allow for
    inference of the lazy arguments at call time.

    Args:
        klass: the class to wrap
        is_lazy: a function that takes the name of the argument and the value of the argument and
            returns True if the argument is lazy
        infer_func: the function that is applied to the input of `infer_method` to infer the lazy arguments
            should return a tuple of the same length as `lazy_keywords`
        hook_name: the method name in which `infer_func` is applied to infer the lazy arguments
    """
    # in essence we are trying to infer some value from the input of the call method to fully initialize the class
    # till then, we store the partialized init function and call it in the call method once we have
    # the input to infer the lazy arguments. However, this apporach must respect the jax transformations, so we
    # make sure that the initialized class does not undergo any transformations before the the class is fully
    # initialized. This is done by checking that the input to the call method is not a Tracer.

    init_sig = inspect.signature(klass.__init__)
    params = list(init_sig.parameters.values())[1:]  # skip self

    def lazy_init(init_func: Callable) -> Callable:
        @ft.wraps(init_func)
        def wrapper(self, *args, **kwargs):
            margs, mkwargs = list(), dict()

            for i, param in enumerate(params):
                # mask args and kwargs for partialization of the init function
                name, default, kind = param.name, param.default, param.kind

                if kind == param.POSITIONAL_ONLY:
                    value = args[i] if len(args) > i else default
                    margs += [LAZY] if is_lazy(name, value) else [value]
                elif kind == param.POSITIONAL_OR_KEYWORD and len(args) > i:
                    value = args[i] if len(args) > i else kwargs.get(name, default)
                    margs += [LAZY] if is_lazy(name, value) else [value]
                else:
                    value = kwargs.get(name, default)
                    mkwargs[name] = LAZY if is_lazy(name, value) else value

            if LAZY in margs or LAZY in mkwargs.values():
                for key in self._fields:
                    # temporarily populate missing fields to the instance to
                    # avoid AttributeError: Uninitialized fields
                    vars(self)[key] = LAZY

                partial_init = LazyPartial(init_func, self, *margs, **mkwargs)
                _lazy_init_registry[id(self)] = partial_init
                return

            return init_func(self, *args, **kwargs)

        return wrapper

    def lazy_call(call_func):
        @ft.wraps(call_func)
        def wrapper(self, *args, **kwargs):
            if id(self) not in _lazy_init_registry:
                return call_func(self, *args, **kwargs)

            msg = (
                f"Using Tracers as input to a lazy layer is not supported. "
                "Use non-Tracer input to initialize the layer.\n"
                "This error can occur if jax transformations are applied to a layer before "
                "calling it with a non Tracer input.\n"
                "Example: \n"
                ">>> # This will fail\n"
                ">>> x = jax.numpy.ones(...)\n"
                f">>> layer = {type(self).__name__}(None, ...)\n"
                ">>> layer = jax.jit(layer)\n"
                ">>> layer(x) \n"
                ">>> # Instead, first initialize the layer with a non Tracer input\n"
                ">>> # and then apply jax transformations\n"
                f">>> layer = {type(self).__name__}(None, ...)\n"
                ">>> layer(x) # dry run to initialize the layer\n"
                ">>> layer = jax.jit(layer)\n"
            )

            # per: https://github.com/google/jax/issues/15625
            # prevent jax transformations from being applied to the layer
            # before the layer is fully initialized, otherwise tracer leaks will occur
            jtu.tree_map(lambda arg: concrete_or_error(None, arg, msg), (args, kwargs))

            # we are lazy, so we can call the original call function
            # we are forwarded from the init method, so we need
            # to infer the lazy arguments and call the init method
            # the self input here is the self path
            partial_func = _lazy_init_registry.pop(id(self))

            # get the inferred arguments
            output = infer_func(self, *args, **kwargs)

            # in essence, we need to decide how to merge the output with the masked args and kwargs
            fargs, fkwargs = partial_func.args, partial_func.keywords
            lazy_args = [None for arg in fargs if arg is LAZY]
            lazy_kwargs = {k: None for k in fkwargs if fkwargs[k] is LAZY}

            keys = list(lazy_kwargs.keys())

            for i, item in enumerate(output):
                # the output of infer func should be a tuple for each lazy arg
                # merge the output with the masked args and kwargs
                if i < len(lazy_args):
                    lazy_args[i] = item  # handle args first

                elif (index := i - len(lazy_args)) < len(keys):
                    lazy_kwargs[keys[index]] = item  # handle kwargs next

            with _mutable_context(self):
                # since we are calling the init method, we need to be within the mutable context
                # to allow the init method to mutate the instance
                partial_func(*lazy_args, **lazy_kwargs)
            return call_func(self, *args, **kwargs)

        return wrapper

    for name, wrapper in (("__init__", lazy_init), (hook_name, lazy_call)):
        setattr(klass, name, wrapper(getattr(klass, name)))

    return klass


class LazyInFeatures:
    """A lazy layer that infers the in_features argument from the input shape at call time"""

    def __init_subclass__(klass: type[T]) -> None:
        super().__init_subclass__()
        lazy_class(
            klass,
            hook_name="__call__",
            infer_func=lambda _, x, *a, **k: (x.shape[0],),
            is_lazy=lambda name, value: (name == "in_features" and value is None),
        )
