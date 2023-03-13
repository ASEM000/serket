from __future__ import annotations

import functools as ft
import inspect
from typing import Any, Callable, Sequence

import jax
import pytreeclass as pytc

_lazy_placeholder = object()
_LAZY_KW = "_lazy_init"


class LazyPartial(ft.partial):
    def __call__(self, *args, **keywords):
        # https://stackoverflow.com/a/7811270
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is _lazy_placeholder else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)


def _lazy_init_wrapper(
    init_func: Callable,
    non_self_params: Sequence[inspect.Parameter],
    lazy_keywords: Sequence[str],
) -> Callable:
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
                is_lazy_names += [param.name in lazy_keywords and value is None]
                masked_args += [_lazy_placeholder] if is_lazy_names[-1] else [value]
            elif param.kind == param.POSITIONAL_OR_KEYWORD and len(args) > i:
                # value might be in args or kwargs if in `POSITIONAL_OR_KEYWORD`
                # so we double check the length of args to make sure we don't fetch from kwargs
                value = args[i] if len(args) > i else param.default
                is_lazy_names += [param.name in lazy_keywords and value is None]
                masked_args += [_lazy_placeholder] if is_lazy_names[-1] else [value]
            else:
                # value must be in kwargs or in defaults
                value = kwargs[param.name] if param.name in kwargs else param.default
                is_lazy_names += [param.name in lazy_keywords and value is None]
                masked_kwargs[param.name] = (
                    _lazy_placeholder if is_lazy_names[-1] else value
                )

        # partialize the init function
        partial_init = LazyPartial(init_func, self, *masked_args, **masked_kwargs)

        if True in is_lazy_names:
            # we are lazy, so we need to store the partialized init function
            # somewhere in the instance
            # set all fields to None to mark the instance as lazy
            for field in pytc.fields(self):
                vars(self)[field.name] = None

            # store the partialized init function in the instance
            setattr(self, _LAZY_KW, partial_init)
            # return to the call method
            return
        else:
            # not lazy, so delete the lazy func
            del partial_init

        if hasattr(self, _LAZY_KW):
            # we came from the call method, so we can delete the lazy
            # partialized init function
            del vars(self)[_LAZY_KW]

        # call the original init function
        return init_func(self, *args, **kwargs)

    return wrapper


def _check_non_tracer(args: Any, name: str = "Class") -> Any:
    # check if any of the inputs are Tracers
    if any(isinstance(arg, jax.core.Tracer) for arg in args):
        raise ValueError(
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
    return args


def _lazy_call_wrapper(call_func, infer_func):
    @ft.wraps(call_func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, _LAZY_KW):
            # make sure the user doesn't call the call at this point
            # with `jax` transformations
            # in case we have more than input argument
            # then we need to check if any of them is a tracer
            # tracer means that some `jax` transformation was applied
            args = _check_non_tracer(args, name=type(self).__name__)

            # we are forwarded from the init method, so we need
            # to infer the lazy arguments and call the init method
            partial_func = getattr(self, _LAZY_KW)

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


def lazy_class(klass, lazy_keywords: Sequence[str], infer_func: Callable):
    """
    wrap init and call methods of a class to allow inference of lazy arguments
    at call time
    this is done by wrapping the init method to allow for partialization of the
    lazy arguments, and wrapping the call method to allow for inference of the
    lazy arguments at call time.
    the lazy decorator achieves the following pattern:

    >>> @ft.partial(pytc.treeclass, leafwise=True, indexing=True)
    ... class Test:
    ...    a : int
    ...    b : float
    ...    def __init__(self, a,b):
    ...        if a is None :
    ...            for k in self.__field_map__:
    ...                setattr(self, k, None)
    ...            # partialize all args except for the None ones
    ...            self._lazy_init= ft.partial(type(self).__init__,self=self, b=b)
    ...            return # move on to call method

    ...        self.a = a
    ...        self.b = b

    ...    def __call__(self,x):
    ...        if hasattr(self, "_lazy_init"):
    ...            self._lazy_init(a=x.shape[0])
    ...        return x

    >>> tree = Test(None,2) # Test(a=None, b=None)
    >>> tree(jnp.ones([1]))
    >>> tree # Test(a=1, b=2)

    ** usage **
    >>> lazy_keywords = ["kernel_size"]
    >>> infer_func = lambda self, *args, **kwargs:(args[0].shape[-1],)
    >>> @ft.partial(lazy_class, lazy_keywords=lazy_keywords, infer_func=infer_func)
    ... @ft.partial(pytc.treeclass, leafwise=True, indexing=True)
    ... class Test:
    ...     ...
    """
    init_sig = inspect.signature(klass.__init__)
    non_self_params = list(init_sig.parameters.values())[1:]
    klass.__init__ = _lazy_init_wrapper(klass.__init__, non_self_params, lazy_keywords)
    klass.__call__ = _lazy_call_wrapper(klass.__call__, infer_func)
    return klass
