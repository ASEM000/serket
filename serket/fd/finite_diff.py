from __future__ import annotations

import functools as ft
from typing import Callable

import jax
import jax.numpy as jnp

from .utils import _generate_central_offsets, generate_finitediff_coeffs


@ft.partial(jax.jit, static_argnames=("accuracy", "derivative"))
def fdiff(
    func: Callable,
    *,
    argnum: int = 0,
    step_size: float = None,
    offsets: tuple[float | int, ...] = None,
    derivative: int = 1,
    accuracy: int = 2,
) -> Callable:
    """Finite difference derivative of a function

    Args:
        func: function to differentiate
        argnum: argument number to differentiate. Defaults to 0.
        step_size: step size for the finite difference stencil. Defaults to None.
        offsets: offsets for the finite difference stencil. Defaults to None.
        derivative: derivative order. Defaults to 1.
        accuracy: accuracy of the finite difference stencil. Defaults to 2. used to generate offsets if not provided.

    Returns:
        Callable: derivative of the function

    Example:
        >>> def f(x):
        ...     return x**2
        >>> df = fdiff(f)
        >>> df(2.0)
        DeviceArray(4., dtype=float32)
    """

    if not isinstance(argnum, int) or argnum < 0:
        raise ValueError(f"argnum must be a non-negative integer, got {argnum}")

    if derivative < 1 or not isinstance(derivative, int):
        raise ValueError(f"derivative must be a positive integer, got {derivative}")

    if offsets is None:
        if accuracy < 2:
            raise ValueError(f"accuracy must be >= 2, got {accuracy}")

        offsets = _generate_central_offsets(derivative, accuracy=accuracy)

    if step_size is None:
        step_size = 1e-3 * (10 ** (derivative))

    # finite difference coefficients
    coeffs = generate_finitediff_coeffs(offsets, derivative)

    # infinitisimal step size along the axis
    DX = jnp.array(offsets) * step_size

    def diff_func(*args, **kwargs):
        # evaluate function at shifted points
        for coeff, dx in zip(coeffs, DX):
            shifted_args = list(args)
            shifted_args[argnum] += dx
            yield coeff * func(*shifted_args, **kwargs) / (step_size**derivative)

    return lambda *args, **kwargs: sum(diff_func(*args, **kwargs))
