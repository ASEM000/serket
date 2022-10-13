from __future__ import annotations

import functools as ft
from typing import Callable

import jax
import jax.numpy as jnp


def _generate_central_offsets(
    derivative: int, accuracy: int = 2
) -> tuple[float | int, ...]:
    """Generate central difference offsets

    Args:
        derivative (int): derivative order

    Returns:
        tuple[float | int, ...]: central difference offsets
    """
    if derivative < 1:
        raise ValueError(f"derivative must be >= 1, got {derivative}")
    if accuracy < 2:
        raise ValueError(f"accuracy must be >= 2, got {accuracy}")

    return tuple(
        range(-((derivative + accuracy - 1) // 2), (derivative + accuracy - 1) // 2 + 1)
    )


@ft.partial(jax.jit, static_argnums=(1,))
def generate_finitediff_coeffs(
    offsets: tuple[float | int, ...], derivative: int
) -> tuple[float]:
    """Generate FD coeffs

    Args:
        offsets (tuple[float | int, ...]): offsets of the finite difference stencil
        derivative (int): derivative order

    Returns:
        tuple[float]: finite difference coefficients

    Example:
        >>> generate_finitediff_coeffs(offsets=(-1, 0, 1), derivative=1)
        (-0.5, 0.0, 0.5)

        >>> generate_finitediff_coeffs(offsets=(-1, 0, 1), derivative=2)
        (1.0, -2.0, 1.0)
        # translates to  1*f(x-1) - 2*f(x) + 1*f(x+1)

    See: https://en.wikipedia.org/wiki/Finite_difference_coefficient
    """

    N = len(offsets)

    if derivative >= N:
        raise ValueError(
            "Sampling points must be larger than derivative order."
            f" len(offsets)={len(offsets)} must be less than {derivative}"
        )

    A = jnp.repeat(jnp.array(offsets)[None, :], repeats=N, axis=0)
    A **= jnp.arange(0, N).reshape(-1, 1)
    index = jnp.arange(N)
    factorial = jnp.prod(jnp.arange(1, derivative + 1))
    B = jnp.where(index == derivative, factorial, 0)[:, None]
    C = jnp.linalg.inv(A) @ B  # solve Ax = B
    return C.flatten()


def fdiff(
    func: Callable,
    *,
    argnum: int = 0,
    step_size: float = None,
    offsets: tuple[float | int, ...] = None,
    derivative: int = 1,
    **kwargs,
) -> Callable:
    """Finite difference derivative of a function

    Args:
        func (Callable): function to differentiate
        argnum (int, optional): argument number to differentiate. Defaults to 0.
        step_size (float, optional): step size for the finite difference stencil. Defaults to None.
        offsets (tuple[float | int, ...], optional): offsets for the finite difference stencil. Defaults to None.
        derivative (int, optional): derivative order. Defaults to 1.

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
        offsets = _generate_central_offsets(derivative, accuracy=2)

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
