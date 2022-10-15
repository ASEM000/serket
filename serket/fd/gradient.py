from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import pytreeclass as pytc

from .utils import (
    _generate_backward_offsets,
    _generate_central_offsets,
    _generate_forward_offsets,
    generate_finitediff_coeffs,
)


@ft.partial(jax.jit, static_argnames=("accuracy", "axis", "derivative"))
def gradient(
    x: jnp.ndarray,
    *,
    axis: int = 0,
    accuracy: int = 1,
    step_size: float = 1,
    derivative: int = 1,
) -> jnp.ndarray:
    """Compute the gradient along a given axis
    Similar to np.gradient, but with the option to specify accuracy, derivative and step size
    See: https://github.com/google/jax/blob/main/jax/_src/numpy/lax_numpy.py

    The function add support for

    Args:
        x: input array
        axis: axis along which to compute the gradient. Default is 0
        accuracy: accuracy order of the gradient. Default is 1
        step_size: step size. Default is 1
        derivative: derivative order of the gradient. Default is 1
    Returns:
        gradient: gradient along the given axis

    Example:
        # dydx of a 2D array
        >>> x, y = [jnp.linspace(0, 1, 100)] * 2
        >>> dx, dy = x[1] - x[0], y[1] - y[0]
        >>> X, Y = jnp.meshgrid(x, y, indexing="ij")
        >>> F =  jnp.sin(X) * jnp.cos(Y)
        >>> dFdX = gradient(F, step_size=dx, axis=0, accuracy=3)
        >>> dFdXdY = gradient(dFdX, step_size=dy, axis=1, accuracy=3)

    """
    size = x.shape[axis]

    left_offsets = _generate_forward_offsets(derivative, accuracy)
    left_coeffs = generate_finitediff_coeffs(left_offsets, derivative)

    right_offsets = _generate_backward_offsets(derivative, accuracy)
    right_coeffs = generate_finitediff_coeffs(right_offsets, derivative)

    center_offsets = _generate_central_offsets(derivative, accuracy + 1)
    center_coeffs = generate_finitediff_coeffs(center_offsets, derivative)

    sliced = ft.partial(jax.lax.slice_in_dim, x, axis=axis)

    if accuracy > (size // 2 - 1):
        raise ValueError(f"accuracy must be <= {(size//2-1)}, got {accuracy}")

    left_x = sum(
        coeff * sliced(offset, offset - center_offsets[0])
        for offset, coeff in zip(left_offsets, left_coeffs)
    )

    right_x = sum(
        coeff * sliced(size + (offset - center_offsets[-1]), size + (offset))
        for offset, coeff in zip(right_offsets, right_coeffs)
    )

    center_x = sum(
        coeff * sliced(offset - center_offsets[0], size + (offset - center_offsets[-1]))
        for offset, coeff in zip(center_offsets, center_coeffs)
    )

    return jnp.concatenate([left_x, center_x, right_x], axis=axis) / (
        step_size**derivative
    )


@pytc.treeclass
class Gradient:
    axis: int = pytc.nondiff_field()
    accuracy: int = pytc.nondiff_field()
    step_size: float = pytc.nondiff_field()
    derivative: int = pytc.nondiff_field()

    def __init__(
        self,
        *,
        axis=0,
        accuracy=1,
        step_size=1,
        derivative=1,
    ):
        """Compute the gradient along a given axis as a Layer
        Similar to np.gradient, but with the option to specify accuracy, derivative and step size
        See: https://github.com/google/jax/blob/main/jax/_src/numpy/lax_numpy.py

        Args:
            axis: axis along which to compute the gradient. Default is 0
            accuracy: accuracy order of the gradient. Default is 1
            step_size: step size. Default is 1
            derivative: derivative order of the gradient. Default is 1

        Returns:
            gradient: gradient along the given axis
        """
        self.axis = axis
        self.accuracy = accuracy
        self.step_size = step_size
        self.derivative = derivative

        self._func = jax.jit(
            ft.partial(
                gradient,
                axis=self.axis,
                accuracy=self.accuracy,
                step_size=self.step_size,
                derivative=self.derivative,
            )
        )

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self._func(x, **kwargs)
