# credits to Mahmoud Asem 2022 @KAIST
# functions that operate on arrays
# higher accuracy finite difference gradient might require
# setting jax.config.update("jax_enable_x64", True)

from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp
import pytreeclass as pytc

from serket.fd.utils import (
    _generate_backward_offsets,
    _generate_central_offsets,
    _generate_forward_offsets,
    generate_finitediff_coeffs,
)
from serket.nn.utils import _check_and_return


@ft.partial(jax.jit, static_argnames=("accuracy", "axis", "derivative"))
def difference(
    x: jnp.ndarray,
    *,
    axis: int = 0,
    accuracy: int = 1,
    step_size: float | jnp.ndarray = 1,
    derivative: int = 1,
) -> jnp.ndarray:
    """Compute the finite difference derivative along a given axis with a given accuracy
    using central difference for interior points and forward/backward difference for boundary points
    Similar to np.gradient, but with the option to specify accuracy, derivative and step size
    See: https://github.com/google/jax/blob/main/jax/_src/numpy/lax_numpy.py

    Args:
        x: input array
        axis: axis along which to compute the gradient. Default is 0
        accuracy: accuracy order of the gradient. Default is 1
        step_size: step size. Default is 1
        derivative: derivative order of the gradient. Default is 1
    Returns:
        Finite difference derivative along the given axis

    Example:
        # dydx of a 2D array
        >>> x, y = [jnp.linspace(0, 1, 100)] * 2
        >>> dx, dy = x[1] - x[0], y[1] - y[0]
        >>> X, Y = jnp.meshgrid(x, y, indexing="ij")
        >>> F =  jnp.sin(X) * jnp.cos(Y)
        >>> dFdX = difference(F, step_size=dx, axis=0, accuracy=3)
        >>> dFdXdY = difference(dFdX, step_size=dy, axis=1, accuracy=3)

        # 1d finite difference derivative
        >>> x = jnp.array([1.2, 1.3, 2.2, 3., 4.5, 5.5, 6., 7., 8., 20.])
        >>> difference(x, accuracy=1)
        [ 0.0999999  0.5        0.85       1.15       1.25       0.75 0.75       1.         6.5       12.       ]


        # apply forward difference to the first element with accuracy 1
        x_1 = 1.3-1.2 = 0.1

        # apply central difference to interior elements with accuracy 2
        x_2 = (2.2-1.2)/2 = 0.5
        x_3 = (3.-1.3)/2 = 0.85
        x_4 = (4.5-2.2)/2 = 1.15
        x_5 = (5.5-3.)/2 = 1.25
        x_6 = (6.-4.5)/2 = 0.75
        x_7 = (7.-5.5)/2 = 0.75
        x_8 = (8.-6.)/2 = 1.
        x_9 = (20.-7.)/2 = 6.5

        # apply backward difference to the last element with accuracy 1
        x_10 = 20.-8. = 12.
    """
    size = x.shape[axis]
    sliced = ft.partial(jax.lax.slice_in_dim, x, axis=axis)

    left_offsets = _generate_forward_offsets(derivative, accuracy)
    left_coeffs = generate_finitediff_coeffs(left_offsets, derivative)

    right_offsets = _generate_backward_offsets(derivative, accuracy)
    right_coeffs = generate_finitediff_coeffs(right_offsets, derivative)

    center_offsets = _generate_central_offsets(derivative, accuracy + 1)
    center_coeffs = generate_finitediff_coeffs(center_offsets, derivative)

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


@ft.partial(jax.jit, static_argnames=("accuracy"))
def gradient(
    x: jnp.ndarray,
    *,
    accuracy: int | tuple[int, ...] = 1,
    step_size: float | tuple[float, ...] | jnp.ndarry = 1,
) -> jnp.ndarray:
    """Compute the ∇F of input array where F is a scalar function of x and
    returns vectors of the same shape as x stacked along the first axis.

    Args:
        x: input array
        accuracy: accuracy order of the gradient. Default is 1, can be a tuple for each axis
        step_size: step size. Default is 1, can be a tuple for each axis

    Index notation : dF/dxi

    Example:
        # ∇F of a 2D array
        >>> x, y = [jnp.linspace(0, 1, 100)] * 2
        >>> dx, dy = x[1] - x[0], y[1] - y[0]
        >>> X, Y = jnp.meshgrid(x, y, indexing="ij")
        >>> Z = X**2 + Y**3
        >>> dZdX , dZdY = gradient(Z, step_size=(dx,dy))
        >>> dZdx_true, dZdy_true= 2*X , 3*Y**2
        >>> numpy.testing.assert_allclose(dZdx, dZdx_true, atol=1e-4)
        >>> numpy.testing.assert_allclose(dZdy, dZdy_true, atol=1e-4)
    """
    accuracy = _check_and_return(accuracy, x.ndim, "accuracy")
    step_size = _check_and_return(step_size, x.ndim, "step_size")

    return jnp.stack(
        [
            difference(x, accuracy=acc, step_size=step, derivative=1, axis=axis)
            for axis, (acc, step) in enumerate(zip(accuracy, step_size))
        ],
        axis=0,
    )


@ft.partial(jax.jit, static_argnames=("accuracy", "keepdims"))
def divergence(
    x: jnp.ndarray,
    *,
    accuracy: int | tuple[int, ...] = 1,
    step_size: float | tuple[float, ...] = 1,
    keepdims: bool = True,
) -> jnp.ndarray:
    """Compute the ∇.F of input array where F is a vector field whose components are the first axis of x
    and returns a scalar field

    Args:
        x: input array where the leading axis is the dimension of the vector field
        accuracy: accuracy order of the gradient. Default is 1, can be a tuple for each axis
        step_size: step size. Default is 1, can be a tuple for each axis
        keepdims: whether to keep the leading dimension. Default is True.

    Index notation: dFi/dxi

    Example:
        # ∇.F of a 2D array
        >>> x, y = [jnp.linspace(0, 1, 100)] * 2
        >>> dx, dy = x[1] - x[0], y[1] - y[0]
        >>> X, Y = jnp.meshgrid(x, y, indexing="ij")
        >>> F1 = X**2 + Y**3
        >>> F2 = X**4 + Y**3
        >>> F = jnp.stack([F1, F2], axis=0) # 2D vector field F = (F1, F2)
        >>> divZ = divergence(F,step_size=(dx,dy), accuracy=7, keepdims=False)
        >>> divZ_true = 2*X + 3*Y**2  # (dF1/dx) + (dF2/dy)
        >>> numpy.testing.assert_allclose(divZ, divZ_true, atol=5e-4)
    """
    accuracy = _check_and_return(accuracy, x.ndim - 1, "accuracy")
    step_size = _check_and_return(step_size, x.ndim - 1, "step_size")

    result = sum(
        difference(
            x.at[axis].get(), accuracy=acc, step_size=step, derivative=1, axis=axis
        )
        for axis, (acc, step) in enumerate(zip(accuracy, step_size))
    )

    if keepdims:
        return jnp.expand_dims(result, axis=0)
    return result


@ft.partial(jax.jit, static_argnames=("accuracy"))
def laplacian(
    x: jnp.ndarray,
    *,
    accuracy: int | tuple[int, ...] = 1,
    step_size: float | tuple[float, ...] | jnp.ndarry = 1,
) -> jnp.ndarray:
    """Compute the ΔF of input array.
    Args:
        x: input array
        accuracy: accuracy order of the gradient. Default is 1, can be a tuple for each axis
        step_size: step size. Default is 1, can be a tuple for each axis

    Index notation: d2F/dxi2
    Example:
        # ΔF array
        >>> x, y = [jnp.linspace(0, 1, 100)] * 2
        >>> dx, dy = x[1] - x[0], y[1] - y[0]
        >>> X, Y = jnp.meshgrid(x, y, indexing="ij")
        >>> Z = X**4 + Y**3
        >>> laplacianZ = laplacian(Z, step_size=(dx,dy))
        >>> laplacianZ_true = 12*X**2 + 6*Y
        >>> numpy.testing.assert_allclose(laplacianZ, laplacianZ_true, atol=1e-4)
    """
    accuracy = _check_and_return(accuracy, x.ndim, "accuracy")
    step_size = _check_and_return(step_size, x.ndim, "step_size")

    return sum(
        difference(x, accuracy=acc, step_size=step, derivative=2, axis=axis)
        for axis, (acc, step) in enumerate(zip(accuracy, step_size))
    )


@ft.partial(jax.jit, static_argnames=("accuracy"))
def curl(
    x: jnp.ndarray,
    *,
    accuracy: int | tuple[int, ...] = 1,
    step_size: float | tuple[float, ...] | jnp.ndarry = 1,
) -> jnp.ndarray:
    """Compute the ∇×F of input array where F is a vector field whose components are the first axis of x
    and returns a vector field

    Index notation: εijk dFk/dxj

    Args:
        x: input array where the leading axis is the dimension of the vector field
        accuracy: accuracy order of the gradient. Default is 1, can be a tuple for each axis
        step_size: step size. Default is 1, can be a tuple for each axis

    Example:
        Curl for a 3D vector field is defined as
        F = (F1, F2, F3)
        ∇×F = (dF3/dy - dF2/dz, dF1/dz - dF3/dx, dF2/dx - dF1/dy)
        >>> jax.config.update("jax_enable_x64", True)
        >>> x,y,z = [jnp.linspace(0, 1, 100)] * 3
        >>> dx,dy,dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]
        >>> X,Y,Z = jnp.meshgrid(x,y,z, indexing="ij")
        >>> F1 = X**2 + Y**3
        >>> F2 = X**4 + Y**3
        >>> F3 = jnp.zeros_like(F1)
        >>> F = jnp.stack([F1,F2,F3], axis=0)
        >>> curlF = sk.fd.curl(F, step_size=(dx,dy,dz),  accuracy=6)
        >>> curlF_exact = jnp.stack([F1*0,F1*0, 4*X**3 - 3*Y**2], axis=0)
        >>> npt.assert_allclose(curlF, curlF_exact, atol=1e-7)
    """

    accuracy = _check_and_return(accuracy, x.ndim - 1, "accuracy")
    step_size = _check_and_return(step_size, x.ndim - 1, "step_size")

    if not (x.ndim == 4 and x.shape[0] == 3):
        raise ValueError(
            "Input array must be composed of 3 vector fields of 3D shape (3, nx, ny, nz)"
        )

    dF1dY = difference(x[0], accuracy=accuracy[1], step_size=step_size[1], axis=1)
    dF1dZ = difference(x[0], accuracy=accuracy[2], step_size=step_size[2], axis=2)

    dF2dX = difference(x[1], accuracy=accuracy[0], step_size=step_size[0], axis=0)
    dF2dZ = difference(x[1], accuracy=accuracy[2], step_size=step_size[2], axis=2)

    dF3dX = difference(x[2], accuracy=accuracy[0], step_size=step_size[0], axis=0)
    dF3dY = difference(x[2], accuracy=accuracy[1], step_size=step_size[1], axis=1)

    return jnp.stack(
        [
            dF3dY - dF2dZ,
            dF1dZ - dF3dX,
            dF2dX - dF1dY,
        ],
        axis=0,
    )


@pytc.treeclass
class Difference:
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
        """wrap difference as a layer"""
        self.axis = axis
        self.accuracy = accuracy
        self.step_size = step_size
        self.derivative = derivative

        self._func = ft.partial(
            difference,
            axis=self.axis,
            accuracy=self.accuracy,
            step_size=self.step_size,
            derivative=self.derivative,
        )

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self._func(x)


@pytc.treeclass
class Gradient:
    accuracy: int | tuple[int, ...] = pytc.nondiff_field()
    step_size: float | tuple[float, ...] = pytc.nondiff_field()

    def __init__(
        self,
        *,
        accuracy=1,
        step_size=1,
    ):
        """wrap gradient as a layer"""
        self.accuracy = accuracy
        self.step_size = step_size

        self._func = ft.partial(
            gradient,
            accuracy=self.accuracy,
            step_size=self.step_size,
        )

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self._func(x)


@pytc.treeclass
class Divergence:
    accuracy: int | tuple[int, ...] = pytc.nondiff_field()
    step_size: float | tuple[float, ...] = pytc.nondiff_field()
    keepdims: bool = pytc.nondiff_field()

    def __init__(
        self,
        *,
        accuracy=1,
        step_size=1,
        keepdims=True,
    ):
        """wrap divergence as a layer"""
        self.accuracy = accuracy
        self.step_size = step_size
        self.keepdims = keepdims

        self._func = ft.partial(
            divergence,
            accuracy=self.accuracy,
            step_size=self.step_size,
            keepdims=self.keepdims,
        )

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self._func(x)


@pytc.treeclass
class Laplacian:
    accuracy: int | tuple[int, ...] = pytc.nondiff_field()
    step_size: float | tuple[float, ...] = pytc.nondiff_field()

    def __init__(
        self,
        *,
        accuracy=1,
        step_size=1,
    ):
        """wrap laplacian as a layer"""
        self.accuracy = accuracy
        self.step_size = step_size

        self._func = ft.partial(
            laplacian,
            accuracy=self.accuracy,
            step_size=self.step_size,
        )

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self._func(x)


@pytc.treeclass
class Curl:
    accuracy: int | tuple[int, ...] = pytc.nondiff_field()
    step_size: float | tuple[float, ...] = pytc.nondiff_field()

    def __init__(
        self,
        *,
        accuracy=1,
        step_size=1,
    ):
        """wrap curl as a layer"""
        self.accuracy = accuracy
        self.step_size = step_size

        self._func = ft.partial(
            curl,
            accuracy=self.accuracy,
            step_size=self.step_size,
        )

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return self._func(x)
