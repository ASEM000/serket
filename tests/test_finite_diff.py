# import jax
import jax.numpy as jnp
import numpy.testing as npt

from serket.fd import curl, difference, divergence, gradient, laplacian

# jax.config.update("jax_enable_x64", True)  # to test for higher accuracy


def test_difference():
    x, y = [jnp.linspace(0, 1, 1000)] * 2
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    F = jnp.sin(X) * jnp.cos(Y)
    dFdX = difference(F, step_size=dx, axis=0, accuracy=3)
    npt.assert_allclose(dFdX, jnp.cos(X) * jnp.cos(Y), rtol=1e-3)

    dFdY = difference(F, step_size=dy, axis=1, accuracy=3)
    npt.assert_allclose(dFdY, -jnp.sin(X) * jnp.sin(Y), atol=1e-3)

    # 1 4 6 3 2
    # left = (4-1) = 3 -> forward difference
    # center = (6-1)/2, (3-4)/2 -> central difference
    # right = (2-3) = -1 -> backward difference
    npt.assert_allclose(
        difference(jnp.array([1, 4, 6, 3, 2]), accuracy=1, axis=0, derivative=1),
        jnp.array([3, 2.5, -0.5, -2, -1]),
    )

    # 1 4 6 3 2 4
    # [-1.5  2.  -0.5] (0, 1, 2)
    # [-0.5  0.   0.5] (-1, 0, 1)
    # [ 0.5 -2.   1.5] (-2, -1, 0)
    npt.assert_allclose(
        difference(jnp.array([1, 4, 6, 3, 2, 4]), accuracy=2, axis=0, derivative=1),
        jnp.array([3.5, 2.5, -0.5, -2.0, 0.5, 3.5]),
    )


def test_divergence():
    x, y = [jnp.linspace(0, 1, 100)] * 2
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    F1 = X**2 + Y**3
    F2 = X**4 + Y**3
    F = jnp.stack([F1, F2], axis=0)  # 2D vector field F = (F1, F2)
    divZ = divergence(F, step_size=(dx, dy), accuracy=7, keepdims=False)
    divZ_true = 2 * X + 3 * Y**2  # (dF1/dx) + (dF2/dy)
    npt.assert_allclose(divZ, divZ_true, atol=5e-4)


def test_gradient():
    x, y = [jnp.linspace(0, 1, 100)] * 2
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    F = X**2 + Y**3
    gradF = gradient(F, step_size=(dx, dy), accuracy=7)
    gradF_true = jnp.stack([2 * X, 3 * Y**2], axis=0)
    npt.assert_allclose(gradF, gradF_true, atol=5e-4)


# def test_laplacian():
#     # needs float64 for higher accuracy
#     x, y = [jnp.linspace(0, 1, 100)] * 2
#     dx, dy = x[1] - x[0], y[1] - y[0]
#     X, Y = jnp.meshgrid(x, y, indexing="ij")
#     Z = X**4 + Y**3
#     laplacianZ = laplacian(Z, step_size=(dx, dy), accuracy=10)
#     laplacianZ_true = 12 * X**2 + 6 * Y
#     npt.assert_allclose(laplacianZ, laplacianZ_true, atol=1e-4)


def test_curl():
    x, y, z = [jnp.linspace(0, 1, 100)] * 3
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    F1 = X**2 + Y**2
    F2 = X**4 + Y**3
    F3 = 0 * Z
    F = jnp.stack([F1, F2, F3], axis=0)
    curl_Z = curl(F, step_size=(dx, dy, dz), accuracy=3)

    F1 = 0 * X
    F2 = 0 * Y
    F3 = 4 * X**3 - 2 * Y
    curl_Z_true = jnp.stack([F1, F2, F3], axis=0)

    npt.assert_allclose(curl_Z, curl_Z_true, atol=1e-2)
