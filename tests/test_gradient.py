import jax.numpy as jnp
import numpy.testing as npt

from serket import diff
from serket.fd import difference


def test_difference_first_derivative():
    # test against analytical solution
    all_correct = lambda lhs, rhs: npt.assert_allclose(lhs, rhs, atol=1e-4)
    for func in [
        lambda x: x,
        lambda x: x**2,
        lambda x: x + 2,
        lambda x: jnp.sin(x),
        lambda x: jnp.cos(x) * jnp.sin(x),
    ]:
        x = jnp.linspace(0, 1, 100)
        dx = x[1] - x[0]
        y = func(x)
        f1 = jnp.gradient(y) / dx
        f2 = difference(y, step_size=dx)

        all_correct(f1, f2)

    for func in [
        lambda x, y: x + y,
        lambda x, y: x**2 + y**3,
        lambda x, y: x + 2 + y * x,
    ]:
        x, y = [jnp.linspace(0, 1, 100)] * 2
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        f = func(X, Y)
        f1 = jnp.gradient(f, axis=0) / dx
        f2 = difference(f, step_size=dx, axis=0)
        all_correct(f1, f2)

        f1 = jnp.gradient(f, axis=1) / dy
        f2 = difference(f, step_size=dy, axis=1)
        all_correct(f1, f2)


def test_difference_second_derivative():

    x = jnp.linspace(0, 2 * jnp.pi, 100)

    for func in [
        lambda x: x,
        lambda x: x**2,
        lambda x: x + 2,
        lambda x: jnp.sin(x),
        lambda x: jnp.cos(x) * jnp.sin(x),
    ]:

        y = func(x)

        y_xx_fd = difference(
            y, axis=0, accuracy=4, derivative=2, step_size=(x[1] - x[0])
        )
        y_xx_an = (diff(diff(func)))(x)
        npt.assert_allclose(y_xx_fd, y_xx_an, atol=1e-1)


def test_difference_argnum():
    all_correct = lambda lhs, rhs: npt.assert_allclose(lhs, rhs, atol=1e-4)

    for func in [
        lambda x, y: x + y,
        lambda x, y: x**2 + y**3,
        lambda x, y: x + 2 + y * x,
    ]:
        x, y = [jnp.linspace(0, 1, 100)] * 2
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        f = func(X, Y)
        f1 = jnp.gradient(f, axis=0) / dx
        f2 = difference(f, step_size=dx, axis=0)
        all_correct(f1, f2)

        f1 = jnp.gradient(f, axis=1) / dy
        f2 = difference(f, step_size=dy, axis=1)
        all_correct(f1, f2)


def test_difference_dx_dy():

    x, y = [jnp.linspace(0, 1, 100)] * 2
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    for func in [
        lambda x, y: jnp.sin(x) * jnp.cos(y),
        lambda x, y: x**2 + y**2,
        lambda x, y: x**2 + y**2 + x * y,
        lambda x, y: x**2 + y**2 + x * y + x**3 + y**3,
    ]:

        f = func(X, Y)
        f_an = diff(diff(func), argnums=1)(X, Y)  # df/dxdy
        f_ff = difference(f, step_size=dx, axis=0, accuracy=3)
        f_ff = difference(f_ff, step_size=dy, axis=1, accuracy=3)

        npt.assert_allclose(
            f_an,
            f_ff,
            atol=1e-1,
        )


# def test_Gradient():
#     all_correct = lambda lhs, rhs: npt.assert_allclose(lhs, rhs, atol=1e-4)

#     for func in [
#         lambda x, y: x + y,
#         lambda x, y: x**2 + y**3,
#         lambda x, y: x + 2 + y * x,
#     ]:
#         x, y = [jnp.linspace(0, 1, 100)] * 2
#         dx, dy = x[1] - x[0], y[1] - y[0]
#         X, Y = jnp.meshgrid(x, y, indexing="ij")
#         f = func(X, Y)
#         f1 = jnp.gradient(f, axis=0) / dx
#         f2 = Gradient(step_size=dx, axis=0)(f)
#         all_correct(f1, f2)

#         f1 = jnp.gradient(f, axis=1) / dy
#         f2 = Gradient(step_size=dy, axis=1)(f)
#         all_correct(f1, f2)
