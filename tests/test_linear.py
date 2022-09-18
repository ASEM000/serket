import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytreeclass as pytc

from serket.nn import FNN


def test_linear():
    x = jnp.linspace(0, 1, 100)[:, None]
    y = x**3 + jax.random.uniform(jax.random.PRNGKey(0), (100, 1)) * 0.01

    @jax.value_and_grad
    def loss_func(model, x, y):
        return jnp.mean((model(x) - y) ** 2)

    @jax.jit
    def update(model, x, y):
        value, grad = loss_func(model, x, y)
        return value, model - 1e-3 * grad

    model = FNN([1, 128, 128, 1])

    model = pytc.filter_nondiff(model)
    print(model.tree_diagram())
    for _ in range(20_000):
        value, model = update(model, x, y)

    npt.assert_allclose(jnp.array(4.933563e-05), value, atol=1e-3)
