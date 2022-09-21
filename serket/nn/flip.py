import jax
import jax.numpy as jnp
import pytreeclass as pytc


@pytc.treeclass
class FlipLeftRight2D:
    """Flip channels left to right."""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 3, "x must be 3D"
        flip = lambda x: jnp.flip(x, axis=1)
        return jax.vmap(flip)(x)


@pytc.treeclass
class FlipUpDown2D:
    """Flip channels up to down."""

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        assert x.ndim == 3, "x must be 3D"
        flip = lambda x: jnp.flip(x, axis=0)
        return jax.vmap(flip)(x)
