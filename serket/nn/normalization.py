from __future__ import annotations

import jax
import jax.numpy as jnp
import pytreeclass as pytc


@pytc.treeclass
class LayerNorm:
    γ: jnp.ndarray = None
    β: jnp.ndarray = None

    ε: float = pytc.nondiff_field()
    affine: bool = pytc.nondiff_field()
    normalized_shape: int | tuple[int] = pytc.nondiff_field()

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        *,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """Layer Normalization
        See: https://nn.labml.ai/normalization/layer_norm/index.html
        transform the input by scaling and shifting to have zero mean and unit variance.

        Args:
            normalized_shape (int | tuple[int,...]): the shape of the input to be normalized.
            eps (float, optional): . Defaults to 1e-5.
            affine (bool, optional): whether to apply affine transformation. Defaults to True.
        """

        self.normalized_shape = (
            normalized_shape
            if isinstance(normalized_shape, tuple)
            else (normalized_shape,)
        )
        self.ε = eps
        self.affine = affine

        if self.affine:
            # make γ and β trainable
            self.γ = jnp.ones(normalized_shape)
            self.β = jnp.zeros(normalized_shape)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:

        dims = tuple(range(len(x.shape) - len(self.normalized_shape), len(x.shape)))

        μ = jnp.mean(x, axis=dims, keepdims=True)
        σ_2 = jnp.var(x, axis=dims, keepdims=True)
        x̂ = (x - μ) * jax.lax.rsqrt((σ_2 + self.ε))

        x̂ = (x̂ * self.γ + self.β) if self.affine else x̂

        return x̂
