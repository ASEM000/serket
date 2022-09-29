from __future__ import annotations

import jax.numpy as jnp

from serket.nn import Padding1D, Padding2D, Padding3D


def test_padding1d():
    layer = Padding1D(padding=1)
    assert layer(jnp.ones((1, 1))).shape == (1, 3)


def test_padding2d():
    layer = Padding2D(padding=1)
    assert layer(jnp.ones((1, 1, 1))).shape == (1, 3, 3)

    layer = Padding2D(padding=((1, 2), (3, 4)))
    assert layer(jnp.ones((1, 1, 1))).shape == (1, 4, 8)


def test_padding3d():
    layer = Padding3D(padding=1)
    assert layer(jnp.ones((1, 1, 1, 1))).shape == (1, 3, 3, 3)

    layer = Padding3D(padding=((1, 2), (3, 4), (5, 6)))
    assert layer(jnp.ones((1, 1, 1, 1))).shape == (1, 4, 8, 12)
