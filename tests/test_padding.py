from __future__ import annotations

import jax.numpy as jnp

from serket.nn import Pad1D, Pad2D, Pad3D


def test_padding1d():
    layer = Pad1D(padding=1)
    assert layer(jnp.ones((1, 1))).shape == (1, 3)


def test_padding2d():
    layer = Pad2D(padding=1)
    assert layer(jnp.ones((1, 1, 1))).shape == (1, 3, 3)

    layer = Pad2D(padding=((1, 2), (3, 4)))
    assert layer(jnp.ones((1, 1, 1))).shape == (1, 4, 8)


def test_padding3d():
    layer = Pad3D(padding=1)
    assert layer(jnp.ones((1, 1, 1, 1))).shape == (1, 3, 3, 3)

    layer = Pad3D(padding=((1, 2), (3, 4), (5, 6)))
    assert layer(jnp.ones((1, 1, 1, 1))).shape == (1, 4, 8, 12)
