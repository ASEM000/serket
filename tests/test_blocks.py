from __future__ import annotations

import jax
import jax.numpy as jnp

from serket.nn.blocks import UNetBlock, VGG16Block, VGG19Block


def count_parameters(model):
    is_array = lambda x: isinstance(x, jax.Array)
    count = 0

    def map_func(leaf):
        nonlocal count
        if is_array(leaf):
            count += leaf.size
        return leaf

    jax.tree_map(map_func, model)
    return count


def test_vgg16_block():
    model = VGG16Block(3)
    assert count_parameters(model) == 14_714_688
    assert model(jnp.ones([3, 224, 224])).shape == (512, 1, 1)


def test_vgg19_block():
    model = VGG19Block(3)
    assert count_parameters(model) == 20_024_384
    assert model(jnp.ones([3, 224, 224])).shape == (512, 1, 1)


def test_unet_block():
    # assert count_parameters(UNetBlock(3, 1, 32)) == 7_757_153
    model = UNetBlock(3, 1, 2)
    assert model(jnp.ones((3, 320, 320))).shape == (1, 320, 320)
