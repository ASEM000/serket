from __future__ import annotations

# from serket.nn.blocks import UNetBlock, VGG16Block, VGG19Block

# def test_vgg16_block():
#     block = VGG16Block(3)
#     count, _ = _reduce_count_and_size(block)
#     assert count.real == 14_714_688
#     model = VGG16Block(3)
#     assert model(jnp.ones([3, 224, 224])).shape == (512, 1, 1)


# def test_vgg19_block():
#     block = VGG19Block(3)
#     count, _ = _reduce_count_and_size(block)
#     assert count.real == 20_024_384

#     model = VGG19Block(3)
#     assert model(jnp.ones([3, 224, 224])).shape == (512, 1, 1)


# def test_unet_block():
#     block = UNetBlock(3, 1, init_features=32)
#     count, _ = _reduce_count_and_size(block)
#     assert count.real == 7_757_153

#     model = UNetBlock(3, 1, 2)

#     assert model(jnp.ones((3, 320, 320))).shape == (1, 320, 320)
