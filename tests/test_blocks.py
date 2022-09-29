from pytreeclass.tree_viz.utils import _reduce_count_and_size

from serket.nn.blocks import VGG16Block, VGG19Block


def test_vgg16_block():
    block = VGG16Block(3)
    count, _ = _reduce_count_and_size(block)
    assert count.real == 14_714_688


def test_vgg19_block():
    block = VGG19Block(3)
    count, _ = _reduce_count_and_size(block)
    assert count.real == 20_024_384
