import jax
import jax.random as jr
import jax.tree_util as jtu
import pytest

from serket.nn.utils import (
    _canonicalize_init_func,
    _canonicalize_input_dilation,
    _canonicalize_kernel,
    _canonicalize_padding,
    _canonicalize_strides,
)


def test_canonicalize_init_func():
    def _check_partial(f):
        return _canonicalize_init_func(f, "test")

    k = jr.PRNGKey(0)

    assert _check_partial("he_normal")(k, (2, 2)).shape == (2, 2)
    assert _check_partial("he_uniform")(k, (2, 2)).shape == (2, 2)
    assert _check_partial("glorot_normal")(k, (2, 2)).shape == (2, 2)
    assert _check_partial("glorot_uniform")(k, (2, 2)).shape == (2, 2)
    assert _check_partial("lecun_normal")(k, (2, 2)).shape == (2, 2)
    assert _check_partial("lecun_uniform")(k, (2, 2)).shape == (2, 2)
    assert _check_partial("normal")(k, (2, 2)).shape == (2, 2)
    assert _check_partial("uniform")(k, (2, 2)).shape == (2, 2)
    assert _check_partial("ones")(k, (2, 2)).shape == (2, 2)
    assert _check_partial("zeros")(k, (2, 2)).shape == (2, 2)
    assert _check_partial("xavier_normal")(k, (2, 2)).shape == (2, 2)
    assert _check_partial("xavier_uniform")(k, (2, 2)).shape == (2, 2)

    assert isinstance(_check_partial(jax.nn.initializers.he_normal()), jtu.Partial)
    assert isinstance(_check_partial(None), type(None))

    with pytest.raises(ValueError):
        _check_partial("invalid")

    with pytest.raises(ValueError):
        _check_partial(1)


def test_canonicalize():
    assert _canonicalize_kernel(3, 2) == (3, 3)
    assert _canonicalize_kernel((3, 3), 2) == (3, 3)
    assert _canonicalize_kernel((3, 3, 3), 3) == (3, 3, 3)

    with pytest.raises(AssertionError):
        _canonicalize_kernel((3, 3), 3)

    with pytest.raises(AssertionError):
        _canonicalize_kernel((3, 3, 3), 2)

    with pytest.raises(AssertionError):
        _canonicalize_kernel((3, 3, 3), 1)

    assert _canonicalize_input_dilation(3, 2) == (3, 3)
    assert _canonicalize_input_dilation((3, 3), 2) == (3, 3)
    assert _canonicalize_input_dilation((3, 3, 3), 3) == (3, 3, 3)

    assert _canonicalize_strides(3, 2) == (3, 3)
    assert _canonicalize_strides((3, 3), 2) == (3, 3)
    assert _canonicalize_strides((3, 3, 3), 3) == (3, 3, 3)


def test_canonicalize_padding():
    assert _canonicalize_padding(1, (3, 3)) == ((1, 1), (1, 1))
    assert _canonicalize_padding(0, (3, 3)) == ((0, 0), (0, 0))
    assert _canonicalize_padding(2, (3, 3)) == ((2, 2), (2, 2))

    assert _canonicalize_padding((1, 1), (3, 3)) == ((1, 1), (1, 1))
    assert _canonicalize_padding(((1, 1), (1, 1)), (3, 3)) == ((1, 1), (1, 1))
    assert _canonicalize_padding(("same", "same"), (3, 3)) == ((1, 1), (1, 1))
    assert _canonicalize_padding(("valid", "valid"), (3, 3)) == ((0, 0), (0, 0))
    with pytest.raises(ValueError):
        _canonicalize_padding(("invalid", "valid"), (3, 3))

    with pytest.raises(ValueError):
        _canonicalize_padding(("valid", "invalid"), (3, 3))

    with pytest.raises(ValueError):
        _canonicalize_padding(("invalid", ()), (3, 3))
